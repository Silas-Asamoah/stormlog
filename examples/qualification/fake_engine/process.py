"""Start the fake engine as a subprocess that signals can reach."""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import IO, Sequence

ROOT = Path(__file__).resolve().parents[3]
MARKER = "FAKE_ENGINE_URL="


class FakeEngineProcess:
    """``with FakeEngineProcess(["--step-seconds", "0.001"]) as server:``.

    ``server.base_url`` and ``server.pid`` are known once it serves. Stopping
    sends SIGCONT first, so a process left stopped still ends.
    """

    def __init__(self, args: Sequence[str] = (), *, startup_seconds: float = 30.0):
        self.args = list(args)
        self.startup_seconds = startup_seconds
        self.process: subprocess.Popen[str] | None = None
        self.base_url = ""
        self.pid = 0

    def start(self) -> FakeEngineProcess:
        environment = {**os.environ, "PYTHONPATH": str(ROOT)}
        self.process = subprocess.Popen(
            [sys.executable, "-m", "examples.qualification.fake_engine", *self.args],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert self.process.stdout is not None
        line = _first_marker(self.process.stdout, self.startup_seconds)
        if line is None:
            self.stop()
            raise RuntimeError("the fake engine did not report its URL in time")
        url, pid = line.split()
        self.base_url = url[len(MARKER) :]
        self.pid = int(pid.split("=", 1)[1])
        return self

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def stop(self, timeout: float = 10.0) -> int | None:
        process = self.process
        if process is None or process.poll() is not None:
            return None if process is None else process.returncode
        if hasattr(signal, "SIGCONT"):
            os.kill(process.pid, signal.SIGCONT)
        process.terminate()
        try:
            return process.wait(timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait(timeout)

    def __enter__(self) -> FakeEngineProcess:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()


def _first_marker(stream: IO[str], timeout: float) -> str | None:
    lines: queue.Queue[str] = queue.Queue()

    def reader() -> None:
        for line in stream:
            lines.put(line)
            if line.startswith(MARKER):
                return

    threading.Thread(target=reader, daemon=True).start()
    while True:
        try:
            line = lines.get(timeout=timeout)
        except queue.Empty:
            return None
        if line.startswith(MARKER):
            return line.strip()


__all__ = ["FakeEngineProcess"]
