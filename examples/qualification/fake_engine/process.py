"""Start the fake engine as a subprocess that signals can reach."""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import IO, Sequence

ROOT = Path(__file__).resolve().parents[3]
MARKER = "FAKE_ENGINE_URL="


class FakeEngineProcess:
    """``with FakeEngineProcess(["--step-seconds", "0.001"]) as server:``.

    ``server.base_url`` and ``server.pid`` are known once it serves. Stopping
    sends SIGCONT first, so a process left stopped still ends. Its output is
    read to the end the whole time, keeping the last lines in ``output_tail``:
    a pipe nobody reads fills with the tracebacks of clients that gave up,
    and the child then blocks on every write, its exit included.
    """

    def __init__(self, args: Sequence[str] = (), *, startup_seconds: float = 30.0):
        self.args = list(args)
        self.startup_seconds = startup_seconds
        self.process: subprocess.Popen[str] | None = None
        self.base_url = ""
        self.pid = 0
        self._reader: _OutputReader | None = None

    def start(self) -> FakeEngineProcess:
        self.process = subprocess.Popen(
            [sys.executable, "-m", "examples.qualification.fake_engine", *self.args],
            cwd=ROOT,
            env=_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert self.process.stdout is not None
        self._reader = _OutputReader(self.process.stdout)
        line = self._reader.marker(self.startup_seconds)
        if line is None:
            self.stop()
            raise RuntimeError("the fake engine did not report its URL in time")
        url, pid = line.split()
        self.base_url = url[len(MARKER) :]
        self.pid = int(pid.split("=", 1)[1])
        return self

    @property
    def output_tail(self) -> list[str]:
        """The child's last lines of output, for a test's diagnostics."""
        return [] if self._reader is None else list(self._reader.tail)

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


def _environment() -> dict[str, str]:
    """The caller's environment, with the checkout first on PYTHONPATH and
    whatever the caller had there kept after it."""
    inherited = os.environ.get("PYTHONPATH")
    path = str(ROOT) if not inherited else os.pathsep.join([str(ROOT), inherited])
    return {**os.environ, "PYTHONPATH": path}


class _OutputReader:
    """Reads a child's output until it ends, handing over the URL line and
    keeping the last ``keep`` lines."""

    def __init__(self, stream: IO[str], keep: int = 200) -> None:
        self.tail: deque[str] = deque(maxlen=keep)
        self._markers: queue.Queue[str] = queue.Queue()
        self._stream = stream
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self._stream:
            self.tail.append(line)
            if line.startswith(MARKER):
                self._markers.put(line.strip())

    def marker(self, timeout: float) -> str | None:
        try:
            return self._markers.get(timeout=timeout)
        except queue.Empty:
            return None


__all__ = ["FakeEngineProcess"]
