"""Start, stop and kill a local collector, Prometheus and Jaeger from their binaries.

Each service runs from its binary with the config beside this script:
``otelcol-contrib`` with ``otelcol.yaml`` (or ``otelcol-x1.yaml`` with
``--x1``), ``prometheus`` with ``prometheus.yml``, and ``jaeger`` with
``jaeger.yaml``. A binary is found on ``PATH``, or named by
``STORMLOG_OTELCOL``, ``STORMLOG_PROMETHEUS`` or ``STORMLOG_JAEGER``; a
missing one is skipped with a message.

Each started service's pid, process start time and command line are kept
in the state directory, and ``stop`` and ``kill`` signal only that process,
and only while its command line matches and its start time is within a
second, which a stepped wall clock can move on Linux: a pid the system has
since reused is never signalled. ``stop`` sends ``SIGTERM``, then
``SIGKILL`` if the service is still running after 10 s. ``kill`` is
``SIGKILL``, for outage episodes::

    python -m examples.observability.local_stack start --x1
    python -m examples.observability.local_stack kill otelcol    # the outage
    python -m examples.observability.local_stack start --x1 --only otelcol
    python -m examples.observability.local_stack status
    python -m examples.observability.local_stack stop
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
DEFAULT_STATE = Path("artifacts") / "observability-stack"
STOP_SECONDS = 10.0
# How far a process's start time may move and still be the same process:
# on Linux it is derived from the boot time, which moves when the wall
# clock is stepped. The command line must match exactly as well.
START_TOLERANCE_SECONDS = 1.0


@dataclass(frozen=True)
class Service:
    name: str
    binary: str
    variable: str

    def command(self, binary: str, *, x1: bool, state: Path) -> list[str]:
        if self.name == "otelcol":
            config = HERE / ("otelcol-x1.yaml" if x1 else "otelcol.yaml")
            return [binary, f"--config={config}"]
        if self.name == "prometheus":
            return [
                binary,
                f"--config.file={HERE / 'prometheus.yml'}",
                f"--storage.tsdb.path={state / 'prometheus-data'}",
                "--web.listen-address=127.0.0.1:9090",
            ]
        return [binary, f"--config={HERE / 'jaeger.yaml'}"]


SERVICES = (
    Service("otelcol", "otelcol-contrib", "STORMLOG_OTELCOL"),
    Service("prometheus", "prometheus", "STORMLOG_PROMETHEUS"),
    Service("jaeger", "jaeger", "STORMLOG_JAEGER"),
)


def _binary(service: Service) -> str | None:
    named = os.environ.get(service.variable)
    if named:
        return named if Path(named).exists() else None
    return shutil.which(service.binary)


def _pid_file(state: Path, name: str) -> Path:
    return state / f"{name}.pid.json"


def _running(state: Path, name: str) -> psutil.Process | None:
    """The service's process, if it still runs as the one this script started."""
    path = _pid_file(state, name)
    try:
        saved = json.loads(path.read_text())
        process = psutil.Process(int(saved["pid"]))
        moved = abs(process.create_time() - float(saved["started"]))
        if moved > START_TOLERANCE_SECONDS or process.cmdline() != saved["cmdline"]:
            return None  # the pid now belongs to another process
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process
    except (OSError, ValueError, KeyError, psutil.Error):
        return None


def start(state: Path, *, x1: bool, only: set[str]) -> int:
    state.mkdir(parents=True, exist_ok=True)
    failed = 0
    for service in SERVICES:
        if only and service.name not in only:
            continue
        if _running(state, service.name) is not None:
            print(f"{service.name}: already running")
            continue
        binary = _binary(service)
        if binary is None:
            print(f"{service.name}: no {service.binary} binary; skipped")
            continue
        with (state / f"{service.name}.log").open("ab") as log:
            process = subprocess.Popen(
                service.command(binary, x1=x1, state=state),
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=state,
                start_new_session=True,
            )
        started = psutil.Process(process.pid)
        _pid_file(state, service.name).write_text(
            json.dumps(
                {
                    "pid": process.pid,
                    "started": started.create_time(),
                    "cmdline": started.cmdline(),
                    "x1": x1,
                }
            )
        )
        time.sleep(0.5)
        if process.poll() is not None:
            print(f"{service.name}: exited at once; see {state / service.name}.log")
            failed += 1
        else:
            print(f"{service.name}: started, pid {process.pid}")
    return 1 if failed else 0


def stop(state: Path, *, only: set[str], kill: bool = False) -> int:
    for service in SERVICES:
        if only and service.name not in only:
            continue
        process = _running(state, service.name)
        if process is None:
            _pid_file(state, service.name).unlink(missing_ok=True)
            continue
        if kill:
            process.send_signal(signal.SIGKILL)
        else:
            process.terminate()
            try:
                process.wait(STOP_SECONDS)
            except psutil.TimeoutExpired:
                process.send_signal(signal.SIGKILL)
        try:
            process.wait(STOP_SECONDS)
        except psutil.Error:
            pass
        _pid_file(state, service.name).unlink(missing_ok=True)
        print(f"{service.name}: {'killed' if kill else 'stopped'}")
    return 0


def status(state: Path) -> int:
    for service in SERVICES:
        process = _running(state, service.name)
        print(
            f"{service.name}: "
            + (f"running, pid {process.pid}" if process else "not running")
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("start", "stop", "kill", "status"))
    parser.add_argument(
        "names", nargs="*", help="Services: otelcol, prometheus, jaeger"
    )
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--x1", action="store_true", help="Use otelcol-x1.yaml.")
    parser.add_argument("--only", default="", help="Comma-separated services.")
    args = parser.parse_args(argv)
    only = {name for name in args.only.split(",") if name} | set(args.names)
    unknown = only - {service.name for service in SERVICES}
    if unknown:
        parser.error(f"unknown services: {', '.join(sorted(unknown))}")
    state = args.state_dir.resolve()
    if args.action == "start":
        return start(state, x1=args.x1, only=only)
    if args.action == "status":
        return status(state)
    return stop(state, only=only, kill=args.action == "kill")


if __name__ == "__main__":
    sys.exit(main())
