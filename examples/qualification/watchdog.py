"""The pulse watchdog: continues a target its harness can no longer continue.

The pulser starts it in its own session, so a signal to the harness's process
group (Ctrl+C, a job's SIGTERM, an ssh disconnect's SIGHUP) never reaches it,
and it ignores those signals anyway. It reads a pipe from the harness: end of
file means the harness is gone, whatever happened to its pid, and the target
is continued at once. It also continues a target stopped well past the
longest pulse. It says ``ready`` once it watches, and the pulser waits for
that before its first stop.

    python -m examples.qualification.watchdog --pid PID --start-time T --limit S
"""

from __future__ import annotations

import argparse
import os
import select
import signal
import sys
import time
from typing import Sequence

from .pulser import Target

POLL_SECONDS = 0.05


def main(argv: Sequence[str] | None = None) -> int:
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, signal.SIG_IGN)
    args = _parser().parse_args(argv)
    target = Target(args.pid, args.start_time)
    print("ready", flush=True)
    watch(target, args.limit, sys.stdin.fileno())
    return 0


def watch(target: Target, limit_seconds: float, harness_pipe: int) -> None:
    """Until the target is gone: continue it when the harness's pipe closes
    (then stop watching), or when it has been stopped past the limit."""
    stopped_since: float | None = None
    while target.is_alive():
        if _closed(harness_pipe):
            _continue(target)
            return
        if not target.is_stopped():
            stopped_since = None
            continue
        stopped_since = stopped_since or time.monotonic()
        if time.monotonic() - stopped_since > limit_seconds:
            _continue(target)
            stopped_since = None


def _closed(pipe: int) -> bool:
    """Whether the harness closed its end, waiting at most one poll."""
    readable, _w, _x = select.select([pipe], [], [], POLL_SECONDS)
    return bool(readable) and os.read(pipe, 4096) == b""


def _continue(target: Target) -> None:
    if target.is_alive():
        os.kill(target.pid, signal.SIGCONT)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m examples.qualification.watchdog")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--start-time", type=float, required=True)
    parser.add_argument("--limit", type=float, required=True)
    return parser


if __name__ == "__main__":
    sys.exit(main())
