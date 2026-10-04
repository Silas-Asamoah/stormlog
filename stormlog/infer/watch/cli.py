"""``stormlog infer watch``: the command line."""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..errors import InferUsageError
from .config import load_watch_config
from .watcher import Watcher, WatchOptions, WatchOutcome

HELP = "Watch a vLLM server and record its incidents"


def add_watch_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("watch", help=HELP, description=HELP)
    parser.add_argument("--root", required=True, help="Watch root directory")
    parser.add_argument(
        "--config", default=None, help="stormlog.infer.watch_config JSON file"
    )
    parser.add_argument("--base-url", default=None, help="The vLLM server's URL")
    parser.add_argument(
        "--metrics-url",
        default=None,
        help="Its /metrics URL (default: the server's origin plus /metrics)",
    )
    parser.add_argument(
        "--interval", type=float, default=None, help="Seconds between ticks"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds (default: until Ctrl+C or SIGTERM)",
    )
    parser.add_argument(
        "--ready-file",
        type=Path,
        default=None,
        help="Written once the first scrape has succeeded",
    )
    parser.add_argument(
        "--test-trigger",
        default=None,
        help="For qualification only: every=SECONDS, or file (fires when "
        "ROOT/test-trigger appears)",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable holding the server's bearer token",
    )


def cmd_watch(args: argparse.Namespace) -> int:
    config = load_watch_config(
        args.config,
        overrides={
            "server.base_url": args.base_url,
            "server.metrics_url": args.metrics_url,
            "tick_seconds": args.interval,
        },
    )
    every, from_file = _test_trigger(args.test_trigger)
    if args.duration is not None and not (
        math.isfinite(args.duration) and args.duration > 0
    ):
        raise InferUsageError("--duration must be a finite number > 0")
    options = WatchOptions(
        duration_seconds=args.duration,
        ready_file=args.ready_file,
        test_trigger_every=every,
        test_trigger_file=from_file,
        api_key=_api_key(args.api_key_env),
    )
    watcher = Watcher(config, Path(args.root), options=options)
    # The watch leaves SIGINT and SIGTERM ignored: the process is about to
    # exit with the code its report holds, and a late signal must not turn
    # that into an interrupt.
    outcome = asyncio.run(_run(watcher))
    if outcome.report_path is not None:
        print(f"Watch report: {outcome.report_path}")
    if outcome.unsound:
        print(f"Watch unsound: {', '.join(outcome.unsound)}", file=sys.stderr)
    return outcome.exit_code


_SIGNALS = (signal.SIGINT, signal.SIGTERM)


async def _run(watcher: Watcher) -> WatchOutcome:
    """Run until done: the first SIGINT or SIGTERM is the documented end, a
    second cuts the shutdown short, and a third is the default interrupt.
    Once the watch has returned, both are ignored for good, so the exit code
    stays the one the report holds."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed: list[signal.Signals] = []
    received: list[int] = []

    def on_signal(signum: int) -> None:
        received.append(signum)
        if len(received) == 1:
            stop.set()
            return
        watcher.hurry()
        for installed_signum in installed:  # a third is the default interrupt
            loop.remove_signal_handler(installed_signum)

    for signum in _SIGNALS:
        try:
            loop.add_signal_handler(signum, on_signal, signum)
            installed.append(signum)
        except (NotImplementedError, RuntimeError):
            pass
    try:
        return await watcher.run(stop)
    finally:
        _ignore(loop, installed)


def _ignore(
    loop: asyncio.AbstractEventLoop, installed: Sequence[signal.Signals]
) -> None:
    """Swap the loop's handlers for SIG_IGN with the signals blocked, so none
    can arrive in between and find the default interrupt."""
    block = getattr(signal, "pthread_sigmask", None)
    if block is not None:
        block(signal.SIG_BLOCK, installed)
    try:
        for signum in installed:
            loop.remove_signal_handler(signum)
            signal.signal(signum, signal.SIG_IGN)
    finally:
        if block is not None:
            block(signal.SIG_UNBLOCK, installed)


def _api_key(name: str | None) -> str | None:
    if name is None:
        return None
    value = os.environ.get(name)
    if not value:
        raise InferUsageError(f"--api-key-env: {name} is not set")
    return value


def _test_trigger(value: str | None) -> tuple[float | None, bool]:
    if value is None:
        return None, False
    if value == "file":
        return None, True
    prefix = "every="
    if value.startswith(prefix):
        try:
            seconds = float(value[len(prefix) :])
        except ValueError:
            seconds = 0.0
        if math.isfinite(seconds) and seconds > 0:
            return seconds, False
    raise InferUsageError("--test-trigger must be every=SECONDS or file")


__all__ = ["add_watch_parser", "cmd_watch"]
