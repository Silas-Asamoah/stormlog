"""The victim: an ``infer profile`` run with three probes in its own process.

``python -m examples.qualification.victim --probes DIR -- <infer profile
arguments>`` runs the profile exactly as ``stormlog infer profile`` would,
and adds (#221 design A.2):

- **phase markers**: ``DIR/markers/<case>_<phase>_<event>.json``, written as
  each phase starts and ends, so the harness times its episodes against
  the measured window while it runs;
- **the append-time probe**: each artifact line's index and the time its
  append was flushed, in ``DIR/append-times.jsonl``. A client record is
  analyzer-ready then, which the replay needs (R4);
- **the client idle probe**: gaps of more than 20 ms in a 10 ms timer, in
  ``DIR/client-idle.jsonl``, so a stall on the client's own host is seen.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

from stormlog.infer import events
from stormlog.infer.cli import (
    _profile_config,
    _validate_profile_arguments,
    build_parser,
)
from stormlog.infer.profile import InferenceProfiler, PhaseEvent

IDLE_TICK_SECONDS = 0.01
IDLE_GAP_SECONDS = 0.02


class Markers:
    """Phase events, each in its own file, written whole."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def write(self, event: PhaseEvent) -> None:
        name = f"{event.case_id}_{event.phase}_{event.event}.json"
        partial = self.directory / f".{name}.tmp"
        partial.write_text(json.dumps(asdict(event), sort_keys=True), encoding="utf-8")
        os.replace(partial, self.directory / name)


def read_marker(directory: Path, phase: str, event: str) -> dict[str, Any] | None:
    """The first marker for ``phase`` and ``event``, if written yet."""
    for path in sorted(directory.glob(f"*_{phase}_{event}.json")):
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded
    return None


class AppendProbe:
    """Notes each artifact line's index and when its append was flushed."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self._lines: dict[Path, int] = {}
        self._original: Callable[..., None] | None = None
        self._handle = out.open("a", encoding="utf-8")

    def install(self) -> None:
        original = events.JsonlEventWriter.append
        self._original = original
        probe = self

        def append(
            self: events.JsonlEventWriter,
            record: dict[str, Any],
            *args: Any,
            **kwargs: Any,
        ) -> None:
            # Whatever else append takes (#220 adds an argument) passes on.
            original(self, record, *args, **kwargs)
            probe.note(self.path, record)

        events.JsonlEventWriter.append = append  # type: ignore[method-assign]

    def note(self, path: Path, record: dict[str, Any]) -> None:
        flushed = time.time_ns()
        index = self._lines.get(path)
        if index is None:
            index = len(path.read_bytes().splitlines()) - 1
        self._lines[path] = index + 1
        line = {
            "line": index,
            "appended_ns": flushed,
            "event_type": record.get("event_type"),
        }
        self._handle.write(json.dumps(line, sort_keys=True) + "\n")
        self._handle.flush()

    def uninstall(self) -> None:
        if self._original is not None:
            events.JsonlEventWriter.append = self._original  # type: ignore[method-assign]
        self._handle.close()


class IdleProbe:
    """A 10 ms timer on its own thread; a tick that comes 20 ms late or more
    means the client's host stalled."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="client-idle", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        with self.out.open("a", encoding="utf-8") as handle:
            while not self._stop.is_set():
                before = time.monotonic()
                time.sleep(IDLE_TICK_SECONDS)
                late = time.monotonic() - before - IDLE_TICK_SECONDS
                if late >= IDLE_GAP_SECONDS:
                    record = {"at_ns": time.time_ns(), "late_ms": late * 1000}
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()


def run(probes: Path, profile_arguments: Sequence[str]) -> dict[str, Any]:
    """Run the victim's profile with its probes; return its report."""
    args = build_parser().parse_args(["profile", *profile_arguments])
    _validate_profile_arguments(args)
    probes.mkdir(parents=True, exist_ok=True)
    markers = Markers(probes / "markers")
    appends = AppendProbe(probes / "append-times.jsonl")
    idle = IdleProbe(probes / "client-idle.jsonl")
    appends.install()
    idle.start()
    try:
        profiler = InferenceProfiler(_profile_config(args), on_phase=markers.write)
        return profiler.run()
    finally:
        idle.stop()
        appends.uninstall()


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3 or argv[0] != "--probes" or argv[2] != "--":
        print(
            "usage: python -m examples.qualification.victim --probes DIR -- "
            "<infer profile arguments>",
            file=sys.stderr,
        )
        return 2
    run(Path(argv[1]), argv[3:])
    return 0


__all__ = ["AppendProbe", "IdleProbe", "Markers", "main", "read_marker", "run"]


if __name__ == "__main__":
    sys.exit(main())
