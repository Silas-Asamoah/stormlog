"""The victim's profile and its three probes, against the fake vLLM engine."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from examples.qualification.fake_engine import FakeEngine, FakeEngineConfig
from examples.qualification.fake_engine.process import _environment
from examples.qualification.victim import AppendProbe, main, read_marker, run


def _arguments(engine: FakeEngine, output: Path) -> list[str]:
    return [
        "--endpoint", engine.endpoint,
        "--model", engine.config.model,
        "--concurrency", "1",
        "--input-tokens", "8",
        "--output-tokens", "2",
        "--requests", "3",
        "--output", str(output),
        "--tokenizer", "none",
        "--system-sampler", "none",
        "--run-id", "victim",
    ]  # fmt: skip


def test_the_victim_marks_its_phases_and_times_each_append(tmp_path: Path) -> None:
    probes, output = tmp_path / "probes", tmp_path / "victim.jsonl"
    with FakeEngine(FakeEngineConfig(step_seconds=0.001)) as engine:
        run(probes, _arguments(engine, output))
    started = read_marker(probes / "markers", "measured", "started")
    ended = read_marker(probes / "markers", "measured", "ended")
    assert started is not None and ended is not None
    assert started["at_ns"] < ended["at_ns"]
    lines = output.read_text().splitlines()
    appends = [
        json.loads(line)
        for line in (probes / "append-times.jsonl").read_text().splitlines()
    ]
    # Every line the run appended, in order, with its own event type.
    assert [a["line"] for a in appends] == list(range(len(appends)))
    for append in appends:
        record = json.loads(lines[append["line"]])
        assert record.get("event_type") == append["event_type"]
    assert all(
        b["appended_ns"] >= a["appended_ns"] for a, b in zip(appends, appends[1:])
    )
    assert (probes / "client-idle.jsonl").exists()


def test_the_victim_runs_as_its_own_process(tmp_path: Path) -> None:
    probes, output = tmp_path / "probes", tmp_path / "victim.jsonl"
    with FakeEngine(FakeEngineConfig(step_seconds=0.001)) as engine:
        command = [
            sys.executable, "-m", "examples.qualification.victim",
            "--probes", str(probes), "--", *_arguments(engine, output),
        ]  # fmt: skip
        finished = subprocess.run(
            command, env=_environment(), capture_output=True, timeout=60
        )
    assert finished.returncode == 0, finished.stderr.decode()
    assert read_marker(probes / "markers", "measured", "ended") is not None


def test_a_bad_command_line_is_refused() -> None:
    assert main(["--probes", "x"]) == 2


def test_the_append_probe_passes_on_whatever_append_takes(tmp_path: Path) -> None:
    # #220's Prometheus export adds an argument to JsonlEventWriter.append;
    # the probe wraps it without knowing its signature.
    from stormlog.infer import events

    seen: list[tuple[Any, ...]] = []
    real = events.JsonlEventWriter.append

    def append(self: Any, record: dict[str, Any], *extra: Any, **named: Any) -> None:
        seen.append((extra, named))
        real(self, record)

    events.JsonlEventWriter.append = append  # type: ignore[method-assign]
    probe = AppendProbe(tmp_path / "append-times.jsonl")
    try:
        probe.install()
        with events.JsonlEventWriter(tmp_path / "artifact.jsonl") as writer:
            writer.append({"event_type": "x"}, {"extra": 1}, flag=True)  # type: ignore[call-arg]
    finally:
        probe.uninstall()
        events.JsonlEventWriter.append = real  # type: ignore[method-assign]
    assert seen == [(({"extra": 1},), {"flag": True})]
    assert (tmp_path / "append-times.jsonl").read_text().count("\n") == 1
