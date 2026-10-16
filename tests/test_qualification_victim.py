"""The victim's profile and its three probes, against the fake vLLM engine."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from examples.qualification.fake_engine import FakeEngine, FakeEngineConfig
from examples.qualification.fake_engine.process import _environment
from examples.qualification.victim import (
    SENDS,
    AppendProbe,
    main,
    read_marker,
    run,
)
from stormlog.infer.correlation_events import ArtifactIdentityEvent, CorrelationContext
from stormlog.infer.events import JsonlEventWriter


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
    # Every line flushed through the event writer, in order, with its own
    # event type. Final summary/state rows are appended separately.
    assert [a["line"] for a in appends] == list(range(len(appends)))
    for append in appends:
        record = json.loads(lines[append["line"]])
        assert record.get("event_type") == append["event_type"]
    assert all(
        b["appended_ns"] >= a["appended_ns"] for a, b in zip(appends, appends[1:])
    )
    assert (probes / "client-idle.jsonl").exists()
    # Each request's send, noted before its answer: in flight from then.
    sends = {
        send["x_request_id"]: send["sent_ns"]
        for send in map(json.loads, (probes / SENDS).read_text().splitlines())
    }
    # The request records only: #218's dispatch and first-content records
    # carry the same ID.
    requests = [
        r
        for r in map(json.loads, lines)
        if r.get("event_type") == "infer.request" and r.get("x_request_id")
    ]
    assert sorted(sends) == sorted(r["x_request_id"] for r in requests)
    for request in requests:
        assert sends[request["x_request_id"]] <= request["started_at_ns"]


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


def test_the_append_probe_tracks_new_and_reused_contexts(tmp_path: Path) -> None:
    context = CorrelationContext(
        run_id="run", session_id="session", producer_id="client", source="test",
        clock_domain="host/wall", clock_kind="wall", collection_mode="active",
        provenance="observed",
    )  # fmt: skip
    event = ArtifactIdentityEvent(
        context=context,
        event_id="first",
        artifact_kind="inference_jsonl",
        created_at_ns=1,
    )
    path = tmp_path / "artifact.jsonl"
    timings = tmp_path / "append-times.jsonl"
    probe = AppendProbe(timings)
    probe.install()
    try:
        with JsonlEventWriter(path) as writer:
            writer.append(event.to_record())
            writer.append(replace(event, event_id="same-context").to_record())
            writer.append(
                replace(
                    event,
                    event_id="new-context",
                    context=replace(context, run_id="other"),
                ).to_record()
            )
            writer.append({"event_type": "legacy", "schema_version": 1})
    finally:
        probe.uninstall()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    appends = [json.loads(line) for line in timings.read_text().splitlines()]
    assert len(rows) == len(appends) == 6
    assert [a["line"] for a in appends] == list(range(6))
    assert [a["event_type"] for a in appends] == [r["event_type"] for r in rows]
    assert appends[0]["appended_ns"] == appends[1]["appended_ns"]
    assert appends[3]["appended_ns"] == appends[4]["appended_ns"]
