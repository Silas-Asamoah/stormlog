"""A profile tells its caller when each phase's arrivals start and end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.qualification.fake_engine import FakeEngine, FakeEngineConfig
from stormlog.infer.config import ProfileConfig
from stormlog.infer.profile import InferenceProfiler, PhaseEvent


def _config(engine: FakeEngine, output: Path, **changes: object) -> ProfileConfig:
    values: dict[str, object] = {
        "endpoint": engine.endpoint,
        "model": engine.config.model,
        "concurrency": (2,),
        "input_tokens": (8,),
        "output_tokens": (2,),
        "output_path": str(output),
        "request_count": 4,
        "tokenizer": "none",
        "system_sampler": "none",
        "run_id": "run-1",
        "prompt_mode": "unique",
    }
    values.update(changes)
    return ProfileConfig(**values)  # type: ignore[arg-type]


def _phase_windows(output: Path) -> dict[str, dict[str, object]]:
    records = [json.loads(line) for line in output.read_text().splitlines() if line]
    return {
        record["phase"]: record
        for record in records
        if record.get("event_type") == "infer.phase_window"
    }


def test_each_phase_reports_its_start_and_end_in_order(tmp_path: Path) -> None:
    output = tmp_path / "infer.jsonl"
    events: list[PhaseEvent] = []
    with FakeEngine(FakeEngineConfig(step_seconds=0.001)) as engine:
        config = _config(engine, output, warmup_requests=2)
        InferenceProfiler(config, on_phase=events.append).run()
    assert [(e.phase, e.event) for e in events] == [
        ("warmup", "started"),
        ("warmup", "ended"),
        ("measured", "started"),
        ("measured", "ended"),
    ]
    windows = _phase_windows(output)
    for event in events:
        window = windows[event.phase]
        assert event.case_id == window["case_id"]
        if event.event == "started":
            # Told just before the first arrival is dispatched.
            assert event.at_ns <= window["started_at_ns"]  # type: ignore[operator]
        else:
            assert event.window == {
                key: window[key]
                for key in ("started_at_ns", "window_ended_at_ns", "drained_at_ns")
            }


def test_a_failing_callback_stops_the_profile(tmp_path: Path) -> None:
    # The caller relies on the events, so losing one is not silent.
    def refuse(event: PhaseEvent) -> None:
        raise RuntimeError("marker not written")

    with FakeEngine(FakeEngineConfig(step_seconds=0.001)) as engine:
        config = _config(engine, tmp_path / "infer.jsonl")
        with pytest.raises(RuntimeError, match="marker not written"):
            InferenceProfiler(config, on_phase=refuse).run()
