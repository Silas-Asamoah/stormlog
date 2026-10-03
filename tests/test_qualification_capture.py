"""I1's profiler window, against the fake vLLM engine."""

from __future__ import annotations

from pathlib import Path

from examples.qualification.capture import capture_window
from examples.qualification.fake_engine import FakeEngine, FakeEngineConfig


def test_a_window_stamps_both_calls_and_its_stop_pauses(tmp_path: Path) -> None:
    config = FakeEngineConfig(step_seconds=0.001, trace_dir=tmp_path / "traces")
    with FakeEngine(config) as engine:
        engine.controls.stop_pause_seconds = 0.2
        window = capture_window(engine.base_url, 0.1)
    assert window.ok
    assert window.stop is not None
    assert window.start.returned_ns <= window.stop.requested_ns
    assert window.stop.returned_ns - window.stop.requested_ns >= 200_000_000
    record = window.to_record()
    assert record["stop"]["status"] == 200


def test_a_refused_start_is_never_stopped(tmp_path: Path) -> None:
    with FakeEngine(FakeEngineConfig(step_seconds=0.001)) as engine:
        window = capture_window(engine.base_url, 0.1)
    assert (window.start.status, window.stop, window.ok) == (404, None, False)
