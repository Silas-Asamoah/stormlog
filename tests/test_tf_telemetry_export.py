"""Telemetry export tests for TensorFlow tracking paths."""

from __future__ import annotations

import json
import time
from argparse import Namespace
from pathlib import Path

import pytest

import tfmemprof.cli as tf_cli
import tfmemprof.tracker as tf_tracker
from gpumemprof.telemetry import validate_telemetry_record


def test_tf_tracker_emits_v2_event_records(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tf_tracker, "TF_AVAILABLE", True)

    tracker = tf_tracker.MemoryTracker(
        sampling_interval=0.01,
        device="/GPU:0",
        enable_logging=False,
    )
    monkeypatch.setattr(tracker, "_get_current_memory", lambda: 32.0)

    tracker.start_tracking()
    time.sleep(0.03)
    result = tracker.stop_tracking()

    assert result.events
    first = result.events[0]
    assert first["schema_version"] == 2
    assert first["collector"] == "tfmemprof.memory_tracker"
    validate_telemetry_record(first)


def test_tf_cli_track_output_normalizes_legacy_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tf_cli, "TF_AVAILABLE", True)

    class _FakeResult:
        peak_memory = 2.0
        average_memory = 2.0
        duration = 1.0
        memory_usage = [2.0]
        timestamps = [1700000000.0]
        alerts_triggered: list[object] = []
        events = [
            {
                "timestamp": 1700000000.0,
                "type": "sample",
                "memory_mb": 2.0,
                "device": "/GPU:0",
            }
        ]

    class _FakeTracker:
        def __init__(self, *args: object, **kwargs: object) -> None:
            _ = args
            _ = kwargs

        def add_alert_callback(self, callback: object) -> None:
            _ = callback

        def start_tracking(self) -> None:
            return None

        def get_current_memory(self) -> float:
            return 2.0

        def stop_tracking(self) -> "_FakeResult":
            return _FakeResult()

    def _interrupt(_: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(tf_cli, "MemoryTracker", _FakeTracker)
    monkeypatch.setattr(tf_cli.time, "sleep", _interrupt)

    output = tmp_path / "tf_track.json"
    args = Namespace(
        interval=0.25,
        threshold=4000,
        device="/GPU:0",
        output=str(output),
    )

    exit_code = tf_cli.cmd_track(args)
    assert exit_code == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["events"]
    event = payload["events"][0]
    assert event["schema_version"] == 2
    assert event["collector"] == "tfmemprof.memory_tracker"
    validate_telemetry_record(event)
