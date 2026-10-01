"""Telemetry export tests for TensorFlow tracking paths."""

from __future__ import annotations

import json
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterator, cast

import pytest

import stormlog.tensorflow.cli as tf_cli
import stormlog.tensorflow.tracker as tf_tracker
from stormlog.collector_health import COLLECTOR_HEALTH_UNHEALTHY
from stormlog.phases import parse_phase_boundary
from stormlog.session import create_session_summary
from stormlog.telemetry import validate_telemetry_record
from stormlog.telemetry_sink import TelemetrySinkConfig


def _wait_until_events(
    tracker: tf_tracker.MemoryTracker, *, timeout: float = 1.0, interval: float = 0.01
) -> bool:
    """Wait until at least one event is collected or timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if tracker.events:
            return True
        time.sleep(interval)
    return bool(tracker.events)


class _NoOpThread:
    def __init__(self, *args: object, **kwargs: object) -> None:
        _ = args
        self.daemon = bool(kwargs.get("daemon", False))

    def start(self) -> None:
        return None

    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float | None = None) -> None:
        _ = timeout


class _FailingSink:
    def __init__(self, *, fail_on: set[str]) -> None:
        self.fail_on = fail_on
        self.append_calls = 0
        self.flush_calls = 0
        self.close_calls = 0

    def append(self, record: dict[str, object]) -> None:
        _ = record
        self.append_calls += 1
        if "append" in self.fail_on:
            raise OSError("disk full")

    def flush(self, *, force: bool = False) -> None:
        _ = force
        self.flush_calls += 1
        if "flush" in self.fail_on:
            raise OSError("disk full")

    def close(self) -> None:
        self.close_calls += 1
        if "close" in self.fail_on:
            raise OSError("disk full")


def test_tf_main_parses_wandb_track_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_cmd_track(args: object) -> int:
        captured["args"] = args
        return 0

    monkeypatch.setattr(tf_cli, "cmd_track", _fake_cmd_track)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tfmemprof",
            "track",
            "--output",
            "tf_track.json",
            "--wandb",
            "--wandb-project",
            "stormlog-tests",
            "--wandb-entity",
            "team",
            "--wandb-mode",
            "offline",
            "--wandb-run-id",
            "run-123",
            "--wandb-name",
            "tf smoke",
            "--wandb-group",
            "job-42",
            "--wandb-job-type",
            "tf-track",
            "--wandb-log-artifacts",
            "--wandb-log-attribution",
        ],
    )

    assert tf_cli.main() == 0

    args = captured["args"]
    assert args.wandb is True
    assert args.wandb_project == "stormlog-tests"
    assert args.wandb_entity == "team"
    assert args.wandb_mode == "offline"
    assert args.wandb_run_id == "run-123"
    assert args.wandb_name == "tf smoke"
    assert args.wandb_group == "job-42"
    assert args.wandb_job_type == "tf-track"
    assert args.wandb_log_artifacts is True
    assert args.wandb_log_attribution is True


def test_tf_tracker_emits_v3_event_records(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tf_tracker, "TF_AVAILABLE", True)

    tracker = tf_tracker.MemoryTracker(
        sampling_interval=0.01,
        device="/GPU:0",
        enable_logging=False,
    )
    monkeypatch.setattr(tracker, "_get_current_memory", lambda: 32.0)

    tracker.start_tracking()
    assert _wait_until_events(tracker), "tracker did not emit an event before timeout"
    result = tracker.get_tracking_results()

    assert result.events
    first = result.events[0]
    assert first["schema_version"] == 4
    assert isinstance(first["session_id"], str)
    assert first["session_id"]
    assert first["collector"] == "stormlog.tensorflow.memory_tracker"
    assert first["job_id"] is None
    assert first["rank"] == 0
    assert first["local_rank"] == 0
    assert first["world_size"] == 1
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

        def get_statistics(self) -> dict[str, object]:
            return {
                "current_memory_mb": 2.0,
                "collector_health_status": "healthy",
            }

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
    assert event["schema_version"] == 4
    assert isinstance(event["session_id"], str)
    assert event["session_id"]
    assert event["collector"] == "stormlog.tensorflow.memory_tracker"
    validate_telemetry_record(event)


def test_tf_cli_track_passes_distributed_identity_to_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tf_cli, "TF_AVAILABLE", True)
    created: dict[str, object] = {}

    class _FakeResult:
        peak_memory = 0.0
        average_memory = 0.0
        duration = 0.0
        memory_usage: list[float] = []
        timestamps: list[float] = []
        alerts_triggered: list[object] = []
        events: list[dict[str, object]] = []

    class _FakeTracker:
        def __init__(self, *args: object, **kwargs: object) -> None:
            _ = args
            created.update(kwargs)

        def add_alert_callback(self, callback: object) -> None:
            _ = callback

        def start_tracking(self) -> None:
            return None

        def get_current_memory(self) -> float:
            return 0.0

        def get_statistics(self) -> dict[str, object]:
            return {
                "current_memory_mb": 0.0,
                "collector_health_status": "healthy",
            }

        def stop_tracking(self) -> "_FakeResult":
            return _FakeResult()

    def _interrupt(_: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(tf_cli, "MemoryTracker", _FakeTracker)
    monkeypatch.setattr(tf_cli.time, "sleep", _interrupt)

    exit_code = tf_cli.cmd_track(
        Namespace(
            interval=0.25,
            threshold=4000,
            device="/GPU:0",
            output=None,
            job_id="train-42",
            rank=3,
            local_rank=1,
            world_size=8,
        )
    )

    assert exit_code == 0
    assert created["job_id"] == "train-42"
    assert created["rank"] == 3
    assert created["local_rank"] == 1
    assert created["world_size"] == 8


def test_tf_cli_track_exports_results_to_wandb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tf_cli, "TF_AVAILABLE", True)
    exported: dict[str, Any] = {}
    wandb_config = Namespace(enabled=True)
    session_summary = create_session_summary(
        source="stormlog.tensorflow.memory_tracker",
        session_id="tf-session-12345678",
        job_id="job-42",
    )

    class _FakeResult:
        peak_memory = 2.0
        average_memory = 2.0
        duration = 1.0
        memory_usage = [2.0]
        timestamps = [1700000000.0]
        alerts_triggered: list[object] = []
        events = [{"event_type": "warning", "context": "memory high"}]

    class _FakeTracker:
        def __init__(self, *args: object, **kwargs: object) -> None:
            _ = args, kwargs

        def add_alert_callback(self, callback: object) -> None:
            _ = callback

        def start_tracking(self) -> None:
            return None

        def get_current_memory(self) -> float:
            return 2.0

        def get_statistics(self) -> dict[str, object]:
            return {
                "current_memory_mb": 2.0,
                "collector_health_status": "healthy",
                "total_events": 1,
            }

        def get_session_summary(self) -> object:
            return session_summary

        def stop_tracking(self) -> "_FakeResult":
            return _FakeResult()

    monkeypatch.setattr(tf_cli, "MemoryTracker", _FakeTracker)
    monkeypatch.setattr(
        tf_cli.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt)
    )
    monkeypatch.setattr(
        tf_cli, "wandb_config_from_namespace", lambda args: wandb_config
    )
    monkeypatch.setattr(
        tf_cli,
        "ensure_wandb_available",
        lambda config: exported.setdefault("ensured", config),
    )
    monkeypatch.setattr(
        tf_cli,
        "export_tracking_run_to_wandb",
        lambda config, **kwargs: exported.update(config=config, kwargs=kwargs),
    )

    exit_code = tf_cli.cmd_track(
        Namespace(
            interval=0.25,
            threshold=4000,
            device="/GPU:0",
            output=None,
            job_id="job-42",
            rank=1,
            local_rank=0,
            world_size=2,
            telemetry_sink_dir="tf_sink",
            telemetry_flush_seconds=2.0,
            telemetry_rollover_mb=64,
            telemetry_retention_files=8,
            telemetry_retention_total_mb=512,
            wandb=True,
        )
    )

    assert exit_code == 0
    assert exported["ensured"] is wandb_config
    assert exported["config"] is wandb_config
    assert exported["kwargs"]["command_name"] == "tfmemprof-track"
    assert exported["kwargs"]["session_summary"] == session_summary
    assert exported["kwargs"]["stats"]["total_events"] == 1
    assert exported["kwargs"]["events"] == [
        {"event_type": "warning", "context": "memory high"}
    ]
    assert exported["kwargs"]["output_path"] is None
    assert exported["kwargs"]["telemetry_sink_dir"] == "tf_sink"
    assert exported["kwargs"]["oom_dump_path"] is None


def test_tf_cli_track_passes_telemetry_sink_config_to_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tf_cli, "TF_AVAILABLE", True)
    created: dict[str, object] = {}

    class _FakeResult:
        peak_memory = 0.0
        average_memory = 0.0
        duration = 0.0
        memory_usage: list[float] = []
        timestamps: list[float] = []
        alerts_triggered: list[object] = []
        events: list[dict[str, object]] = []

    class _FakeTracker:
        def __init__(self, *args: object, **kwargs: object) -> None:
            _ = args
            created.update(kwargs)

        def add_alert_callback(self, callback: object) -> None:
            _ = callback

        def start_tracking(self) -> None:
            return None

        def get_current_memory(self) -> float:
            return 0.0

        def get_statistics(self) -> dict[str, object]:
            return {
                "current_memory_mb": 0.0,
                "collector_health_status": "healthy",
            }

        def stop_tracking(self) -> "_FakeResult":
            return _FakeResult()

    monkeypatch.setattr(tf_cli, "MemoryTracker", _FakeTracker)
    monkeypatch.setattr(
        tf_cli.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt)
    )

    exit_code = tf_cli.cmd_track(
        Namespace(
            interval=0.25,
            threshold=4000,
            device="/GPU:0",
            output="ignored.json",
            job_id=None,
            rank=None,
            local_rank=None,
            world_size=None,
            telemetry_sink_dir="tf_sink",
            telemetry_flush_seconds=4.0,
            telemetry_rollover_mb=12,
            telemetry_retention_files=5,
            telemetry_retention_total_mb=96,
        )
    )

    assert exit_code == 0
    telemetry_sink_config = cast(TelemetrySinkConfig, created["telemetry_sink_config"])
    assert telemetry_sink_config.root_dir == tf_cli.Path("tf_sink")
    assert telemetry_sink_config.flush_every_seconds == 4.0
    assert telemetry_sink_config.rollover_max_bytes == 12 * 1024 * 1024
    assert telemetry_sink_config.retention_max_files == 5
    assert telemetry_sink_config.retention_max_total_bytes == 96 * 1024 * 1024


def test_tf_tracker_records_degrade_and_recover_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tf_tracker, "TF_AVAILABLE", True)

    tracker = tf_tracker.MemoryTracker(
        sampling_interval=0.01,
        device="/GPU:0",
        enable_logging=False,
    )
    samples: Iterator[float | RuntimeError] = iter(
        [RuntimeError("memory probe failed"), 16.0]
    )
    current_time = {"value": 50.0}
    monkeypatch.setattr(tf_tracker.time, "time", lambda: current_time["value"])

    def _sequence() -> float:
        value = next(samples)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(tracker, "_get_current_memory", _sequence)

    tracker._run_tracking_iteration()
    degraded = tracker.events[-1]
    assert degraded["event_type"] == "collector_degraded"
    assert degraded["metadata"]["collector_health_status"] == "unhealthy"
    assert degraded["metadata"]["telemetry_partial"] is True

    current_time["value"] = 51.1
    tracker._run_tracking_iteration()
    event_types = [event["event_type"] for event in tracker.events]
    assert event_types == ["collector_degraded", "collector_recovered", "sample"]
    assert tracker.memory_usage == [16.0]
    assert tracker.get_statistics()["collector_health_status"] == "healthy"


def test_tf_tracker_bounds_history_and_reports_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tf_tracker, "TF_AVAILABLE", True)
    tracker = tf_tracker.MemoryTracker(
        sampling_interval=0.01,
        device="/GPU:0",
        enable_logging=False,
        max_history=3,
    )
    samples: Iterator[float] = iter([8.0, 16.0, 24.0, 32.0, 40.0])
    current_time = {"value": 0.0}
    monkeypatch.setattr(tf_tracker.time, "time", lambda: current_time["value"])

    def _next_sample() -> float:
        return next(samples)

    monkeypatch.setattr(tracker, "_get_current_memory", _next_sample)

    for index in range(5):
        current_time["value"] = float(index + 1)
        tracker._run_tracking_iteration()

    stats = tracker.get_statistics()
    assert tracker.memory_usage == [24.0, 32.0, 40.0]
    assert len(tracker.timestamps) == 3
    assert stats["history_window_limit"] == 3
    assert stats["history_retained_samples"] == 3
    assert stats["history_dropped_samples"] == 2
    assert stats["history_retained_events"] == 3
    assert stats["history_dropped_events"] == 2
    assert stats["peak_memory_mb"] == 40.0
    assert stats["average_memory_mb"] == pytest.approx(
        (8.0 + 16.0 + 24.0 + 32.0 + 40.0) / 5
    )
    assert stats["min_memory_mb"] == 8.0

    result = tracker.get_tracking_results()
    assert result.memory_usage == [24.0, 32.0, 40.0]
    assert result.history_window_limit == 3
    assert result.history_retained_samples == 3
    assert result.history_dropped_samples == 2
    assert result.history_retained_events == 3
    assert result.history_dropped_events == 2
    assert result.peak_memory == 40.0
    assert result.average_memory == pytest.approx((8.0 + 16.0 + 24.0 + 32.0 + 40.0) / 5)
    assert result.min_memory == 8.0


def test_tf_tracker_streams_events_to_append_only_sink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(tf_tracker, "TF_AVAILABLE", True)

    tracker = tf_tracker.MemoryTracker(
        sampling_interval=0.01,
        device="/GPU:0",
        enable_logging=False,
        telemetry_sink_config=TelemetrySinkConfig(
            root_dir=tmp_path / "sink",
            flush_every_events=1,
            flush_every_seconds=1.0,
            rollover_max_bytes=1024,
            retention_max_total_bytes=1024 * 1024,
        ),
    )
    monkeypatch.setattr(tracker, "_get_current_memory", lambda: 32.0)

    tracker._run_tracking_iteration()
    tracker._close_telemetry_sink()

    segment = tmp_path / "sink" / "segment-000001.jsonl"
    payload = json.loads(segment.read_text(encoding="utf-8").splitlines()[0])
    assert payload["collector"] == "stormlog.tensorflow.memory_tracker"
    assert payload["event_type"] == "sample"
    assert payload["allocator_allocated_bytes"] == 32 * 1024 * 1024
    stats = tracker.get_statistics()
    assert stats["final_retained_files"] == 1
    assert stats["rollover_count"] == 1


def test_tf_tracker_disables_sink_after_append_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tf_tracker, "TF_AVAILABLE", True)

    tracker = tf_tracker.MemoryTracker(
        sampling_interval=0.01,
        device="/GPU:0",
        enable_logging=False,
    )
    sink = _FailingSink(fail_on={"append"})
    tracker._telemetry_sink = sink  # type: ignore[assignment]

    tracker._append_event(
        timestamp=1.0,
        memory_mb=32.0,
        event_type="sample",
    )

    assert tracker.events
    assert tracker._telemetry_sink is None
    assert sink.append_calls == 1

    tracker.tracking = True
    result = tracker.stop_tracking()

    assert [event["event_type"] for event in result.events] == ["sample", "stop"]
    assert tracker._session_summary is not None
    assert tracker._session_summary.status == "incomplete"
    assert tracker.get_statistics()["session_status"] == "incomplete"


def test_tf_tracker_disables_sink_after_flush_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tf_tracker, "TF_AVAILABLE", True)

    tracker = tf_tracker.MemoryTracker(
        sampling_interval=0.01,
        device="/GPU:0",
        enable_logging=False,
    )
    sink = _FailingSink(fail_on={"flush"})
    tracker._telemetry_sink = sink  # type: ignore[assignment]

    tracker._flush_telemetry_sink()

    assert tracker._telemetry_sink is None
    assert sink.flush_calls == 1


def test_tf_tracker_stop_tracking_disables_sink_after_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tf_tracker, "TF_AVAILABLE", True)
    monkeypatch.setattr(tf_tracker.threading, "Thread", _NoOpThread)

    tracker = tf_tracker.MemoryTracker(
        sampling_interval=0.01,
        device="/GPU:0",
        enable_logging=False,
    )
    tracker.start_tracking()
    sink = _FailingSink(fail_on={"close"})
    tracker._telemetry_sink = sink  # type: ignore[assignment]

    result = tracker.stop_tracking()

    assert [event["event_type"] for event in result.events] == ["start", "stop"]
    assert tracker._telemetry_sink is None
    assert sink.close_calls >= 1
    assert tracker._session_summary is not None
    assert tracker._session_summary.status == "incomplete"
    assert tracker.get_statistics()["session_status"] == "incomplete"


def test_tf_tracker_persistent_failure_preserves_status_events_without_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tf_tracker, "TF_AVAILABLE", True)

    tracker = tf_tracker.MemoryTracker(
        sampling_interval=0.01,
        device="/GPU:0",
        enable_logging=False,
    )
    tracker._collector_retry_backoff_initial_s = 0.1
    tracker._collector_retry_backoff_cap_s = 0.4
    current_time = {"value": 60.0}
    monkeypatch.setattr(tf_tracker.time, "time", lambda: current_time["value"])

    def _fail() -> float:
        raise RuntimeError("memory probe failed")

    monkeypatch.setattr(tracker, "_get_current_memory", _fail)

    tracker._run_tracking_iteration()
    current_time["value"] = 60.11
    tracker._run_tracking_iteration()
    current_time["value"] = 60.32
    tracker._run_tracking_iteration()

    result = tracker.get_tracking_results()

    assert result.events
    assert [event["event_type"] for event in result.events] == ["collector_degraded"]
    assert result.memory_usage == []
    assert result.history_window_limit == 10_000
    assert tracker.get_statistics()["collector_health_status"] == (
        COLLECTOR_HEALTH_UNHEALTHY
    )
    assert tracker.get_statistics()["collector_consecutive_failures"] == 3


def test_tf_tracker_uses_session_wall_clock_for_failure_only_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tf_tracker, "TF_AVAILABLE", True)

    tracker = tf_tracker.MemoryTracker(
        sampling_interval=0.01,
        device="/GPU:0",
        enable_logging=False,
    )
    current_time = {"value": 100.0}
    monkeypatch.setattr(tf_tracker.time, "time", lambda: current_time["value"])
    monkeypatch.setattr(tf_tracker.threading, "Thread", _NoOpThread)

    def _fail() -> float:
        raise RuntimeError("memory probe failed")

    monkeypatch.setattr(tracker, "_get_current_memory", _fail)

    tracker.start_tracking()
    current_time["value"] = 101.5
    tracker._run_tracking_iteration()

    stats = tracker.get_statistics()
    assert stats["tracking_duration_seconds"] == pytest.approx(1.5)

    current_time["value"] = 103.0
    result = tracker.stop_tracking()

    assert result.start_time == pytest.approx(100.0)
    assert result.end_time == pytest.approx(103.0)
    assert result.duration == pytest.approx(3.0)
    assert result.memory_usage == []
    assert [event["event_type"] for event in result.events] == [
        "start",
        "collector_degraded",
        "stop",
    ]


def test_tf_tracker_recreates_sink_on_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(tf_tracker, "TF_AVAILABLE", True)
    monkeypatch.setattr(tf_tracker.threading, "Thread", _NoOpThread)

    tracker = tf_tracker.MemoryTracker(
        sampling_interval=0.01,
        device="/GPU:0",
        enable_logging=False,
        telemetry_sink_config=TelemetrySinkConfig(
            root_dir=tmp_path / "sink",
            flush_every_events=1,
            flush_every_seconds=1.0,
            rollover_max_bytes=1024 * 1024,
            retention_max_total_bytes=1024 * 1024,
        ),
    )
    monkeypatch.setattr(tracker, "_get_current_memory", lambda: 32.0)

    tracker.start_tracking()
    first_sink = tracker._telemetry_sink
    assert first_sink is not None
    tracker.stop_tracking()
    assert tracker._telemetry_sink is None

    tracker.start_tracking()
    second_sink = cast(object, tracker._telemetry_sink)

    assert second_sink is not None
    assert second_sink is not first_sink


def test_tf_tracker_emits_structured_phase_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tf_tracker, "TF_AVAILABLE", True)
    monkeypatch.setattr(tf_tracker.threading, "Thread", _NoOpThread)

    tracker = tf_tracker.MemoryTracker(
        sampling_interval=0.01,
        device="/GPU:0",
        enable_logging=False,
    )

    tracker.start_tracking()
    with tracker.phase("train_step", metadata={"epoch": 2}) as handle:
        assert handle.phase_path == "train_step"
    result = tracker.stop_tracking()

    phase_events = [
        event for event in result.events if event["event_type"].startswith("phase_")
    ]
    assert [event["event_type"] for event in phase_events] == [
        "phase_enter",
        "phase_exit",
    ]

    enter_scope = parse_phase_boundary(phase_events[0])
    exit_scope = parse_phase_boundary(phase_events[1])
    assert enter_scope is not None
    assert exit_scope is not None
    assert enter_scope.path == ("train_step",)
    assert enter_scope.attributes == {"epoch": 2}
    assert exit_scope.scope_id == enter_scope.scope_id
    assert exit_scope.path == enter_scope.path
