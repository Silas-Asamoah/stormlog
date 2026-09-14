import logging
import types
from pathlib import Path

import pytest

from stormlog.telemetry_sink import TelemetrySinkConfig
from stormlog.tui import monitor


class DummyCPUTracker:
    """Minimal CPUMemoryTracker stand-in for TUI unit tests."""

    def __init__(
        self,
        sampling_interval: float = 0.5,
        max_events: int = 10_000,
        enable_alerts: bool = True,
        telemetry_sink_config: object | None = None,
    ) -> None:
        self.sampling_interval = sampling_interval
        self.max_events = max_events
        self.enable_alerts = enable_alerts
        self.telemetry_sink_config = telemetry_sink_config
        self.is_tracking = False
        self.events: list[object] = []

    def start_tracking(self) -> None:
        self.is_tracking = True

    def stop_tracking(self) -> None:
        self.is_tracking = False

    def get_statistics(self) -> dict[str, object]:
        return {"mode": "cpu"}

    def get_memory_timeline(self, interval: float = 1.0) -> dict[str, object]:
        return {}

    def get_events(self, since: float | None = None) -> list[object]:
        _ = since
        return list(self.events)

    def clear_events(self) -> None:
        self.events.clear()

    def export_events(self, *args: object, **kwargs: object) -> None:
        return None


class BrokenGPUTracker:
    """GPU tracker stub that always fails to initialize."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("No CUDA available")


def _stub_torch(cuda_available: bool) -> object:
    cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    return types.SimpleNamespace(cuda=cuda)


def test_tracker_session_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure we gracefully fall back to CPU tracking when GPU tracker fails."""

    monkeypatch.setattr(monitor, "MemoryTracker", BrokenGPUTracker)
    monkeypatch.setattr(monitor, "MemoryWatchdog", None)
    monkeypatch.setattr(monitor, "CPUMemoryTracker", DummyCPUTracker)
    monkeypatch.setattr(monitor, "torch", _stub_torch(False))

    session = monitor.TrackerSession()
    session.start()

    assert session.backend == "cpu"
    assert isinstance(session._tracker, DummyCPUTracker)
    assert session._tracker.max_events == session.max_events
    assert session.is_active

    session.stop()
    assert not session.is_active


def test_tracker_session_works_without_gpu_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TrackerSession should still operate when the GPU tracker cannot import."""

    monkeypatch.setattr(monitor, "MemoryTracker", None)
    monkeypatch.setattr(monitor, "MemoryWatchdog", None)
    monkeypatch.setattr(monitor, "CPUMemoryTracker", DummyCPUTracker)
    monkeypatch.setattr(monitor, "torch", None)

    session = monitor.TrackerSession()
    session.start()

    assert session.backend == "cpu"
    assert isinstance(session._tracker, DummyCPUTracker)

    session.stop()


def test_tracker_session_requires_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate that we surface a helpful error when no backends exist."""

    monkeypatch.setattr(monitor, "MemoryTracker", None)
    monkeypatch.setattr(monitor, "CPUMemoryTracker", None)

    with pytest.raises(monitor.TrackerUnavailableError):
        monitor.TrackerSession()


def test_tracker_session_get_telemetry_events_normalizes_cpu_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(monitor, "MemoryTracker", BrokenGPUTracker)
    monkeypatch.setattr(monitor, "MemoryWatchdog", None)
    monkeypatch.setattr(monitor, "CPUMemoryTracker", DummyCPUTracker)
    monkeypatch.setattr(monitor, "torch", _stub_torch(False))

    session = monitor.TrackerSession()
    session.start()

    assert isinstance(session._tracker, DummyCPUTracker)
    session._tracker.events.append(
        types.SimpleNamespace(
            timestamp=1700000000.0,
            event_type="warning",
            memory_allocated=1024,
            memory_reserved=2048,
            memory_change=1024,
            device_id=-1,
            context="cpu warning",
            job_id="job-123",
            rank=2,
            local_rank=0,
            world_size=4,
            metadata={"source": "test"},
        )
    )

    telemetry_events = session.get_telemetry_events()
    assert len(telemetry_events) == 1
    first = telemetry_events[0]
    assert first.schema_version == 4
    assert isinstance(first.session_id, str)
    assert first.session_id
    assert first.collector == "stormlog.cpu_tracker"
    assert first.event_type == "warning"
    assert first.rank == 2
    assert first.local_rank == 0
    assert first.world_size == 4
    assert first.job_id == "job-123"

    telemetry_records = session.telemetry_records()
    assert len(telemetry_records) == 1
    assert telemetry_records[0].source_kind == "cpu"
    assert telemetry_records[0].resource["collector"] == "stormlog.cpu_tracker"
    assert telemetry_records[0].attributes["memory.allocator.allocated_bytes"] == 1024


def test_tracker_session_preserves_collector_health_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(monitor, "MemoryTracker", BrokenGPUTracker)
    monkeypatch.setattr(monitor, "MemoryWatchdog", None)
    monkeypatch.setattr(monitor, "CPUMemoryTracker", DummyCPUTracker)
    monkeypatch.setattr(monitor, "torch", _stub_torch(False))

    session = monitor.TrackerSession()
    session.start()

    assert isinstance(session._tracker, DummyCPUTracker)
    session._tracker.events.append(
        types.SimpleNamespace(
            timestamp=1700000001.0,
            event_type="collector_degraded",
            memory_allocated=1024,
            memory_reserved=1024,
            memory_change=0,
            device_id=-1,
            device_total=None,
            context="collector degraded",
            metadata={
                "collector_health_status": "degraded",
                "telemetry_partial": True,
                "collector_partial_fields": ["device_total_bytes"],
            },
        )
    )

    telemetry_events = session.get_telemetry_events()

    assert len(telemetry_events) == 1
    first = telemetry_events[0]
    assert first.event_type == "collector_degraded"
    assert first.device_total_bytes is None
    assert first.metadata["collector_health_status"] == "degraded"
    assert first.metadata["telemetry_partial"] is True
    assert first.metadata["collector_partial_fields"] == ["device_total_bytes"]


def test_tracker_session_passes_telemetry_sink_config_to_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(monitor, "MemoryTracker", BrokenGPUTracker)
    monkeypatch.setattr(monitor, "MemoryWatchdog", None)
    monkeypatch.setattr(monitor, "CPUMemoryTracker", DummyCPUTracker)
    monkeypatch.setattr(monitor, "torch", _stub_torch(False))

    sink_config = TelemetrySinkConfig(root_dir=Path("/tmp/tui_sink"))
    session = monitor.TrackerSession(telemetry_sink_config=sink_config)
    session.start()

    assert isinstance(session._tracker, DummyCPUTracker)
    assert session._tracker.telemetry_sink_config == sink_config

    session.stop()


@pytest.mark.parametrize("conversion", ["valid", "invalid", "raises"])
def test_tracker_session_preserves_canonical_conversion_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    conversion: str,
) -> None:
    monkeypatch.setattr(monitor, "CPUMemoryTracker", DummyCPUTracker)
    session = monitor.TrackerSession()
    tracker = DummyCPUTracker()
    session._tracker = tracker
    session.backend = "cpu"
    raw_event = types.SimpleNamespace(
        timestamp=1700000000.0,
        event_type="sample",
        memory_allocated=1024,
        memory_reserved=2048,
        memory_change=1024,
        context="legacy",
    )
    tracker.events.append(raw_event)

    def convert(event: object) -> dict[str, object]:
        if conversion == "raises":
            raise ValueError("conversion failed")
        if conversion == "invalid":
            return {"schema_version": 999}
        return {**vars(event), "context": "canonical"}

    monkeypatch.setattr(tracker, "_telemetry_record_from_event", convert, raising=False)
    with caplog.at_level(logging.DEBUG, logger=monitor.__name__):
        events = session.get_telemetry_events()

    assert len(events) == 1
    assert events[0].context == ("canonical" if conversion == "valid" else "legacy")
    assert events[0].allocator_allocated_bytes == 1024
    assert ("canonical event conversion failed" in caplog.text) is (
        conversion != "valid"
    )
