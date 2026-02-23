import types

import pytest

from gpumemprof.tui import monitor


class DummyCPUTracker:
    """Minimal CPUMemoryTracker stand-in for TUI unit tests."""

    def __init__(
        self,
        sampling_interval: float = 0.5,
        max_events: int = 10_000,
        enable_alerts: bool = True,
    ) -> None:
        self.sampling_interval = sampling_interval
        self.max_events = max_events
        self.enable_alerts = enable_alerts
        self.is_tracking = False
        self.events: list[object] = []

    def start_tracking(self) -> None:
        self.is_tracking = True

    def stop_tracking(self) -> None:
        self.is_tracking = False

    def get_statistics(self) -> dict[str, str]:
        return {"mode": "cpu"}

    def get_memory_timeline(self, interval: float = 1.0) -> dict[str, object]:
        return {}

    def get_events(self, since: float | None = None) -> list[object]:
        return []

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
