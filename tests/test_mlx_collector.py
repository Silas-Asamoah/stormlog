import pytest

from stormlog.derived_fields import compute_event_fields
from stormlog.mlx.collector import MLXCollector
from stormlog.session import create_session_summary
from stormlog.telemetry import telemetry_event_from_record
from tests.mlx_fakes import FakeCore, make_runtime


def test_passive_mapping_and_independent_host_memory():
    core = FakeCore()
    collector = MLXCollector(
        make_runtime(core),
        host_reader=lambda: {
            "system_memory_total_bytes": 9000,
            "system_memory_available_bytes": 6000,
            "process_rss_bytes": 1000,
        },
    )
    snapshot = collector.capture_snapshot()
    assert snapshot.active_bytes == 100
    assert snapshot.cache_bytes == 20
    assert snapshot.process_rss_bytes == 1000
    assert snapshot.metadata["mlx"]["allocator_held_bytes"] == 120
    record = collector.telemetry_record(snapshot, create_session_summary(source="test"))
    assert record["allocator_change_bytes"] is None
    for key in (
        "allocator_reserved_bytes",
        "allocator_active_bytes",
        "allocator_inactive_bytes",
        "device_used_bytes",
        "device_free_bytes",
        "device_total_bytes",
    ):
        assert record[key] is None
    event = telemetry_event_from_record(record)
    assert event.metadata["framework"] == "mlx"
    assert not record["metadata"]["memory_capabilities"]["supports_bounded_profiling"]
    assert record["metadata"]["mlx"]["capabilities"]["bounded_sampling"]
    assert "eval" not in core.calls and "reset" not in core.calls
    assert not any(isinstance(call, tuple) and call[0] == "sync" for call in core.calls)
    derived = compute_event_fields(record)
    assert derived["fragmentation_ratio"] is None
    assert derived["utilization_ratio"] is None


@pytest.mark.parametrize("value", [-1, True, 2.5, RuntimeError("broken")])
def test_invalid_active_never_becomes_zero(value):
    core = FakeCore()
    core.values["active_memory"] = value
    snapshot = MLXCollector(make_runtime(core)).capture_snapshot()
    assert snapshot.active_bytes is None
    assert "active_memory" in snapshot.unavailable
    assert snapshot.cache_bytes == 20


def test_zero_partial_and_optional_absence():
    core = FakeCore()
    core.values["active_memory"] = 0
    core.values["cache_memory"] = RuntimeError("transient")
    core.get_memory_limit = None
    collector = MLXCollector(make_runtime(core))
    first = collector.capture_snapshot()
    assert first.active_bytes == 0 and first.cache_bytes is None
    assert collector.capabilities["supports_allocator_allocated"]
    assert first.metadata["mlx"]["memory_limit_bytes"] is None
    core.values["cache_memory"] = 12
    assert collector.capture_snapshot().cache_bytes == 12
