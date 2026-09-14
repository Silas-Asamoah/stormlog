"""Tests for stormlog/derived_fields.py."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from stormlog.derived_fields import (
    compute_event_fields,
    compute_session_fields,
    enrich_event,
)
from stormlog.telemetry import SCHEMA_VERSION_V2, TelemetryEventV2

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BASE_NS = 1_700_000_000_000_000_000
_DEVICE_TOTAL = 16 * 1024**3  # 16 GiB


def _make_event(
    *,
    allocated: int = 4 * 1024**3,
    reserved: int = 6 * 1024**3,
    device_total: int | None = _DEVICE_TOTAL,
    collector: str = "stormlog.cuda_tracker",
    event_type: str = "sample",
    index: int = 0,
) -> TelemetryEventV2:
    """Build a minimal valid TelemetryEventV2 for derived-field tests."""
    ts = _BASE_NS + index * 100_000_000
    device_used = allocated
    device_free = (device_total - device_used) if device_total is not None else None
    return TelemetryEventV2(
        schema_version=SCHEMA_VERSION_V2,
        timestamp_ns=ts,
        event_type=event_type,
        collector=collector,
        sampling_interval_ms=100,
        pid=1,
        host="test-host",
        device_id=0,
        allocator_allocated_bytes=allocated,
        allocator_reserved_bytes=reserved,
        allocator_active_bytes=None,
        allocator_inactive_bytes=None,
        allocator_change_bytes=0,
        device_used_bytes=device_used,
        device_free_bytes=device_free,
        device_total_bytes=device_total,
        context=None,
    )


# ---------------------------------------------------------------------------
# compute_event_fields — allocator_gap_bytes
# ---------------------------------------------------------------------------


def test_allocator_gap_bytes_is_reserved_minus_allocated() -> None:
    event = _make_event(allocated=4 * 1024**3, reserved=6 * 1024**3)
    fields = compute_event_fields(event)
    assert fields["allocator_gap_bytes"] == 2 * 1024**3


def test_allocator_gap_bytes_is_zero_when_equal() -> None:
    event = _make_event(allocated=4 * 1024**3, reserved=4 * 1024**3)
    fields = compute_event_fields(event)
    assert fields["allocator_gap_bytes"] == 0


def test_allocator_gap_bytes_always_present_even_with_no_total() -> None:
    event = _make_event(allocated=1024, reserved=2048, device_total=None)
    fields = compute_event_fields(event)
    assert "allocator_gap_bytes" in fields
    assert fields["allocator_gap_bytes"] == 1024


def test_allocator_gap_bytes_clamped_to_zero_when_allocated_exceeds_reserved() -> None:
    """Negative gaps indicate stale data; clamp to zero."""
    event = _make_event(allocated=8 * 1024**3, reserved=4 * 1024**3)
    fields = compute_event_fields(event)
    assert fields["allocator_gap_bytes"] == 0


# ---------------------------------------------------------------------------
# compute_event_fields — utilization_ratio
# ---------------------------------------------------------------------------


def test_utilization_ratio_when_device_total_is_known() -> None:
    allocated = 4 * 1024**3
    event = _make_event(allocated=allocated, device_total=_DEVICE_TOTAL)
    fields = compute_event_fields(event)
    expected = allocated / _DEVICE_TOTAL
    assert fields["utilization_ratio"] == pytest.approx(expected)


def test_utilization_ratio_is_none_when_device_total_is_none() -> None:
    event = _make_event(device_total=None)
    fields = compute_event_fields(event)
    assert fields["utilization_ratio"] is None


def test_utilization_ratio_is_none_when_device_total_is_zero() -> None:
    # device_total_bytes of 0 is treated as unknown (guard against div-by-zero)
    event = _make_event(allocated=0, reserved=0, device_total=0)
    fields = compute_event_fields(event)
    assert fields["utilization_ratio"] is None


# ---------------------------------------------------------------------------
# compute_event_fields — fragmentation_ratio (div-by-zero guard)
# ---------------------------------------------------------------------------


def test_fragmentation_ratio_normal_case() -> None:
    allocated = 2 * 1024**3
    reserved = 4 * 1024**3
    event = _make_event(allocated=allocated, reserved=reserved)
    fields = compute_event_fields(event)
    # gap = 2 GiB, reserved = 4 GiB → ratio = 0.5
    assert fields["fragmentation_ratio"] == pytest.approx(0.5)


def test_fragmentation_ratio_is_none_when_reserved_is_zero() -> None:
    """Must not divide by zero."""
    event = _make_event(allocated=0, reserved=0)
    fields = compute_event_fields(event)
    assert fields["fragmentation_ratio"] is None


def test_fragmentation_ratio_is_zero_when_fully_allocated() -> None:
    n = 4 * 1024**3
    event = _make_event(allocated=n, reserved=n)
    fields = compute_event_fields(event)
    assert fields["fragmentation_ratio"] == pytest.approx(0.0)


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("as_mapping", [True, False])
def test_fragmentation_ratio_honors_explicit_capability(
    enabled: bool, as_mapping: bool
) -> None:
    event = _make_event(allocated=100, reserved=1000)
    event.metadata["memory_capabilities"] = {"supports_fragmentation_analysis": enabled}
    source = event.__dict__ if as_mapping else event

    fields = compute_event_fields(source)

    assert fields["fragmentation_ratio"] == (0.9 if enabled else None)
    assert fields["allocator_gap_bytes"] == 900
    assert compute_session_fields([source])["avg_fragmentation_ratio"] == (
        0.9 if enabled else None
    )


# ---------------------------------------------------------------------------
# compute_event_fields — is_degraded_collector
# ---------------------------------------------------------------------------

_DEGRADED_COLLECTORS = [
    "stormlog.fallback_tracker",
    "stormlog.degraded",
    "mps_unavailable",
    "partial_collector",
    "legacy.unknown",
    "FALLBACK",  # case-insensitive
]

_HEALTHY_COLLECTORS = [
    "stormlog.cuda_tracker",
    "stormlog.rocm_tracker",
    "stormlog.mps_tracker",
    "stormlog.cpu_tracker",
    "stormlog.tensorflow.memory_tracker",
]


@pytest.mark.parametrize("collector", _DEGRADED_COLLECTORS)  # type: ignore[misc, unused-ignore]
def test_is_degraded_collector_true_for_known_degraded_strings(
    collector: str,
) -> None:
    event = _make_event(collector=collector)
    fields = compute_event_fields(event)
    assert fields["is_degraded_collector"] is True, collector


@pytest.mark.parametrize("collector", _HEALTHY_COLLECTORS)  # type: ignore[misc, unused-ignore]
def test_is_degraded_collector_false_for_healthy_collectors(
    collector: str,
) -> None:
    event = _make_event(collector=collector)
    fields = compute_event_fields(event)
    assert fields["is_degraded_collector"] is False, collector


# ---------------------------------------------------------------------------
# compute_event_fields — dict input (CLI use case)
# ---------------------------------------------------------------------------


def test_compute_event_fields_accepts_plain_dict() -> None:
    event_dict = {
        "allocator_allocated_bytes": 1024,
        "allocator_reserved_bytes": 2048,
        "device_total_bytes": 8192,
        "collector": "stormlog.cuda_tracker",
    }
    fields = compute_event_fields(event_dict)
    assert fields["allocator_gap_bytes"] == 1024
    assert fields["utilization_ratio"] == pytest.approx(1024 / 8192)
    assert fields["fragmentation_ratio"] == pytest.approx(0.5)
    assert fields["is_degraded_collector"] is False


def test_derived_field_helpers_accept_read_only_mapping() -> None:
    event_mapping = MappingProxyType(
        {
            "allocator_allocated_bytes": 1024,
            "allocator_reserved_bytes": 2048,
            "device_total_bytes": 8192,
            "collector": "stormlog.cuda_tracker",
            "event_type": "stop",
        }
    )

    fields = compute_event_fields(event_mapping)
    session = compute_session_fields([event_mapping])
    enriched = enrich_event(event_mapping)

    assert fields["allocator_gap_bytes"] == 1024
    assert fields["utilization_ratio"] == pytest.approx(1024 / 8192)
    assert fields["fragmentation_ratio"] == pytest.approx(0.5)
    assert session["is_session_interrupted"] is False
    assert enriched["derived"] == fields


# ---------------------------------------------------------------------------
# TensorFlow synthetic events — reserved == allocated (no separate counter)
# ---------------------------------------------------------------------------


def test_tensorflow_synthetic_event_gap_is_zero() -> None:
    """TF aliases reserved == allocated; gap must be 0, not a real reservation."""
    tf_event = {
        "allocator_allocated_bytes": 1024,
        "allocator_reserved_bytes": 1024,
        "device_total_bytes": 4096,
        "collector": None,
    }
    fields = compute_event_fields(tf_event)
    assert fields["allocator_gap_bytes"] == 0
    assert fields["fragmentation_ratio"] == 0.0
    assert fields["utilization_ratio"] == pytest.approx(0.25)


def test_tensorflow_gap_zero_even_without_device_total() -> None:
    """TF gap is always 0 regardless of whether device_total is known."""
    tf_event = {
        "allocator_allocated_bytes": 512,
        "allocator_reserved_bytes": 512,
        "device_total_bytes": None,
        "collector": None,
    }
    fields = compute_event_fields(tf_event)
    assert fields["allocator_gap_bytes"] == 0
    assert fields["utilization_ratio"] is None


def test_tensorflow_session_no_spurious_gap_or_frag() -> None:
    """TF session with reserved==allocated must not report fake gaps or fragmentation."""
    tf_events = [
        {
            "allocator_allocated_bytes": 1024,
            "allocator_reserved_bytes": 1024,
            "device_total_bytes": 4096,
            "collector": None,
            "event_type": "sample",
        },
        {
            "allocator_allocated_bytes": 2048,
            "allocator_reserved_bytes": 2048,
            "device_total_bytes": 4096,
            "collector": None,
            "event_type": "stop",
        },
    ]
    session = compute_session_fields(tf_events)
    assert session["avg_fragmentation_ratio"] == 0.0
    assert session["peak_utilization_ratio"] == pytest.approx(0.5)
    assert session["is_session_interrupted"] is False


# ---------------------------------------------------------------------------
# compute_session_fields — multi-event fixture
# ---------------------------------------------------------------------------


def _build_session_events() -> list[TelemetryEventV2]:
    """Three-event session: two samples, then a stop."""
    gb = 1024**3
    return [
        _make_event(allocated=2 * gb, reserved=4 * gb, event_type="sample", index=0),
        _make_event(allocated=6 * gb, reserved=8 * gb, event_type="sample", index=1),
        _make_event(allocated=6 * gb, reserved=8 * gb, event_type="stop", index=2),
    ]


def test_compute_session_fields_peak_utilization() -> None:
    events = _build_session_events()
    session = compute_session_fields(events)
    # Peak allocated = 6 GiB / 16 GiB device
    expected_peak = 6 * 1024**3 / _DEVICE_TOTAL
    assert session["peak_utilization_ratio"] == pytest.approx(expected_peak)


def test_compute_session_fields_avg_fragmentation() -> None:
    events = _build_session_events()
    session = compute_session_fields(events)
    # Event 0: gap=2 GiB, reserved=4 GiB → 0.5
    # Event 1 & 2: gap=2 GiB, reserved=8 GiB → 0.25 each
    expected_avg = (0.5 + 0.25 + 0.25) / 3
    assert session["avg_fragmentation_ratio"] == pytest.approx(expected_avg)


def test_compute_session_fields_not_interrupted_when_last_is_stop() -> None:
    events = _build_session_events()
    session = compute_session_fields(events)
    assert session["is_session_interrupted"] is False


def test_compute_session_fields_interrupted_when_no_stop() -> None:
    gb = 1024**3
    events = [
        _make_event(allocated=2 * gb, reserved=4 * gb, event_type="sample", index=0),
        _make_event(allocated=4 * gb, reserved=6 * gb, event_type="sample", index=1),
    ]
    session = compute_session_fields(events)
    assert session["is_session_interrupted"] is True


def test_compute_session_fields_empty_sequence() -> None:
    session = compute_session_fields([])
    assert session["peak_utilization_ratio"] is None
    assert session["avg_fragmentation_ratio"] is None
    assert session["is_session_interrupted"] is True  # no stop event


def test_compute_session_fields_no_device_total() -> None:
    gb = 1024**3
    events = [
        _make_event(allocated=2 * gb, reserved=4 * gb, device_total=None, index=0),
        _make_event(allocated=4 * gb, reserved=8 * gb, device_total=None, index=1),
    ]
    session = compute_session_fields(events)
    assert session["peak_utilization_ratio"] is None


# ---------------------------------------------------------------------------
# enrich_event
# ---------------------------------------------------------------------------


def test_enrich_event_contains_raw_fields() -> None:
    event = _make_event()
    enriched = enrich_event(event)
    assert "allocator_allocated_bytes" in enriched
    assert "allocator_reserved_bytes" in enriched
    assert "device_total_bytes" in enriched


def test_enrich_event_contains_derived_key() -> None:
    event = _make_event()
    enriched = enrich_event(event)
    assert "derived" in enriched


def test_enrich_event_derived_contains_expected_fields() -> None:
    event = _make_event()
    enriched = enrich_event(event)
    derived = enriched["derived"]
    assert "allocator_gap_bytes" in derived
    assert "utilization_ratio" in derived
    assert "fragmentation_ratio" in derived
    assert "is_degraded_collector" in derived


def test_enrich_event_derived_values_match_compute_event_fields() -> None:
    event = _make_event(allocated=3 * 1024**3, reserved=5 * 1024**3)
    enriched = enrich_event(event)
    direct = compute_event_fields(event)
    assert enriched["derived"]["allocator_gap_bytes"] == direct["allocator_gap_bytes"]
    assert enriched["derived"]["utilization_ratio"] == pytest.approx(
        direct["utilization_ratio"]
    )


def test_enrich_event_accepts_plain_dict() -> None:
    event_dict = {
        "allocator_allocated_bytes": 512,
        "allocator_reserved_bytes": 1024,
        "device_total_bytes": 4096,
        "collector": "stormlog.cpu_tracker",
    }
    enriched = enrich_event(event_dict)
    assert enriched["allocator_allocated_bytes"] == 512
    assert "derived" in enriched
    assert enriched["derived"]["allocator_gap_bytes"] == 512


def test_enrich_event_accepts_plain_object() -> None:
    """Exercise the vars() fallback for non-dataclass, non-dict objects."""
    from types import SimpleNamespace

    event = SimpleNamespace(
        allocator_allocated_bytes=256,
        allocator_reserved_bytes=512,
        device_total_bytes=2048,
        collector="stormlog.cpu_tracker",
    )
    enriched = enrich_event(event)
    assert enriched["allocator_allocated_bytes"] == 256
    assert "derived" in enriched
    assert enriched["derived"]["allocator_gap_bytes"] == 256


def test_is_degraded_detects_collector_health_degraded_constant() -> None:
    """Guard against drift: COLLECTOR_HEALTH_DEGRADED must be recognized."""
    from stormlog.collector_health import COLLECTOR_HEALTH_DEGRADED

    event = _make_event(collector=f"stormlog.{COLLECTOR_HEALTH_DEGRADED}_tracker")
    fields = compute_event_fields(event)
    assert fields["is_degraded_collector"] is True
