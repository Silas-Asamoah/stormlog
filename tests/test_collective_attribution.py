"""Tests for collective-memory attribution heuristics."""

from __future__ import annotations

import pytest

from gpumemprof.collective_attribution import (
    attribute_collective_memory,
    resolve_collective_attribution_config,
)
from gpumemprof.telemetry import SCHEMA_VERSION_V2, TelemetryEventV2

BASE_NS = 1_700_000_000_000_000_000
STEP_NS = 100_000_000
DEVICE_TOTAL_BYTES = 16 * 1024**3


def _make_event(
    *,
    timestamp_ns: int,
    rank: int,
    world_size: int,
    allocator_reserved: int,
    device_used: int,
    event_type: str = "sample",
    context: str | None = None,
    metadata: dict | None = None,
    device_total: int | None = DEVICE_TOTAL_BYTES,
) -> TelemetryEventV2:
    allocator_allocated = max(0, allocator_reserved - 100_000_000)
    device_free = None
    if device_total is not None:
        device_free = max(0, device_total - device_used)
    return TelemetryEventV2(
        schema_version=SCHEMA_VERSION_V2,
        timestamp_ns=timestamp_ns,
        event_type=event_type,
        collector="gpumemprof.cuda_tracker",
        sampling_interval_ms=100,
        pid=1,
        host="test",
        device_id=0,
        allocator_allocated_bytes=allocator_allocated,
        allocator_reserved_bytes=allocator_reserved,
        allocator_active_bytes=None,
        allocator_inactive_bytes=None,
        allocator_change_bytes=0,
        device_used_bytes=device_used,
        device_free_bytes=device_free,
        device_total_bytes=device_total,
        context=context,
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        metadata=metadata or {},
    )


def _build_sync_spike_events(
    *,
    world_size: int,
    with_markers: bool,
    include_rank1_spike: bool = True,
    device_total: int | None = DEVICE_TOTAL_BYTES,
) -> list[TelemetryEventV2]:
    events: list[TelemetryEventV2] = []

    for rank in range(world_size):
        rank_offset_ns = rank * 2_000_000
        for sample_index in range(12):
            timestamp_ns = BASE_NS + sample_index * STEP_NS + rank_offset_ns
            reserved = 2_000_000_000
            used = reserved + 40_000_000
            if sample_index == 6:
                if rank != 1 or include_rank1_spike:
                    used = reserved + 1_600_000_000
            events.append(
                _make_event(
                    timestamp_ns=timestamp_ns,
                    rank=rank,
                    world_size=world_size,
                    allocator_reserved=reserved,
                    device_used=used,
                    device_total=device_total,
                )
            )

            if with_markers and sample_index == 6:
                events.append(
                    _make_event(
                        timestamp_ns=timestamp_ns + 10_000_000,
                        rank=rank,
                        world_size=world_size,
                        allocator_reserved=reserved,
                        device_used=reserved,
                        event_type="collective",
                        context="NCCL all_reduce phase",
                        metadata={"phase": "communication.collective"},
                        device_total=device_total,
                    )
                )

    return events


def test_resolve_collective_config_supports_preset_and_overrides() -> None:
    config = resolve_collective_attribution_config(
        "high",
        overrides={"min_confidence": 0.41, "min_gap_ratio": 0.03},
    )
    assert config.preset == "high"
    assert config.min_confidence == pytest.approx(0.41)
    assert config.min_gap_ratio == pytest.approx(0.03)


def test_resolve_collective_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Unknown collective attribution override"):
        resolve_collective_attribution_config("medium", {"unknown_key": 1})


def test_collective_attribution_positive_marker_and_synchrony_case() -> None:
    events = _build_sync_spike_events(world_size=4, with_markers=True)

    results = attribute_collective_memory(events)

    assert {item.rank for item in results} == {0, 1, 2, 3}
    assert all(item.confidence >= 0.7 for item in results)
    assert all("marker_collective_token" in item.reason_codes for item in results)
    assert all("cross_rank_synchrony" in item.reason_codes for item in results)


def test_collective_attribution_negative_unsynchronized_spike() -> None:
    events = _build_sync_spike_events(
        world_size=2,
        with_markers=False,
        include_rank1_spike=False,
    )

    results = attribute_collective_memory(events)

    assert results == []


def test_collective_attribution_ambiguous_timing_only_case() -> None:
    events = _build_sync_spike_events(world_size=4, with_markers=False)

    results = attribute_collective_memory(events)

    assert len(results) == 4
    assert all(0.35 <= item.confidence < 0.7 for item in results)
    assert all("weak_marker_signal" in item.reason_codes for item in results)
    assert all("cross_rank_synchrony" in item.reason_codes for item in results)


def test_collective_attribution_single_rank_is_tagged_ambiguous() -> None:
    events = _build_sync_spike_events(world_size=1, with_markers=True)

    results = attribute_collective_memory(events)

    assert len(results) == 1
    result = results[0]
    assert "single_rank_only" in result.reason_codes
    assert "marker_collective_token" in result.reason_codes
    assert 0.5 <= result.confidence < 0.8


def test_collective_attribution_handles_missing_device_total_with_lower_confidence() -> (
    None
):
    events = _build_sync_spike_events(
        world_size=3, with_markers=True, device_total=None
    )

    results = attribute_collective_memory(events)

    assert len(results) == 3
    assert all(item.evidence is not None for item in results)
    assert all(
        item.evidence and item.evidence.peak_gap_ratio is None for item in results
    )
    assert all(item.confidence < 1.0 for item in results)


def test_collective_attribution_stability_no_significant_hidden_gap() -> None:
    events: list[TelemetryEventV2] = []
    for rank in range(2):
        for sample_index in range(12):
            ts = BASE_NS + sample_index * STEP_NS + rank * 1_000_000
            reserved = 1_500_000_000 + sample_index * 1_000_000
            used = reserved + 5_000_000
            events.append(
                _make_event(
                    timestamp_ns=ts,
                    rank=rank,
                    world_size=2,
                    allocator_reserved=reserved,
                    device_used=used,
                )
            )

    results = attribute_collective_memory(events)
    assert results == []


def test_collective_attribution_keeps_marker_evidence_rank_local() -> None:
    events = _build_sync_spike_events(world_size=2, with_markers=False)
    marker_time_ns = BASE_NS + 6 * STEP_NS + 10_000_000
    events.append(
        _make_event(
            timestamp_ns=marker_time_ns,
            rank=0,
            world_size=2,
            allocator_reserved=2_000_000_000,
            device_used=2_000_000_000,
            event_type="collective",
            context="NCCL all_reduce phase",
            metadata={"phase": "communication.collective"},
        )
    )

    results = sorted(attribute_collective_memory(events), key=lambda item: item.rank)

    assert len(results) == 2
    rank0 = results[0]
    rank1 = results[1]
    assert rank0.rank == 0
    assert rank1.rank == 1
    assert "marker_collective_token" in rank0.reason_codes
    assert "marker_collective_token" not in rank1.reason_codes
    assert "weak_marker_signal" in rank1.reason_codes


def test_collective_attribution_accepts_uppercase_sample_event_type() -> None:
    events = _build_sync_spike_events(world_size=2, with_markers=True)
    normalized: list[TelemetryEventV2] = []
    for event in events:
        if event.event_type == "sample":
            normalized.append(
                TelemetryEventV2(
                    schema_version=event.schema_version,
                    timestamp_ns=event.timestamp_ns,
                    event_type="SAMPLE",
                    collector=event.collector,
                    sampling_interval_ms=event.sampling_interval_ms,
                    pid=event.pid,
                    host=event.host,
                    device_id=event.device_id,
                    allocator_allocated_bytes=event.allocator_allocated_bytes,
                    allocator_reserved_bytes=event.allocator_reserved_bytes,
                    allocator_active_bytes=event.allocator_active_bytes,
                    allocator_inactive_bytes=event.allocator_inactive_bytes,
                    allocator_change_bytes=event.allocator_change_bytes,
                    device_used_bytes=event.device_used_bytes,
                    device_free_bytes=event.device_free_bytes,
                    device_total_bytes=event.device_total_bytes,
                    context=event.context,
                    job_id=event.job_id,
                    rank=event.rank,
                    local_rank=event.local_rank,
                    world_size=event.world_size,
                    metadata=dict(event.metadata),
                )
            )
        else:
            normalized.append(event)

    results = attribute_collective_memory(normalized)
    assert len(results) == 2
