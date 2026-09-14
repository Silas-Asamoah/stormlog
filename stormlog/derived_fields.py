"""Registry-driven derived-field layer for Stormlog telemetry.

Centralises the common investigation formulas (allocator gap, utilisation %,
fragmentation ratio, degraded-collector state) so that ``diagnose.py``,
``gap_analysis.py``, and future TUI surfaces all use a single, tested source
of truth instead of copy-pasting the same arithmetic.

Usage::

    from stormlog.derived_fields import compute_event_fields, enrich_event

    fields = compute_event_fields(event)
    # fields["allocator_gap_bytes"] always present
    # fields["utilization_ratio"] is None when device_total_bytes is unknown

    enriched = enrich_event(event)
    # {"allocator_allocated_bytes": ..., ..., "allocator_gap_bytes": ..., ...}
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping as MappingABC
from typing import Any, Dict, Sequence, TypedDict, cast

from .collector_health import COLLECTOR_HEALTH_DEGRADED

# ---------------------------------------------------------------------------
# Public contract
# ---------------------------------------------------------------------------


class DerivedFields(TypedDict, total=False):
    """Computed fields derived from a single telemetry event.

    ``allocator_gap_bytes`` is always present.  All other fields are present only
    when the underlying raw counters are available; their absence is encoded as
    ``None`` in the returned dict rather than a missing key.
    """

    allocator_gap_bytes: int | None
    utilization_ratio: None | float
    fragmentation_ratio: None | float
    is_degraded_collector: bool


class SessionDerivedFields(TypedDict, total=False):
    """Session-scoped rollups computed across a sequence of events."""

    peak_utilization_ratio: None | float
    avg_fragmentation_ratio: None | float
    is_session_interrupted: bool


# ---------------------------------------------------------------------------
# Degraded-collector registry
# ---------------------------------------------------------------------------

# Collector strings that indicate the collector is operating in a degraded /
# fallback mode.  Cross-referenced against ``collector_health.py`` status
# constants and the known collector name fragments used in ``telemetry.py``.
_DEGRADED_COLLECTOR_SUBSTRINGS: frozenset[str] = frozenset(
    {
        "fallback",
        COLLECTOR_HEALTH_DEGRADED,  # canonical constant from collector_health.py
        "unavailable",
        "partial",
        "legacy.unknown",
    }
)


def _collector_is_degraded(collector: None | str) -> bool:
    """Return True when the collector name signals a degraded / fallback mode."""
    if not collector:
        return False
    lower = collector.lower()
    return any(fragment in lower for fragment in _DEGRADED_COLLECTOR_SUBSTRINGS)


# ---------------------------------------------------------------------------
# Per-field compute helpers (the "registry")
# ---------------------------------------------------------------------------


def _compute_allocator_gap_bytes(
    allocator_allocated_bytes: int | None,
    allocator_reserved_bytes: int | None,
) -> int | None:
    """Reserved memory that the allocator holds but has not actively allocated.

    Clamped to zero - negative values indicate a stale or inconsistent snapshot
    rather than a meaningful metric.
    """
    if allocator_allocated_bytes is None or allocator_reserved_bytes is None:
        return None
    return max(0, allocator_reserved_bytes - allocator_allocated_bytes)


def _compute_utilization_ratio(
    used_bytes: int | None,
    device_total_bytes: None | int,
) -> None | float:
    """Fraction of device capacity currently allocated (0.0 - 1.0)."""
    if used_bytes is None or device_total_bytes is None or device_total_bytes <= 0:
        return None
    return used_bytes / device_total_bytes


def _compute_fragmentation_ratio(
    allocator_allocated_bytes: int | None,
    allocator_reserved_bytes: int | None,
) -> None | float:
    """Fraction of reserved memory that is fragmented (unused gap / reserved).

    Returns ``None`` when ``allocator_reserved_bytes`` is zero to avoid
    division-by-zero.
    """
    if allocator_allocated_bytes is None or not allocator_reserved_bytes:
        return None
    gap = _compute_allocator_gap_bytes(
        allocator_allocated_bytes, allocator_reserved_bytes
    )
    return gap / allocator_reserved_bytes if gap is not None else None


# ---------------------------------------------------------------------------
# Shared accessor
# ---------------------------------------------------------------------------


def _event_get(event: Any, key: str, default: Any = None) -> Any:
    """Uniform attribute access for dataclasses and mapping inputs."""
    if isinstance(event, MappingABC):
        return event.get(key, default)
    return getattr(event, key, default)


def memory_feature_enabled(event: Any, feature: str) -> bool:
    """Honor explicit capability flags while preserving unannotated legacy events."""
    metadata = _event_get(event, "metadata", {})
    capabilities = (
        metadata.get("memory_capabilities", {})
        if isinstance(metadata, MappingABC)
        else {}
    )
    return not (
        isinstance(capabilities, MappingABC) and capabilities.get(feature) is False
    )


def fragmentation_unavailable_reason(events: Sequence[Any]) -> str | None:
    """Explain when every supplied sample explicitly disables fragmentation."""
    if events and not any(
        memory_feature_enabled(event, "supports_fragmentation_analysis")
        for event in events
    ):
        return "Collector capabilities disable fragmentation analysis."
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_event_fields(event: Any) -> DerivedFields:
    """Compute row-scoped derived fields from a single telemetry event.

    ``event`` may be a ``TelemetryEventV2``, ``TelemetryEventV3``, or any
    object / mapping that exposes the standard allocator counter attributes.

    ``allocator_gap_bytes`` is always present in the returned dict.  Fields that
    require raw counters which may be absent (e.g. ``device_total_bytes``) are
    returned as ``None`` rather than being omitted.

    Args:
        event: A telemetry event object or mapping.

    Returns:
        A :class:`DerivedFields` dict.
    """
    raw_allocated = _event_get(event, "allocator_allocated_bytes")
    raw_reserved = _event_get(event, "allocator_reserved_bytes")
    raw_device_used = _event_get(event, "device_used_bytes")
    allocated = int(raw_allocated) if raw_allocated is not None else None
    reserved = int(raw_reserved) if raw_reserved is not None else None
    device_used = int(raw_device_used) if raw_device_used is not None else allocated
    use_device_utilization = allocated is None or not memory_feature_enabled(
        event, "supports_allocator_allocated"
    )
    device_total: None | int = _event_get(event, "device_total_bytes")
    collector: None | str = _event_get(event, "collector")

    result: DerivedFields = {
        "allocator_gap_bytes": _compute_allocator_gap_bytes(allocated, reserved),
        "utilization_ratio": _compute_utilization_ratio(
            device_used if use_device_utilization else allocated,
            device_total,
        ),
        "fragmentation_ratio": (
            _compute_fragmentation_ratio(allocated, reserved)
            if memory_feature_enabled(event, "supports_fragmentation_analysis")
            else None
        ),
        "is_degraded_collector": _collector_is_degraded(collector),
    }
    return result


def compute_session_fields(events: Sequence[Any]) -> SessionDerivedFields:
    """Compute session-scoped rollups across an ordered sequence of events.

    Args:
        events: Ordered sequence of telemetry event objects or mappings.

    Returns:
        A :class:`SessionDerivedFields` dict with:

        * ``peak_utilization_ratio`` - highest per-event utilization seen; or
          ``None`` if ``device_total_bytes`` was unavailable in every event.
        * ``avg_fragmentation_ratio`` - mean fragmentation ratio across events
          where ``allocator_reserved_bytes > 0``; ``None`` if none qualify.
        * ``is_session_interrupted`` - ``True`` when the last event is not a
          ``"stop"`` event.
    """
    util_values: list[float] = []
    frag_values: list[float] = []

    for event in events:
        per_event = compute_event_fields(event)
        u = per_event.get("utilization_ratio")
        if u is not None:
            util_values.append(u)
        f = per_event.get("fragmentation_ratio")
        if f is not None:
            frag_values.append(f)

    peak_utilization: None | float = max(util_values) if util_values else None
    avg_fragmentation: None | float = (
        sum(frag_values) / len(frag_values) if frag_values else None
    )

    last_event_type: str = ""
    if events:
        last_event_type = _event_get(events[-1], "event_type", "")

    is_interrupted = last_event_type != "stop"

    return {
        "peak_utilization_ratio": peak_utilization,
        "avg_fragmentation_ratio": avg_fragmentation,
        "is_session_interrupted": is_interrupted,
    }


def enrich_event(event: Any) -> Dict[str, Any]:
    """Return a flat dict merging raw event fields with derived fields.

    Derived fields are nested under a ``"derived"`` key so that raw and
    computed fields never collide.

    Args:
        event: A telemetry event object or mapping.

    Returns:
        ``{"schema_version": ..., ..., "derived": {"allocator_gap_bytes": ..., ...}}``
    """
    if isinstance(event, MappingABC):
        raw: Dict[str, Any] = dict(event)
    elif dataclasses.is_dataclass(event):
        raw = dataclasses.asdict(cast(Any, event))
    else:
        raw = dict(vars(event))

    raw["derived"] = dict(compute_event_fields(event))
    return raw


__all__ = [
    "DerivedFields",
    "SessionDerivedFields",
    "compute_event_fields",
    "compute_session_fields",
    "enrich_event",
]
