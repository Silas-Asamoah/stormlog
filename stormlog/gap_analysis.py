"""Shared hidden-memory gap analysis utilities."""

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Sequence, cast

import numpy as np
from scipy import stats

from .derived_fields import compute_event_fields

try:
    from .phases import PhaseAttribution, PhaseReplayIndex
except ImportError:  # pragma: no cover - phase package may land in another slice
    PhaseAttribution = Any  # type: ignore[assignment,misc]
    PhaseReplayIndex = Any  # type: ignore[assignment,misc]
from .telemetry import TelemetryEventLike

_LOGGER = logging.getLogger(__name__)


@dataclass
class GapFinding:
    """A classified finding from hidden-memory gap analysis."""

    classification: str  # 'transient_spike' | 'persistent_drift' | 'fragmentation_like'
    severity: str  # 'info', 'warning', 'critical'
    confidence: float  # 0.0 to 1.0
    evidence: dict[str, Any]
    description: str
    remediation: List[str]
    evidence_timestamp_ns: int | None = None
    phase_attribution: PhaseAttribution | None = None


def analyze_hidden_memory_gaps(
    events: Sequence[TelemetryEventLike],
    thresholds: Mapping[str, float],
    format_memory: Callable[[int], str],
    remediation_by_classification: Mapping[str, List[str]],
    phase_resolver: PhaseReplayIndex | None = None,
) -> List[GapFinding]:
    """Classify allocator-vs-device hidden memory gaps over time."""
    gaps: List[float] = []
    normalized: List[float] = []
    timestamps_ns: List[int] = []
    usable_events: List[TelemetryEventLike] = []

    for event in events:
        if str(event.event_type).strip().lower() != "sample":
            continue
        if event.device_total_bytes is None or event.device_total_bytes <= 0:
            continue
        if event.device_used_bytes is None or event.allocator_reserved_bytes is None:
            continue
        gap = event.device_used_bytes - event.allocator_reserved_bytes
        gaps.append(float(gap))
        normalized.append(gap / event.device_total_bytes)
        timestamps_ns.append(event.timestamp_ns)
        usable_events.append(event)

    if len(gaps) < 3:
        return []

    threshold = thresholds["gap_ratio_threshold"]
    if max(abs(value) for value in normalized) < threshold:
        return []

    _LOGGER.debug(
        "Running hidden-memory gap classifiers on %d samples", len(usable_events)
    )

    findings: List[GapFinding] = []
    findings.extend(
        _detect_gap_transient_spikes(
            events=usable_events,
            gaps=gaps,
            z_threshold=thresholds["gap_spike_zscore"],
            remediation=remediation_by_classification.get("transient_spike", []),
            phase_resolver=phase_resolver,
        )
    )
    findings.extend(
        _detect_gap_persistent_drift(
            events=usable_events,
            gaps=gaps,
            timestamps_ns=timestamps_ns,
            drift_r_squared_threshold=thresholds["gap_drift_r_squared"],
            format_memory=format_memory,
            remediation=remediation_by_classification.get("persistent_drift", []),
            phase_resolver=phase_resolver,
        )
    )
    findings.extend(
        _detect_gap_fragmentation_pattern(
            events=usable_events,
            gaps=gaps,
            fragmentation_threshold=thresholds["gap_fragmentation_ratio"],
            remediation=remediation_by_classification.get("fragmentation_like", []),
            phase_resolver=phase_resolver,
        )
    )

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(
        key=lambda finding: (
            severity_order.get(finding.severity, 9),
            -finding.confidence,
        )
    )

    _LOGGER.debug("Hidden-memory gap analysis produced %d finding(s)", len(findings))
    return findings


def _detect_gap_transient_spikes(
    events: Sequence[TelemetryEventLike],
    gaps: List[float],
    z_threshold: float,
    remediation: List[str],
    phase_resolver: PhaseReplayIndex | None,
) -> List[GapFinding]:
    """Detect transient spikes in the gap series using z-score."""
    arr = np.asarray(gaps, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std == 0 or np.isnan(std):
        return []

    spike_indices = [
        index
        for index, gap_value in enumerate(arr)
        if (gap_value - mean) / std > z_threshold
    ]
    if not spike_indices:
        return []

    max_idx = int(np.argmax(arr))
    max_gap = float(arr[max_idx])
    max_z = (max_gap - mean) / std
    phase_event = events[max_idx]

    severity = "critical" if max_z > 2 * z_threshold else "warning"
    confidence = min(1.0, max_z / (3 * z_threshold))

    return [
        GapFinding(
            classification="transient_spike",
            severity=severity,
            confidence=round(confidence, 3),
            evidence={
                "spike_count": len(spike_indices),
                "max_gap_bytes": max_gap,
                "max_zscore": round(max_z, 3),
                "mean_gap_bytes": round(mean, 1),
                "std_gap_bytes": round(std, 1),
            },
            description=(
                f"Detected {len(spike_indices)} transient spike(s) in the "
                f"device-vs-allocator gap (max z-score {max_z:.1f})."
            ),
            remediation=list(remediation),
            evidence_timestamp_ns=phase_event.timestamp_ns,
            phase_attribution=_resolve_phase_attribution(phase_resolver, phase_event),
        )
    ]


def _detect_gap_persistent_drift(
    events: Sequence[TelemetryEventLike],
    gaps: List[float],
    timestamps_ns: List[int],
    drift_r_squared_threshold: float,
    format_memory: Callable[[int], str],
    remediation: List[str],
    phase_resolver: PhaseReplayIndex | None,
) -> List[GapFinding]:
    """Detect persistent upward drift in the gap via linear regression."""
    if len(gaps) < 5:
        return []

    x = np.asarray(timestamps_ns, dtype=float)
    x = x - x[0]  # relative time
    y = np.asarray(gaps, dtype=float)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value**2

    if slope <= 0 or r_squared < drift_r_squared_threshold:
        return []

    severity = "critical" if r_squared > 0.85 else "warning"
    confidence = round(min(1.0, r_squared), 3)
    slope_bytes_per_sec = slope * 1e9
    phase_event = events[-1]

    return [
        GapFinding(
            classification="persistent_drift",
            severity=severity,
            confidence=confidence,
            evidence={
                "slope_bytes_per_sec": round(slope_bytes_per_sec, 1),
                "r_squared": round(r_squared, 4),
                "p_value": round(p_value, 6),
                "gap_start_bytes": round(float(y[0]), 1),
                "gap_end_bytes": round(float(y[-1]), 1),
            },
            description=(
                f"Device-vs-allocator gap is drifting upward at "
                f"{format_memory(int(abs(slope_bytes_per_sec)))}/s "
                f"(R²={r_squared:.2f})."
            ),
            remediation=list(remediation),
            evidence_timestamp_ns=phase_event.timestamp_ns,
            phase_attribution=_resolve_phase_attribution(phase_resolver, phase_event),
        )
    ]


def _detect_gap_fragmentation_pattern(
    events: Sequence[TelemetryEventLike],
    gaps: List[float],
    fragmentation_threshold: float,
    remediation: List[str],
    phase_resolver: PhaseReplayIndex | None,
) -> List[GapFinding]:
    """Detect fragmentation-like behaviour: high reserved-allocated ratio."""
    frag_ratios: List[float] = []
    frag_events: List[TelemetryEventLike] = []
    for event in events:
        derived = compute_event_fields(event)
        frag = derived.get("fragmentation_ratio")
        if frag is not None:
            frag_ratios.append(frag)
            frag_events.append(event)

    if not frag_ratios:
        return []

    avg_frag = float(np.mean(frag_ratios))
    max_frag = float(np.max(frag_ratios))
    if avg_frag < fragmentation_threshold:
        return []

    severity = "critical" if avg_frag > 0.5 else "warning"
    confidence = round(min(1.0, avg_frag / 0.6), 3)
    phase_event = frag_events[int(np.argmax(frag_ratios))]

    return [
        GapFinding(
            classification="fragmentation_like",
            severity=severity,
            confidence=confidence,
            evidence={
                "avg_fragmentation_ratio": round(avg_frag, 4),
                "max_fragmentation_ratio": round(max_frag, 4),
                "avg_gap_bytes": round(float(np.mean(gaps)), 1),
                "sample_count": len(frag_ratios),
            },
            description=(
                f"Allocator fragmentation averaging {avg_frag:.0%} suggests "
                f"reserved-but-unused memory is inflating the device footprint."
            ),
            remediation=list(remediation),
            evidence_timestamp_ns=phase_event.timestamp_ns,
            phase_attribution=_resolve_phase_attribution(phase_resolver, phase_event),
        )
    ]


def _resolve_phase_attribution(
    phase_resolver: PhaseReplayIndex | None,
    event: TelemetryEventLike,
) -> PhaseAttribution | None:
    if phase_resolver is None:
        return None
    if not hasattr(phase_resolver, "resolve_for_event"):
        return None
    return cast("PhaseAttribution | None", phase_resolver.resolve_for_event(event))
