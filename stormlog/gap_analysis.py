"""Shared hidden-memory gap analysis utilities."""

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Sequence

import numpy as np
from scipy import stats

from .telemetry import TelemetryEventV2

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


def analyze_hidden_memory_gaps(
    events: Sequence[TelemetryEventV2],
    thresholds: Mapping[str, float],
    format_memory: Callable[[int], str],
    remediation_by_classification: Mapping[str, List[str]],
) -> List[GapFinding]:
    """Classify allocator-vs-device hidden memory gaps over time."""
    gaps: List[float] = []
    normalized: List[float] = []
    timestamps_ns: List[int] = []
    usable_events: List[TelemetryEventV2] = []

    for event in events:
        if event.device_total_bytes is None or event.device_total_bytes <= 0:
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
            gaps=gaps,
            z_threshold=thresholds["gap_spike_zscore"],
            remediation=remediation_by_classification.get("transient_spike", []),
        )
    )
    findings.extend(
        _detect_gap_persistent_drift(
            gaps=gaps,
            timestamps_ns=timestamps_ns,
            drift_r_squared_threshold=thresholds["gap_drift_r_squared"],
            format_memory=format_memory,
            remediation=remediation_by_classification.get("persistent_drift", []),
        )
    )
    findings.extend(
        _detect_gap_fragmentation_pattern(
            events=usable_events,
            gaps=gaps,
            fragmentation_threshold=thresholds["gap_fragmentation_ratio"],
            remediation=remediation_by_classification.get("fragmentation_like", []),
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
    gaps: List[float],
    z_threshold: float,
    remediation: List[str],
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
        )
    ]


def _detect_gap_persistent_drift(
    gaps: List[float],
    timestamps_ns: List[int],
    drift_r_squared_threshold: float,
    format_memory: Callable[[int], str],
    remediation: List[str],
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
        )
    ]


def _detect_gap_fragmentation_pattern(
    events: Sequence[TelemetryEventV2],
    gaps: List[float],
    fragmentation_threshold: float,
    remediation: List[str],
) -> List[GapFinding]:
    """Detect fragmentation-like behaviour: high reserved-allocated ratio."""
    frag_ratios: List[float] = []
    for event in events:
        reserved = event.allocator_reserved_bytes
        allocated = event.allocator_allocated_bytes
        if reserved > 0:
            frag_ratios.append((reserved - allocated) / reserved)

    if not frag_ratios:
        return []

    avg_frag = float(np.mean(frag_ratios))
    max_frag = float(np.max(frag_ratios))
    if avg_frag < fragmentation_threshold:
        return []

    severity = "critical" if avg_frag > 0.5 else "warning"
    confidence = round(min(1.0, avg_frag / 0.6), 3)

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
        )
    ]
