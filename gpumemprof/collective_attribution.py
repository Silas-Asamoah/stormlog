"""Heuristics for attributing hidden-memory spikes to collective communication."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .telemetry import TelemetryEventV2

_COLLECTIVE_TOKENS = (
    "nccl",
    "collective",
    "communication",
    "all_reduce",
    "allreduce",
    "all_gather",
    "allgather",
    "reduce_scatter",
    "reducescatter",
    "broadcast",
    "barrier",
)


@dataclass
class CollectiveAttributionConfig:
    """Runtime knobs for collective-memory attribution heuristics."""

    preset: str = "medium"
    min_samples_per_rank: int = 6
    min_gap_bytes: int = 128 * 1024 * 1024
    min_gap_ratio: float = 0.04
    robust_zscore_threshold: float = 2.5
    marker_window_ns: int = 150_000_000
    interval_padding_ns: int = 120_000_000
    synchrony_window_ns: int = 120_000_000
    min_synchrony_ratio: float = 0.5
    min_confidence: float = 0.5


@dataclass
class CollectiveAttributionEvidence:
    """Evidence fields backing one collective-attribution output."""

    marker_hits: int
    synchronized_ranks: int
    expected_world_size: int
    synchrony_ratio: float
    peak_gap_bytes: int
    peak_gap_ratio: float | None
    robust_zscore: float


@dataclass
class CollectiveAttributionResult:
    """Communication-attributed hidden-memory interval."""

    rank: int
    interval_start_ns: int
    interval_end_ns: int
    classification: str
    confidence: float
    reason_codes: list[str] = field(default_factory=list)
    evidence: CollectiveAttributionEvidence | None = None


@dataclass(frozen=True)
class _RankSpike:
    key: tuple[int, int, int]
    rank: int
    timestamp_ns: int
    peak_gap_bytes: int
    peak_gap_ratio: float | None
    robust_zscore: float
    marker_times: tuple[int, ...]


_PRESET_CONFIGS: dict[str, CollectiveAttributionConfig] = {
    "low": CollectiveAttributionConfig(
        preset="low",
        min_samples_per_rank=8,
        min_gap_bytes=256 * 1024 * 1024,
        min_gap_ratio=0.06,
        robust_zscore_threshold=3.0,
        marker_window_ns=120_000_000,
        interval_padding_ns=100_000_000,
        synchrony_window_ns=100_000_000,
        min_synchrony_ratio=0.6,
        min_confidence=0.7,
    ),
    "medium": CollectiveAttributionConfig(),
    "high": CollectiveAttributionConfig(
        preset="high",
        min_samples_per_rank=5,
        min_gap_bytes=64 * 1024 * 1024,
        min_gap_ratio=0.025,
        robust_zscore_threshold=1.8,
        marker_window_ns=180_000_000,
        interval_padding_ns=160_000_000,
        synchrony_window_ns=180_000_000,
        min_synchrony_ratio=0.34,
        min_confidence=0.35,
    ),
}

_OVERRIDABLE_FIELDS = frozenset(
    {
        "min_samples_per_rank",
        "min_gap_bytes",
        "min_gap_ratio",
        "robust_zscore_threshold",
        "marker_window_ns",
        "interval_padding_ns",
        "synchrony_window_ns",
        "min_synchrony_ratio",
        "min_confidence",
    }
)


def resolve_collective_attribution_config(
    preset: str = "medium",
    overrides: Mapping[str, Any] | None = None,
) -> CollectiveAttributionConfig:
    """Resolve a preset config with optional per-threshold overrides."""

    normalized_preset = (preset or "medium").strip().lower()
    if normalized_preset not in _PRESET_CONFIGS:
        known = ", ".join(sorted(_PRESET_CONFIGS))
        raise ValueError(
            f"Unknown collective attribution preset: {preset!r} (known: {known})"
        )

    config = replace(_PRESET_CONFIGS[normalized_preset])
    config.preset = normalized_preset

    if overrides:
        unknown = sorted(key for key in overrides if key not in _OVERRIDABLE_FIELDS)
        if unknown:
            raise ValueError(
                "Unknown collective attribution override fields: " + ", ".join(unknown)
            )
        typed_overrides = {key: overrides[key] for key in overrides}
        config = replace(config, **typed_overrides)

    _validate_collective_config(config)
    return config


def attribute_collective_memory(
    events: Sequence[TelemetryEventV2],
    *,
    config: CollectiveAttributionConfig | None = None,
    preset: str = "medium",
    overrides: Mapping[str, Any] | None = None,
) -> list[CollectiveAttributionResult]:
    """Attribute hidden-memory spikes to communication phases using hybrid signals."""

    if not events:
        return []

    resolved = config or resolve_collective_attribution_config(preset, overrides)
    ordered_events = sorted(events, key=lambda item: item.timestamp_ns)
    marker_timestamps = tuple(
        event.timestamp_ns
        for event in ordered_events
        if _event_has_collective_marker(event)
    )
    grouped_samples = _group_sample_events_by_rank(ordered_events)
    if not grouped_samples:
        return []

    spikes_by_rank: dict[int, list[_RankSpike]] = {}
    for rank, rank_samples in grouped_samples.items():
        spikes = _detect_rank_spikes(
            rank=rank,
            rank_events=rank_samples,
            marker_timestamps=marker_timestamps,
            config=resolved,
        )
        if spikes:
            spikes_by_rank[rank] = spikes

    if not spikes_by_rank:
        return []

    expected_world_size = _expected_world_size(ordered_events)
    synchrony_by_spike = _build_synchrony_lookup(
        spikes_by_rank, resolved.synchrony_window_ns
    )

    results: list[CollectiveAttributionResult] = []
    for rank, spikes in spikes_by_rank.items():
        for spike in spikes:
            synchronized_ranks = synchrony_by_spike.get(spike.key, {rank})
            scored = _score_spike(
                spike=spike,
                synchronized_ranks=synchronized_ranks,
                expected_world_size=expected_world_size,
                config=resolved,
            )
            if scored is not None and scored.confidence >= resolved.min_confidence:
                results.append(scored)

    return _merge_rank_intervals(results)


def _validate_collective_config(config: CollectiveAttributionConfig) -> None:
    if config.min_samples_per_rank < 3:
        raise ValueError("min_samples_per_rank must be >= 3")
    if config.min_gap_bytes < 0:
        raise ValueError("min_gap_bytes must be >= 0")
    if config.min_gap_ratio < 0:
        raise ValueError("min_gap_ratio must be >= 0")
    if config.robust_zscore_threshold <= 0:
        raise ValueError("robust_zscore_threshold must be > 0")
    if config.marker_window_ns < 0:
        raise ValueError("marker_window_ns must be >= 0")
    if config.interval_padding_ns < 0:
        raise ValueError("interval_padding_ns must be >= 0")
    if config.synchrony_window_ns < 0:
        raise ValueError("synchrony_window_ns must be >= 0")
    if not 0 <= config.min_synchrony_ratio <= 1:
        raise ValueError("min_synchrony_ratio must be in [0, 1]")
    if not 0 <= config.min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1]")


def _group_sample_events_by_rank(
    events: Iterable[TelemetryEventV2],
) -> dict[int, list[TelemetryEventV2]]:
    grouped: dict[int, list[TelemetryEventV2]] = {}
    for event in events:
        if event.event_type != "sample":
            continue
        grouped.setdefault(event.rank, []).append(event)
    return grouped


def _detect_rank_spikes(
    *,
    rank: int,
    rank_events: Sequence[TelemetryEventV2],
    marker_timestamps: Sequence[int],
    config: CollectiveAttributionConfig,
) -> list[_RankSpike]:
    if len(rank_events) < config.min_samples_per_rank:
        return []

    positive_gaps = np.asarray(
        [
            max(0, event.device_used_bytes - event.allocator_reserved_bytes)
            for event in rank_events
        ],
        dtype=float,
    )
    if positive_gaps.size < config.min_samples_per_rank:
        return []

    spikes: list[_RankSpike] = []
    for sample_index, event in enumerate(rank_events):
        gap_bytes = max(0, event.device_used_bytes - event.allocator_reserved_bytes)
        gap_ratio = _compute_gap_ratio(event, gap_bytes)
        if not _is_significant_gap(
            gap_bytes=gap_bytes,
            gap_ratio=gap_ratio,
            config=config,
        ):
            continue

        robust_zscore = _robust_zscore(positive_gaps, float(gap_bytes))
        if robust_zscore < config.robust_zscore_threshold:
            continue

        nearby_markers = tuple(
            ts
            for ts in marker_timestamps
            if abs(ts - event.timestamp_ns) <= config.marker_window_ns
        )
        spikes.append(
            _RankSpike(
                key=(rank, event.timestamp_ns, sample_index),
                rank=rank,
                timestamp_ns=event.timestamp_ns,
                peak_gap_bytes=int(gap_bytes),
                peak_gap_ratio=gap_ratio,
                robust_zscore=round(float(robust_zscore), 4),
                marker_times=nearby_markers,
            )
        )

    return spikes


def _compute_gap_ratio(event: TelemetryEventV2, gap_bytes: int) -> float | None:
    if event.device_total_bytes is None or event.device_total_bytes <= 0:
        return None
    return abs(gap_bytes) / event.device_total_bytes


def _is_significant_gap(
    *,
    gap_bytes: int,
    gap_ratio: float | None,
    config: CollectiveAttributionConfig,
) -> bool:
    ratio_significant = gap_ratio is not None and gap_ratio >= config.min_gap_ratio
    bytes_significant = gap_bytes >= config.min_gap_bytes
    return ratio_significant or bytes_significant


def _robust_zscore(values: np.ndarray, value: float) -> float:
    if values.size < 3:
        return 0.0

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0:
        return max(0.0, 0.6745 * (value - median) / mad)

    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    if std > 0:
        mean = float(np.mean(values))
        return max(0.0, (value - mean) / std)
    return 0.0


def _expected_world_size(events: Sequence[TelemetryEventV2]) -> int:
    world_sizes = [event.world_size for event in events if event.world_size > 0]
    if world_sizes:
        return max(world_sizes)
    ranks = {event.rank for event in events}
    return max(len(ranks), 1)


def _build_synchrony_lookup(
    spikes_by_rank: Mapping[int, Sequence[_RankSpike]],
    synchrony_window_ns: int,
) -> dict[tuple[int, int, int], set[int]]:
    all_spikes = [spike for spikes in spikes_by_rank.values() for spike in spikes]
    lookup: dict[tuple[int, int, int], set[int]] = {}
    for spike in all_spikes:
        synchronized = {
            other.rank
            for other in all_spikes
            if abs(other.timestamp_ns - spike.timestamp_ns) <= synchrony_window_ns
        }
        if not synchronized:
            synchronized = {spike.rank}
        lookup[spike.key] = synchronized
    return lookup


def _score_spike(
    *,
    spike: _RankSpike,
    synchronized_ranks: set[int],
    expected_world_size: int,
    config: CollectiveAttributionConfig,
) -> CollectiveAttributionResult | None:
    marker_overlap = bool(spike.marker_times)
    synchronized_count = max(len(synchronized_ranks), 1)

    if expected_world_size <= 1:
        synchrony_ratio = 0.0
    else:
        synchrony_ratio = min(1.0, (synchronized_count - 1) / (expected_world_size - 1))

    zscore_strength = min(1.0, spike.robust_zscore / config.robust_zscore_threshold)
    bytes_strength = (
        min(1.0, spike.peak_gap_bytes / config.min_gap_bytes)
        if config.min_gap_bytes > 0
        else 1.0
    )
    ratio_strength = (
        min(1.0, (spike.peak_gap_ratio or 0.0) / config.min_gap_ratio)
        if config.min_gap_ratio > 0
        else 1.0
    )
    divergence_strength = max(bytes_strength, ratio_strength)

    confidence = 0.0
    confidence += 0.34 if marker_overlap else 0.0
    confidence += 0.34 * synchrony_ratio
    confidence += 0.16 * zscore_strength
    confidence += 0.16 * divergence_strength

    reason_codes: list[str] = ["gap_spike_statistical_outlier"]

    if marker_overlap:
        reason_codes.extend(["marker_collective_token", "marker_spike_overlap"])
    else:
        confidence -= 0.15
        reason_codes.append("weak_marker_signal")

    if synchrony_ratio >= config.min_synchrony_ratio:
        reason_codes.append("cross_rank_synchrony")

    if divergence_strength >= 0.5:
        reason_codes.append("allocator_device_divergence")

    if expected_world_size <= 1 or synchronized_count <= 1:
        confidence -= 0.15
        reason_codes.append("single_rank_only")

    if spike.peak_gap_ratio is None:
        confidence -= 0.05

    confidence = max(0.0, min(1.0, confidence))

    if confidence <= 0:
        return None

    marker_start = min(spike.marker_times) if spike.marker_times else spike.timestamp_ns
    marker_end = max(spike.marker_times) if spike.marker_times else spike.timestamp_ns
    interval_start = min(marker_start, spike.timestamp_ns) - config.interval_padding_ns
    interval_end = max(marker_end, spike.timestamp_ns) + config.interval_padding_ns

    evidence = CollectiveAttributionEvidence(
        marker_hits=len(spike.marker_times),
        synchronized_ranks=synchronized_count,
        expected_world_size=expected_world_size,
        synchrony_ratio=round(float(synchrony_ratio), 4),
        peak_gap_bytes=spike.peak_gap_bytes,
        peak_gap_ratio=(
            round(float(spike.peak_gap_ratio), 6)
            if spike.peak_gap_ratio is not None
            else None
        ),
        robust_zscore=round(float(spike.robust_zscore), 4),
    )

    if confidence >= 0.8:
        classification = "collective_confident"
    elif confidence >= 0.6:
        classification = "collective_likely"
    else:
        classification = "collective_suspect"

    return CollectiveAttributionResult(
        rank=spike.rank,
        interval_start_ns=interval_start,
        interval_end_ns=interval_end,
        classification=classification,
        confidence=round(float(confidence), 3),
        reason_codes=sorted(set(reason_codes)),
        evidence=evidence,
    )


def _merge_rank_intervals(
    results: Sequence[CollectiveAttributionResult],
) -> list[CollectiveAttributionResult]:
    if not results:
        return []

    merged: list[CollectiveAttributionResult] = []
    for result in sorted(results, key=lambda item: (item.rank, item.interval_start_ns)):
        if not merged:
            merged.append(result)
            continue

        prev = merged[-1]
        same_rank = prev.rank == result.rank
        overlaps = result.interval_start_ns <= prev.interval_end_ns
        if not same_rank or not overlaps:
            merged.append(result)
            continue

        prev_evidence = prev.evidence
        curr_evidence = result.evidence
        if prev_evidence is None:
            merged_evidence = curr_evidence
        elif curr_evidence is None:
            merged_evidence = prev_evidence
        else:
            merged_evidence = CollectiveAttributionEvidence(
                marker_hits=max(prev_evidence.marker_hits, curr_evidence.marker_hits),
                synchronized_ranks=max(
                    prev_evidence.synchronized_ranks, curr_evidence.synchronized_ranks
                ),
                expected_world_size=max(
                    prev_evidence.expected_world_size, curr_evidence.expected_world_size
                ),
                synchrony_ratio=max(
                    prev_evidence.synchrony_ratio, curr_evidence.synchrony_ratio
                ),
                peak_gap_bytes=max(
                    prev_evidence.peak_gap_bytes, curr_evidence.peak_gap_bytes
                ),
                peak_gap_ratio=_max_optional_ratio(
                    prev_evidence.peak_gap_ratio,
                    curr_evidence.peak_gap_ratio,
                ),
                robust_zscore=max(
                    prev_evidence.robust_zscore, curr_evidence.robust_zscore
                ),
            )

        merged[-1] = CollectiveAttributionResult(
            rank=prev.rank,
            interval_start_ns=min(prev.interval_start_ns, result.interval_start_ns),
            interval_end_ns=max(prev.interval_end_ns, result.interval_end_ns),
            classification=_merge_classification(
                prev.classification, result.classification
            ),
            confidence=max(prev.confidence, result.confidence),
            reason_codes=sorted(set(prev.reason_codes + result.reason_codes)),
            evidence=merged_evidence,
        )

    return merged


def _max_optional_ratio(first: float | None, second: float | None) -> float | None:
    if first is None:
        return second
    if second is None:
        return first
    return max(first, second)


def _merge_classification(first: str, second: str) -> str:
    order = {
        "collective_suspect": 0,
        "collective_likely": 1,
        "collective_confident": 2,
    }
    return first if order.get(first, -1) >= order.get(second, -1) else second


def _event_has_collective_marker(event: TelemetryEventV2) -> bool:
    event_type = str(getattr(event, "event_type", ""))
    context = getattr(event, "context", None)
    metadata = getattr(event, "metadata", {})

    text_fragments: list[str] = [event_type]
    if isinstance(context, str) and context:
        text_fragments.append(context)
    text_fragments.extend(_iter_string_values(metadata))

    for fragment in text_fragments:
        if _contains_collective_token(fragment):
            return True
    return False


def _iter_string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str):
                yield key
            yield from _iter_string_values(nested)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_string_values(item)


def _contains_collective_token(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return False

    normalized = lowered.replace("-", "_")
    collapsed = normalized.replace("_", "")
    return any(
        token in normalized or token.replace("_", "") in collapsed
        for token in _COLLECTIVE_TOKENS
    )


__all__ = [
    "CollectiveAttributionConfig",
    "CollectiveAttributionEvidence",
    "CollectiveAttributionResult",
    "attribute_collective_memory",
    "resolve_collective_attribution_config",
]
