"""Phase attribution policy and formatting helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from .replay import PhaseReplayIndex as ReplayPhaseReplayIndex
    from .replay import PhaseSpan


@dataclass(frozen=True)
class PhaseSummary:
    """Optional presentation-oriented phase winner when canonical attribution stays ambiguous."""

    phase_path: str
    source: str


@dataclass(frozen=True)
class PhaseAttribution:
    """Resolved workload phase attribution for an anomaly or report item."""

    phase_resolution: str
    phase_source: str | None = None
    phase_path: str | None = None
    phase_paths: list[str] = field(default_factory=list)
    scope_id: str | None = None
    thread_id: int | None = None
    thread_name: str | None = None
    phase_summary: PhaseSummary | None = None


def format_phase_path(path: Sequence[str]) -> str:
    """Return a human-readable phase path label."""
    return " / ".join(part for part in path if part)


def attribute_active_spans(
    spans: Sequence["PhaseSpan"],
    *,
    strategy: str = "deepest_per_thread",
    origin_thread_id: int | None = None,
    origin_phase_scope_id: str | None = None,
) -> PhaseAttribution | None:
    """Collapse active spans into a unique or ambiguity-preserving attribution."""
    if strategy != "deepest_per_thread":
        raise ValueError(f"Unsupported phase attribution strategy: {strategy}")

    if not spans:
        return None

    origin_attribution = _attribute_origin_spans(
        spans,
        origin_thread_id=origin_thread_id,
        origin_phase_scope_id=origin_phase_scope_id,
    )
    if origin_attribution is not None:
        return origin_attribution

    thread_matches, ambiguous_labels = _deepest_thread_matches(spans)
    if not thread_matches and not ambiguous_labels:
        return None

    if ambiguous_labels or len(thread_matches) != 1:
        phase_paths = sorted(
            ambiguous_labels | {format_phase_path(span.path) for span in thread_matches}
        )
        summary = _build_phase_summary(
            spans=spans,
            phase_paths=phase_paths,
        )
        return PhaseAttribution(
            phase_resolution="ambiguous",
            phase_paths=phase_paths,
            phase_summary=summary,
        )

    return _build_unique_attribution(thread_matches[0], source="heuristic")


def _attribute_origin_spans(
    spans: Sequence["PhaseSpan"],
    *,
    origin_thread_id: int | None,
    origin_phase_scope_id: str | None,
) -> PhaseAttribution | None:
    """Prefer an exact scope identity, falling back to its originating thread."""
    if origin_phase_scope_id:
        exact_matches = [
            span for span in spans if span.scope_id == origin_phase_scope_id
        ]
        if len(exact_matches) == 1:
            return _build_unique_attribution(exact_matches[0], source="exact")

    if origin_thread_id is not None:
        thread_spans = [span for span in spans if span.thread_id == origin_thread_id]
        if thread_spans:
            return _build_unique_attribution(
                _most_recent_active_span(thread_spans),
                source="thread_local",
            )
    return None


def _deepest_thread_matches(
    spans: Sequence["PhaseSpan"],
) -> tuple[list["PhaseSpan"], set[str]]:
    """Choose deepest spans per thread while retaining conflicting path labels."""
    thread_matches: list["PhaseSpan"] = []
    ambiguous_labels: set[str] = set()
    deepest_by_thread: dict[int, list["PhaseSpan"]] = {}
    for span in spans:
        deepest_by_thread.setdefault(span.thread_id, []).append(span)

    for thread_spans in deepest_by_thread.values():
        max_depth = max(item.depth for item in thread_spans)
        deepest = [item for item in thread_spans if item.depth == max_depth]
        labels = {format_phase_path(item.path) for item in deepest}
        if len(labels) != 1:
            ambiguous_labels.update(labels)
            continue
        thread_matches.append(
            max(deepest, key=lambda item: (item.sequence, item.scope_id))
        )

    return thread_matches, ambiguous_labels


def resolve_phase_for_event(
    index: "ReplayPhaseReplayIndex", event: Any
) -> PhaseAttribution | None:
    """Resolve a phase attribution for one event-like object using replay data."""
    timestamp_ns = _event_field(event, "timestamp_ns")
    session_id = _event_field(event, "session_id")
    if not isinstance(timestamp_ns, int) or not isinstance(session_id, str):
        return None
    rank = _coerce_rank(_event_field(event, "rank", 0))
    if rank is None:
        return None
    origin_thread_id = _origin_thread_id_for_event(event)
    origin_phase_scope_id = _origin_phase_scope_id_for_event(event)
    spans = index.active_spans(
        timestamp_ns=timestamp_ns,
        session_id=session_id,
        rank=rank,
    )
    return attribute_active_spans(
        spans,
        origin_thread_id=origin_thread_id,
        origin_phase_scope_id=origin_phase_scope_id,
    )


def summarize_phase_attribution(attribution: PhaseAttribution | None) -> str | None:
    """Return a user-facing summary string for one phase attribution."""
    if attribution is None:
        return None
    if attribution.phase_summary is not None:
        if attribution.phase_summary.source == "heuristic":
            return f"(likely) {attribution.phase_summary.phase_path}"
        return attribution.phase_summary.phase_path
    return summarize_phase_resolution(
        phase_resolution=attribution.phase_resolution,
        phase_path=attribution.phase_path,
        phase_paths=attribution.phase_paths,
    )


def summarize_phase_resolution(
    *,
    phase_resolution: str | None,
    phase_path: str | None = None,
    phase_paths: Sequence[str] | None = None,
) -> str | None:
    """Render one phase resolution without hiding ambiguity semantics."""
    labels = [path for path in (phase_paths or ()) if path]
    if phase_resolution == "unique":
        if phase_path:
            return phase_path
        if len(labels) == 1:
            return labels[0]
        return None
    if phase_resolution == "ambiguous" and labels:
        return f"(ambiguous) {' | '.join(labels)}"
    return None


def merge_phase_attributions(
    first: PhaseAttribution | None,
    second: PhaseAttribution | None,
) -> PhaseAttribution | None:
    """Merge two attribution candidates without inventing a false unique path."""
    if first is None:
        return second
    if second is None:
        return first

    first_paths = _phase_paths(first)
    second_paths = _phase_paths(second)
    if not first_paths:
        return second
    if not second_paths:
        return first

    merged_paths = sorted(set(first_paths) | set(second_paths))
    if len(merged_paths) == 1:
        phase_path = merged_paths[0]
        if _same_unique_scope(first, second):
            return first
        return PhaseAttribution(
            phase_resolution="ambiguous",
            phase_paths=[phase_path],
        )

    return PhaseAttribution(
        phase_resolution="ambiguous",
        phase_paths=merged_paths,
    )


def phase_attribution_to_payload(
    attribution: PhaseAttribution | None,
) -> dict[str, Any] | None:
    """Serialize a phase attribution without emitting redundant summary fields."""
    if attribution is None:
        return None

    payload: dict[str, Any] = {
        "phase_resolution": attribution.phase_resolution,
        "phase_paths": [path for path in attribution.phase_paths if path],
    }
    if attribution.phase_source is not None:
        payload["phase_source"] = attribution.phase_source
    if attribution.phase_path is not None:
        payload["phase_path"] = attribution.phase_path
    if attribution.scope_id is not None:
        payload["scope_id"] = attribution.scope_id
    if attribution.thread_id is not None:
        payload["thread_id"] = attribution.thread_id
    if attribution.thread_name is not None:
        payload["thread_name"] = attribution.thread_name
    if attribution.phase_summary is not None:
        payload["phase_summary"] = {
            "phase_path": attribution.phase_summary.phase_path,
            "source": attribution.phase_summary.source,
        }
    return payload


def _build_unique_attribution(span: "PhaseSpan", *, source: str) -> PhaseAttribution:
    phase_path = format_phase_path(span.path)
    return PhaseAttribution(
        phase_resolution="unique",
        phase_source=source,
        phase_path=phase_path,
        phase_paths=[phase_path],
        scope_id=span.scope_id,
        thread_id=span.thread_id,
        thread_name=span.thread_name,
    )


def _build_phase_summary(
    *,
    spans: Sequence["PhaseSpan"],
    phase_paths: Sequence[str],
) -> PhaseSummary | None:
    unique_paths = sorted({path for path in phase_paths if path})
    if len(unique_paths) <= 1:
        return None
    winner = _most_recent_active_span(spans)
    winner_path = format_phase_path(winner.path)
    if not winner_path:
        return None
    return PhaseSummary(phase_path=winner_path, source="heuristic")


def _most_recent_active_span(spans: Sequence["PhaseSpan"]) -> "PhaseSpan":
    return max(spans, key=lambda item: (item.sequence, item.scope_id))


def _phase_paths(attribution: PhaseAttribution) -> list[str]:
    if attribution.phase_paths:
        return [path for path in attribution.phase_paths if path]
    if attribution.phase_path:
        return [attribution.phase_path]
    return []


def _origin_thread_id_for_event(event: Any) -> int | None:
    origin_thread_id = _event_field(event, "origin_thread_id")
    if isinstance(origin_thread_id, int):
        return origin_thread_id
    metadata = _event_field(event, "metadata", {})
    if isinstance(metadata, dict):
        raw_origin = metadata.get("origin_thread_id")
        if isinstance(raw_origin, int):
            return raw_origin
    return None


def _origin_phase_scope_id_for_event(event: Any) -> str | None:
    origin_phase_scope_id = _event_field(event, "origin_phase_scope_id")
    if isinstance(origin_phase_scope_id, str) and origin_phase_scope_id:
        return origin_phase_scope_id
    metadata = _event_field(event, "metadata", {})
    if isinstance(metadata, dict):
        raw_origin = metadata.get("origin_phase_scope_id")
        if isinstance(raw_origin, str) and raw_origin:
            return raw_origin
    return None


def _event_field(event: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(event, dict):
        return event.get(field_name, default)
    return getattr(event, field_name, default)


def _coerce_rank(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _same_unique_scope(first: PhaseAttribution, second: PhaseAttribution) -> bool:
    return (
        first.phase_resolution == "unique"
        and second.phase_resolution == "unique"
        and first.phase_path == second.phase_path
        and first.scope_id == second.scope_id
        and first.thread_id == second.thread_id
    )


__all__ = [
    "PhaseAttribution",
    "PhaseSummary",
    "attribute_active_spans",
    "format_phase_path",
    "merge_phase_attributions",
    "phase_attribution_to_payload",
    "resolve_phase_for_event",
    "summarize_phase_attribution",
    "summarize_phase_resolution",
]
