"""Phase boundary parsing and replay index helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .runtime import (
    PHASE_ENTER_EVENT,
    PHASE_EXIT_EVENT,
    PHASE_SCOPE_ATTRIBUTES_KEY,
    PHASE_SCOPE_METADATA_KEY,
)


@dataclass(frozen=True)
class PhaseBoundaryRecord:
    action: str
    name: str
    path: tuple[str, ...]
    depth: int
    scope_id: str
    parent_scope_id: str | None
    thread_id: int
    thread_name: str
    sequence: int
    session_id: str
    timestamp_ns: int
    attributes: dict[str, Any]


@dataclass(frozen=True)
class PhaseSpan:
    session_id: str
    rank: int
    thread_id: int
    thread_name: str
    scope_id: str
    path: tuple[str, ...]
    start_ns: int
    end_ns: int
    sequence: int
    synthetic_end: bool = False

    @property
    def depth(self) -> int:
        return len(self.path)


class PhaseReplayIndex:
    """Replay phase boundaries into queryable active spans."""

    def __init__(
        self, intervals_by_group: Mapping[tuple[str, int], Sequence[PhaseSpan]]
    ) -> None:
        self._intervals_by_group = {
            key: tuple(sorted(intervals, key=lambda item: (item.start_ns, item.end_ns)))
            for key, intervals in intervals_by_group.items()
        }

    @classmethod
    def from_events(cls, events: Sequence[Any]) -> "PhaseReplayIndex":
        """Build a replay index from telemetry events."""
        boundaries, session_end_by_group = _replay_boundaries(events)

        active_by_thread: dict[tuple[str, int, int], list[PhaseBoundaryRecord]] = {}
        intervals_by_group: dict[tuple[str, int], list[PhaseSpan]] = {}
        for timestamp_ns, _, _, rank, scope in boundaries:
            thread_key = (scope.session_id, rank, scope.thread_id)
            stack = active_by_thread.setdefault(thread_key, [])
            if scope.action == "enter":
                stack.append(scope)
                continue
            if not stack or stack[-1].scope_id != scope.scope_id:
                continue
            opened = stack.pop()
            if not stack:
                active_by_thread.pop(thread_key, None)
            intervals_by_group.setdefault((scope.session_id, rank), []).append(
                PhaseSpan(
                    session_id=scope.session_id,
                    rank=rank,
                    thread_id=scope.thread_id,
                    thread_name=scope.thread_name,
                    scope_id=scope.scope_id,
                    path=opened.path,
                    start_ns=opened.timestamp_ns,
                    end_ns=timestamp_ns,
                    sequence=opened.sequence,
                )
            )

        for (session_id, rank, _thread_id), stack in active_by_thread.items():
            session_end_ns = session_end_by_group.get((session_id, rank))
            if session_end_ns is None:
                continue
            for scope in stack:
                intervals_by_group.setdefault((session_id, rank), []).append(
                    PhaseSpan(
                        session_id=session_id,
                        rank=rank,
                        thread_id=scope.thread_id,
                        thread_name=scope.thread_name,
                        scope_id=scope.scope_id,
                        path=scope.path,
                        start_ns=scope.timestamp_ns,
                        end_ns=session_end_ns,
                        sequence=scope.sequence,
                        synthetic_end=True,
                    )
                )

        return cls(intervals_by_group)

    def spans_for(
        self,
        *,
        session_id: str,
        rank: int | None = None,
    ) -> list[PhaseSpan]:
        """Return all reconstructed spans for one session/rank selection."""
        spans: list[PhaseSpan] = []
        for (
            group_session_id,
            group_rank,
        ), intervals in self._intervals_by_group.items():
            if group_session_id != session_id:
                continue
            if rank is not None and group_rank != rank:
                continue
            spans.extend(intervals)
        return list(spans)

    def active_spans(
        self,
        *,
        timestamp_ns: int,
        session_id: str,
        rank: int | None = None,
    ) -> list[PhaseSpan]:
        """Return active spans at a timestamp before attribution policy is applied."""
        matches: list[PhaseSpan] = []
        for (
            group_session_id,
            group_rank,
        ), intervals in self._intervals_by_group.items():
            if group_session_id != session_id:
                continue
            if rank is not None and group_rank != rank:
                continue
            for interval in intervals:
                if interval.start_ns <= timestamp_ns <= interval.end_ns:
                    matches.append(interval)
        return matches

    def resolve(
        self,
        *,
        timestamp_ns: int,
        session_id: str | None,
        rank: int | None = None,
        origin_thread_id: int | None = None,
        origin_phase_scope_id: str | None = None,
    ) -> Any:
        """Resolve one timestamp against the replay index."""
        if session_id is None:
            return None
        from .policy import attribute_active_spans

        return attribute_active_spans(
            self.active_spans(
                timestamp_ns=timestamp_ns,
                session_id=session_id,
                rank=rank,
            ),
            origin_thread_id=origin_thread_id,
            origin_phase_scope_id=origin_phase_scope_id,
        )

    def resolve_for_event(self, event: Any) -> Any:
        """Resolve one event-like object against the replay index."""
        from .policy import resolve_phase_for_event

        return resolve_phase_for_event(self, event)


def parse_phase_boundary(event: Any) -> PhaseBoundaryRecord | None:
    """Extract one normalized phase payload from an event-like object."""
    event_type = _event_field(event, "event_type", "")
    raw_scope = _phase_scope_metadata(event, event_type)
    if raw_scope is None:
        return None
    session_id = _event_field(event, "session_id")
    timestamp_ns = _event_field(event, "timestamp_ns")
    if not isinstance(session_id, str) or not isinstance(timestamp_ns, int):
        return None
    if not _valid_phase_scope_labels(raw_scope, event_type):
        return None
    normalized_path = tuple(
        str(part) for part in raw_scope["path"] if isinstance(part, str) and part
    )
    if not normalized_path:
        return None
    if not _valid_phase_scope_position(raw_scope):
        return None
    return _build_phase_boundary_record(
        raw_scope,
        normalized_path=normalized_path,
        session_id=session_id,
        timestamp_ns=timestamp_ns,
    )


def _phase_scope_metadata(event: Any, event_type: str) -> Mapping[str, Any] | None:
    if event_type not in {PHASE_ENTER_EVENT, PHASE_EXIT_EVENT}:
        return None
    metadata = _event_field(event, "metadata", {})
    if not isinstance(metadata, Mapping):
        return None
    raw_scope = metadata.get(PHASE_SCOPE_METADATA_KEY)
    if not isinstance(raw_scope, Mapping):
        return None
    return raw_scope


def _valid_phase_scope_labels(raw_scope: Mapping[str, Any], event_type: str) -> bool:
    action = raw_scope.get("action")
    name = raw_scope.get("name")
    scope_id = raw_scope.get("scope_id")
    path = raw_scope.get("path")
    thread_name = raw_scope.get("thread_name")
    if (
        not isinstance(action, str)
        or not isinstance(name, str)
        or not name.strip()
        or not isinstance(scope_id, str)
        or not isinstance(thread_name, str)
        or not isinstance(path, list)
    ):
        return False
    expected_action = "enter" if event_type == PHASE_ENTER_EVENT else "exit"
    return action == expected_action


def _valid_phase_scope_position(raw_scope: Mapping[str, Any]) -> bool:
    sequence_value = raw_scope.get("sequence")
    thread_id_value = raw_scope.get("thread_id")
    parent_scope_id_value = raw_scope.get("parent_scope_id")
    if not isinstance(sequence_value, int):
        return False
    if not isinstance(thread_id_value, int):
        return False
    if parent_scope_id_value is not None and not isinstance(parent_scope_id_value, str):
        return False
    return True


def _build_phase_boundary_record(
    raw_scope: Mapping[str, Any],
    *,
    normalized_path: tuple[str, ...],
    session_id: str,
    timestamp_ns: int,
) -> PhaseBoundaryRecord:
    depth_value = raw_scope.get("depth")
    if not isinstance(depth_value, int) or depth_value != len(normalized_path):
        depth_value = len(normalized_path)
    raw_attributes = raw_scope.get(PHASE_SCOPE_ATTRIBUTES_KEY, {})
    attributes = dict(raw_attributes) if isinstance(raw_attributes, Mapping) else {}
    return PhaseBoundaryRecord(
        action=raw_scope["action"],
        name=raw_scope["name"].strip(),
        path=normalized_path,
        depth=depth_value,
        scope_id=raw_scope["scope_id"],
        parent_scope_id=raw_scope.get("parent_scope_id"),
        thread_id=raw_scope["thread_id"],
        thread_name=raw_scope["thread_name"],
        sequence=raw_scope["sequence"],
        session_id=session_id,
        timestamp_ns=timestamp_ns,
        attributes=attributes,
    )


def is_phase_boundary_event(event: Any) -> bool:
    """Return ``True`` when the event is a structured phase boundary."""
    event_type = _event_field(event, "event_type", "")
    if event_type not in {PHASE_ENTER_EVENT, PHASE_EXIT_EVENT}:
        return False
    return parse_phase_boundary(event) is not None


def _event_field(event: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(event, Mapping):
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


def _replay_boundaries(
    events: Sequence[Any],
) -> tuple[
    list[tuple[int, int, str, int, PhaseBoundaryRecord]],
    dict[tuple[str, int], int],
]:
    session_end_by_group: dict[tuple[str, int], int] = {}
    boundaries: list[tuple[int, int, str, int, PhaseBoundaryRecord]] = []
    for event in events:
        session_id = _event_field(event, "session_id")
        timestamp_ns = _event_field(event, "timestamp_ns")
        if not isinstance(session_id, str) or not isinstance(timestamp_ns, int):
            continue
        rank = _coerce_rank(_event_field(event, "rank", 0))
        if rank is not None:
            group_key = (session_id, rank)
            previous_end = session_end_by_group.get(group_key)
            if previous_end is None or timestamp_ns > previous_end:
                session_end_by_group[group_key] = timestamp_ns

        event_type = _event_field(event, "event_type", "")
        if event_type not in {PHASE_ENTER_EVENT, PHASE_EXIT_EVENT}:
            continue
        if rank is None:
            continue

        scope = parse_phase_boundary(event)
        if scope is None:
            continue
        boundaries.append(
            (
                timestamp_ns,
                scope.sequence,
                scope.scope_id,
                rank,
                scope,
            )
        )

    boundaries.sort(
        key=lambda item: (item[4].session_id, item[3], item[0], item[1], item[2])
    )

    return boundaries, session_end_by_group


__all__ = [
    "PhaseBoundaryRecord",
    "PhaseReplayIndex",
    "PhaseSpan",
    "is_phase_boundary_event",
    "parse_phase_boundary",
]
