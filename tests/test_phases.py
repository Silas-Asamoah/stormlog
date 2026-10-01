"""Tests for structured phase replay and attribution."""

from __future__ import annotations

import threading
from typing import Any

import pytest

phases = pytest.importorskip("stormlog.phases")
PHASE_ENTER_EVENT = phases.PHASE_ENTER_EVENT
PHASE_EXIT_EVENT = phases.PHASE_EXIT_EVENT
PHASE_SCOPE_METADATA_KEY = phases.PHASE_SCOPE_METADATA_KEY
PhaseAttribution = phases.PhaseAttribution
PhaseProtocolError = phases.PhaseProtocolError
PhaseRecorder = phases.PhaseRecorder
PhaseSummary = phases.PhaseSummary
PhaseReplayIndex = phases.PhaseReplayIndex
is_phase_boundary_event = phases.is_phase_boundary_event
merge_phase_attributions = phases.merge_phase_attributions
phase_attribution_to_payload = phases.phase_attribution_to_payload
summarize_phase_attribution = phases.summarize_phase_attribution

from stormlog.telemetry import SCHEMA_VERSION_V3, telemetry_event_from_record


def _boundary_record_input() -> dict[str, Any]:
    return {
        "event_type": PHASE_ENTER_EVENT,
        "session_id": "session-1",
        "timestamp_ns": 100,
        "metadata": _phase_scope(
            action="enter",
            name=" train ",
            path=["train"],
            scope_id="scope-1",
            sequence=1,
            thread_id=11,
        ),
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action", "exit"),
        ("action", None),
        ("name", "  "),
        ("name", 12),
        ("scope_id", None),
        ("thread_name", 12),
        ("path", ("train",)),
        ("path", [None, 12, ""]),
        ("sequence", "1"),
        ("thread_id", "11"),
        ("parent_scope_id", 1),
    ],
)
def test_parse_phase_boundary_rejects_malformed_scope_fields(
    field: str, value: Any
) -> None:
    event = _boundary_record_input()
    event["metadata"][PHASE_SCOPE_METADATA_KEY][field] = value

    assert phases.parse_phase_boundary(event) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event_type", "sample"),
        ("metadata", []),
        ("metadata", {PHASE_SCOPE_METADATA_KEY: []}),
        ("session_id", None),
        ("timestamp_ns", "100"),
    ],
)
def test_parse_phase_boundary_rejects_malformed_event_fields(
    field: str, value: Any
) -> None:
    event = _boundary_record_input()
    event[field] = value

    assert phases.parse_phase_boundary(event) is None


def test_parse_phase_boundary_normalizes_path_and_preserves_integer_booleans() -> None:
    event = _boundary_record_input()
    scope = event["metadata"][PHASE_SCOPE_METADATA_KEY]
    scope.update(
        path=["train", 12, "", " ", "step"],
        depth="wrong",
        sequence=True,
        thread_id=False,
        attributes=[("ignored", 1)],
    )

    boundary = phases.parse_phase_boundary(event)

    assert boundary is not None
    assert boundary.name == "train"
    assert boundary.path == ("train", " ", "step")
    assert boundary.depth == 3
    assert boundary.sequence is True
    assert boundary.thread_id is False
    assert boundary.attributes == {}


def test_phase_attribution_preserves_ambiguity_when_scope_ids_are_duplicated() -> None:
    spans = [
        phases.PhaseSpan(
            session_id="session-1",
            rank=0,
            thread_id=thread_id,
            thread_name=f"thread-{thread_id}",
            scope_id="duplicate",
            path=("train",),
            start_ns=0,
            end_ns=100,
            sequence=sequence,
        )
        for thread_id, sequence in ((11, 1), (22, 2))
    ]

    ambiguous = phases.attribute_active_spans(spans, origin_phase_scope_id="duplicate")
    assert ambiguous is not None
    assert ambiguous.phase_resolution == "ambiguous"
    assert ambiguous.phase_paths == ["train"]
    assert ambiguous.phase_summary is None

    thread_local = phases.attribute_active_spans(
        spans, origin_phase_scope_id="duplicate", origin_thread_id=11
    )
    assert thread_local is not None
    assert thread_local.phase_source == "thread_local"
    assert thread_local.thread_id == 11


def _make_event(
    *,
    timestamp_ns: int,
    event_type: str = "sample",
    session_id: str = "session-1",
    rank: int = 0,
    metadata: dict[str, object] | None = None,
) -> object:
    return telemetry_event_from_record(
        {
            "schema_version": SCHEMA_VERSION_V3,
            "session_id": session_id,
            "timestamp_ns": timestamp_ns,
            "event_type": event_type,
            "collector": "stormlog.cuda_tracker",
            "sampling_interval_ms": 100,
            "pid": 1234,
            "host": "test-host",
            "device_id": 0,
            "allocator_allocated_bytes": 1,
            "allocator_reserved_bytes": 1,
            "allocator_active_bytes": 1,
            "allocator_inactive_bytes": 0,
            "allocator_change_bytes": 0,
            "device_used_bytes": 1,
            "device_free_bytes": 1,
            "device_total_bytes": 2,
            "context": event_type,
            "job_id": "job-1",
            "rank": rank,
            "local_rank": rank,
            "world_size": 2,
            "metadata": metadata or {},
        },
        permissive_legacy=False,
    )


def _phase_scope(
    *,
    action: str,
    name: str,
    path: list[str],
    scope_id: str,
    sequence: int,
    thread_id: int,
) -> dict[str, object]:
    return {
        PHASE_SCOPE_METADATA_KEY: {
            "action": action,
            "name": name,
            "path": path,
            "depth": len(path),
            "scope_id": scope_id,
            "parent_scope_id": scope_id.rsplit(":", 1)[0] if len(path) > 1 else None,
            "thread_id": thread_id,
            "thread_name": f"thread-{thread_id}",
            "sequence": sequence,
        }
    }


def test_phase_timeline_resolver_prefers_deepest_nested_path() -> None:
    events = [
        _make_event(
            timestamp_ns=100,
            event_type=PHASE_ENTER_EVENT,
            metadata=_phase_scope(
                action="enter",
                name="train",
                path=["train"],
                scope_id="session-1:1",
                sequence=1,
                thread_id=11,
            ),
        ),
        _make_event(
            timestamp_ns=120,
            event_type=PHASE_ENTER_EVENT,
            metadata=_phase_scope(
                action="enter",
                name="step",
                path=["train", "step"],
                scope_id="session-1:2",
                sequence=2,
                thread_id=11,
            ),
        ),
        _make_event(timestamp_ns=140),
    ]

    resolver = PhaseReplayIndex.from_events(events)
    attribution = resolver.resolve_for_event(events[-1])

    assert attribution is not None
    assert attribution.phase_resolution == "unique"
    assert attribution.phase_source == "heuristic"
    assert attribution.phase_path == "train / step"
    assert attribution.phase_summary is None


def test_phase_timeline_resolver_treats_same_timestamp_exit_as_active() -> None:
    events = [
        _make_event(
            timestamp_ns=100,
            event_type=PHASE_ENTER_EVENT,
            metadata=_phase_scope(
                action="enter",
                name="train",
                path=["train"],
                scope_id="session-1:1",
                sequence=1,
                thread_id=11,
            ),
        ),
        _make_event(
            timestamp_ns=100,
            event_type=PHASE_EXIT_EVENT,
            metadata=_phase_scope(
                action="exit",
                name="train",
                path=["train"],
                scope_id="session-1:1",
                sequence=2,
                thread_id=11,
            ),
        ),
        _make_event(timestamp_ns=100),
    ]

    resolver = PhaseReplayIndex.from_events(events)
    attribution = resolver.resolve_for_event(events[-1])

    assert attribution is not None
    assert attribution.phase_resolution == "unique"
    assert attribution.phase_source == "heuristic"
    assert attribution.phase_path == "train"


def test_phase_timeline_resolver_synthesizes_open_scope_until_session_end() -> None:
    events = [
        _make_event(
            timestamp_ns=100,
            event_type=PHASE_ENTER_EVENT,
            metadata=_phase_scope(
                action="enter",
                name="evaluate",
                path=["evaluate"],
                scope_id="session-1:1",
                sequence=1,
                thread_id=11,
            ),
        ),
        _make_event(timestamp_ns=200),
    ]

    resolver = PhaseReplayIndex.from_events(events)
    attribution = resolver.resolve_for_event(events[-1])

    assert attribution is not None
    assert attribution.phase_resolution == "unique"
    assert attribution.phase_source == "heuristic"
    assert attribution.phase_path == "evaluate"


def test_phase_timeline_resolver_marks_multi_thread_overlap_ambiguous() -> None:
    events = [
        _make_event(
            timestamp_ns=100,
            event_type=PHASE_ENTER_EVENT,
            metadata=_phase_scope(
                action="enter",
                name="train",
                path=["train"],
                scope_id="session-1:1",
                sequence=1,
                thread_id=11,
            ),
        ),
        _make_event(
            timestamp_ns=110,
            event_type=PHASE_ENTER_EVENT,
            metadata=_phase_scope(
                action="enter",
                name="evaluate",
                path=["evaluate"],
                scope_id="session-1:2",
                sequence=2,
                thread_id=12,
            ),
        ),
        _make_event(timestamp_ns=120),
    ]

    resolver = PhaseReplayIndex.from_events(events)
    attribution = resolver.resolve_for_event(events[-1])

    assert attribution is not None
    assert attribution.phase_resolution == "ambiguous"
    assert attribution.phase_paths == ["evaluate", "train"]
    assert attribution.phase_summary is not None
    assert attribution.phase_summary.phase_path == "evaluate"
    assert attribution.phase_summary.source == "heuristic"


def test_phase_timeline_resolver_prefers_origin_thread_id_when_available() -> None:
    events = [
        _make_event(
            timestamp_ns=100,
            event_type=PHASE_ENTER_EVENT,
            metadata=_phase_scope(
                action="enter",
                name="train",
                path=["train"],
                scope_id="session-1:1",
                sequence=1,
                thread_id=11,
            ),
        ),
        _make_event(
            timestamp_ns=110,
            event_type=PHASE_ENTER_EVENT,
            metadata=_phase_scope(
                action="enter",
                name="evaluate",
                path=["evaluate"],
                scope_id="session-1:2",
                sequence=2,
                thread_id=12,
            ),
        ),
        _make_event(timestamp_ns=120, metadata={"origin_thread_id": 11}),
    ]

    resolver = PhaseReplayIndex.from_events(events)
    attribution = resolver.resolve_for_event(events[-1])

    assert attribution is not None
    assert attribution.phase_resolution == "unique"
    assert attribution.phase_source == "thread_local"
    assert attribution.phase_path == "train"
    assert attribution.phase_summary is None


def test_phase_timeline_resolver_prefers_origin_scope_id_when_available() -> None:
    events = [
        _make_event(
            timestamp_ns=100,
            event_type=PHASE_ENTER_EVENT,
            metadata=_phase_scope(
                action="enter",
                name="train",
                path=["train"],
                scope_id="session-1:1",
                sequence=1,
                thread_id=11,
            ),
        ),
        _make_event(
            timestamp_ns=120, metadata={"origin_phase_scope_id": "session-1:1"}
        ),
    ]

    resolver = PhaseReplayIndex.from_events(events)
    attribution = resolver.resolve_for_event(events[-1])

    assert attribution is not None
    assert attribution.phase_resolution == "unique"
    assert attribution.phase_source == "exact"
    assert attribution.phase_path == "train"


def test_phase_timeline_resolver_ignores_boundary_with_malformed_rank() -> None:
    events = [
        {
            "session_id": "session-1",
            "timestamp_ns": 100,
            "event_type": PHASE_ENTER_EVENT,
            "rank": "rank-zero",
            "metadata": _phase_scope(
                action="enter",
                name="train",
                path=["train"],
                scope_id="session-1:1",
                sequence=1,
                thread_id=11,
            ),
        },
        _make_event(timestamp_ns=140),
    ]

    resolver = PhaseReplayIndex.from_events(events)

    assert resolver.spans_for(session_id="session-1") == []


def test_phase_timeline_resolver_ignores_action_event_type_mismatches() -> None:
    event = {
        "session_id": "session-1",
        "timestamp_ns": 100,
        "event_type": PHASE_ENTER_EVENT,
        "rank": 0,
        "metadata": _phase_scope(
            action="exit",
            name="train",
            path=["train"],
            scope_id="session-1:1",
            sequence=1,
            thread_id=11,
        ),
    }

    resolver = PhaseReplayIndex.from_events([event])

    assert is_phase_boundary_event(event) is False
    assert resolver.spans_for(session_id="session-1", rank=0) == []


def test_phase_timeline_resolver_returns_none_for_malformed_event_rank() -> None:
    events = [
        _make_event(
            timestamp_ns=100,
            event_type=PHASE_ENTER_EVENT,
            metadata=_phase_scope(
                action="enter",
                name="train",
                path=["train"],
                scope_id="session-1:1",
                sequence=1,
                thread_id=11,
            ),
        ),
        _make_event(timestamp_ns=120),
    ]

    resolver = PhaseReplayIndex.from_events(events)
    attribution = resolver.resolve_for_event(
        {
            "session_id": "session-1",
            "timestamp_ns": 120,
            "rank": "zero",
        }
    )

    assert attribution is None


def test_phase_recorder_exit_uses_current_thread_for_strict_thread_validation() -> None:
    recorder = PhaseRecorder()
    token, _boundary = recorder.enter(session_id="session-1", rank=0, name="train")
    error: Exception | None = None

    def _close_on_other_thread() -> None:
        nonlocal error
        try:
            recorder.exit(token)
        except Exception as exc:  # pragma: no branch - asserted below
            error = exc

    worker = threading.Thread(target=_close_on_other_thread)
    worker.start()
    worker.join()

    assert isinstance(error, PhaseProtocolError)
    assert "different thread" in str(error)
    assert recorder.exit(token).event_type == PHASE_EXIT_EVENT


def test_merge_phase_attributions_keeps_same_label_multi_thread_overlap_ambiguous() -> (
    None
):
    merged = merge_phase_attributions(
        PhaseAttribution(
            phase_resolution="unique",
            phase_path="train / step",
            phase_paths=["train / step"],
            scope_id="scope-1",
            thread_id=11,
            thread_name="thread-11",
        ),
        PhaseAttribution(
            phase_resolution="unique",
            phase_path="train / step",
            phase_paths=["train / step"],
            scope_id="scope-2",
            thread_id=12,
            thread_name="thread-12",
        ),
    )

    assert merged is not None
    assert merged.phase_resolution == "ambiguous"
    assert merged.phase_paths == ["train / step"]
    assert merged.phase_path is None


@pytest.mark.parametrize(
    "overrides,resolution,paths",
    [
        ({}, "unique", ["train"]),
        ({"scope_id": "other"}, "ambiguous", ["train"]),
        ({"thread_id": 12}, "ambiguous", ["train"]),
        ({"phase_resolution": "ambiguous"}, "ambiguous", ["train"]),
        (
            {"phase_path": "eval", "phase_paths": ["eval"]},
            "ambiguous",
            ["eval", "train"],
        ),
    ],
)
def test_merge_phase_attributions_preserves_unique_scope_identity(
    overrides: dict[str, Any], resolution: str, paths: list[str]
) -> None:
    fields = dict(
        phase_resolution="unique",
        phase_path="train",
        phase_paths=["train"],
        scope_id="scope",
        thread_id=11,
    )
    first = PhaseAttribution(**fields)
    second = PhaseAttribution(**(fields | overrides))
    merged = merge_phase_attributions(first, second)
    assert merged is not None
    assert merged.phase_resolution == resolution
    assert merged.phase_paths == paths
    assert (merged is first) is (resolution == "unique")


def test_summarize_phase_attribution_marks_ambiguous_single_label() -> None:
    summary = summarize_phase_attribution(
        PhaseAttribution(
            phase_resolution="ambiguous",
            phase_paths=["train / step"],
        )
    )

    assert summary == "(ambiguous) train / step"


def test_summarize_phase_attribution_prefers_heuristic_summary_when_present() -> None:
    summary = summarize_phase_attribution(
        PhaseAttribution(
            phase_resolution="ambiguous",
            phase_paths=["evaluate", "train"],
            phase_summary=PhaseSummary(
                phase_path="evaluate",
                source="heuristic",
            ),
        )
    )

    assert summary == "(likely) evaluate"


def test_phase_attribution_to_payload_omits_phase_summary_when_not_needed() -> None:
    payload = phase_attribution_to_payload(
        PhaseAttribution(
            phase_resolution="unique",
            phase_source="thread_local",
            phase_path="train",
            phase_paths=["train"],
            scope_id="scope-1",
            thread_id=11,
            thread_name="thread-11",
        )
    )

    assert payload is not None
    assert payload["phase_source"] == "thread_local"
    assert "phase_summary" not in payload
