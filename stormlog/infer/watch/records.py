"""The watcher's records: ``stormlog.infer.watch/1``.

Every record the watcher writes to its ledger, and hands to an exporter, is
built here and checked against closed vocabularies, so a value that becomes a
metric label (#220) can only be one of the listed ones. The fixture
``tests/fixtures/watch/records_v1.jsonl`` holds one record of each type and
is regenerated from this module; a test keeps them equal.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

WATCH_FORMAT = "stormlog.infer.watch/1"
WATCH_SCHEMA_VERSION = 1

TRIGGER_STATE = "infer.trigger_state"
INCIDENT_EVENT = "infer.incident_event"
INCIDENT = "infer.incident"
INCIDENT_ASSOCIATION = "infer.incident_association"
INCIDENT_FINALIZED = "infer.incident_finalized"
INCIDENT_PRUNED = "infer.incident_pruned"
WATCH_HEALTH = "infer.watch_health"
WATCH_SESSION = "infer.watch_session"
EVENT_TYPES = (
    WATCH_SESSION,
    TRIGGER_STATE,
    INCIDENT_EVENT,
    INCIDENT,
    INCIDENT_ASSOCIATION,
    INCIDENT_FINALIZED,
    INCIDENT_PRUNED,
    WATCH_HEALTH,
)

TRIGGER_KINDS = ("metric", "slo", "signal", "health", "test")
TRIGGER_EVENTS = ("pending", "fired", "resolving", "reentered", "resolved", "reset")
TRIGGER_STATES = ("inactive", "pending", "firing", "resolving")
INCIDENT_EVENTS = ("opened", "capture_started", "capture_stopped")
CAPTURE_STATUSES = (
    "captured",
    "partial",
    "failed",
    "interrupted",
    "disabled",
    "skipped_cooldown",
    "skipped_rate_limit",
    "skipped_budget",
    "skipped_unavailable",
    "skipped_owner",
    "health_only",
    "recorded_by_policy",
)
FIDELITY = ("complete", "partial", "missing")
SEAL_STATUSES = ("completed", "interrupted")
DETAIL_LEVELS = ("metrics", "spans", "execution", "kernel_trace")
SUPPRESSION_REASONS = (
    "cooldown",
    "rate_limit",
    "budget",
    "owner",
    "unavailable",
    "open_limit",
    "join_limit",
)
FINALIZED_OUTCOMES = ("ok", "failed", "timeout", "deferred")
SESSION_PHASES = ("started", "ended")
UNSOUND_REASONS = (
    "no_successful_scrape",
    "engine_required",
    "ledger_failing",
    "ledger_close_timeout",
    "incident_writes_failing",
    "store_writer_timeout",
)
PRUNE_REASONS = ("max_age_hours", "max_incidents", "max_total_bytes")
REARM_BASES = ("exact_cohort", "horizon", "cap")
SELF_INDUCED_REASONS = (
    "attribution_window",
    "completion_horizon",
    "unfinished_cohort",
    "outstanding_cohort",
)
START_OUTCOMES = ("acknowledged", "rejected", "unknown")
CAPTURE_TIMING_KEYS = (
    "start_requested_at_ns",
    "start_returned_at_ns",
    "start_outcome",
    "stop_requested_at_ns",
    "stop_returned_at_ns",
    "start_call_ns",
    "stop_call_ns",
)
LOSS_KEYS = (
    "scrapes_failed",
    "scrapes_oversized",
    "scrape_ticks_missed",
    "scrape_frozen_ticks",
    "spans_refused_connections",
    "spans_busy",
    "spans_body_timeouts",
    "spans_too_many",
    "spans_dropped_queue_full",
    "spans_decode_failures",
    "history_evicted_age",
    "history_evicted_bytes",
    "span_ring_evicted",
    "hook_dropped",
    "hook_errors",
    "hook_seq_gaps",
    "hook_capped",
    "trace_files_unowned",
    "trace_bytes_discarded",
    "trace_exports_missing",
    "io_rejected",
    "ledger_dropped",
    "export_failures",
)
MAX_JOINED_TRIGGERS = 16
MAX_ENGINE_SPAN_LINKS = 32
MAX_REQUEST_REFS = 32


def envelope(
    event_type: str, *, session_id: str, run_id: str, timestamp_ns: int
) -> dict[str, Any]:
    """The fields every watcher record starts with."""
    return {
        "schema_version": WATCH_SCHEMA_VERSION,
        "format": WATCH_FORMAT,
        "event_type": event_type,
        "session_id": session_id,
        "run_id": run_id,
        "timestamp_ns": timestamp_ns,
    }


def trigger_fields(
    *,
    trigger_id: str,
    kind: str,
    reason: str,
    fired_at_ns: int,
    counts_toward_exit: bool,
    threshold: float | None = None,
    observed: float | None = None,
    observed_bounds: Sequence[float | None] | None = None,
    samples: float | None = None,
    pending_since_ns: int | None = None,
    sustained_ns: int = 0,
    window_seconds: float | None = None,
    hold_seconds: float | None = None,
    clear_seconds: float | None = None,
    requested_at_ns: int | None = None,
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One trigger of an incident, with every key present."""
    return {
        "trigger_id": trigger_id,
        "kind": kind,
        "reason": reason,
        "threshold": finite(threshold),
        "observed": finite(observed),
        "observed_bounds": (
            [finite(bound) for bound in observed_bounds]
            if observed_bounds is not None
            else None
        ),
        "samples": finite(samples),
        "pending_since_ns": pending_since_ns,
        "fired_at_ns": fired_at_ns,
        "resolved_at_ns": None,
        "sustained_ns": sustained_ns,
        "window_seconds": window_seconds,
        "hold_seconds": hold_seconds,
        "clear_seconds": clear_seconds,
        "counts_toward_exit": counts_toward_exit,
        "requested_at_ns": requested_at_ns,
        "detail": dict(detail or {}),
    }


def capture_fields(
    status: str, *, owner: str, stop_reason: str | None = None, **timing: Any
) -> dict[str, Any]:
    """An incident's ``capture``: every timing key, null when not measured."""
    unknown = set(timing) - set(CAPTURE_TIMING_KEYS)
    if unknown:
        raise ValueError(f"unknown capture fields: {', '.join(sorted(unknown))}")
    capture: dict[str, Any] = {
        "status": status,
        "stop_reason": stop_reason,
        "owner": owner,
    }
    capture.update({key: timing.get(key) for key in CAPTURE_TIMING_KEYS})
    return capture


def empty_loss() -> dict[str, int | None]:
    """Every loss key, null: a key whose source was not running stays null."""
    return {key: None for key in LOSS_KEYS}


def finite(value: float | None) -> float | None:
    """A float JSON can carry; NaN and infinities become null."""
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def validate_record(record: Mapping[str, Any]) -> None:
    """Check a record against the closed vocabularies; raise ``ValueError``."""
    if record.get("format") != WATCH_FORMAT:
        raise ValueError("not a stormlog.infer.watch/1 record")
    event_type = record.get("event_type")
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown watch record type {event_type!r}")
    for key in ("session_id", "run_id"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"{event_type} needs {key}")
    _CHECKS.get(str(event_type), _no_check)(record)


def _no_check(_record: Mapping[str, Any]) -> None:
    return None


def _one_of(
    value: Any, allowed: Sequence[str], name: str, *, null: bool = False
) -> None:
    if value is None and null:
        return
    if value not in allowed:
        raise ValueError(f"{name} {value!r} is not one of {', '.join(allowed)}")


def _check_trigger_state(record: Mapping[str, Any]) -> None:
    _one_of(record.get("kind"), TRIGGER_KINDS, "kind")
    _one_of(record.get("event"), TRIGGER_EVENTS, "event")
    _one_of(record.get("state"), TRIGGER_STATES, "state")


def _check_incident_event(record: Mapping[str, Any]) -> None:
    _one_of(record.get("event"), INCIDENT_EVENTS, "event")
    _one_of(record.get("kind"), TRIGGER_KINDS, "kind")
    _one_of(record.get("rearm_basis"), REARM_BASES, "rearm_basis", null=True)


def _check_trigger(trigger: Mapping[str, Any]) -> None:
    _one_of(trigger.get("kind"), TRIGGER_KINDS, "trigger kind")


def _check_window(window: Mapping[str, Any] | None) -> None:
    if window is None:
        return
    _one_of(window.get("fidelity"), FIDELITY, "fidelity")
    _one_of(window.get("detail_requested"), DETAIL_LEVELS, "detail_requested")
    _one_of(
        window.get("detail_collected"), DETAIL_LEVELS, "detail_collected", null=True
    )


def _check_incident(record: Mapping[str, Any]) -> None:
    _one_of(record.get("status"), SEAL_STATUSES, "status")
    _check_trigger(record["trigger"])
    for trigger in _bounded(record, "joined_triggers", MAX_JOINED_TRIGGERS):
        _check_trigger(trigger)
    for name in ("pre_window", "post_window", "deep_window"):
        _check_window(record.get(name))
    capture = record["capture"]
    _one_of(capture.get("status"), CAPTURE_STATUSES, "capture status")
    _one_of(capture.get("start_outcome"), START_OUTCOMES, "start_outcome", null=True)
    _one_of(record.get("rearm_basis"), REARM_BASES, "rearm_basis", null=True)
    _one_of(
        record.get("self_induced_reason"),
        SELF_INDUCED_REASONS,
        "self_induced_reason",
        null=True,
    )
    for reason in record.get("suppressed") or {}:
        _one_of(reason, SUPPRESSION_REASONS, "suppression reason")
    _check_loss(record.get("loss") or {})
    _bounded(record, "engine_span_links", MAX_ENGINE_SPAN_LINKS)
    _bounded(record, "request_refs", MAX_REQUEST_REFS)


def _bounded(record: Mapping[str, Any], name: str, limit: int) -> Sequence[Any]:
    items: Sequence[Any] = record.get(name) or []
    if len(items) > limit:
        raise ValueError(f"at most {limit} {name}")
    return items


def _check_loss(loss: Mapping[str, Any]) -> None:
    if set(loss) != set(LOSS_KEYS):
        raise ValueError("loss must hold exactly the closed set of keys")
    for value in loss.values():
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError("loss values are integers or null")


def _check_association(record: Mapping[str, Any]) -> None:
    _check_trigger(record["trigger"])


def _check_finalized(record: Mapping[str, Any]) -> None:
    _one_of(record.get("outcome"), FINALIZED_OUTCOMES, "outcome")


def _check_pruned(record: Mapping[str, Any]) -> None:
    _one_of(record.get("reason"), PRUNE_REASONS, "prune reason")


def _check_session(record: Mapping[str, Any]) -> None:
    _one_of(record.get("phase"), SESSION_PHASES, "phase")
    for reason in record.get("unsound") or ():
        _one_of(reason, UNSOUND_REASONS, "unsound reason")


_CHECKS = {
    TRIGGER_STATE: _check_trigger_state,
    INCIDENT_EVENT: _check_incident_event,
    INCIDENT: _check_incident,
    INCIDENT_ASSOCIATION: _check_association,
    INCIDENT_FINALIZED: _check_finalized,
    INCIDENT_PRUNED: _check_pruned,
    WATCH_SESSION: _check_session,
}


__all__ = [
    "CAPTURE_STATUSES",
    "CAPTURE_TIMING_KEYS",
    "DETAIL_LEVELS",
    "EVENT_TYPES",
    "FIDELITY",
    "FINALIZED_OUTCOMES",
    "INCIDENT",
    "INCIDENT_ASSOCIATION",
    "INCIDENT_EVENT",
    "INCIDENT_EVENTS",
    "INCIDENT_FINALIZED",
    "INCIDENT_PRUNED",
    "LOSS_KEYS",
    "SEAL_STATUSES",
    "MAX_ENGINE_SPAN_LINKS",
    "MAX_JOINED_TRIGGERS",
    "MAX_REQUEST_REFS",
    "PRUNE_REASONS",
    "REARM_BASES",
    "SELF_INDUCED_REASONS",
    "START_OUTCOMES",
    "SUPPRESSION_REASONS",
    "TRIGGER_EVENTS",
    "TRIGGER_KINDS",
    "TRIGGER_STATE",
    "TRIGGER_STATES",
    "UNSOUND_REASONS",
    "WATCH_FORMAT",
    "WATCH_HEALTH",
    "WATCH_SESSION",
    "WATCH_SCHEMA_VERSION",
    "capture_fields",
    "empty_loss",
    "envelope",
    "finite",
    "trigger_fields",
    "validate_record",
]
