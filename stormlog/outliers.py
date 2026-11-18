"""Selection-versus-baseline outlier comparison for Stormlog telemetry.

Given two sets of telemetry events, a selection (one rank, one time window,
one session, or the events behind one grouped issue) and a baseline, this
module ranks the event fields whose distributions differ most between them.

Every field gets a score in ``[0, 1]`` so categorical and numeric fields rank
on one scale:

* categorical fields use the total variation distance between the two value
  distributions, ``0.5 * sum(|p_selection(v) - p_baseline(v)|)``;
* numeric fields use the two-sample Kolmogorov-Smirnov statistic over present
  values, raised to the change in missing ratio when a field is present far
  more often on one side.

Scores are effect sizes, not significance tests. Telemetry samples form an
autocorrelated time series, so the independence that p-values assume does not
hold; trust is expressed instead through support counts and a ``confidence``
of ``high`` or ``low`` with reasons.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import numpy as np

from .correlation import DEFAULT_CORRELATION_WINDOW_NS
from .derived_fields import DerivedFields, compute_event_fields
from .issues import normalized_error_stem
from .telemetry import TelemetryEvent
from .telemetry_classification import event_backend, event_severity, is_alert_event

COMPARISON_FORMAT = "stormlog.outlier_comparison"
COMPARISON_SCHEMA_VERSION = 1
DEFAULT_MAX_VALUES = 50
DEFAULT_MIN_EVENTS = 30
DEFAULT_NEIGHBOR_WINDOW_NS = DEFAULT_CORRELATION_WINDOW_NS
MISSING_VALUE = "(missing)"
TOP_VALUE_COUNT = 5

ComparisonScope = Literal["rank", "window", "session", "fingerprint"]
FieldKind = Literal["categorical", "numeric"]
Confidence = Literal["high", "low"]

_FieldExtractor = Callable[[TelemetryEvent, DerivedFields], object]


def _metadata_value(key: str) -> _FieldExtractor:
    return lambda event, _derived: event.metadata.get(key)


def _attribute(name: str) -> _FieldExtractor:
    return lambda event, _derived: getattr(event, name)


def _derived(name: str) -> _FieldExtractor:
    return lambda _event, derived: derived.get(name)


def _collector_error_stem(event: TelemetryEvent, _derived: DerivedFields) -> object:
    error = event.metadata.get("collector_last_error")
    return None if error is None else normalized_error_stem(error)


def _hidden_gap_bytes(event: TelemetryEvent, _derived: DerivedFields) -> object:
    if event.device_used_bytes is None or event.allocator_reserved_bytes is None:
        return None
    return event.device_used_bytes - event.allocator_reserved_bytes


_CATEGORICAL_FIELDS: Mapping[str, _FieldExtractor] = {
    "event_type": _attribute("event_type"),
    "collector": _attribute("collector"),
    "backend": lambda event, _derived: event_backend(event),
    "severity": lambda event, _derived: event_severity(event),
    "is_alert": lambda event, _derived: is_alert_event(event),
    "collector_health_status": _metadata_value("collector_health_status"),
    "collector_last_error_stem": _collector_error_stem,
    "is_degraded_collector": _derived("is_degraded_collector"),
    "context": _attribute("context"),
    "host": _attribute("host"),
    "device_id": _attribute("device_id"),
    "rank": _attribute("rank"),
    "local_rank": _attribute("local_rank"),
    "world_size": _attribute("world_size"),
}

_NUMERIC_FIELDS: Mapping[str, _FieldExtractor] = {
    "allocator_allocated_bytes": _attribute("allocator_allocated_bytes"),
    "allocator_reserved_bytes": _attribute("allocator_reserved_bytes"),
    "allocator_active_bytes": _attribute("allocator_active_bytes"),
    "allocator_inactive_bytes": _attribute("allocator_inactive_bytes"),
    "allocator_change_bytes": _attribute("allocator_change_bytes"),
    "device_used_bytes": _attribute("device_used_bytes"),
    "device_free_bytes": _attribute("device_free_bytes"),
    "hidden_gap_bytes": _hidden_gap_bytes,
    "allocator_gap_bytes": _derived("allocator_gap_bytes"),
    "utilization_ratio": _derived("utilization_ratio"),
    "fragmentation_ratio": _derived("fragmentation_ratio"),
    "allocator_cache_ratio": _derived("allocator_cache_ratio"),
    "collector_consecutive_failures": _metadata_value("collector_consecutive_failures"),
}

CATEGORICAL_FIELDS: tuple[str, ...] = tuple(_CATEGORICAL_FIELDS)
NUMERIC_FIELDS: tuple[str, ...] = tuple(_NUMERIC_FIELDS)
ELIGIBLE_FIELDS: tuple[str, ...] = CATEGORICAL_FIELDS + NUMERIC_FIELDS

# The field that defines each scope cannot explain its own selection.
SCOPE_DEFINING_FIELDS: Mapping[ComparisonScope, tuple[str, ...]] = {
    "rank": ("rank", "local_rank"),
    "window": (),
    "session": (),
    "fingerprint": (),
}

_REQUIRED_SCOPE_ARGUMENTS: Mapping[ComparisonScope, tuple[str, ...]] = {
    "rank": ("rank",),
    "window": ("session_id", "time_start_ns", "time_end_ns"),
    "session": ("session_id",),
    "fingerprint": ("fingerprint_id",),
}


@dataclass(frozen=True)
class ComparisonFilter:
    """Selection, baseline and tuning knobs for one outlier comparison.

    Attributes:
        scope: Which selection to compare against its baseline.
        session_id: The session to select (``window``, ``session``) or to
            limit the comparison to (``rank``).
        job_id: Limits ``rank`` and ``session`` comparisons to one job.
        rank: The rank to select (``rank``) or to limit a window to.
        time_start_ns: Inclusive start of the selected ``window``.
        time_end_ns: Exclusive end of the selected ``window``.
        baseline_session_ids: Explicit baseline sessions for ``session``
            scope. Empty means every other loaded session.
        fingerprint_id: The grouped issue to select (``fingerprint``).
        window_ns: How far around each issue hit the ``fingerprint``
            baseline reaches.
        fields: Restricts the comparison to these eligible fields.
        max_values: Categorical fields with more distinct values are skipped.
        min_events: Support below this marks results as low confidence.
        limit: Keeps only the highest scoring fields.
    """

    scope: ComparisonScope
    session_id: str | None = None
    job_id: str | None = None
    rank: int | None = None
    time_start_ns: int | None = None
    time_end_ns: int | None = None
    baseline_session_ids: tuple[str, ...] = ()
    fingerprint_id: str | None = None
    window_ns: int = DEFAULT_NEIGHBOR_WINDOW_NS
    fields: tuple[str, ...] = ()
    max_values: int = DEFAULT_MAX_VALUES
    min_events: int = DEFAULT_MIN_EVENTS
    limit: int | None = None

    def __post_init__(self) -> None:
        if self.scope not in get_args(ComparisonScope):
            raise ValueError(
                f"scope must be one of {', '.join(get_args(ComparisonScope))}"
            )
        missing = [
            name
            for name in _REQUIRED_SCOPE_ARGUMENTS[self.scope]
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(f"{self.scope} scope requires {', '.join(missing)}")
        self._validate_scope_arguments()
        _validate_tuning(
            fields=self.fields,
            max_values=self.max_values,
            min_events=self.min_events,
            limit=self.limit,
        )

    def _validate_scope_arguments(self) -> None:
        if (
            self.time_start_ns is not None
            and self.time_end_ns is not None
            and self.time_start_ns >= self.time_end_ns
        ):
            raise ValueError("time_start_ns must be earlier than time_end_ns")
        if self.baseline_session_ids and self.scope != "session":
            raise ValueError("baseline sessions only apply to session scope")
        if self.session_id is not None and self.session_id in self.baseline_session_ids:
            raise ValueError("the selected session cannot also be a baseline session")
        if self.window_ns <= 0:
            raise ValueError("window_ns must be > 0")


@dataclass(frozen=True)
class ComparisonGroup:
    """Size and identity of one side of a comparison."""

    event_count: int
    session_ids: tuple[str, ...] = ()
    ranks: tuple[int, ...] = ()
    start_ns: int | None = None
    end_ns: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize the group to a JSON-safe dictionary."""
        return {
            "event_count": self.event_count,
            "session_ids": list(self.session_ids),
            "ranks": list(self.ranks),
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
        }


@dataclass(frozen=True)
class ValueShare:
    """How often one categorical value appears on each side."""

    value: str
    selection_count: int
    baseline_count: int
    selection_share: float
    baseline_share: float

    @property
    def delta(self) -> float:
        """Selection share minus baseline share."""
        return self.selection_share - self.baseline_share

    def as_dict(self) -> dict[str, Any]:
        """Serialize the value share to a JSON-safe dictionary."""
        return {
            "value": self.value,
            "selection_count": self.selection_count,
            "baseline_count": self.baseline_count,
            "selection_share": self.selection_share,
            "baseline_share": self.baseline_share,
            "delta": self.delta,
        }


@dataclass(frozen=True)
class FieldComparison:
    """How strongly one field separates the selection from the baseline.

    ``top_values`` is set for categorical fields; the median and p95 pairs
    are set for numeric fields.
    """

    field: str
    kind: FieldKind
    score: float
    confidence: Confidence
    reasons: tuple[str, ...]
    selection_present: int
    baseline_present: int
    selection_missing_ratio: float
    baseline_missing_ratio: float
    top_values: tuple[ValueShare, ...] = ()
    selection_median: float | None = None
    baseline_median: float | None = None
    selection_p95: float | None = None
    baseline_p95: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize the field comparison to a JSON-safe dictionary."""
        return {
            "field": self.field,
            "kind": self.kind,
            "score": self.score,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "selection_present": self.selection_present,
            "baseline_present": self.baseline_present,
            "selection_missing_ratio": self.selection_missing_ratio,
            "baseline_missing_ratio": self.baseline_missing_ratio,
            "top_values": [value.as_dict() for value in self.top_values],
            "selection_median": self.selection_median,
            "baseline_median": self.baseline_median,
            "selection_p95": self.selection_p95,
            "baseline_p95": self.baseline_p95,
        }


@dataclass(frozen=True)
class SkippedField:
    """An eligible field left out of the ranking, and why."""

    field: str
    reason: Literal["high_cardinality", "constant", "no_data"]
    distinct_values: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Serialize the skipped field to a JSON-safe dictionary."""
        return {
            "field": self.field,
            "reason": self.reason,
            "distinct_values": self.distinct_values,
        }


@dataclass(frozen=True)
class ComparisonResult:
    """Ranked fields that separate a selection from its baseline."""

    scope: ComparisonScope
    selection: ComparisonGroup
    baseline: ComparisonGroup
    fields: Sequence[FieldComparison] = ()
    skipped: Sequence[SkippedField] = ()
    criteria: Mapping[str, Any] = field(default_factory=dict)
    warnings: Sequence[str] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the result to a JSON-safe dictionary."""
        return {
            "format": COMPARISON_FORMAT,
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "scope": self.scope,
            "criteria": dict(self.criteria),
            "selection": self.selection.as_dict(),
            "baseline": self.baseline.as_dict(),
            "fields": [row.as_dict() for row in self.fields],
            "skipped": [row.as_dict() for row in self.skipped],
            "warnings": list(self.warnings),
        }


def split_events(
    filters: ComparisonFilter,
    events: Sequence[TelemetryEvent],
) -> tuple[list[TelemetryEvent], list[TelemetryEvent]]:
    """Split events into the selection and baseline a filter describes.

    Handles the ``rank``, ``window`` and ``session`` scopes. ``fingerprint``
    scope needs issue evidence; use :func:`split_fingerprint_events`.

    Args:
        filters: The comparison to run.
        events: Candidate events, already loaded.

    Returns:
        The selected events and the baseline events, in input order. Events
        outside the filter's session, rank or job limits are in neither.

    Raises:
        ValueError: If called with ``fingerprint`` scope.
    """
    if filters.scope == "fingerprint":
        raise ValueError("fingerprint scope needs issue evidence")
    selection: list[TelemetryEvent] = []
    baseline: list[TelemetryEvent] = []
    for event in events:
        if not _within_limits(filters, event):
            continue
        side = _side(filters, event)
        if side is not None:
            (selection if side else baseline).append(event)
    return selection, baseline


def _within_limits(filters: ComparisonFilter, event: TelemetryEvent) -> bool:
    if filters.job_id is not None and event.job_id != filters.job_id:
        return False
    if filters.scope == "rank":
        return filters.session_id is None or event.session_id == filters.session_id
    if filters.scope == "window":
        return event.session_id == filters.session_id and (
            filters.rank is None or event.rank == filters.rank
        )
    return True


def _side(filters: ComparisonFilter, event: TelemetryEvent) -> bool | None:
    """Return True for the selection, False for the baseline, None for neither."""
    if filters.scope == "rank":
        return event.rank == filters.rank
    if filters.scope == "window":
        start, end = filters.time_start_ns, filters.time_end_ns
        if start is None or end is None:
            return None
        return start <= event.timestamp_ns < end
    if event.session_id == filters.session_id:
        return True
    if filters.baseline_session_ids:
        return False if event.session_id in filters.baseline_session_ids else None
    return False


def split_fingerprint_events(
    events: Sequence[TelemetryEvent],
    hits: Iterable[tuple[str | None, int | None, int | None]],
    *,
    window_ns: int = DEFAULT_NEIGHBOR_WINDOW_NS,
) -> tuple[list[TelemetryEvent], list[TelemetryEvent]]:
    """Split events into one issue's hits and their neighboring events.

    Args:
        events: Candidate events, already loaded.
        hits: ``(session_id, rank, timestamp_ns)`` for each piece of issue
            evidence. Hits without a session or timestamp cannot be matched to
            telemetry and are ignored; a ``None`` rank matches any rank.
        window_ns: How far before and after each hit a neighbor may be.

    Returns:
        The events that are issue hits, and the other events of the same
        session that lie within ``window_ns`` of a hit.
    """
    hit_keys: set[tuple[str, int | None, int]] = set()
    hit_times: dict[str, list[int]] = defaultdict(list)
    for session_id, rank, timestamp_ns in hits:
        if session_id is None or timestamp_ns is None:
            continue
        hit_keys.add((session_id, rank, timestamp_ns))
        hit_times[session_id].append(timestamp_ns)
    for times in hit_times.values():
        times.sort()
    selection: list[TelemetryEvent] = []
    baseline: list[TelemetryEvent] = []
    for event in events:
        session_times = hit_times.get(event.session_id)
        if not session_times:
            continue
        key = (event.session_id, event.rank, event.timestamp_ns)
        if key in hit_keys or (event.session_id, None, event.timestamp_ns) in hit_keys:
            selection.append(event)
        elif _near_any(session_times, event.timestamp_ns, window_ns):
            baseline.append(event)
    return selection, baseline


def _near_any(sorted_times: Sequence[int], timestamp_ns: int, window_ns: int) -> bool:
    index = bisect_left(sorted_times, timestamp_ns - window_ns)
    return index < len(sorted_times) and sorted_times[index] <= timestamp_ns + window_ns


def describe_events(events: Sequence[TelemetryEvent]) -> ComparisonGroup:
    """Summarize which sessions, ranks and time span a set of events covers."""
    if not events:
        return ComparisonGroup(event_count=0)
    timestamps = [event.timestamp_ns for event in events]
    return ComparisonGroup(
        event_count=len(events),
        session_ids=tuple(sorted({event.session_id for event in events})),
        ranks=tuple(sorted({event.rank for event in events})),
        start_ns=min(timestamps),
        end_ns=max(timestamps),
    )


def compare_events(
    selection: Sequence[TelemetryEvent],
    baseline: Sequence[TelemetryEvent],
    *,
    scope: ComparisonScope,
    fields: Sequence[str] = (),
    max_values: int = DEFAULT_MAX_VALUES,
    min_events: int = DEFAULT_MIN_EVENTS,
    limit: int | None = None,
    criteria: Mapping[str, Any] | None = None,
    warnings: Sequence[str] = (),
) -> ComparisonResult:
    """Rank the fields whose distributions differ most between two event sets.

    Args:
        selection: The events under suspicion.
        baseline: The events they are compared against.
        scope: The scope that produced the two sets. Fields that define it
            (for example ``rank`` in rank scope) are not compared.
        fields: Restricts the comparison to these eligible fields. Empty
            compares every eligible field.
        max_values: Categorical fields with more distinct values than this
            are skipped as high cardinality.
        min_events: Sides or fields with fewer events than this are marked
            low confidence.
        limit: Keeps only the highest scoring fields.
        criteria: Echoed back in the result so it can be reproduced.
        warnings: Caller warnings to carry into the result.

    Returns:
        The ranked comparison. Either side being empty yields no fields and
        a warning rather than an error.

    Raises:
        ValueError: If a field is not eligible or a tuning value is invalid.
    """
    _validate_tuning(
        fields=fields, max_values=max_values, min_events=min_events, limit=limit
    )
    all_warnings = list(warnings) + _size_warnings(
        len(selection), len(baseline), min_events
    )
    ranked: list[FieldComparison] = []
    skipped: list[SkippedField] = []
    if selection and baseline:
        ranked, skipped = _rank_fields(
            _field_table(selection),
            _field_table(baseline),
            names=_compared_fields(scope, fields),
            max_values=max_values,
            min_events=min_events,
        )
    return ComparisonResult(
        scope=scope,
        selection=describe_events(selection),
        baseline=describe_events(baseline),
        fields=ranked[:limit] if limit is not None else ranked,
        skipped=skipped,
        criteria=dict(criteria or {}),
        warnings=all_warnings,
    )


def _validate_tuning(
    *,
    fields: Sequence[str],
    max_values: int,
    min_events: int,
    limit: int | None,
) -> None:
    unknown = sorted(set(fields) - set(ELIGIBLE_FIELDS))
    if unknown:
        raise ValueError(
            f"unknown comparison field(s): {', '.join(unknown)}; "
            f"choose from {', '.join(ELIGIBLE_FIELDS)}"
        )
    if max_values < 1:
        raise ValueError("max_values must be >= 1")
    if min_events < 1:
        raise ValueError("min_events must be >= 1")
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")


def _size_warnings(selected: int, baseline: int, min_events: int) -> list[str]:
    warnings: list[str] = []
    for side, count in (("selection", selected), ("baseline", baseline)):
        if count == 0:
            warnings.append(f"{side} has no events; nothing to compare")
        elif count < min_events:
            warnings.append(
                f"{side} has {count} events, fewer than min_events={min_events}; "
                "every field is low confidence"
            )
    return warnings


def _compared_fields(scope: ComparisonScope, fields: Sequence[str]) -> list[str]:
    requested = set(fields) if fields else set(ELIGIBLE_FIELDS)
    excluded = set(SCOPE_DEFINING_FIELDS[scope])
    return [name for name in ELIGIBLE_FIELDS if name in requested - excluded]


def _field_table(events: Sequence[TelemetryEvent]) -> dict[str, list[object]]:
    extractors = {**_CATEGORICAL_FIELDS, **_NUMERIC_FIELDS}
    table: dict[str, list[object]] = {name: [] for name in extractors}
    for event in events:
        derived = compute_event_fields(event)
        for name, extract in extractors.items():
            table[name].append(extract(event, derived))
    return table


def _rank_fields(
    selection: Mapping[str, list[object]],
    baseline: Mapping[str, list[object]],
    *,
    names: Sequence[str],
    max_values: int,
    min_events: int,
) -> tuple[list[FieldComparison], list[SkippedField]]:
    ranked: list[FieldComparison] = []
    skipped: list[SkippedField] = []
    for name in names:
        if name in _CATEGORICAL_FIELDS:
            outcome = _categorical_comparison(
                name, selection[name], baseline[name], max_values, min_events
            )
        else:
            outcome = _numeric_comparison(
                name, selection[name], baseline[name], min_events
            )
        if isinstance(outcome, SkippedField):
            skipped.append(outcome)
        else:
            ranked.append(outcome)
    ranked.sort(key=lambda row: (-row.score, row.field))
    return ranked, skipped


def _category(value: object) -> str:
    if value is None:
        return MISSING_VALUE
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _categorical_comparison(
    name: str,
    selection: Sequence[object],
    baseline: Sequence[object],
    max_values: int,
    min_events: int,
) -> FieldComparison | SkippedField:
    selected = Counter(_category(value) for value in selection)
    reference = Counter(_category(value) for value in baseline)
    values = sorted(selected.keys() | reference.keys())
    if len(values) > max_values:
        return SkippedField(name, "high_cardinality", distinct_values=len(values))
    if len(values) == 1:
        return SkippedField(name, "constant", distinct_values=1)
    shares = [
        ValueShare(
            value=value,
            selection_count=selected[value],
            baseline_count=reference[value],
            selection_share=selected[value] / len(selection),
            baseline_share=reference[value] / len(baseline),
        )
        for value in values
    ]
    shares.sort(key=lambda share: (-abs(share.delta), share.value))
    return FieldComparison(
        field=name,
        kind="categorical",
        score=0.5 * sum(abs(share.delta) for share in shares),
        **_support(name, selection, baseline, min_events),
        top_values=tuple(shares[:TOP_VALUE_COUNT]),
    )


def _numeric_values(values: Sequence[object]) -> np.ndarray:
    present = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return np.asarray(present, dtype=float)


def _numeric_comparison(
    name: str,
    selection: Sequence[object],
    baseline: Sequence[object],
    min_events: int,
) -> FieldComparison | SkippedField:
    selected = _numeric_values(selection)
    reference = _numeric_values(baseline)
    if selected.size == 0 and reference.size == 0:
        return SkippedField(name, "no_data")
    support = _support(name, selection, baseline, min_events, numeric=True)
    missing_delta = abs(
        support["selection_missing_ratio"] - support["baseline_missing_ratio"]
    )
    combined = np.concatenate([selected, reference])
    if missing_delta == 0 and np.ptp(combined) == 0:
        return SkippedField(name, "constant", distinct_values=1)
    statistic = (
        ks_statistic(selected, reference) if selected.size and reference.size else 0.0
    )
    return FieldComparison(
        field=name,
        kind="numeric",
        score=max(statistic, missing_delta),
        **support,
        selection_median=_percentile(selected, 50),
        baseline_median=_percentile(reference, 50),
        selection_p95=_percentile(selected, 95),
        baseline_p95=_percentile(reference, 95),
    )


def _support(
    name: str,
    selection: Sequence[object],
    baseline: Sequence[object],
    min_events: int,
    *,
    numeric: bool = False,
) -> dict[str, Any]:
    if numeric:
        selection_present = int(_numeric_values(selection).size)
        baseline_present = int(_numeric_values(baseline).size)
    else:
        selection_present = sum(value is not None for value in selection)
        baseline_present = sum(value is not None for value in baseline)
    reasons: list[str] = []
    if len(selection) < min_events:
        reasons.append("small_selection")
    if len(baseline) < min_events:
        reasons.append("small_baseline")
    if numeric and min(selection_present, baseline_present) < min_events:
        reasons.append("sparse_field")
    return {
        "confidence": "low" if reasons else "high",
        "reasons": tuple(reasons),
        "selection_present": selection_present,
        "baseline_present": baseline_present,
        "selection_missing_ratio": 1 - selection_present / len(selection),
        "baseline_missing_ratio": 1 - baseline_present / len(baseline),
    }


def ks_statistic(first: np.ndarray, second: np.ndarray) -> float:
    """Return the two-sample Kolmogorov-Smirnov statistic of two samples.

    This is the largest vertical distance between the two empirical CDFs,
    in ``[0, 1]``. Both samples must be non-empty.
    """
    first = np.sort(first)
    second = np.sort(second)
    grid = np.concatenate([first, second])
    first_cdf = np.searchsorted(first, grid, side="right") / first.size
    second_cdf = np.searchsorted(second, grid, side="right") / second.size
    return float(np.max(np.abs(first_cdf - second_cdf)))


def _percentile(values: np.ndarray, percent: float) -> float | None:
    return float(np.percentile(values, percent)) if values.size else None


__all__ = [
    "CATEGORICAL_FIELDS",
    "COMPARISON_FORMAT",
    "COMPARISON_SCHEMA_VERSION",
    "DEFAULT_MAX_VALUES",
    "DEFAULT_MIN_EVENTS",
    "DEFAULT_NEIGHBOR_WINDOW_NS",
    "ELIGIBLE_FIELDS",
    "MISSING_VALUE",
    "NUMERIC_FIELDS",
    "SCOPE_DEFINING_FIELDS",
    "TOP_VALUE_COUNT",
    "ComparisonFilter",
    "ComparisonGroup",
    "ComparisonResult",
    "ComparisonScope",
    "Confidence",
    "FieldComparison",
    "FieldKind",
    "SkippedField",
    "ValueShare",
    "compare_events",
    "describe_events",
    "ks_statistic",
    "split_events",
    "split_fingerprint_events",
]
