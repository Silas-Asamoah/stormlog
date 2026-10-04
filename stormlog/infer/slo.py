"""SLO policies: latency criteria declared at an explicit boundary.

A criterion is keyed ``boundary.metric``. ``client`` criteria are measured by
Stormlog's own client; ``server`` criteria are what vLLM reports, per request
through its span attributes or in aggregate through its histograms. The two
are never merged or relabelled. There is no client inter-token latency:
streamed chunks are not tokens.

A policy is a versioned JSON document (``stormlog.infer.slo`` v1) or a list
of ``KEY:MS`` flags. ``infer profile --slo`` records it in the artifact as an
``infer.slo`` record, so an artifact can carry its own declared SLO.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, TypeGuard

from .errors import InferInputError, InferUsageError

if TYPE_CHECKING:
    from .populations import MeasuredInterval
    from .vllm_analysis import JoinedSpans

SLO_FORMAT = "stormlog.infer.slo"
SLO_VERSION = 1
SLO_EVENT_TYPE = "infer.slo"
EVALUATION_FORMAT = "stormlog.infer.slo_evaluation"
EVALUATION_VERSION = 1

CLIENT: Final = "client"
SERVER: Final = "server"
MEASURED_WINDOW: Final = "measured_window"
SLIDING: Final = "sliding"
OFFERED: Final = "offered"
BOUNDS: Final = "bounds"

Boundary = Literal["client", "server"]

_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}")
_POLICY_KEYS = frozenset(
    {
        "format",
        "version",
        "name",
        "criteria",
        "attainment_target",
        "population",
        "unknown_policy",
        "interval",
    }
)
_CRITERION_KEYS = frozenset({"metric", "boundary", "max_ms", "attainment_target"})
_INTERVAL_KEYS = frozenset({"kind", "seconds"})


@dataclass(frozen=True)
class CriterionDef:
    """What a criterion measures, and where its value comes from."""

    boundary: Boundary
    metric: str
    definition: str
    per_request: str | None
    aggregate: str | None

    @property
    def key(self) -> str:
        return f"{self.boundary}.{self.metric}"


_DEFINITIONS = (
    CriterionDef(
        CLIENT,
        "ttft",
        "send to the first non-empty content delta, on the client",
        "infer.request ttft_ms",
        None,
    ),
    CriterionDef(
        CLIENT,
        "ttft_from_intended",
        "intended arrival to the first non-empty content delta, on the client",
        "infer.request ttft_ms + dispatch_lag_ms",
        None,
    ),
    CriterionDef(
        CLIENT,
        "e2e",
        "send to the end of the response, after [DONE] is parsed, on the client",
        "infer.request e2e_latency_ms",
        None,
    ),
    CriterionDef(
        CLIENT,
        "e2e_from_intended",
        "intended arrival to the end of the response, on the client",
        "infer.request ended_at_ns - intended_at_ns",
        None,
    ),
    CriterionDef(
        CLIENT,
        "tpot",
        "(e2e - ttft) / (output_tokens - 1) with server-reported output tokens; "
        "a per-request mean, not inter-token latency",
        "infer.request with output_token_source server_usage",
        None,
    ),
    CriterionDef(
        SERVER,
        "ttft",
        "vLLM's own time to first token",
        "span gen_ai.latency.time_to_first_token",
        "vllm:time_to_first_token_seconds",
    ),
    CriterionDef(
        SERVER,
        "e2e",
        "vLLM's own end-to-end latency",
        "span gen_ai.latency.e2e",
        "vllm:e2e_request_latency_seconds",
    ),
    CriterionDef(
        SERVER,
        "queue",
        "vLLM's first wait in the scheduler queue",
        "span gen_ai.latency.time_in_queue",
        "vllm:request_queue_time_seconds",
    ),
    CriterionDef(
        SERVER,
        "itl",
        "vLLM's inter-token latency, as vLLM defines it",
        None,
        "vllm:inter_token_latency_seconds",
    ),
    CriterionDef(
        SERVER,
        "tpot",
        "vLLM's time per output token",
        None,
        "vllm:request_time_per_output_token_seconds",
    ),
)

CRITERIA: Mapping[str, CriterionDef] = MappingProxyType(
    {definition.key: definition for definition in _DEFINITIONS}
)


@dataclass(frozen=True)
class Criterion:
    """One latency limit: a value passes when it is at most ``max_ms``."""

    metric: str
    boundary: Boundary
    max_ms: float
    # A marginal target for this criterion alone, as in MLPerf's separate
    # per-metric percentile limits.
    attainment_target: float | None = None

    @property
    def key(self) -> str:
        return f"{self.boundary}.{self.metric}"

    @property
    def definition(self) -> CriterionDef:
        return CRITERIA[self.key]

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "metric": self.metric,
            "boundary": self.boundary,
            "max_ms": self.max_ms,
        }
        if self.attainment_target is not None:
            record["attainment_target"] = self.attainment_target
        return record


@dataclass(frozen=True)
class SloInterval:
    """The interval attainment and goodput are judged over.

    ``measured_window`` is a case's declared interval, for offline analysis;
    ``sliding`` is a window of ``seconds``, for an online watcher.
    """

    kind: Literal["measured_window", "sliding"] = MEASURED_WINDOW
    seconds: float | None = None

    def to_record(self) -> dict[str, Any]:
        if self.kind == SLIDING:
            return {"kind": self.kind, "seconds": self.seconds}
        return {"kind": self.kind}


@dataclass(frozen=True)
class SloSpec:
    """A named SLO policy.

    ``attainment_target`` is joint: every criterion passes. Each criterion's
    own target is marginal. Version 1 has one population, ``offered``, and
    one unknown policy, ``bounds``: an outcome that cannot be judged widens
    the reported bounds and is never silently counted as met or missed.
    """

    name: str
    criteria: tuple[Criterion, ...]
    attainment_target: float | None = None
    interval: SloInterval = SloInterval()
    population: Literal["offered"] = OFFERED
    unknown_policy: Literal["bounds"] = BOUNDS

    def to_record(self) -> dict[str, Any]:
        """The policy as its versioned JSON document."""
        return {
            "format": SLO_FORMAT,
            "version": SLO_VERSION,
            "name": self.name,
            "criteria": [criterion.to_record() for criterion in self.criteria],
            "attainment_target": self.attainment_target,
            "population": self.population,
            "unknown_policy": self.unknown_policy,
            "interval": self.interval.to_record(),
        }

    def digest(self) -> str:
        """SHA-256 of the canonical document.

        500 and 500.0 digest alike, and so do the same criteria in another
        order: they are joined by AND.
        """
        record = self.to_record()
        record["criteria"] = sorted(
            record["criteria"], key=lambda item: (item["boundary"], item["metric"])
        )
        canonical = json.dumps(
            _integral_floats_as_ints(record), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def criterion(self, key: str) -> Criterion | None:
        return next((item for item in self.criteria if item.key == key), None)


def load_slo(path: str | Path) -> SloSpec:
    """Read a policy file; a file that cannot be read or is invalid exits 5."""
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys
        )
    except OSError as exc:
        raise InferInputError(f"SLO policy {path}: {exc.strerror or exc}") from exc
    except (ValueError, RecursionError) as exc:
        raise InferInputError(f"SLO policy {path}: not valid JSON ({exc})") from exc
    try:
        return slo_from_document(payload)
    except ValueError as exc:
        raise InferInputError(f"SLO policy {path}: {exc}") from exc


def slo_from_document(payload: Any) -> SloSpec:
    """Validate a ``stormlog.infer.slo`` v1 document.

    Raises:
        ValueError: naming the first rule the document breaks.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("the policy must be a JSON object")
    _reject_unknown(payload, _POLICY_KEYS, "the policy")
    if payload.get("format") != SLO_FORMAT:
        raise ValueError(f"format must be {SLO_FORMAT!r}")
    if payload.get("version") != SLO_VERSION or isinstance(
        payload.get("version"), bool
    ):
        raise ValueError(f"version must be {SLO_VERSION}")
    if payload.get("population", OFFERED) != OFFERED:
        raise ValueError(f"population must be {OFFERED!r} in version 1")
    if payload.get("unknown_policy", BOUNDS) != BOUNDS:
        raise ValueError(f"unknown_policy must be {BOUNDS!r} in version 1")
    criteria = payload.get("criteria")
    if not isinstance(criteria, list):
        raise ValueError("criteria must be a list")
    return _spec(
        name=payload.get("name"),
        criteria=tuple(
            _criterion_from_document(item, index) for index, item in enumerate(criteria)
        ),
        attainment_target=_target(
            payload.get("attainment_target"), "attainment_target"
        ),
        interval=_interval_from_document(
            payload.get("interval", {"kind": MEASURED_WINDOW})
        ),
    )


def parse_slo_flags(items: Sequence[str], *, name: str = "cli") -> SloSpec:
    """Build a policy from ``KEY:MS`` flags such as ``ttft:500``.

    A key without a boundary is a client criterion; ``server.ttft:400`` names
    the server one. Flags have no targets and judge the measured window.

    Raises:
        InferUsageError: for a flag that is not ``KEY:MS`` or names no
            criterion, a repeated key, or a limit that is not a positive number.
    """
    try:
        return _spec(
            name=name,
            criteria=tuple(_criterion_from_flag(item) for item in items),
            attainment_target=None,
            interval=SloInterval(),
        )
    except InferUsageError:
        raise
    except ValueError as exc:
        raise InferUsageError(f"--slo: {exc}") from exc


def slo_record(spec: SloSpec, *, session_id: str, source: str) -> dict[str, Any]:
    """The ``infer.slo`` record ``infer profile --slo`` writes."""
    return {
        "schema_version": 1,
        "event_type": SLO_EVENT_TYPE,
        "session_id": session_id,
        "source": source,
        "digest": spec.digest(),
        "slo": spec.to_record(),
    }


def slo_from_artifact(records: Sequence[Mapping[str, Any]]) -> SloSpec | None:
    """The policy an artifact declared, or None when it declared none.

    Raises:
        InferInputError: when the artifact holds more than one ``infer.slo``
            record, or one that is not a valid policy.
    """
    found = [record for record in records if record.get("event_type") == SLO_EVENT_TYPE]
    if not found:
        return None
    if len(found) > 1:
        raise InferInputError(
            f"the artifact holds {len(found)} infer.slo records; expected one"
        )
    try:
        return slo_from_document(found[0].get("slo"))
    except ValueError as exc:
        raise InferInputError(f"the artifact's infer.slo record: {exc}") from exc


Outcome = Literal["pass", "fail", "not_applicable", "unknown"]
CriteriaVerdict = Literal["criteria_met", "criteria_missed", "unknown"]

# vLLM span attributes for the server criteria judged per request, in seconds.
SPAN_ATTRIBUTES: Mapping[str, str] = MappingProxyType(
    {
        "server.ttft": "gen_ai.latency.time_to_first_token",
        "server.e2e": "gen_ai.latency.e2e",
        "server.queue": "gen_ai.latency.time_in_queue",
    }
)


@dataclass(frozen=True)
class CriterionValue:
    """A value to judge in milliseconds, or the reason there is none."""

    value_ms: float | None
    reason: str | None = None
    not_applicable: bool = False


@dataclass(frozen=True)
class CriterionOutcome:
    """How one criterion judged one request or span."""

    outcome: Outcome
    value_ms: float | None
    max_ms: float
    reason: str | None = None


@dataclass(frozen=True)
class CriteriaOutcome:
    """The criteria alone, with no claim about whether the service succeeded."""

    outcome: CriteriaVerdict
    criteria: Mapping[str, CriterionOutcome]


RequestVerdict = Literal["met", "missed", "unknown"]
_REQUEST_OUTCOMES: Mapping[CriteriaVerdict, RequestVerdict] = MappingProxyType(
    {"criteria_met": "met", "criteria_missed": "missed", "unknown": "unknown"}
)


@dataclass(frozen=True)
class RequestSloOutcome:
    """A client request judged against a policy.

    ``met`` needs a successful request and every criterion passing; any other
    status is ``missed``; a successful request with a criterion that cannot be
    judged, and none failing, is ``unknown``.
    """

    outcome: RequestVerdict
    status: str
    criteria: Mapping[str, CriterionOutcome]

    @property
    def met(self) -> bool | None:
        return {"met": True, "missed": False}.get(self.outcome)


@dataclass(frozen=True)
class SpanSloOutcome:
    """An engine-finished span judged against a policy's server criteria.

    A vLLM span carries no finish reason, and vLLM emits one for aborted and
    failed requests too, so a span shows whether the criteria were met, never
    whether the request succeeded.
    """

    outcome: CriteriaVerdict
    criteria: Mapping[str, CriterionOutcome]
    service_success: Literal["unverified"] = "unverified"


@dataclass(frozen=True)
class CriterionCounts:
    """One criterion over a case: outcomes among successful requests, and
    its marginal attainment bounds over every offered request."""

    passed: int
    failed: int
    not_applicable: int
    unknown: int
    attainment_lower: float | None
    attainment_upper: float | None

    def to_record(self) -> dict[str, Any]:
        return {
            "pass": self.passed,
            "fail": self.failed,
            "not_applicable": self.not_applicable,
            "unknown": self.unknown,
            "attainment_lower": self.attainment_lower,
            "attainment_upper": self.attainment_upper,
        }


@dataclass(frozen=True)
class SloEvaluation:
    """A case judged against a policy (``stormlog.infer.slo_evaluation`` v1).

    ``attainment_lower`` counts unknown outcomes as missed and
    ``attainment_upper`` as met, so missing evidence widens the bounds and
    never moves a single figure. Goodput is SLO goodput at the offered load:
    good requests per second of the case's rate interval, as vLLM's benchmark
    computes it, not the highest rate that meets a target. An unmeasurable
    evaluation (a criterion no successful request could be judged on, or one
    that is aggregate-only) has no attainment or goodput, never zero.
    ``cohort_valid`` and ``cohort_issues`` repeat the case's cohort checks
    (None when the caller gave no cohort): the figures are over the records
    the run holds, which an invalid cohort does not vouch for.
    """

    slo_name: str
    slo_digest: str
    slo_source: str
    status: Literal["evaluated", "unmeasurable"]
    reason: str | None
    population_declared: str
    population_evaluated: str
    offered: int
    met: int
    missed: int
    unknown: int
    attainment_lower: float | None
    attainment_upper: float | None
    evidence_coverage: float | None
    per_criterion: Mapping[str, CriterionCounts]
    interval: MeasuredInterval | None
    goodput_lower_rps: float | None
    goodput_upper_rps: float | None
    goodput_lower_output_tps: float | None
    cohort_valid: bool | None = None
    cohort_issues: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "format": EVALUATION_FORMAT,
            "version": EVALUATION_VERSION,
            "slo_name": self.slo_name,
            "slo_digest": self.slo_digest,
            "slo_source": self.slo_source,
            "status": self.status,
            "reason": self.reason,
            "population_declared": self.population_declared,
            "population_evaluated": self.population_evaluated,
            "offered": self.offered,
            "met": self.met,
            "missed": self.missed,
            "unknown": self.unknown,
            "attainment_lower": self.attainment_lower,
            "attainment_upper": self.attainment_upper,
            "evidence_coverage": self.evidence_coverage,
            "per_criterion": {
                key: counts.to_record() for key, counts in self.per_criterion.items()
            },
            "interval": None if self.interval is None else self.interval.to_record(),
            "goodput_lower_rps": self.goodput_lower_rps,
            "goodput_upper_rps": self.goodput_upper_rps,
            "goodput_lower_output_tps": self.goodput_lower_output_tps,
            "cohort_valid": self.cohort_valid,
            "cohort_issues": list(self.cohort_issues),
        }


def evaluate_criteria(
    values: Mapping[str, float | CriterionValue | None],
    spec: SloSpec,
    *,
    boundary: Literal["client", "server", "both"] = "both",
) -> CriteriaOutcome:
    """Judge each criterion of ``spec`` against ``values`` keyed by criterion.

    A criterion outside ``boundary`` is unknown, as is one with no value or a
    value that is not finite. A failing criterion makes the whole verdict
    ``criteria_missed``; otherwise an unknown one makes it ``unknown``.
    """
    outcomes = {
        criterion.key: _judge(criterion, values.get(criterion.key), boundary)
        for criterion in spec.criteria
    }
    return CriteriaOutcome(outcome=_combine(outcomes.values()), criteria=outcomes)


def evaluate_request(
    record: Mapping[str, Any],
    spec: SloSpec,
    *,
    span: Mapping[str, Any] | None = None,
    missing_span_reason: str = "no_joined_span",
) -> RequestSloOutcome:
    """Judge one ``infer.request`` record; ``span`` is its joined vLLM span.

    Client criteria read the record and server criteria read only ``span``'s
    attributes, so a missing span leaves the server criteria unknown, with
    ``missing_span_reason``, and is never filled from client values. A request
    that did not succeed is ``missed`` whatever its criteria say; they are
    still judged on what was recorded, for diagnosis.
    """
    values = {
        **client_values(record),
        **server_values(span, missing_span_reason=missing_span_reason),
    }
    criteria = evaluate_criteria(values, spec)
    status = str(record.get("status"))
    outcome = _REQUEST_OUTCOMES[criteria.outcome] if status == "ok" else "missed"
    return RequestSloOutcome(outcome=outcome, status=status, criteria=criteria.criteria)


def evaluate_span(span_attributes: Mapping[str, Any], spec: SloSpec) -> SpanSloOutcome:
    """Judge an engine-finished span against the policy's server criteria.

    Client criteria are unknown on a span. The outcome is ``criteria_met``,
    ``criteria_missed`` or ``unknown``, never ``met``: success is unverified.
    """
    criteria = evaluate_criteria(server_values(span_attributes), spec, boundary=SERVER)
    return SpanSloOutcome(outcome=criteria.outcome, criteria=criteria.criteria)


def span_attributes_by_request(
    records: Sequence[Mapping[str, Any]], span_paths: Sequence[str | Path] = ()
) -> JoinedSpans:
    """Each measured request's trusted vLLM span attributes, by ``x_request_id``.

    Spans come from the artifact and from ``span_paths``. A request whose span
    arrived again with different content, or that has several spans, is left
    out and listed in ``quarantined`` with the reason; pass that reason to
    ``evaluate_request`` as ``missing_span_reason``.

    Raises:
        InferInputError: for a span file or span record that cannot be read.
    """
    from .vllm_analysis import joined_span_attributes, load_external_spans

    rows = [dict(record) for record in records]
    try:
        return joined_span_attributes(rows, load_external_spans(rows, span_paths))
    except (OSError, ValueError) as exc:
        raise InferInputError(f"vLLM spans: {exc}") from exc


def request_span(
    record: Mapping[str, Any], spans: JoinedSpans
) -> tuple[Mapping[str, Any] | None, str]:
    """A request's trusted span, or None and the reason it has none."""
    request_id = str(record.get("x_request_id"))
    span = spans.by_request.get(request_id)
    return span, spans.quarantined.get(request_id, "no_joined_span")


def slo_attained(
    record: Mapping[str, Any], spec: SloSpec, *, span: Mapping[str, Any] | None = None
) -> bool | None:
    """True when met, False when missed, None when it cannot be judged."""
    return evaluate_request(record, spec, span=span).met


def client_values(record: Mapping[str, Any]) -> dict[str, CriterionValue]:
    """The client criteria's values for one ``infer.request`` record."""
    ttft = _number(record.get("ttft_ms"))
    e2e = _number(record.get("e2e_latency_ms"))
    return {
        "client.ttft": _value(ttft, "no_client_ttft"),
        "client.ttft_from_intended": _from_intended_ttft(ttft, record),
        "client.e2e": _value(e2e, "no_client_e2e"),
        "client.e2e_from_intended": _from_intended_e2e(record),
        "client.tpot": _client_tpot(record, ttft, e2e),
    }


def server_values(
    span_attributes: Mapping[str, Any] | None,
    *,
    missing_span_reason: str = "no_joined_span",
) -> dict[str, CriterionValue]:
    """The per-request server criteria's values from a span's attributes."""
    if span_attributes is None:
        return {
            key: CriterionValue(None, reason=missing_span_reason)
            for key in SPAN_ATTRIBUTES
        }
    values = {}
    for key, attribute in SPAN_ATTRIBUTES.items():
        seconds = _number(span_attributes.get(attribute))
        values[key] = _value(
            None if seconds is None else seconds * 1000.0,
            f"span_attribute_missing:{attribute}",
        )
    return values


def _judge(
    criterion: Criterion,
    supplied: float | CriterionValue | None,
    boundary: Literal["client", "server", "both"],
) -> CriterionOutcome:
    limit = criterion.max_ms
    if boundary != "both" and criterion.boundary != boundary:
        return CriterionOutcome(
            "unknown", None, limit, f"{criterion.boundary}_boundary"
        )
    if criterion.definition.per_request is None:
        return CriterionOutcome("unknown", None, limit, "aggregate_only")
    if isinstance(supplied, CriterionValue):
        return _judge_value(supplied, limit)
    return _judge_value(CriterionValue(supplied), limit)


def _judge_value(value: CriterionValue, limit: float) -> CriterionOutcome:
    if value.not_applicable:
        return CriterionOutcome("not_applicable", None, limit, value.reason)
    if value.value_ms is None:
        return CriterionOutcome("unknown", None, limit, value.reason or "no_value")
    if not math.isfinite(value.value_ms):
        return CriterionOutcome("unknown", value.value_ms, limit, "non_finite_value")
    outcome: Outcome = "pass" if value.value_ms <= limit else "fail"
    return CriterionOutcome(outcome, value.value_ms, limit)


def _combine(outcomes: Iterable[CriterionOutcome]) -> CriteriaVerdict:
    kinds = {item.outcome for item in outcomes}
    if "fail" in kinds:
        return "criteria_missed"
    if "unknown" in kinds:
        return "unknown"
    return "criteria_met"


def _from_intended_ttft(
    ttft: float | None, record: Mapping[str, Any]
) -> CriterionValue:
    lag = _number(record.get("dispatch_lag_ms"))
    if ttft is None:
        return CriterionValue(None, reason="no_client_ttft")
    if lag is None:
        return CriterionValue(None, reason="no_dispatch_lag")
    return CriterionValue(ttft + lag)


def _from_intended_e2e(record: Mapping[str, Any]) -> CriterionValue:
    intended, ended = record.get("intended_at_ns"), record.get("ended_at_ns")
    if not _is_real(intended) or not _is_real(ended):
        return CriterionValue(None, reason="no_intended_or_end_time")
    return CriterionValue((ended - intended) / 1_000_000.0)


def _client_tpot(
    record: Mapping[str, Any], ttft: float | None, e2e: float | None
) -> CriterionValue:
    """vLLM's formula, only with output tokens the server itself reported.

    A local tokenizer's count is deterministic but is not the served count,
    so it leaves TPOT unknown. One output token or none has no TPOT; vLLM's
    benchmark counts that as meeting the limit, and so does this.
    """
    if record.get("output_token_source") != "server_usage":
        return CriterionValue(None, reason="output_tokens_not_server_reported")
    tokens = record.get("output_tokens")
    if not isinstance(tokens, int) or isinstance(tokens, bool):
        return CriterionValue(None, reason="no_output_tokens")
    if tokens <= 1:
        return CriterionValue(None, reason="single_output_token", not_applicable=True)
    if ttft is None or e2e is None:
        return CriterionValue(None, reason="no_client_ttft_or_e2e")
    return CriterionValue((e2e - ttft) / (tokens - 1))


def _value(value: float | None, reason: str) -> CriterionValue:
    return CriterionValue(value) if value is not None else CriterionValue(None, reason)


def _number(value: Any) -> float | None:
    return float(value) if _is_real(value) else None


def _spec(
    *,
    name: Any,
    criteria: tuple[Criterion, ...],
    attainment_target: float | None,
    interval: SloInterval,
) -> SloSpec:
    if not isinstance(name, str) or _NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(
            "name must start with a lowercase letter and use only lowercase "
            "letters, digits and underscores, at most 64 characters"
        )
    if not criteria:
        raise ValueError("a policy needs at least one criterion")
    keys = [criterion.key for criterion in criteria]
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise ValueError(f"criterion {', '.join(repeated)} is given more than once")
    return SloSpec(
        name=name,
        criteria=criteria,
        attainment_target=attainment_target,
        interval=interval,
    )


def _criterion_from_document(item: Any, index: int) -> Criterion:
    label = f"criteria[{index}]"
    if not isinstance(item, Mapping):
        raise ValueError(f"{label} must be an object")
    _reject_unknown(item, _CRITERION_KEYS, label)
    boundary, metric = item.get("boundary"), item.get("metric")
    if not isinstance(boundary, str) or not isinstance(metric, str):
        raise ValueError(f"{label} needs a boundary and a metric")
    return Criterion(
        metric=metric,
        boundary=_known_boundary(boundary, metric),
        max_ms=_limit(item.get("max_ms"), f"{label}.max_ms"),
        attainment_target=_target(
            item.get("attainment_target"), f"{label}.attainment_target"
        ),
    )


def _criterion_from_flag(item: str) -> Criterion:
    key, separator, raw_limit = item.rpartition(":")
    if not separator or not key:
        raise InferUsageError(f"--slo {item!r}: expected KEY:MS, such as ttft:500")
    boundary, dot, metric = key.partition(".")
    if not dot:
        boundary, metric = CLIENT, key
    try:
        limit = float(raw_limit)
    except ValueError as exc:
        raise InferUsageError(f"--slo {item!r}: {raw_limit!r} is not a number") from exc
    try:
        return Criterion(
            metric=metric,
            boundary=_known_boundary(boundary, metric),
            max_ms=_limit(limit, "the limit"),
        )
    except ValueError as exc:
        raise InferUsageError(f"--slo {item!r}: {exc}") from exc


def _known_boundary(boundary: str, metric: str) -> Boundary:
    key = f"{boundary}.{metric}"
    if key in CRITERIA:
        return CRITERIA[key].boundary
    if key == f"{CLIENT}.itl":
        raise ValueError(
            "client.itl does not exist: streamed chunks are not tokens. Use "
            "client.tpot for a per-token mean, or server.itl for vLLM's own"
        )
    raise ValueError(f"unknown criterion {key}; known: {', '.join(sorted(CRITERIA))}")


def _limit(value: Any, label: str) -> float:
    if not _finite(value) or value <= 0:
        raise ValueError(f"{label} must be a positive number of milliseconds")
    return float(value)


def _target(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if not _is_real(value) or not 0 < value <= 1:
        raise ValueError(f"{label} must be a fraction above 0 and at most 1")
    return float(value)


def _interval_from_document(value: Any) -> SloInterval:
    if not isinstance(value, Mapping):
        raise ValueError("interval must be an object")
    _reject_unknown(value, _INTERVAL_KEYS, "interval")
    kind, seconds = value.get("kind"), value.get("seconds")
    if kind == MEASURED_WINDOW:
        if seconds is not None:
            raise ValueError("a measured_window interval takes no seconds")
        return SloInterval()
    if kind == SLIDING:
        if not _finite(seconds) or seconds <= 0:
            raise ValueError("a sliding interval needs seconds > 0")
        return SloInterval(kind=SLIDING, seconds=float(seconds))
    raise ValueError(f"interval.kind must be {MEASURED_WINDOW!r} or {SLIDING!r}")


def _reject_unknown(
    payload: Mapping[str, Any], allowed: frozenset[str], label: str
) -> None:
    unknown = sorted(str(key) for key in set(payload) - allowed)
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")


def _is_real(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: Any) -> TypeGuard[int | float]:
    """A real number a float can hold; an integer too large for one is not."""
    if not _is_real(value):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """A JSON object whose keys are each given once; readers differ on repeats."""
    keys = [key for key, _value in pairs]
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise ValueError(f"{', '.join(repeated)} given more than once")
    return dict(pairs)


def _integral_floats_as_ints(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _integral_floats_as_ints(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_integral_floats_as_ints(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


__all__ = [
    "CRITERIA",
    "CriteriaOutcome",
    "Criterion",
    "CriterionCounts",
    "CriterionDef",
    "CriterionOutcome",
    "CriterionValue",
    "RequestSloOutcome",
    "SPAN_ATTRIBUTES",
    "SpanSloOutcome",
    "SLO_EVENT_TYPE",
    "SLO_FORMAT",
    "SLO_VERSION",
    "EVALUATION_FORMAT",
    "EVALUATION_VERSION",
    "SloEvaluation",
    "SloInterval",
    "SloSpec",
    "load_slo",
    "parse_slo_flags",
    "request_span",
    "span_attributes_by_request",
    "slo_from_artifact",
    "slo_from_document",
    "slo_record",
    "client_values",
    "evaluate_criteria",
    "evaluate_request",
    "evaluate_span",
    "server_values",
    "slo_attained",
]
