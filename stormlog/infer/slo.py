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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, TypeGuard

from .errors import InferInputError, InferUsageError

SLO_FORMAT = "stormlog.infer.slo"
SLO_VERSION = 1
SLO_EVENT_TYPE = "infer.slo"

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
        """SHA-256 of the canonical document; 500 and 500.0 digest alike."""
        canonical = json.dumps(
            _integral_floats_as_ints(self.to_record()),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def criterion(self, key: str) -> Criterion | None:
        return next((item for item in self.criteria if item.key == key), None)


def load_slo(path: str | Path) -> SloSpec:
    """Read a policy file; a file that cannot be read or is invalid exits 5."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise InferInputError(f"SLO policy {path}: {exc.strerror or exc}") from exc
    except ValueError as exc:
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
    if not _is_real(value) or not math.isfinite(value) or value <= 0:
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
        if not _is_real(seconds) or not math.isfinite(seconds) or seconds <= 0:
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
    "Criterion",
    "CriterionDef",
    "SLO_EVENT_TYPE",
    "SLO_FORMAT",
    "SLO_VERSION",
    "SloInterval",
    "SloSpec",
    "load_slo",
    "parse_slo_flags",
    "slo_from_artifact",
    "slo_from_document",
    "slo_record",
]
