"""The watch configuration: ``stormlog.infer.watch_config`` version 1.

A JSON file, strictly validated: an unknown key, a wrong type, a number
that is not finite (JSON readers accept NaN and Infinity), a URL that is
not http or https, or settings that contradict each other (a hold time
shorter than its window, a window, or incident windows, longer than the
history) are refused. Everything the file leaves out comes
from the defaults ``watch_defaults/1``, and the resolved configuration's
digest goes into the watcher's session record, so a run says exactly which
settings it qualified. The ``export`` section belongs to #220 and is passed
through unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from ...scrub import redact_url
from ..diagnosis_signals import SignalConfig
from ..diagnosis_vocabulary import (
    KV_PREEMPTION_PRESSURE,
    PREFIX_CACHE_LOSS,
    QUEUE_SATURATION,
)
from ..errors import InferInputError, InferUsageError
from .disk import StoreLimits
from .evaluate import (
    ACTION_DEEP_CAPTURE,
    ACTION_RECORD,
    KIND_HEALTH,
    KIND_METRIC,
    KIND_SIGNAL,
    HistoryPredicate,
    TriggerSpec,
)
from .predicates import (
    CounterRateAtLeast,
    FrozenExporter,
    GaugeAtLeast,
    HistogramShareAbove,
    ScrapeFailures,
    ScrapeFailureShare,
    SignalExceeds,
)
from .triggers import Sustain

CONFIG_FORMAT = "stormlog.infer.watch_config"
CONFIG_VERSION = 1
DEFAULTS_VERSION = "watch_defaults/1"
# A shorter tick would only measure the watcher's own overhead.
MIN_TICK_SECONDS = 0.05
# Upper bounds, refused with exit 2: past them a setting is a mistake, and
# its nanosecond conversions overflow (1e300 s crashed the watch with exit
# 1). A tick or scrape timeout of an hour; seven days of history, incident
# window or trigger time; retention of ten years.
MAX_TICK_SECONDS = 3600.0
MAX_SPAN_SECONDS = 7 * 24 * 3600.0
MAX_AGE_HOURS = 10 * 365 * 24.0
SIGNALS = (QUEUE_SATURATION, KV_PREEMPTION_PRESSURE, PREFIX_CACHE_LOSS)
_PREDICATE_KEYS = (
    "signal",
    "gauge",
    "counter_rate",
    "histogram_share",
    "scrape_failures",
    "scrape_failure_share",
    "frozen_exporter",
)
_TRIGGER_KEYS = {
    "id",
    "kind",
    "engine",
    "window_seconds",
    "hold_seconds",
    "clear_seconds",
    "action",
    "deep_capture_when",
    "counts_toward_exit",
    *_PREDICATE_KEYS,
}

DEFAULT_TRIGGERS: tuple[Mapping[str, Any], ...] = (
    {"id": "queue_saturation", "kind": "signal", "signal": QUEUE_SATURATION},
    {"id": "kv_preemption", "kind": "signal", "signal": KV_PREEMPTION_PRESSURE},
    {
        "id": "scrape_failures",
        "kind": "health",
        "window_seconds": 3,
        "hold_seconds": 3,
        "scrape_failures": {"consecutive": 3},
    },
    {
        "id": "scrape_failure_share",
        "kind": "health",
        "window_seconds": 60,
        "hold_seconds": 60,
        "scrape_failure_share": {"share": 0.05, "scrapes": 60},
    },
    {
        "id": "frozen_exporter",
        "kind": "health",
        "window_seconds": 5,
        "hold_seconds": 5,
        "frozen_exporter": {"ticks": 5},
    },
)

_DEFAULT_HEALTH_IDS = frozenset(
    t["id"] for t in DEFAULT_TRIGGERS if t["kind"] == "health"
)


# Incidents are admitted in lanes, each with the limits above: one for
# metric and signal triggers, and one side lane that health and test
# triggers share, so either lane can be full without turning the other's
# away.
INCIDENT_LANES = 2


@dataclass(frozen=True)
class IncidentLimits:
    """How long an incident's windows are and how many may exist, per lane."""

    pre_seconds: float = 120.0
    post_seconds: float = 60.0
    max_open_incidents: int = 2
    max_incidents_per_hour: int = 30

    def totals(self) -> dict[str, int]:
        """The bounds across both lanes, which a consumer sizes by."""
        return {
            "max_open_incidents_total": INCIDENT_LANES * self.max_open_incidents,
            "max_incidents_per_hour_total": (
                INCIDENT_LANES * self.max_incidents_per_hour
            ),
        }


@dataclass(frozen=True)
class WatchConfig:
    """A resolved, validated watch configuration."""

    base_url: str
    metrics_url: str | None
    engine: str | None
    tick_seconds: float
    scrape_timeout_seconds: float
    history_seconds: float
    history_bytes: int
    incident: IncidentLimits
    store: StoreLimits
    triggers: tuple[TriggerSpec, ...]
    trigger_settings: tuple[Mapping[str, Any], ...]
    export: Mapping[str, Any] = field(default_factory=dict)

    def resolved(self) -> dict[str, Any]:
        """The settings as recorded in the session record, every trigger as
        resolved (its sustain, policy and predicate), so two configs that
        watch the same way have the same digest. Server URLs are redacted;
        the digest describes these recorded settings. The ``export`` section
        is #220's and may hold credentials: it is left out."""
        return {
            "format": CONFIG_FORMAT,
            "version": CONFIG_VERSION,
            "defaults": DEFAULTS_VERSION,
            "server": {
                "base_url": redact_url(self.base_url),
                "metrics_url": (
                    "auto"
                    if self.metrics_url == "auto"
                    else redact_url(self.metrics_url)
                ),
                "engine": self.engine,
            },
            "tick_seconds": self.tick_seconds,
            "scrape_timeout_seconds": self.scrape_timeout_seconds,
            "history": {"seconds": self.history_seconds, "bytes": self.history_bytes},
            "incident": {**asdict(self.incident), **self.incident.totals()},
            "store": asdict(self.store),
            "triggers": [_resolved_trigger(spec) for spec in self.triggers],
            "guarantees": {
                spec.trigger_id: {
                    "shortest_firing_violation_seconds": (
                        spec.sustain.shortest_firing_violation(self.tick_seconds)
                    ),
                    # Each evaluation can run up to a scrape timeout late.
                    "detection_bound_seconds": spec.sustain.detection_bound(
                        self.tick_seconds, late=self.scrape_timeout_seconds
                    ),
                }
                for spec in self.triggers
            },
        }

    def digest(self) -> str:
        canonical = json.dumps(self.resolved(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def _resolved_trigger(spec: TriggerSpec) -> dict[str, Any]:
    return {
        "id": spec.trigger_id,
        "kind": spec.kind,
        "action": spec.action,
        "counts_toward_exit": spec.counts_toward_exit,
        "deep_capture_when": spec.deep_capture_when,
        "completion_recorded": spec.completion_recorded,
        "sustain": asdict(spec.sustain),
        # Every predicate is a dataclass; the protocols it is typed by are not.
        "predicate": {
            "type": type(spec.predicate).__name__,
            **asdict(cast(Any, spec.predicate)),
        },
    }


def load_watch_config(
    path: str | Path | None, *, overrides: Mapping[str, Any] | None = None
) -> WatchConfig:
    """Read a config file, apply command-line overrides, and resolve it.

    An unreadable file, or one that is not a version-1 watch config, is an
    :class:`InferInputError` (exit 5); a setting the watcher cannot use is an
    :class:`InferUsageError` (exit 2).
    """
    payload: dict[str, Any] = {"format": CONFIG_FORMAT, "version": CONFIG_VERSION}
    if path is not None:
        payload = _read(Path(path))
    for key, value in (overrides or {}).items():
        if value is not None:
            _set_path(payload, key, value)
    return resolve_watch_config(payload)


def resolve_watch_config(payload: Mapping[str, Any]) -> WatchConfig:
    """Validate a parsed config and fill it from the defaults."""
    if payload.get("format") != CONFIG_FORMAT or payload.get("version") != (
        CONFIG_VERSION
    ):
        raise InferInputError(
            f"not a {CONFIG_FORMAT} version {CONFIG_VERSION} document"
        )
    _only(
        payload,
        {
            "format",
            "version",
            "server",
            "tick_seconds",
            "scrape_timeout_seconds",
            "history",
            "incident",
            "store",
            "triggers",
            "default_health_triggers",
            "export",
        },
        "config",
    )
    server = _section(payload, "server", {"base_url", "metrics_url", "engine"})
    base_url = server.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        raise InferUsageError("server.base_url is required")
    _http_url(base_url, "server.base_url")
    metrics_url = server.get("metrics_url", "auto")
    if metrics_url not in (None, "auto"):
        _http_url(metrics_url, "server.metrics_url")
    engine = _engine(server.get("engine"), "server.engine")
    tick = _positive(
        payload.get("tick_seconds", 1.0), "tick_seconds", most=MAX_TICK_SECONDS
    )
    if tick < MIN_TICK_SECONDS:
        raise InferUsageError(f"tick_seconds must be at least {MIN_TICK_SECONDS}")
    history = _section(payload, "history", {"seconds", "bytes"})
    history_seconds = _positive(
        history.get("seconds", 600.0), "history.seconds", most=MAX_SPAN_SECONDS
    )
    export = payload.get("export", {})
    if not isinstance(export, Mapping):
        raise InferUsageError("export must be an object")
    settings = _trigger_settings(payload)
    specs = tuple(_trigger(t, tick, engine) for t in settings)
    _check_triggers(specs, history_seconds, tick)
    return WatchConfig(
        base_url=base_url,
        metrics_url=metrics_url,
        engine=engine,
        tick_seconds=tick,
        scrape_timeout_seconds=_positive(
            payload.get("scrape_timeout_seconds", min(2.0, tick)),
            "scrape_timeout_seconds",
            most=MAX_TICK_SECONDS,
        ),
        history_seconds=history_seconds,
        history_bytes=_count(history.get("bytes", 32 * 1024 * 1024), "history.bytes"),
        incident=_incident(payload, history_seconds),
        store=_store(payload),
        triggers=specs,
        trigger_settings=settings,
        export=dict(export),
    )


def _trigger_settings(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """The config's triggers, and the default health triggers it does not
    replace (same id) or turn off (``default_health_triggers: false``): a
    config naming its own triggers keeps the watch on its own scraper."""
    triggers = payload.get("triggers")
    if triggers is None:
        return DEFAULT_TRIGGERS
    if not isinstance(triggers, (list, tuple)):
        raise InferUsageError("triggers must be a list")
    if not _keep_health_defaults(payload):
        return tuple(triggers)
    ids = {t.get("id") for t in triggers if isinstance(t, Mapping)}
    health = [t for t in DEFAULT_TRIGGERS if t["kind"] == "health"]
    return (*triggers, *(t for t in health if t["id"] not in ids))


def _keep_health_defaults(payload: Mapping[str, Any]) -> bool:
    keep = payload.get("default_health_triggers", True)
    if not isinstance(keep, bool):
        raise InferUsageError("default_health_triggers must be a JSON boolean")
    return keep


def _read(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InferInputError(f"cannot read watch config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise InferInputError(f"watch config {path} is not a JSON object")
    return payload


def _set_path(payload: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node = payload
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def _only(section: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise InferUsageError(f"{name}: unknown key(s) {', '.join(unknown)}")


def _section(
    payload: Mapping[str, Any], name: str, allowed: set[str]
) -> Mapping[str, Any]:
    section = payload.get(name) or {}
    if not isinstance(section, Mapping):
        raise InferUsageError(f"{name} must be an object")
    _only(section, allowed, name)
    return section


def _positive(value: Any, name: str, *, most: float) -> float:
    try:
        number = _finite(value, name)
    except ValueError as exc:
        raise InferUsageError(f"{name} must be a finite number > 0") from exc
    if number <= 0:
        raise InferUsageError(f"{name} must be a finite number > 0")
    if number > most:
        raise InferUsageError(f"{name} must be at most {most:g}")
    return number


def _finite(value: Any, name: str) -> float:
    """A JSON number that is finite; ``ValueError`` otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _http_url(value: Any, name: str) -> None:
    parts = urllib.parse.urlsplit(value) if isinstance(value, str) else None
    if parts is None or parts.scheme not in ("http", "https") or not parts.hostname:
        raise InferUsageError(f"{name} must be an http or https URL with a host")


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InferUsageError(f"{name} must be an integer > 0")
    return int(value)


def _incident(payload: Mapping[str, Any], history_seconds: float) -> IncidentLimits:
    section = _section(
        payload,
        "incident",
        {"pre_seconds", "post_seconds", "max_open_incidents", "max_incidents_per_hour"},
    )
    limits = IncidentLimits(
        pre_seconds=_positive(
            section.get("pre_seconds", 120.0), "pre_seconds", most=MAX_SPAN_SECONDS
        ),
        post_seconds=_positive(
            section.get("post_seconds", 60.0), "post_seconds", most=MAX_SPAN_SECONDS
        ),
        max_open_incidents=_count(
            section.get("max_open_incidents", 2), "max_open_incidents"
        ),
        max_incidents_per_hour=_count(
            section.get("max_incidents_per_hour", 30), "max_incidents_per_hour"
        ),
    )
    # The seal reads both windows from the history: the oldest scrape it
    # needs is pre_seconds plus post_seconds old by then.
    if limits.pre_seconds + limits.post_seconds > history_seconds:
        raise InferUsageError(
            "incident.pre_seconds plus post_seconds must be <= history.seconds"
        )
    return limits


def _store(payload: Mapping[str, Any]) -> StoreLimits:
    section = _section(
        payload,
        "store",
        {"max_total_bytes", "max_incident_bytes", "max_incidents", "max_age_hours"},
    )
    defaults = StoreLimits()
    counts = {
        key: _count(section.get(key, getattr(defaults, key)), f"store.{key}")
        for key in ("max_total_bytes", "max_incident_bytes", "max_incidents")
    }
    age = _positive(
        section.get("max_age_hours", defaults.max_age_hours),
        "store.max_age_hours",
        most=MAX_AGE_HOURS,
    )
    try:
        return StoreLimits(**counts, max_age_hours=age)
    except (TypeError, ValueError) as exc:
        raise InferUsageError(f"store: {exc}") from exc


def _engine(value: Any, name: str) -> str | None:
    """An engine label value, as vLLM's ``engine`` label gives it, or None."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)) or value == "":
        raise InferUsageError(f'{name} must be an engine label, such as "0"')
    return str(value)


def _trigger(
    settings: Mapping[str, Any], tick: float, default_engine: str | None
) -> TriggerSpec:
    if not isinstance(settings, Mapping):
        raise InferUsageError("each trigger must be an object")
    name = str(settings.get("id") or "")
    _only(settings, _TRIGGER_KEYS, f"trigger {name or '?'}")
    present = [key for key in _PREDICATE_KEYS if key in settings]
    if len(present) != 1:
        raise InferUsageError(f"trigger {name}: give exactly one predicate")
    engine = _engine(settings.get("engine", default_engine), f"trigger {name}: engine")
    try:
        clear = settings.get("clear_seconds")
        sustain = Sustain.with_defaults(
            window=_seconds(settings.get("window_seconds", 30.0), "window_seconds"),
            hold=_seconds(settings.get("hold_seconds", 60.0), "hold_seconds"),
            clear=None if clear is None else _seconds(clear, "clear_seconds"),
            tick=tick,
        )
        return TriggerSpec(
            trigger_id=name,
            kind=str(settings.get("kind", "")),
            sustain=sustain,
            predicate=_predicate(present[0], settings[present[0]], engine),
            action=str(settings.get("action", ACTION_RECORD)),
            # Left out, each takes its kind's default from TriggerSpec, which
            # also refuses what the exit and capture policy forbids.
            deep_capture_when=_optional(settings, "deep_capture_when", str),
            counts_toward_exit=_optional(settings, "counts_toward_exit", bool),
            completion_recorded=_completion_recorded(settings),
        )
    except KeyError as exc:
        raise InferUsageError(f"trigger {name}: missing {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise InferUsageError(f"trigger {name}: {exc}") from exc


def _optional(settings: Mapping[str, Any], key: str, kind: type) -> Any:
    """A setting given as a JSON ``kind``, or None when left out."""
    value = settings.get(key)
    if value is not None and not isinstance(value, kind):
        name = "boolean" if kind is bool else "string"
        raise TypeError(f"{key} must be a JSON {name}")
    return value


def _seconds(value: Any, name: str) -> float:
    seconds = _finite(value, name)
    if not 0 < seconds <= MAX_SPAN_SECONDS:
        raise ValueError(f"{name} must be > 0 and at most {MAX_SPAN_SECONDS:g}")
    return seconds


def _share(value: Any, name: str) -> float:
    share = _finite(value, name)
    if not 0 < share <= 1:
        raise ValueError(f"{name} must be in (0, 1]")
    return share


def _whole(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be an integer > 0")
    return int(value)


def _family(options: Mapping[str, Any]) -> str:
    family = options["family"]
    if not isinstance(family, str) or not family:
        raise ValueError("family must be a metric name")
    return family


def _predicate(key: str, value: Any, engine: str | None) -> Any:
    options = value if isinstance(value, Mapping) else {}
    if key == "signal":
        if value not in SIGNALS:
            raise ValueError(f"signal must be one of {', '.join(SIGNALS)}")
        return SignalExceeds(str(value), SignalConfig(engine=engine))
    if key in ("gauge", "counter_rate", "histogram_share"):
        return _metric_predicate(key, options, engine)
    if key == "scrape_failures":
        return ScrapeFailures(
            consecutive=_whole(options.get("consecutive", 3), "consecutive")
        )
    if key == "scrape_failure_share":
        return ScrapeFailureShare(
            share=_share(options.get("share", 0.05), "share"),
            scrapes=_whole(options.get("scrapes", 60), "scrapes"),
        )
    return FrozenExporter(ticks=_whole(options.get("ticks", 5), "ticks"), engine=engine)


def _metric_predicate(key: str, options: Mapping[str, Any], engine: str | None) -> Any:
    if key == "gauge":
        return GaugeAtLeast(
            family=_family(options),
            threshold=_finite(options["at_least"], "at_least"),
            share=_share(options.get("share", 1.0), "share"),
            min_samples=_whole(options.get("min_samples", 2), "min_samples"),
            engine=engine,
        )
    if key == "counter_rate":
        rate = _finite(options["at_least_per_s"], "at_least_per_s")
        if rate < 0:
            raise ValueError("at_least_per_s must be >= 0")
        return CounterRateAtLeast(
            family=_family(options), rate_per_s=rate, engine=engine
        )
    return HistogramShareAbove(
        family=_family(options),
        value=_finite(options["above"], "above"),
        share=_share(options["share"], "share"),
        min_samples=_whole(options.get("min_samples", 20), "min_samples"),
        engine=engine,
    )


_COMPLETION_FAMILIES = (
    "vllm:e2e_request_latency_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:request_",
)


def _completion_recorded(settings: Mapping[str, Any]) -> bool:
    """Histograms vLLM observes when a request completes."""
    options = settings.get("histogram_share")
    if not isinstance(options, Mapping):
        return False
    family = str(options.get("family", ""))
    return any(family.startswith(prefix) for prefix in _COMPLETION_FAMILIES)


def _check_triggers(
    specs: Sequence[TriggerSpec], history_seconds: float, tick: float
) -> None:
    ids = [spec.trigger_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise InferUsageError("trigger ids must be unique")
    for spec in specs:
        if spec.sustain.window > history_seconds:
            raise InferUsageError(
                f"trigger {spec.trigger_id}: window_seconds is longer than "
                "history.seconds" + _default_hint(spec)
            )
        _check_tail(spec, history_seconds, tick)
        if spec.kind not in (KIND_METRIC, KIND_SIGNAL, KIND_HEALTH):
            raise InferUsageError(
                f"trigger {spec.trigger_id}: kind {spec.kind!r} is not available yet"
            )
        if spec.action == ACTION_DEEP_CAPTURE:
            raise InferUsageError(
                f"trigger {spec.trigger_id}: deep capture is not available yet"
            )


def _check_tail(spec: TriggerSpec, history_seconds: float, tick: float) -> None:
    """A health trigger that reads more scrapes than the history holds could
    never be judged; at 10**19 its count crashed the watch with exit 1."""
    predicate = spec.predicate
    if not isinstance(predicate, HistoryPredicate):
        return
    held = int(history_seconds / tick + 1e-9)
    if predicate.tail_scrapes > held:
        raise InferUsageError(
            f"trigger {spec.trigger_id}: it reads {predicate.tail_scrapes} "
            f"scrapes, more than history.seconds holds at tick_seconds ({held})"
            + _default_hint(spec)
        )


def _default_hint(spec: TriggerSpec) -> str:
    if spec.trigger_id not in _DEFAULT_HEALTH_IDS:
        return ""
    return (
        " (a default health trigger: raise history.seconds, give a "
        "trigger of that id, or set default_health_triggers to false)"
    )


__all__ = [
    "CONFIG_FORMAT",
    "CONFIG_VERSION",
    "DEFAULTS_VERSION",
    "DEFAULT_TRIGGERS",
    "IncidentLimits",
    "WatchConfig",
    "load_watch_config",
    "resolve_watch_config",
]
