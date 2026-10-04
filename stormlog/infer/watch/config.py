"""The watch configuration: ``stormlog.infer.watch_config`` version 1.

A JSON file, strictly validated: an unknown key, a wrong type, or settings
that contradict each other (a hold time shorter than its window, a window
longer than the history) are refused. Everything the file leaves out comes
from the defaults ``watch_defaults/1``, and the resolved configuration's
digest goes into the watcher's session record, so a run says exactly which
settings it qualified. The ``export`` section belongs to #220 and is passed
through unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class IncidentLimits:
    """How long an incident's windows are and how many may exist."""

    pre_seconds: float = 120.0
    post_seconds: float = 60.0
    max_open_incidents: int = 2
    max_incidents_per_hour: int = 30


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
        """The settings as recorded in the session record."""
        return {
            "format": CONFIG_FORMAT,
            "version": CONFIG_VERSION,
            "defaults": DEFAULTS_VERSION,
            "server": {
                "base_url": self.base_url,
                "metrics_url": self.metrics_url,
                "engine": self.engine,
            },
            "tick_seconds": self.tick_seconds,
            "scrape_timeout_seconds": self.scrape_timeout_seconds,
            "history": {"seconds": self.history_seconds, "bytes": self.history_bytes},
            "incident": asdict(self.incident),
            "store": asdict(self.store),
            "triggers": [dict(t) for t in self.trigger_settings],
            "guarantees": {
                spec.trigger_id: {
                    "shortest_firing_violation_seconds": (
                        spec.sustain.shortest_firing_violation()
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
            "export",
        },
        "config",
    )
    server = _section(payload, "server", {"base_url", "metrics_url", "engine"})
    base_url = server.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        raise InferUsageError("server.base_url is required")
    engine = _engine(server.get("engine"), "server.engine")
    tick = _positive(payload.get("tick_seconds", 1.0), "tick_seconds")
    history = _section(payload, "history", {"seconds", "bytes"})
    history_seconds = _positive(history.get("seconds", 600.0), "history.seconds")
    triggers = payload.get("triggers", DEFAULT_TRIGGERS)
    if not isinstance(triggers, (list, tuple)):
        raise InferUsageError("triggers must be a list")
    export = payload.get("export", {})
    if not isinstance(export, Mapping):
        raise InferUsageError("export must be an object")
    settings = tuple(triggers)
    specs = tuple(_trigger(t, tick, engine) for t in settings)
    _check_triggers(specs, history_seconds)
    return WatchConfig(
        base_url=base_url,
        metrics_url=server.get("metrics_url", "auto"),
        engine=engine,
        tick_seconds=tick,
        scrape_timeout_seconds=_positive(
            payload.get("scrape_timeout_seconds", min(2.0, tick)),
            "scrape_timeout_seconds",
        ),
        history_seconds=history_seconds,
        history_bytes=int(
            _positive(history.get("bytes", 32 * 1024 * 1024), "history.bytes")
        ),
        incident=_incident(payload, history_seconds),
        store=_store(payload),
        triggers=specs,
        trigger_settings=settings,
        export=dict(export),
    )


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


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise InferUsageError(f"{name} must be a number > 0")
    return float(value)


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
        pre_seconds=_positive(section.get("pre_seconds", 120.0), "pre_seconds"),
        post_seconds=_positive(section.get("post_seconds", 60.0), "post_seconds"),
        max_open_incidents=_count(
            section.get("max_open_incidents", 2), "max_open_incidents"
        ),
        max_incidents_per_hour=_count(
            section.get("max_incidents_per_hour", 30), "max_incidents_per_hour"
        ),
    )
    if limits.pre_seconds > history_seconds:
        raise InferUsageError("incident.pre_seconds must be <= history.seconds")
    return limits


def _store(payload: Mapping[str, Any]) -> StoreLimits:
    section = _section(
        payload,
        "store",
        {"max_total_bytes", "max_incident_bytes", "max_incidents", "max_age_hours"},
    )
    defaults = StoreLimits()
    try:
        return StoreLimits(
            max_total_bytes=int(
                section.get("max_total_bytes", defaults.max_total_bytes)
            ),
            max_incident_bytes=int(
                section.get("max_incident_bytes", defaults.max_incident_bytes)
            ),
            max_incidents=int(section.get("max_incidents", defaults.max_incidents)),
            max_age_hours=float(section.get("max_age_hours", defaults.max_age_hours)),
        )
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
        sustain = Sustain.with_defaults(
            window=float(settings.get("window_seconds", 30.0)),
            hold=float(settings.get("hold_seconds", 60.0)),
            clear=settings.get("clear_seconds"),
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


def _predicate(key: str, value: Any, engine: str | None) -> Any:
    options = value if isinstance(value, Mapping) else {}
    if key == "signal":
        if value not in SIGNALS:
            raise ValueError(f"signal must be one of {', '.join(SIGNALS)}")
        return SignalExceeds(str(value), SignalConfig(engine=engine))
    if key == "gauge":
        return GaugeAtLeast(
            family=str(options["family"]),
            threshold=float(options["at_least"]),
            share=float(options.get("share", 1.0)),
            min_samples=int(options.get("min_samples", 2)),
            engine=engine,
        )
    if key == "counter_rate":
        return CounterRateAtLeast(
            family=str(options["family"]),
            rate_per_s=float(options["at_least_per_s"]),
            engine=engine,
        )
    if key == "histogram_share":
        return HistogramShareAbove(
            family=str(options["family"]),
            value=float(options["above"]),
            share=float(options["share"]),
            min_samples=int(options.get("min_samples", 20)),
            engine=engine,
        )
    if key == "scrape_failures":
        return ScrapeFailures(consecutive=int(options.get("consecutive", 3)))
    if key == "scrape_failure_share":
        return ScrapeFailureShare(
            share=float(options.get("share", 0.05)),
            scrapes=int(options.get("scrapes", 60)),
        )
    return FrozenExporter(ticks=int(options.get("ticks", 5)), engine=engine)


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


def _check_triggers(specs: Sequence[TriggerSpec], history_seconds: float) -> None:
    ids = [spec.trigger_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise InferUsageError("trigger ids must be unique")
    for spec in specs:
        if spec.sustain.window > history_seconds:
            raise InferUsageError(
                f"trigger {spec.trigger_id}: window_seconds is longer than "
                "history.seconds"
            )
        if spec.kind not in (KIND_METRIC, KIND_SIGNAL, KIND_HEALTH):
            raise InferUsageError(
                f"trigger {spec.trigger_id}: kind {spec.kind!r} is not available yet"
            )
        if spec.action == ACTION_DEEP_CAPTURE:
            raise InferUsageError(
                f"trigger {spec.trigger_id}: deep capture is not available yet"
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
