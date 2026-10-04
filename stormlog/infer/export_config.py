"""Export settings for ``infer profile`` (and ``infer watch``), from flags or JSON.

Nothing is exported unless a flag, or the ``"export"`` section of a watch
configuration, asks for it. Standard ``OTEL_*`` variables never turn export
on. Every setting is checked before the run sends anything; a setting the
exporter cannot use is a usage error (exit 2).
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

from .._export.http_server import parse_listen
from .._export.registry import DEFAULT_MAX_BYTES, DEFAULT_MAX_SAMPLES
from .._export.textfile import validate_slot
from .trace_context import FOLLOW_SAMPLING, OFF, POLICIES

Command = Literal["profile", "watch"]
# Every label value of a profile comes from its configuration, so nothing
# needs room to appear later; a watcher's trigger IDs can change.
DEFAULT_HEADROOM: dict[str, int] = {"profile": 0, "watch": 64}
MIN_MAX_BYTES = 4096
# The longest linger or textfile interval.
MAX_SECONDS = 3600.0


@dataclass(frozen=True)
class ExportConfig:
    """What to export, and where. All off by default."""

    prometheus_listen: str | None = None
    prometheus_linger_seconds: float = 0.0
    prometheus_textfile_dir: Path | None = None
    prometheus_slot: str = "default"
    prometheus_textfile_interval_seconds: float = 15.0
    prometheus_textfile_remove_on_exit: bool = False
    prometheus_max_series: int = DEFAULT_MAX_SAMPLES
    prometheus_max_bytes: int = DEFAULT_MAX_BYTES
    prometheus_series_headroom: int | None = None
    prometheus_case_label: bool = True
    # Profile only: send traceparent with each request, and how to flag it.
    trace_context: str = OFF
    # Stormlog's own head-sampling ratio: what follow-sampling sends, and
    # which request spans an OTLP export keeps.
    sample_ratio: float = 1.0
    # The server's OTEL_TRACES_SAMPLER as the operator declares it; recorded
    # as declared, never verified.
    server_trace_sampler: str | None = None

    @property
    def prometheus_enabled(self) -> bool:
        return (
            self.prometheus_listen is not None
            or self.prometheus_textfile_dir is not None
        )

    @property
    def enabled(self) -> bool:
        return self.prometheus_enabled

    def headroom(self, command: Command) -> int:
        if self.prometheus_series_headroom is not None:
            return self.prometheus_series_headroom
        return DEFAULT_HEADROOM[command]

    def validate(
        self, command: Command = "profile", given: Collection[str] | None = None
    ) -> None:
        """Raise ``ValueError`` for a setting the exporter cannot use.

        ``given`` names the settings that were set, by flag or in JSON, even
        to their defaults; without it, a setting counts as given when it
        differs from its default.
        """
        if self.prometheus_listen is not None:
            parse_listen(self.prometheus_listen)
        validate_slot(self.prometheus_slot)
        _check_dependent(self, _given(self) if given is None else set(given))
        _check_numbers(self)
        _check_trace_context(self, command)

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any], command: Command = "profile"
    ) -> ExportConfig:
        """Settings from the ``"export"`` section of a JSON configuration.

        The keys are the flags' names with underscores, such as
        ``prometheus_listen``; an unknown key is an error.
        """
        names = {item.name for item in fields(cls)}
        unknown = sorted(set(mapping) - names)
        if unknown:
            raise ValueError(f"unknown export settings: {', '.join(unknown)}")
        for name, value in mapping.items():
            _check_type(name, value)
        values = dict(mapping)
        if values.get("prometheus_textfile_dir") is not None:
            values["prometheus_textfile_dir"] = Path(values["prometheus_textfile_dir"])
        config = cls(**values)
        config.validate(command, given=set(mapping))
        return config


# What each JSON setting must be, as the flags' parsers would make it.
_TEXT, _FLAG, _COUNT, _SECONDS = "a string", "true or false", "an integer", "a number"
_SETTING_TYPES = {
    "prometheus_listen": _TEXT,
    "prometheus_linger_seconds": _SECONDS,
    "prometheus_textfile_dir": _TEXT,
    "prometheus_slot": _TEXT,
    "prometheus_textfile_interval_seconds": _SECONDS,
    "prometheus_textfile_remove_on_exit": _FLAG,
    "prometheus_max_series": _COUNT,
    "prometheus_max_bytes": _COUNT,
    "prometheus_series_headroom": _COUNT,
    "prometheus_case_label": _FLAG,
    "trace_context": _TEXT,
    "sample_ratio": _SECONDS,
    "server_trace_sampler": _TEXT,
}


# Settings that may be null, meaning not set.
_NULLABLE = {
    "prometheus_listen",
    "prometheus_textfile_dir",
    "prometheus_series_headroom",
    "server_trace_sampler",
}


def _check_type(name: str, value: Any) -> None:
    if value is None and name in _NULLABLE:
        return
    kind = _SETTING_TYPES[name]
    valid = {
        _TEXT: isinstance(value, str),
        _FLAG: isinstance(value, bool),
        _COUNT: isinstance(value, int) and not isinstance(value, bool),
        _SECONDS: isinstance(value, (int, float)) and not isinstance(value, bool),
    }[kind]
    if not valid:
        raise ValueError(f"export setting {name} must be {kind}, not {value!r}")


# Each setting that needs a destination, and the flag that sets it.
_NEEDS_PROMETHEUS = {
    "prometheus_slot": "--prometheus-slot",
    "prometheus_max_series": "--prometheus-max-series",
    "prometheus_max_bytes": "--prometheus-max-bytes",
    "prometheus_series_headroom": "--prometheus-series-headroom",
    "prometheus_case_label": "--prometheus-case-label",
}
_NEEDS_TEXTFILE = {
    "prometheus_textfile_interval_seconds": "--prometheus-textfile-interval",
    "prometheus_textfile_remove_on_exit": "--prometheus-textfile-remove-on-exit",
}


def _given(config: ExportConfig) -> set[str]:
    defaults = ExportConfig()
    return {
        item.name
        for item in fields(config)
        if getattr(config, item.name) != getattr(defaults, item.name)
    }


def _check_dependent(config: ExportConfig, given: set[str]) -> None:
    if not config.prometheus_enabled:
        for name, flag in _NEEDS_PROMETHEUS.items():
            if name in given:
                raise ValueError(
                    f"{flag} only applies with --prometheus-listen or "
                    "--prometheus-textfile-dir"
                )
    if config.prometheus_listen is None and "prometheus_linger_seconds" in given:
        raise ValueError("--prometheus-linger only applies with --prometheus-listen")
    if config.prometheus_textfile_dir is None:
        for name, flag in _NEEDS_TEXTFILE.items():
            if name in given:
                raise ValueError(f"{flag} only applies with --prometheus-textfile-dir")


def _check_trace_context(config: ExportConfig, command: Command) -> None:
    if config.trace_context not in POLICIES:
        raise ValueError(f"--trace-context must be one of {', '.join(POLICIES)}")
    if command == "watch" and config.trace_context != OFF:
        raise ValueError(
            "--trace-context applies to infer profile only: a watcher sends no "
            "requests"
        )
    if not 0.0 <= config.sample_ratio <= 1.0:
        raise ValueError("--otlp-sample-ratio must be between 0 and 1")
    if config.sample_ratio != 1.0 and config.trace_context != FOLLOW_SAMPLING:
        raise ValueError(
            "--otlp-sample-ratio only applies with --trace-context follow-sampling"
        )


def _check_numbers(config: ExportConfig) -> None:
    _check_time("--prometheus-linger", config.prometheus_linger_seconds, 0.0)
    _check_time(
        "--prometheus-textfile-interval",
        config.prometheus_textfile_interval_seconds,
        1.0,
    )
    if config.prometheus_max_series < 1:
        raise ValueError("--prometheus-max-series must be >= 1")
    if config.prometheus_max_bytes < MIN_MAX_BYTES:
        raise ValueError(f"--prometheus-max-bytes must be >= {MIN_MAX_BYTES}")
    headroom = config.prometheus_series_headroom
    if headroom is not None and headroom < 0:
        raise ValueError("--prometheus-series-headroom must be >= 0")


def _check_time(flag: str, value: float, lowest: float) -> None:
    # A wait on nan returns at once and one on inf or 1e300 overflows; both
    # would break the run's end instead of its start.
    if not (math.isfinite(value) and lowest <= value <= MAX_SECONDS):
        raise ValueError(
            f"{flag} must be finite, from {lowest:g} to {MAX_SECONDS:g} seconds"
        )


def add_trace_context_arguments(parser: argparse.ArgumentParser) -> None:
    """The trace-context flags; ``infer profile`` only."""
    group = parser.add_argument_group(
        "trace context (optional)",
        "Send W3C trace context with each request so a tracing server's span "
        "joins Stormlog's in one trace. Off unless asked for.",
    )
    group.add_argument(
        "--trace-context",
        choices=POLICIES,
        help="off (default) sends no traceparent. preserve-engine always "
        "marks it sampled, so the server records every request as it would "
        "without the header; follow-sampling sends Stormlog's own decision.",
    )
    group.add_argument(
        "--server-trace-sampler",
        metavar="NAME[:ARG]",
        help="The server's OTEL_TRACES_SAMPLER, such as "
        "parentbased_traceidratio:0.1, recorded as declared; Stormlog cannot "
        "read it.",
    )
    group.add_argument(
        "--otlp-sample-ratio",
        type=float,
        metavar="RATIO",
        help="Stormlog's own sampling ratio, 0 to 1 (default 1): what "
        "follow-sampling sends, and which request spans are exported.",
    )


def add_export_arguments(parser: argparse.ArgumentParser) -> None:
    """The export flags, shared by ``infer profile`` and ``infer watch``."""
    group = parser.add_argument_group(
        "export (optional)",
        "Expose Stormlog's own measurements to Prometheus. Off unless asked "
        "for; see docs/inference_export.md.",
    )
    group.add_argument(
        "--prometheus-listen",
        metavar="HOST:PORT",
        help="Serve /metrics here while the run lasts. There is no default "
        "port; a non-loopback address exposes an unauthenticated endpoint.",
    )
    group.add_argument(
        "--prometheus-linger",
        type=float,
        metavar="SECONDS",
        help="Keep /metrics up this long after the run, for a final scrape "
        "(default 0).",
    )
    group.add_argument(
        "--prometheus-textfile-dir",
        metavar="DIR",
        help="Also write DIR/stormlog-<slot>.prom, for node_exporter's "
        "textfile collector or for a run with no server at all.",
    )
    group.add_argument(
        "--prometheus-slot",
        metavar="NAME",
        help="This producer's name: the stormlog_producer label, and the "
        "textfile's and lock's names (default: default).",
    )
    group.add_argument(
        "--prometheus-textfile-interval",
        type=float,
        metavar="SECONDS",
        help="How often the textfile is rewritten (default 15).",
    )
    group.add_argument(
        "--prometheus-textfile-remove-on-exit",
        action="store_true",
        help="Remove the textfile when the run ends instead of keeping its "
        "final values.",
    )
    group.add_argument(
        "--prometheus-max-series",
        type=int,
        metavar="N",
        help=f"Refuse a run whose metrics need more than N samples per scrape "
        f"(default {DEFAULT_MAX_SAMPLES}).",
    )
    group.add_argument(
        "--prometheus-max-bytes",
        type=int,
        metavar="BYTES",
        help=f"Refuse a run whose metrics could exceed BYTES per scrape "
        f"(default {DEFAULT_MAX_BYTES}).",
    )
    group.add_argument(
        "--prometheus-series-headroom",
        type=int,
        metavar="N",
        help="Series a family may add beyond the configured ones before "
        "overflowing (default 0 for profile, 64 for watch).",
    )
    group.add_argument(
        "--prometheus-case-label",
        choices=("on", "off"),
        help='With off, every series has case="all", for matrices too big '
        "for the budget (default on).",
    )


def export_config_from_args(args: argparse.Namespace) -> ExportConfig:
    defaults = ExportConfig()
    textfile_dir = getattr(args, "prometheus_textfile_dir", None)
    config = ExportConfig(
        prometheus_listen=getattr(args, "prometheus_listen", None),
        prometheus_linger_seconds=_or(args, "prometheus_linger", 0.0),
        prometheus_textfile_dir=Path(textfile_dir) if textfile_dir else None,
        prometheus_slot=_or(args, "prometheus_slot", defaults.prometheus_slot),
        prometheus_textfile_interval_seconds=_or(
            args,
            "prometheus_textfile_interval",
            defaults.prometheus_textfile_interval_seconds,
        ),
        prometheus_textfile_remove_on_exit=bool(
            getattr(args, "prometheus_textfile_remove_on_exit", False)
        ),
        prometheus_max_series=_or(args, "prometheus_max_series", DEFAULT_MAX_SAMPLES),
        prometheus_max_bytes=_or(args, "prometheus_max_bytes", DEFAULT_MAX_BYTES),
        prometheus_series_headroom=getattr(args, "prometheus_series_headroom", None),
        prometheus_case_label=getattr(args, "prometheus_case_label", None) != "off",
        trace_context=_or(args, "trace_context", OFF),
        sample_ratio=_or(args, "otlp_sample_ratio", 1.0),
        server_trace_sampler=getattr(args, "server_trace_sampler", None),
    )
    config.validate(given=_given_flags(args))
    return config


# The flags whose value names an ExportConfig field differently.
_FLAG_FIELDS = {
    "prometheus_linger": "prometheus_linger_seconds",
    "prometheus_textfile_interval": "prometheus_textfile_interval_seconds",
    "otlp_sample_ratio": "sample_ratio",
}


def _given_flags(args: argparse.Namespace) -> set[str]:
    """The settings a flag was given for: none of the flags has a default."""
    given = set()
    for item in fields(ExportConfig):
        flag = next((k for k, v in _FLAG_FIELDS.items() if v == item.name), item.name)
        value = getattr(args, flag, None)
        # Not given: None, or False from a store_true flag (0.0 == False).
        if value is not None and value is not False:
            given.add(item.name)
    return given


def _or(args: argparse.Namespace, name: str, default: Any) -> Any:
    value = getattr(args, name, None)
    return default if value is None else value
