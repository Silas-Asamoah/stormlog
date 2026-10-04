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
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

from .._export.http_server import parse_listen
from .._export.otlp_http import Destination
from .._export.registry import DEFAULT_MAX_BYTES, DEFAULT_MAX_SAMPLES
from .._export.textfile import validate_slot
from ..scrub import is_forbidden_key_name
from .export_spans import CONTENT_ITEMS
from .trace_context import (
    FOLLOW_SAMPLING,
    OFF,
    POLICIES,
    PRESERVE_ENGINE,
    parse_server_sampler,
)

Command = Literal["profile", "watch"]
# Every label value of a profile comes from its configuration, so nothing
# needs room to appear later; a watcher's trigger IDs can change.
DEFAULT_HEADROOM: dict[str, int] = {"profile": 0, "watch": 64}
MIN_MAX_BYTES = 4096
# The longest linger or textfile interval.
MAX_SECONDS = 3600.0
DEFAULT_FLUSH_SECONDS = 5.0
DEFAULT_PROBE_SECONDS = 8.0
# A probe interval longer than a batch's retry budget would let a batch
# settle with no attempt at all.
PROBE_SECONDS_RANGE = (0.5, 30.0)


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
    # Spans: to an OTLP/HTTP endpoint, or to a file of OTLP JSON lines.
    otlp_endpoint: str | None = None
    otlp_file: Path | None = None
    otlp_file_fsync: bool = False
    # NAME=VALUE pairs; the values are credentials, never recorded.
    otlp_headers: tuple[str, ...] = ()
    # Send headers over plain http to a host other than this one.
    otlp_allow_insecure_headers: bool = False
    otlp_resource_attributes: tuple[str, ...] = ()
    otlp_resource_attribute_allow: tuple[str, ...] = ()
    otlp_flush_timeout_seconds: float = DEFAULT_FLUSH_SECONDS
    otlp_probe_interval_seconds: float = DEFAULT_PROBE_SECONDS
    # What free text spans may carry: digests, errors, prompts, outputs.
    export_content: frozenset[str] = frozenset()

    @property
    def prometheus_enabled(self) -> bool:
        return (
            self.prometheus_listen is not None
            or self.prometheus_textfile_dir is not None
        )

    @property
    def otlp_enabled(self) -> bool:
        return self.otlp_endpoint is not None or self.otlp_file is not None

    @property
    def enabled(self) -> bool:
        return self.prometheus_enabled or self.otlp_enabled

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
        settings = _given(self) if given is None else set(given)
        _check_dependent(self, settings)
        _check_numbers(self)
        _check_trace_context(self, command, settings)
        _check_otlp(self, settings)

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
        for name in ("prometheus_textfile_dir", "otlp_file"):
            if values.get(name) is not None:
                values[name] = Path(values[name])
        for name in (
            "otlp_headers",
            "otlp_resource_attributes",
            "otlp_resource_attribute_allow",
        ):
            if name in values:
                values[name] = tuple(values[name])
        if "export_content" in values:
            values["export_content"] = frozenset(values["export_content"])
        config = _with_server_ratio(cls(**values), set(mapping))
        config.validate(command, given=set(mapping))
        return config


# What each JSON setting must be, as the flags' parsers would make it.
_TEXT, _FLAG, _COUNT, _SECONDS = "a string", "true or false", "an integer", "a number"
_TEXTS = "a list of strings"
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
    "otlp_endpoint": _TEXT,
    "otlp_file": _TEXT,
    "otlp_file_fsync": _FLAG,
    "otlp_allow_insecure_headers": _FLAG,
    "otlp_headers": _TEXTS,
    "otlp_resource_attributes": _TEXTS,
    "otlp_resource_attribute_allow": _TEXTS,
    "otlp_flush_timeout_seconds": _SECONDS,
    "otlp_probe_interval_seconds": _SECONDS,
    "export_content": _TEXTS,
}


# Settings that may be null, meaning not set.
_NULLABLE = {
    "prometheus_listen",
    "prometheus_textfile_dir",
    "prometheus_series_headroom",
    "server_trace_sampler",
    "otlp_endpoint",
    "otlp_file",
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
        _TEXTS: isinstance(value, (list, tuple))
        and all(isinstance(item, str) for item in value),
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


def _check_trace_context(
    config: ExportConfig, command: Command, given: set[str]
) -> None:
    if config.trace_context not in POLICIES:
        raise ValueError(f"--trace-context must be one of {', '.join(POLICIES)}")
    if command == "watch" and config.trace_context != OFF:
        raise ValueError(
            "--trace-context applies to infer profile only: a watcher sends no "
            "requests"
        )
    if not 0.0 <= config.sample_ratio <= 1.0:
        raise ValueError("--otlp-sample-ratio must be between 0 and 1")
    if (
        "sample_ratio" in given
        and config.trace_context != FOLLOW_SAMPLING
        and not config.otlp_enabled
    ):
        raise ValueError(
            "--otlp-sample-ratio only applies with --otlp-endpoint, --otlp-file "
            "or --trace-context follow-sampling"
        )
    if config.trace_context == PRESERVE_ENGINE and config.sample_ratio != 1.0:
        raise ValueError(
            "--otlp-sample-ratio has no effect with --trace-context "
            "preserve-engine: every request carries a sampled traceparent, and "
            "every request span that did is exported"
        )
    _check_server_sampler(config, given)


def _check_server_sampler(config: ExportConfig, given: set[str]) -> None:
    if config.server_trace_sampler is None:
        return
    sampler = parse_server_sampler(config.server_trace_sampler)
    if (
        config.trace_context == FOLLOW_SAMPLING
        and sampler.parent_based
        and sampler.ratio is not None
        and "sample_ratio" in given
        and config.sample_ratio > sampler.ratio
    ):
        raise ValueError(
            f"--otlp-sample-ratio {config.sample_ratio:g} is above the "
            f"{sampler.ratio:g} the server's sampler declares: follow-sampling "
            "would raise the server's tracing volume"
        )


def _with_server_ratio(config: ExportConfig, given: set[str]) -> ExportConfig:
    """Under follow-sampling, Stormlog's ratio defaults to the server's declared one.

    Only a parent-based sampler follows the flag; any other keeps its own
    share whatever Stormlog sends, so the ratio is left as it is.
    """
    if (
        config.trace_context != FOLLOW_SAMPLING
        or config.server_trace_sampler is None
        or "sample_ratio" in given
    ):
        return config
    try:
        sampler = parse_server_sampler(config.server_trace_sampler)
    except ValueError:
        return config  # refused by validate
    if not sampler.parent_based or sampler.ratio is None:
        return config
    return replace(config, sample_ratio=sampler.ratio)


def sampler_warnings(config: ExportConfig) -> list[str]:
    """Why the trace context would make the server record more, if it would.

    A parent-based server sampler follows a sampled parent, so each request
    marked sampled is recorded whatever the server's own ratio.
    """
    if config.trace_context == OFF or config.server_trace_sampler is None:
        return []
    sampler = parse_server_sampler(config.server_trace_sampler)
    if not sampler.parent_based:
        return []
    marked = 1.0 if config.trace_context == PRESERVE_ENGINE else config.sample_ratio
    if sampler.ratio is not None and marked <= sampler.ratio:
        return []
    share = "every request" if marked == 1.0 else f"{marked * 100:g}% of requests"
    declared = (
        "an unknown share" if sampler.ratio is None else f"{sampler.ratio * 100:g}%"
    )
    return [
        f"--trace-context {config.trace_context} marks {share} "
        f"sampled, and the server's declared sampler "
        f"{config.server_trace_sampler} follows a sampled parent: the server will "
        f"record that share of Stormlog's requests, not the {declared} it keeps "
        "on its own"
    ]


def _check_otlp(config: ExportConfig, given: set[str]) -> None:
    if config.otlp_endpoint is not None and config.otlp_file is not None:
        raise ValueError("use one of --otlp-endpoint and --otlp-file, not both")
    if config.otlp_endpoint is not None:
        Destination.parse(config.otlp_endpoint)
    if not config.otlp_enabled:
        _check_otlp_dependent(given)
    elif config.otlp_file is None and config.otlp_file_fsync:
        raise ValueError("--otlp-file-fsync only applies with --otlp-file")
    _check_otlp_values(config)
    _check_otlp_names(config)


def _check_otlp_values(config: ExportConfig) -> None:
    unknown = sorted(config.export_content - set(CONTENT_ITEMS))
    if unknown:
        raise ValueError(
            f"--export-content takes {', '.join(CONTENT_ITEMS)}; not {', '.join(unknown)}"
        )
    low, high = PROBE_SECONDS_RANGE
    if not low <= config.otlp_probe_interval_seconds <= high:
        raise ValueError(f"--otlp-probe-interval must be between {low:g} and {high:g}")
    if not 0 < config.otlp_flush_timeout_seconds <= 60:
        raise ValueError("--otlp-flush-timeout must be between 0 and 60 seconds")


def _check_otlp_names(config: ExportConfig) -> None:
    for flag, pairs in (
        ("--otlp-header", config.otlp_headers),
        ("--otlp-resource-attribute", config.otlp_resource_attributes),
    ):
        for pair in pairs:
            name, sep, _ = pair.partition("=")
            if not sep or not name.strip():
                # Never echoed: a header's value is a credential.
                raise ValueError(f"{flag} takes NAME=VALUE")
    for key in config.otlp_resource_attribute_allow:
        if is_forbidden_key_name(key):
            raise ValueError(
                f"--otlp-resource-attribute-allow cannot admit {key}: its name "
                "says its value may be a credential"
            )


def _check_otlp_dependent(given: set[str]) -> None:
    # Given, not changed from the default: a flag set to its default does
    # nothing without a destination either.
    for name, flag in (
        ("otlp_file_fsync", "--otlp-file-fsync"),
        ("otlp_headers", "--otlp-header"),
        ("otlp_allow_insecure_headers", "--otlp-allow-insecure-headers"),
        ("otlp_resource_attributes", "--otlp-resource-attribute"),
        ("otlp_resource_attribute_allow", "--otlp-resource-attribute-allow"),
        ("otlp_flush_timeout_seconds", "--otlp-flush-timeout"),
        ("otlp_probe_interval_seconds", "--otlp-probe-interval"),
        ("export_content", "--export-content"),
    ):
        if name in given:
            raise ValueError(f"{flag} only applies with --otlp-endpoint or --otlp-file")


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
        "marks it sampled: a server with a parent-based sampler then records "
        "every Stormlog request, which is what its default sampler does "
        "anyway but more than a parent-based ratio sampler would (a warning "
        "says so). follow-sampling sends Stormlog's own decision, at the "
        "server's declared ratio by default.",
    )
    group.add_argument(
        "--server-trace-sampler",
        metavar="NAME[:ARG]",
        help="The server's OTEL_TRACES_SAMPLER and its ratio, such as "
        "parentbased_traceidratio:0.1: checked against the SDK's sampler names "
        "and recorded as declared, since Stormlog cannot read it.",
    )
    group.add_argument(
        "--otlp-sample-ratio",
        type=float,
        metavar="RATIO",
        help="Stormlog's own sampling ratio, 0 to 1: what follow-sampling "
        "sends (default: the ratio --server-trace-sampler declares, else 1), "
        "and which request spans sent without trace context are exported. "
        "A request span that carried a traceparent is always exported.",
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
    _add_otlp_arguments(parser)


def _add_otlp_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "span export (optional)",
        "Send Stormlog's own spans over OTLP/HTTP, or write them as OTLP "
        "JSON lines. Off unless asked for; OTEL_* variables never turn it on.",
    )
    group.add_argument(
        "--otlp-endpoint",
        metavar="URL",
        help="An OTLP/HTTP traces URL; a bare origin gets /v1/traces. "
        "Credentials belong in --otlp-header, never the URL.",
    )
    group.add_argument(
        "--otlp-file",
        metavar="PATH",
        help="Append spans to PATH as OTLP JSON lines (capped at 256 MiB).",
    )
    group.add_argument(
        "--otlp-file-fsync",
        action="store_true",
        help="fsync the file after each line.",
    )
    group.add_argument(
        "--otlp-allow-insecure-headers",
        action="store_true",
        help="Send the OTLP headers, which hold credentials, over plain http "
        "to a host other than this one. Without it such a run is refused.",
    )
    group.add_argument(
        "--otlp-header",
        action="append",
        metavar="NAME=VALUE",
        help="A request header, such as an API key; repeatable. Its value is "
        "never recorded, and is redacted wherever it would appear. "
        "OTEL_EXPORTER_OTLP_HEADERS and OTEL_EXPORTER_OTLP_TRACES_HEADERS "
        "are read too.",
    )
    group.add_argument(
        "--otlp-resource-attribute",
        action="append",
        metavar="KEY=VALUE",
        help="A resource attribute, such as deployment.environment.name=prod; "
        "repeatable. Only a fixed list of keys is accepted.",
    )
    group.add_argument(
        "--otlp-resource-attribute-allow",
        action="append",
        metavar="KEY",
        help="Accept one more resource attribute key; refused for a name "
        "that suggests a credential.",
    )
    group.add_argument(
        "--otlp-flush-timeout",
        type=float,
        metavar="SECONDS",
        help=f"How long the run waits at its end for spans to leave "
        f"(default {DEFAULT_FLUSH_SECONDS:g}; 2 after Ctrl+C).",
    )
    group.add_argument(
        "--otlp-probe-interval",
        type=float,
        metavar="SECONDS",
        help=f"While the collector is down, how often it is retried "
        f"(default {DEFAULT_PROBE_SECONDS:g}).",
    )
    group.add_argument(
        "--export-content",
        metavar="ITEMS",
        help="Comma-separated free text spans may carry: digests, errors, "
        "prompts, outputs (default none). errors exports server error text, "
        "which can echo the request, prompt included.",
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
        **_otlp_from_args(args, defaults),
    )
    given = _given_flags(args)
    config = _with_server_ratio(config, given)
    config.validate(given=given)
    return config


# The flags whose value names an ExportConfig field differently.
_FLAG_FIELDS = {
    "prometheus_linger": "prometheus_linger_seconds",
    "prometheus_textfile_interval": "prometheus_textfile_interval_seconds",
    "otlp_sample_ratio": "sample_ratio",
    "otlp_header": "otlp_headers",
    "otlp_resource_attribute": "otlp_resource_attributes",
    "otlp_flush_timeout": "otlp_flush_timeout_seconds",
    "otlp_probe_interval": "otlp_probe_interval_seconds",
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


def _otlp_from_args(args: argparse.Namespace, defaults: ExportConfig) -> dict[str, Any]:
    otlp_file = getattr(args, "otlp_file", None)
    content = getattr(args, "export_content", None) or ""
    return {
        "otlp_endpoint": getattr(args, "otlp_endpoint", None),
        "otlp_file": Path(otlp_file) if otlp_file else None,
        "otlp_file_fsync": bool(getattr(args, "otlp_file_fsync", False)),
        "otlp_allow_insecure_headers": bool(
            getattr(args, "otlp_allow_insecure_headers", False)
        ),
        "otlp_headers": tuple(getattr(args, "otlp_header", None) or ()),
        "otlp_resource_attributes": tuple(
            getattr(args, "otlp_resource_attribute", None) or ()
        ),
        "otlp_resource_attribute_allow": tuple(
            getattr(args, "otlp_resource_attribute_allow", None) or ()
        ),
        "otlp_flush_timeout_seconds": _or(
            args, "otlp_flush_timeout", defaults.otlp_flush_timeout_seconds
        ),
        "otlp_probe_interval_seconds": _or(
            args, "otlp_probe_interval", defaults.otlp_probe_interval_seconds
        ),
        "export_content": frozenset(
            item.strip() for item in content.split(",") if item.strip()
        ),
    }


def _or(args: argparse.Namespace, name: str, default: Any) -> Any:
    value = getattr(args, name, None)
    return default if value is None else value
