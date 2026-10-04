"""OTLP span export for an inference profile: headers, resource, and the exporter.

Settings come from flags; headers also from ``OTEL_EXPORTER_OTLP_HEADERS``
and ``OTEL_EXPORTER_OTLP_TRACES_HEADERS``, and resource attributes from
``OTEL_RESOURCE_ATTRIBUTES`` and ``OTEL_SERVICE_NAME``, but no variable
turns export on.

Header values are credentials: they are sent, never recorded, and added to
the known secrets redacted from everything exported. Resource attributes
are operator-declared identity, accepted only for a fixed list of keys
(plus any ``--otlp-resource-attribute-allow`` names), never for a name that
suggests a credential, and only as short printable values. Every key left
out is listed by name, without its value, in the capability record.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .._export.delivery import Breaker
from .._export.otlp_encoding import JsonEncoding, SpanEncoding, encoding_for
from .._export.otlp_http import Destination, OtlpHttpTransport
from .._export.span_export import FileSink, HttpSink, SpanExporter, SpanSink
from .._export.spans import Attributes
from ..scrub import (
    KnownSecrets,
    is_forbidden_key_name,
    redact_url,
    scrub_text,
    truncate_utf8,
)
from .correlation_events import CapabilityEvent, CorrelationContext
from .export_config import ExportConfig
from .export_spans import (
    DIGESTS,
    ERRORS,
    MAX_CONTENT_BYTES,
    OUTPUTS,
    PROMPTS,
    ProfileSpans,
    SpanIdentity,
    error_diagnostics,
    scope,
)

COMPONENT_OTLP = "export.otlp"
HEADER_VARIABLES = ("OTEL_EXPORTER_OTLP_HEADERS", "OTEL_EXPORTER_OTLP_TRACES_HEADERS")
RESOURCE_VARIABLE = "OTEL_RESOURCE_ATTRIBUTES"
SERVICE_NAME_VARIABLE = "OTEL_SERVICE_NAME"
PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
RESOURCE_KEYS = (
    "service.name",
    "service.namespace",
    "service.version",
    "service.instance.id",
    "deployment.environment.name",
    "deployment.environment",
    "host.name",
    "host.id",
    "host.arch",
    "os.type",
    "k8s.cluster.name",
    "k8s.namespace.name",
    "k8s.pod.name",
    "k8s.pod.uid",
    "k8s.node.name",
    "k8s.deployment.name",
    "k8s.statefulset.name",
    "k8s.container.name",
    "cloud.provider",
    "cloud.platform",
    "cloud.region",
    "cloud.availability_zone",
    "container.name",
    "container.id",
)
MAX_RESOURCE_VALUE_CHARS = 128
MAX_RESOURCE_ATTRIBUTES = 32
MAX_LISTED_KEYS = 32
COLLECTOR_MESSAGE_BYTES = 256
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+\Z")


def parse_pairs(text: str) -> list[tuple[str, str]]:
    """``key=value`` members of a comma-separated, percent-encoded list.

    The W3C Baggage-style format the OpenTelemetry variables use. A member
    without ``=`` or without a key is skipped.
    """
    pairs: list[tuple[str, str]] = []
    for member in text.split(","):
        key, sep, value = member.partition("=")
        key = urllib.parse.unquote(key.strip())
        if sep and key:
            pairs.append((key, urllib.parse.unquote(value.strip())))
    return pairs


def resolve_headers(flags: Sequence[str], environ: Mapping[str, str]) -> dict[str, str]:
    """Headers from the variables, then the flags; a later source wins.

    Raises ``ValueError`` for a name that is not an HTTP token, or a value
    with a line break, which would split the request.
    """
    found: list[tuple[str, str]] = []
    for variable in HEADER_VARIABLES:
        found.extend(parse_pairs(environ.get(variable, "")))
    for flag in flags:
        name, _, value = flag.partition("=")
        found.append((name.strip(), value))
    headers: dict[str, str] = {}
    for name, value in found:
        if not _HEADER_NAME.match(name):
            raise ValueError(f"OTLP header name {name!r} is not a valid HTTP name")
        if any(char in value for char in "\r\n\0"):
            raise ValueError(f"OTLP header {name} has a line break in its value")
        headers[name.lower()] = value
    return headers


def _check_header_transport(config: ExportConfig, headers: Mapping[str, str]) -> None:
    """Refuse to send headers, which hold credentials, in clear text off this host.

    ``OTEL_EXPORTER_OTLP_HEADERS`` set up for one collector would otherwise
    go to whatever ``--otlp-endpoint`` names.
    """
    if not headers or config.otlp_endpoint is None:
        return
    destination = Destination.parse(config.otlp_endpoint)
    if (
        destination.scheme == "https"
        or destination.loopback
        or config.otlp_allow_insecure_headers
    ):
        return
    raise ValueError(
        f"the OTLP headers ({', '.join(sorted(headers))}) would go in clear text "
        f"over http to {destination.host}; use https, or pass "
        "--otlp-allow-insecure-headers"
    )


# An HTTP auth scheme, as in "Bearer <token>".
_AUTH_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9._-]*\Z")


def header_secrets(value: str) -> list[str]:
    """The parts of a header value that may be credentials, for ``KnownSecrets``.

    The value; after an auth scheme (``Bearer <token>``, ``ApiKey <key>``),
    the credential on its own, since a collector's auth error often echoes
    only that; and for ``Basic``, the decoded ``user:password`` pair, the
    user and the password, as ``url_secrets`` gives for a URL.
    """
    found = [value]
    scheme, _, credential = value.strip().partition(" ")
    credential = credential.strip()
    if not (credential and _AUTH_SCHEME.match(scheme)):
        return found
    found.append(credential)
    if scheme.lower() == "basic":
        try:
            decoded = base64.b64decode(credential, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return found
        user, separator, password = decoded.partition(":")
        found.extend(part for part in (decoded, user) if part)
        if separator and password:
            found.append(password)
    return found


def resolve_resource(
    base: Attributes,
    *,
    flags: Sequence[str],
    allow: Sequence[str],
    environ: Mapping[str, str],
    secrets: KnownSecrets,
) -> tuple[Attributes, tuple[str, ...]]:
    """Stormlog's resource with the accepted extra attributes; and the keys left out."""
    offered = parse_pairs(environ.get(RESOURCE_VARIABLE, ""))
    if environ.get(SERVICE_NAME_VARIABLE):
        offered.append(("service.name", environ[SERVICE_NAME_VARIABLE]))
    for flag in flags:
        key, _, value = flag.partition("=")
        offered.append((key.strip(), value))
    allowed = set(RESOURCE_KEYS) | set(allow)
    merged: dict[str, Any] = dict(base)
    left_out: set[str] = set()
    for key, value in offered:
        if key not in allowed or is_forbidden_key_name(key) or not _printable(value):
            left_out.add(_listed(key, secrets))
            continue
        merged[key] = secrets.redact(value)
    if len(merged) > MAX_RESOURCE_ATTRIBUTES:
        extra = list(merged)[MAX_RESOURCE_ATTRIBUTES:]
        left_out.update(_listed(key, secrets) for key in extra)
    kept = tuple(list(merged.items())[:MAX_RESOURCE_ATTRIBUTES])
    return kept, tuple(sorted(left_out))[:MAX_LISTED_KEYS]


def _printable(value: str) -> bool:
    return (
        bool(value) and len(value) <= MAX_RESOURCE_VALUE_CHARS and value.isprintable()
    )


def _listed(key: str, secrets: KnownSecrets) -> str:
    # A key's name is shown so the operator can fix it; never its value.
    return truncate_utf8(secrets.redact("".join(c for c in key if c.isprintable())), 64)


def _proxy_set(environ: Mapping[str, str]) -> bool:
    return any(
        environ.get(name) or environ.get(name.lower()) for name in PROXY_VARIABLES
    )


class OtlpExport:
    """The spans of one profile: mapping, exporter, and what they report."""

    def __init__(
        self,
        config: ExportConfig,
        identity: SpanIdentity,
        *,
        host: str,
        version: str,
        secrets: KnownSecrets,
        environ: Mapping[str, str],
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.identity = identity
        self.on_warning = on_warning
        self.headers = resolve_headers(config.otlp_headers, environ)
        _check_header_transport(config, self.headers)
        for value in self.headers.values():
            for part in header_secrets(value):
                secrets.add(part)
        self.spans = ProfileSpans(identity, secrets)
        base: Attributes = (
            ("service.name", "stormlog"),
            ("service.version", version),
            ("service.instance.id", identity.session_id),
            ("host.name", host),
            ("process.pid", os.getpid()),
            ("stormlog.run_id", identity.run_id),
        )
        self.resource, self.dropped_resource_keys = resolve_resource(
            base,
            flags=config.otlp_resource_attributes,
            allow=config.otlp_resource_attribute_allow,
            environ=environ,
            secrets=secrets,
        )
        self.encoding, sink = self._sink(version)
        self.exporter: SpanExporter[Any] = SpanExporter(
            sink,
            self.encoding,
            resource=self.resource,
            scope=scope(version),
            to_span=self.spans.to_span,
            breaker=Breaker(probe_interval=config.otlp_probe_interval_seconds),
            keep_message=self._scrubbed if ERRORS in identity.content else None,
        )
        self.start_error: str | None = None
        self._environ = environ

    def _scrubbed(self, message: str) -> str:
        return scrub_text(
            message, max_bytes=COLLECTOR_MESSAGE_BYTES, secrets=self.spans.secrets
        )

    def _sink(self, version: str) -> tuple[SpanEncoding, SpanSink]:
        config = self.config
        if config.otlp_endpoint is None:
            assert config.otlp_file is not None
            return JsonEncoding(), FileSink(
                config.otlp_file, fsync=config.otlp_file_fsync
            )
        encoding = encoding_for()
        transport = OtlpHttpTransport(
            Destination.parse(config.otlp_endpoint),
            media_type=encoding.media_type,
            headers=self.headers,
            user_agent=f"stormlog/{version}",
        )
        return encoding, HttpSink(transport)

    @property
    def destination(self) -> str:
        config = self.config
        if config.otlp_endpoint is not None:
            return redact_url(config.otlp_endpoint, origin_only=True)
        return str(config.otlp_file)

    # ------------------------------------------------------------- the run
    def start(self) -> None:
        self.start_error = self.exporter.start()
        if self.start_error is not None:
            self._warn(
                f"OTLP span file {self.destination} could not be opened: "
                f"{self.start_error}; spans are not written"
            )
        endpoint = self.config.otlp_endpoint
        if (
            endpoint is not None
            and not Destination.parse(endpoint).loopback
            and _proxy_set(self._environ)
        ):
            self._warn(
                "a proxy variable is set, but spans go straight to "
                f"{self.destination}; a collector reachable only through a "
                "proxy is not supported"
            )

    def observe(self, record: dict[str, Any], extras: Mapping[str, Any] | None) -> None:
        envelope = self.spans.envelope(record, extras)
        if envelope is not None:
            self.exporter.offer(envelope, envelope.size)

    def offer_capture(
        self, *, started_ns: int, ended_ns: int, outcome: str, error_type: str | None
    ) -> None:
        envelope = self.spans.capture_envelope(
            started_ns=started_ns,
            ended_ns=ended_ns,
            outcome=outcome,
            error_type=error_type,
        )
        self.exporter.offer(envelope, envelope.size)

    def request_extras(
        self, prompt: str, output: str | None, error: BaseException | None
    ) -> dict[str, Any]:
        """What a request's span needs beyond its record; on the request's thread."""
        content = self.identity.content
        extras: dict[str, Any] = {}
        if error is not None:
            diagnostics = error_diagnostics(str(error))
            if diagnostics is not None:
                extras["error_diagnostics"] = diagnostics
        if PROMPTS in content:
            extras["prompt"] = truncate_utf8(
                prompt[:MAX_CONTENT_BYTES], MAX_CONTENT_BYTES
            )
        if output is not None:
            if OUTPUTS in content:
                extras["output"] = truncate_utf8(
                    output[:MAX_CONTENT_BYTES], MAX_CONTENT_BYTES
                )
            if DIGESTS in content:
                extras["output_digest"] = hashlib.sha256(
                    output.encode("utf-8", errors="replace")
                ).hexdigest()[:16]
        return extras

    def close(self, deadline: float) -> None:
        self.exporter.close(deadline)

    # ------------------------------------------------------------- reading
    def accounting(self) -> dict[str, Any]:
        accounting = self.exporter.accounting()
        accounting["sampled_out"] = self.spans.sampled_out
        return accounting

    def capability_event(self, context: CorrelationContext) -> CapabilityEvent:
        kind = self.exporter.sink.kind
        summary = self.exporter.summary()
        summary["spans"] = self.accounting()
        available = self.start_error is None
        collected = [kind] if summary["spans"]["exported"] else []
        return CapabilityEvent(
            context=context,
            event_id=f"capability:{COMPONENT_OTLP}",
            component=COMPONENT_OTLP,
            available=available,
            supported=["endpoint", "file"],
            enabled=[kind],
            collected=collected if available else [],
            metadata={
                "destination": self.destination,
                "kind": kind,
                "encoding": self.encoding.name,
                "compression": "gzip" if kind == "endpoint" else None,
                "headers": sorted(self.headers),
                "resource_keys": [key for key, _ in self.resource],
                "dropped_resource_keys": list(self.dropped_resource_keys),
                "sample_ratio": self.identity.sample_ratio,
                "trace_context": self.config.trace_context,
                "server_trace_sampler": self.config.server_trace_sampler,
                "content": sorted(self.identity.content),
                "error": self.start_error,
                "summary": summary,
            },
        )

    def _warn(self, message: str) -> None:
        if self.on_warning is not None:
            try:
                self.on_warning(message)
            except Exception:
                pass
