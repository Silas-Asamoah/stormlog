"""Shared scrubbing primitives for what Stormlog records or sends elsewhere.

An exporter builds its output from allowlists of fields; these helpers are
the layers underneath, shared so that every surface redacts the same way.
Each exporter documents its own allowlists. See ``docs/scrubbing.md``.
"""

from __future__ import annotations

import urllib.parse
from typing import overload

REDACTED = "<redacted>"


@overload
def redact_url(url: str) -> str: ...


@overload
def redact_url(url: None) -> None: ...


def redact_url(url: str | None) -> str | None:
    """A URL as it may be recorded: no credentials and no query string.

    Either can carry a token, so only the scheme, host, port and path are
    kept, and a removed query is marked.
    """
    if url is None:
        return None
    parts = urllib.parse.urlsplit(url)
    query = f"?{REDACTED}" if parts.query else ""
    return f"{parts.scheme}://{_host_and_port(parts)}{parts.path}{query}"


def _host_and_port(parts: urllib.parse.SplitResult) -> str:
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return host
