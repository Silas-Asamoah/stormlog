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
def redact_url(url: str, *, origin_only: bool = False) -> str: ...


@overload
def redact_url(url: None, *, origin_only: bool = False) -> None: ...


def redact_url(url: str | None, *, origin_only: bool = False) -> str | None:
    """A URL as it may be recorded: no credentials and no query string.

    Either can carry a token, so only the scheme, host, port and path are
    kept, and a removed query is marked. ``origin_only`` drops the path too,
    for a destination whose path is not known to be safe: a token can sit
    in a path segment as easily as in a query.
    """
    if url is None:
        return None
    parts = urllib.parse.urlsplit(url)
    origin = f"{parts.scheme}://{_host_and_port(parts)}"
    if origin_only:
        return origin
    query = f"?{REDACTED}" if parts.query else ""
    return f"{origin}{parts.path}{query}"


def _host_and_port(parts: urllib.parse.SplitResult) -> str:
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return host
