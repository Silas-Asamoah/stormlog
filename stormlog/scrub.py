"""Shared scrubbing primitives for what Stormlog records or sends elsewhere.

An exporter builds its output from allowlists of fields; these helpers are
the layers underneath, shared so that every surface redacts the same way.
Each exporter documents its own allowlists. See ``docs/scrubbing.md``.
"""

from __future__ import annotations

import base64
import json
import urllib.parse
from collections.abc import Iterable
from typing import overload

REDACTED = "<redacted>"
# Shorter values are not used for exact-value redaction: they would erase
# ordinary text such as case names and numbers, and a credential that short
# is not protected by redaction anyway.
MIN_SECRET_LENGTH = 8


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


class KnownSecrets:
    """The exact credentials Stormlog was given, redacted wherever they appear.

    Allowlists decide which fields leave; this is the backstop behind them.
    Each value is matched in the forms it most often travels in: as given,
    percent-encoded (inside a URL), JSON-escaped (inside a JSON string), and
    base64 (an encoded token, or a Basic authorization header when the
    ``user:password`` pair is registered).
    """

    def __init__(self, values: Iterable[str | None] = ()) -> None:
        self._forms: tuple[str, ...] = ()
        self.skipped_short = 0
        for value in values:
            self.add(value)

    def add(self, value: str | None) -> None:
        """Register one value; empty and short values are skipped."""
        if not value:
            return
        if len(value) < MIN_SECRET_LENGTH:
            self.skipped_short += 1
            return
        forms = set(self._forms) | _encoded_forms(value)
        # Longest first, so a form that contains another is replaced whole.
        self._forms = tuple(sorted(forms, key=lambda form: (-len(form), form)))

    def redact(self, text: str) -> str:
        """``text`` with every form of every registered value replaced."""
        for form in self._forms:
            if form in text:
                text = text.replace(form, REDACTED)
        return text

    def found_in(self, text: str) -> bool:
        """Whether any form of any registered value occurs in ``text``."""
        return any(form in text for form in self._forms)


def url_secrets(url: str | None) -> list[str]:
    """The values in a URL that may be credentials, for ``KnownSecrets``.

    The password, the ``user:password`` pair a Basic header would encode, a
    user name given without a password (often a token), and every query
    value.
    """
    if not url:
        return []
    parts = urllib.parse.urlsplit(url)
    found: list[str] = []
    if parts.password is not None:
        found.append(urllib.parse.unquote(parts.password))
        found.append(
            f"{urllib.parse.unquote(parts.username or '')}:"
            f"{urllib.parse.unquote(parts.password)}"
        )
    elif parts.username:
        found.append(urllib.parse.unquote(parts.username))
    found.extend(
        value
        for _, value in urllib.parse.parse_qsl(parts.query, keep_blank_values=False)
    )
    return found


def _encoded_forms(value: str) -> set[str]:
    raw = value.encode("utf-8")
    forms = {
        value,
        urllib.parse.quote(value, safe=""),
        urllib.parse.quote_plus(value, safe=""),
        json.dumps(value)[1:-1],
        json.dumps(value, ensure_ascii=False)[1:-1],
    }
    for encoded in (base64.b64encode(raw), base64.urlsafe_b64encode(raw)):
        text = encoded.decode("ascii")
        forms.update({text, text.rstrip("=")})
    return {form for form in forms if form}
