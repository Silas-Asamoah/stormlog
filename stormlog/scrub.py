"""Shared scrubbing primitives for what Stormlog records or sends elsewhere.

An exporter builds its output from allowlists of fields; these helpers are
the layers underneath, shared so that every surface redacts the same way.
Each exporter documents its own allowlists. See ``docs/scrubbing.md``.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.parse
from collections.abc import Iterable
from typing import overload

REDACTED = "<redacted>"
# Shorter values are not used for exact-value redaction: they would erase
# ordinary text such as case names and numbers, and a credential that short
# is not protected by redaction anyway.
MIN_SECRET_LENGTH = 8
# scrub_text cuts a long text this many characters past the requested length
# before matching, so a large body cannot make scrubbing slow. A secret that
# starts inside the kept length ends inside the margin unless it is longer
# than the margin, so the final cut never leaves a fragment of one.
INPUT_MARGIN_CHARS = 4096

# Words that make a key name look like it holds a credential. Matched as
# substrings, case-insensitively, so "api-key", "API_KEY" and "apikey" all
# count; a few innocent names are caught too, which is the safe mistake.
SECRET_KEY_WORDS: tuple[str, ...] = (
    "pass",
    "pwd",
    "secret",
    "token",
    "key",
    "auth",
    "bearer",
    "cred",
    "cookie",
    "session",
    "signature",
    "private",
)
_SECRET_WORDS = "|".join(SECRET_KEY_WORDS)
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # A private key block, including one cut before its END line.
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?"
            r"(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)"
        ),
        REDACTED,
    ),
    # Header-style credentials: "Authorization: Bearer ..." to the end of line.
    (
        re.compile(r"(?i)\b((?:proxy-)?authorization)\s*([:=])\s*[^\r\n]+"),
        r"\1\2 " + REDACTED,
    ),
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"), r"\1 " + REDACTED),
    # URL user information, then URL query strings.
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s@]+@"), r"\1" + REDACTED + "@"),
    (
        re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://[^\s?#\"'<>]*)\?[^\s#\"'<>]*"),
        r"\1?" + REDACTED,
    ),
    # A JSON member whose key looks secret-like: "api_key": "..."
    (
        re.compile(
            r'(?i)("[^"\\]*(?:' + _SECRET_WORDS + r')[^"\\]*"\s*:\s*)'
            r'"(?:[^"\\]|\\.)*"'
        ),
        r'\1"' + REDACTED + '"',
    ),
    # key=value or key: value with a secret-like key.
    (
        re.compile(
            r"(?i)\b([A-Za-z0-9_.-]*(?:" + _SECRET_WORDS + r")[A-Za-z0-9_.-]*)"
            r"(\s*[=:]\s*)([^\s&,;\"']+)"
        ),
        r"\1\2" + REDACTED,
    ),
    # Well-known credential shapes. No word boundary in front: a key glued
    # to the text before it is still a key, and removing a little too much
    # is the safe mistake.
    (re.compile(r"sk-[A-Za-z0-9_-]{16,}"), REDACTED),
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), REDACTED),
    (re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}"), REDACTED),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"), REDACTED),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), REDACTED),
    (re.compile(r"xox[abposr]-[A-Za-z0-9-]{10,}"), REDACTED),
    (
        re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
        REDACTED,
    ),
)


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


def truncate_utf8(text: str, max_bytes: int) -> str:
    """The longest prefix of ``text`` whose UTF-8 encoding fits ``max_bytes``.

    A character is never split. A lone surrogate, which UTF-8 cannot
    encode, becomes ``?``.
    """
    if max_bytes < 0:
        raise ValueError("max_bytes must be >= 0")
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return encoded.decode("utf-8")
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def scrub_text(
    text: str,
    *,
    max_bytes: int | None = None,
    secrets: KnownSecrets | None = None,
) -> str:
    """Free text with credentials removed, then cut to ``max_bytes``.

    For text an exporter has consent to send, such as an error message, and
    only as a layer under its allowlist: patterns catch the common shapes
    of a credential, not every secret. The order matters: the exact values
    in ``secrets`` first, then the patterns, then the cut, so a cut never
    leaves part of a secret that a whole match would have removed.
    """
    if max_bytes is not None:
        text = text[: max_bytes + INPUT_MARGIN_CHARS]
    if secrets is not None:
        text = secrets.redact(text)
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text if max_bytes is None else truncate_utf8(text, max_bytes)


def is_forbidden_key_name(name: str) -> bool:
    """Whether a key's name says its value may be a credential.

    An exporter refuses to admit such a key from any outside source (an
    environment variable or a flag), even one an operator names
    explicitly, because the value cannot be checked.
    """
    lowered = name.lower()
    return any(word in lowered for word in SECRET_KEY_WORDS)
