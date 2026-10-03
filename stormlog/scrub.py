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
from collections.abc import Callable, Iterable, Iterator
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
Span = tuple[int, int]
Finder = Callable[[str], Iterator[Span]]

# Every pattern matches in linear time: each can start only where a run of
# its characters starts (a lookbehind for the same class), and a key's
# words are checked by is_forbidden_key_name on the whole token, never by
# alternation inside the pattern, which backtracks polynomially on runs
# such as "key.key.key.". Each finder reports the spans to redact in the
# text as given; scrub_text merges them with the known secrets' spans and
# replaces them all at once, so no replacement can hide another match.
_KEY_CHARS = "A-Za-z0-9_.-"
_JSON_STRING = r'"((?:[^"\\]|\\.)*)"'
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?"
    r"(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)"
)
_AUTHORIZATION = re.compile(r"(?i)\b(?:proxy-)?authorization\s*[:=]\s*([^\r\n]+)")
_BEARER = re.compile(r"(?i)\b(?:bearer|basic)\s+([A-Za-z0-9._~+/=-]{8,})")
_URL = re.compile(r"(?i)(?<![a-z0-9+.-])[a-z][a-z0-9+.-]*://([^\s\"'<>]*)")
_JSON_MEMBER = re.compile(_JSON_STRING + r"\s*:\s*" + _JSON_STRING)
_KEY_VALUE = re.compile(
    rf"(?<![{_KEY_CHARS}])([{_KEY_CHARS}]+)\s*[=:]\s*([^\s&,;\"']+)"
)
# Well-known credential shapes. No word boundary in front of most: a key
# glued to the text before it is still a key, and removing a little too
# much is the safe mistake. Each class covers the prefix that follows it,
# so a run of repeated prefixes is one match.
_SHAPES = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[abposr]-[A-Za-z0-9-]{10,}"),
    re.compile(
        r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
        r"\.[A-Za-z0-9_-]{8,}"
    ),
)


def _whole(pattern: re.Pattern[str]) -> Finder:
    return lambda text: (match.span() for match in pattern.finditer(text))


def _value(pattern: re.Pattern[str]) -> Finder:
    return lambda text: (match.span(1) for match in pattern.finditer(text))


def _url_spans(text: str) -> Iterator[Span]:
    """A URL's user information and query; its fragment is never sent."""
    for match in _URL.finditer(text):
        rest = match.group(1)
        base = match.start(1)
        authority_end = len(rest)
        for mark in "/?#":
            found = rest.find(mark)
            if found >= 0:
                authority_end = min(authority_end, found)
        at = rest.rfind("@", 0, authority_end)
        if at > 0:
            yield base, base + at
        query = rest.find("?")
        if query >= 0:
            fragment = rest.find("#", query)
            end = fragment if fragment >= 0 else len(rest)
            if end > query + 1:
                yield base + query + 1, base + end


def _json_member_spans(text: str) -> Iterator[Span]:
    for match in _JSON_MEMBER.finditer(text):
        if is_forbidden_key_name(match.group(1)):
            yield match.span(2)


def _key_value_spans(text: str) -> Iterator[Span]:
    for match in _KEY_VALUE.finditer(text):
        if is_forbidden_key_name(match.group(1)):
            yield match.span(2)


_FINDERS: tuple[Finder, ...] = (
    _whole(_PRIVATE_KEY),
    _value(_AUTHORIZATION),
    _value(_BEARER),
    _url_spans,
    _json_member_spans,
    _key_value_spans,
    *(_whole(shape) for shape in _SHAPES),
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
        self._forms = tuple(sorted(forms))

    def redact(self, text: str) -> str:
        """``text`` with every form of every registered value replaced.

        Every occurrence is found in the original text first, overlapping
        ones included, and the spans are merged before anything is
        replaced, so replacing one value can never uncover part of another.
        """
        return replace_spans(text, self.spans(text))

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Where each form of each value occurs in ``text``, as (start, end)."""
        found: list[tuple[int, int]] = []
        for form in self._forms:
            start = text.find(form)
            while start >= 0:
                found.append((start, start + len(form)))
                start = text.find(form, start + 1)
        return found

    def found_in(self, text: str) -> bool:
        """Whether any form of any registered value occurs in ``text``."""
        return any(form in text for form in self._forms)


def replace_spans(text: str, spans: Iterable[tuple[int, int]]) -> str:
    """``text`` with each span, overlapping or touching spans merged, redacted."""
    pieces: list[str] = []
    cursor = 0
    for start, end in _merged(spans):
        pieces.append(text[cursor:start])
        pieces.append(REDACTED)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _merged(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


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
    of a credential, not every secret. Every match, of the exact values in
    ``secrets`` and of the patterns, is found in the same text before any
    is replaced; the cut comes last, so it never leaves part of a secret
    that a whole match would have removed.
    """
    if max_bytes is not None:
        text = text[: max_bytes + INPUT_MARGIN_CHARS]
    spans = secrets.spans(text) if secrets is not None else []
    for finder in _FINDERS:
        spans.extend(finder(text))
    text = replace_spans(text, spans)
    return text if max_bytes is None else truncate_utf8(text, max_bytes)


def is_forbidden_key_name(name: str) -> bool:
    """Whether a key's name says its value may be a credential.

    An exporter refuses to admit such a key from any outside source (an
    environment variable or a flag), even one an operator names
    explicitly, because the value cannot be checked.
    """
    lowered = name.lower()
    return any(word in lowered for word in SECRET_KEY_WORDS)
