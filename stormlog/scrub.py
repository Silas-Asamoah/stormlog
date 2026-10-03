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
BACKSLASH = chr(92)
# Shorter values are not used for exact-value redaction: they would erase
# ordinary text such as case names and numbers, and a credential that short
# is not protected by redaction anyway.
MIN_SECRET_LENGTH = 8
# scrub_text cuts a long text this many characters past the requested length
# before matching, so a large body cannot make scrubbing slow. Only the
# first max_bytes characters can reach the output; the margin is lookahead,
# so a secret that starts inside the kept length is seen whole unless it is
# longer than the margin.
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
# A string never starts at an escaped quote: in valid JSON a backslash never
# comes just before an opening quote, and starting at each escaped quote of
# an unclosed string would rescan it from every one, quadratically.
_JSON_STRING = r'(?<!\\)"((?:[^"\\]|\\.)*)"'
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?"
    r"(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)"
)
_AUTHORIZATION = re.compile(r"(?i)\b(?:proxy-)?authorization\s*[:=]\s*([^\r\n]+)")
_BEARER = re.compile(r"(?i)\b(?:bearer|basic)\s+([A-Za-z0-9._~+/=-]{8,})")
_URL = re.compile(r"(?i)(?<![a-z0-9+.-])[a-z][a-z0-9+.-]*://([^\s\"'<>]*)")
# A member's value is a string, whose inside is redacted, or a bare value
# such as a number or true.
_JSON_MEMBER = re.compile(
    _JSON_STRING + r"\s*:\s*(?:" + _JSON_STRING + r'|([^\s,}\]"]+))'
)
# The value is a double- or single-quoted string, escapes included, or a
# bare word; for a quoted one, what is inside the quotes is redacted.
_KEY_VALUE = re.compile(
    rf"(?<![{_KEY_CHARS}])([{_KEY_CHARS}]+)\s*[=:]\s*"
    r"(?:" + _JSON_STRING + r"|'((?:[^'\\]|\\.)*)'|([^\s&,;\"']+))"
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


def _bearer_spans(text: str) -> Iterator[Span]:
    """A Bearer or Basic token: one with a digit, =, + or /, so prose is not."""
    for match in _BEARER.finditer(text):
        if any(char.isdigit() or char in "=+/" for char in match.group(1)):
            yield match.span(1)


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
        if is_forbidden_key_name(_json_unescaped(match.group(1))):
            group = 2 if match.group(2) is not None else 3
            if match.end(group) > match.start(group):
                yield match.span(group)


def _json_unescaped(key: str) -> str:
    """A JSON string's content with its escapes decoded, or as given."""
    try:
        decoded = json.loads(f'"{key}"')
    except ValueError:
        return key
    return decoded if isinstance(decoded, str) else key


def _key_value_spans(text: str) -> Iterator[Span]:
    for match in _KEY_VALUE.finditer(text):
        if is_forbidden_key_name(match.group(1)):
            group = next(g for g in (2, 3, 4) if match.group(g) is not None)
            if match.end(group) > match.start(group):
                yield match.span(group)


_FINDERS: tuple[Finder, ...] = (
    _whole(_PRIVATE_KEY),
    _value(_AUTHORIZATION),
    _bearer_spans,
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
    Each value is matched in every spelling it most often travels in: as
    given; percent-encoded (inside a URL) in either case of hex digit, with
    any characters left plain, and with ``+`` for a space; JSON-escaped
    (inside a JSON string), with or without unicode escapes in either case,
    surrogate pairs, an escaped ``/`` and the short escapes; and each
    character may be spelled differently from the next. Also base64,
    standard and URL-safe, padded or not, of the value on its own: an
    encoded token, or a Basic authorization header when the
    ``user:password`` pair is registered.
    """

    def __init__(self, values: Iterable[str | None] = ()) -> None:
        self._patterns: dict[str, re.Pattern[str]] = {}
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
        if value not in self._patterns:
            self._patterns[value] = _spellings(value)

    def redact(self, text: str) -> str:
        """``text`` with every form of every registered value replaced.

        Every occurrence is found in the original text first, overlapping
        ones included, and the spans are merged before anything is
        replaced, so replacing one value can never uncover part of another.
        """
        return replace_spans(text, self.spans(text))

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Where each form of each value occurs in ``text``, as (start, end)."""
        # Each pattern is a lookahead, so occurrences that overlap are all
        # found; the spans are merged when they are replaced.
        return [
            (match.start(), match.end(1))
            for pattern in self._patterns.values()
            for match in pattern.finditer(text)
        ]

    def found_in(self, text: str) -> bool:
        """Whether any spelling of any registered value occurs in ``text``."""
        return any(pattern.search(text) for pattern in self._patterns.values())


def replace_spans(
    text: str, spans: Iterable[tuple[int, int]], *, limit: int | None = None
) -> str:
    """``text`` with each span, overlapping or touching spans merged, redacted.

    With ``limit``, only ``text[:limit]`` is kept: a span that starts
    before it is redacted whole, and nothing after it is copied.
    """
    stop = len(text) if limit is None else min(limit, len(text))
    pieces: list[str] = []
    cursor = 0
    for start, end in _merged(spans):
        if start >= stop:
            break
        pieces.append(text[cursor:start])
        pieces.append(REDACTED)
        cursor = end
    if cursor < stop:
        pieces.append(text[cursor:stop])
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

    The user name, on its own whatever the password (it is often a token),
    the password, the ``user:password`` pair a Basic header would encode,
    decoded and also as written in the URL for a client that did not
    decode it, and every query value. A fragment is never sent to a
    server, so it is not read.
    """
    if not url:
        return []
    parts = urllib.parse.urlsplit(url)
    found: list[str] = []
    user = urllib.parse.unquote(parts.username or "")
    if user:
        found.append(user)
    if parts.password is not None:
        password = urllib.parse.unquote(parts.password)
        found.extend([password, f"{user}:{password}"])
        as_written = f"{parts.username or ''}:{parts.password}"
        if as_written != found[-1]:
            found.append(as_written)
    found.extend(
        value
        for _, value in urllib.parse.parse_qsl(parts.query, keep_blank_values=False)
    )
    return found


# JSON's short escapes, as the two characters that spell each.
_JSON_SHORT = {
    '"': BACKSLASH + '"',
    BACKSLASH: BACKSLASH + BACKSLASH,
    "/": BACKSLASH + "/",
    "\b": BACKSLASH + "b",
    "\f": BACKSLASH + "f",
    "\n": BACKSLASH + "n",
    "\r": BACKSLASH + "r",
    "\t": BACKSLASH + "t",
}


def _spellings(value: str) -> re.Pattern[str]:
    """A lookahead matching every spelling of ``value`` that KnownSecrets covers."""
    options = ["".join(_char_spellings(char) for char in value)]
    raw = value.encode("utf-8", "surrogatepass")
    for encoded in (base64.b64encode(raw), base64.urlsafe_b64encode(raw)):
        text = encoded.decode("ascii")
        # Padded first, so the longer spelling is the one matched.
        options.extend(
            re.escape(form) for form in dict.fromkeys((text, text.rstrip("=")))
        )
    return re.compile("(?=(" + "|".join(options) + "))")


def _char_spellings(char: str) -> str:
    """One character as given, percent-encoded or JSON-escaped."""
    options = [re.escape(char)]
    options.append(
        "".join("%" + _hex(byte, 2) for byte in char.encode("utf-8", "surrogatepass"))
    )
    if char == " ":
        options.append(re.escape("+"))
    if char in _JSON_SHORT:
        options.append(re.escape(_JSON_SHORT[char]))
    code = ord(char)
    if code > 0xFFFF:
        code -= 0x10000
        units = [0xD800 + (code >> 10), 0xDC00 + (code & 0x3FF)]
    else:
        units = [code]
    escape = re.escape(BACKSLASH) + "u"
    options.append("".join(escape + _hex(unit, 4) for unit in units))
    return "(?:" + "|".join(options) + ")"


def _hex(number: int, width: int) -> str:
    """``number`` in hex, each letter matching either case."""
    return "".join(
        f"[{digit.lower()}{digit.upper()}]" if digit.isalpha() else digit
        for digit in f"{number:0{width}X}"
    )


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
    # What UTF-8 cannot encode becomes "?" now, one character for one, so
    # the text matched is the text sent: replacing it later could complete
    # a registered value after matching.
    text = text.encode("utf-8", "replace").decode("utf-8")
    spans = secrets.spans(text) if secrets is not None else []
    for finder in _FINDERS:
        spans.extend(finder(text))
    # The margin is lookahead only: it lets a match that starts within the
    # kept length be seen whole. Text from it never reaches the output,
    # even when redactions before it leave room, because the input bound
    # may have cut a secret there before it could match.
    text = replace_spans(text, spans, limit=max_bytes)
    if max_bytes is None:
        return text
    cut = truncate_utf8(text, max_bytes)
    # A marker cut short is dropped rather than left as "<reda".
    start = cut.rfind("<", max(0, len(cut) - len(REDACTED) + 1))
    if start >= 0 and text.startswith(REDACTED, start):
        cut = cut[:start]
    return cut


def is_forbidden_key_name(name: str) -> bool:
    """Whether a key's name says its value may be a credential.

    An exporter refuses to admit such a key from any outside source (an
    environment variable or a flag), even one an operator names
    explicitly, because the value cannot be checked.
    """
    lowered = name.lower()
    return any(word in lowered for word in SECRET_KEY_WORDS)
