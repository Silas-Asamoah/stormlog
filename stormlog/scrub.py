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
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?"
    r"(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)"
)
_AUTHORIZATION = re.compile(r"(?i)\b(?:proxy-)?authorization\s*[:=]\s*([^\r\n]+)")
_BEARER = re.compile(r"(?i)\b(?:bearer|basic)\s+([A-Za-z0-9._~+/=-]{8,})")
_SCHEME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+.-"
)
_URL_END = re.compile(r"[\s\"'<>]")
_AUTHORITY_END = re.compile(r"[/\s]")
_QUESTION = re.compile(r"\?")
_HASH = re.compile("#")
# A member's key, in double quotes (JSON) or single quotes (a Python dict
# repr), and its separator; the value is read separately, like a pair's. A
# key never starts at an escaped quote: valid JSON never has a backslash
# just before an opening quote, and starting at each escaped quote of an
# unclosed string would rescan it from every one, quadratically.
_QUOTED_KEY = re.compile(
    r"""(?<!\\)(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)'|"""
    r"""\\"([^"\\]*)\\")\s*:\s*"""
)
# A key and its separator, without the value: a key that is not
# secret-like must not consume the text after it, which can hold the next
# pair ("error: password=..."). The value is read separately, and only
# for a secret-like key.
_KEY_SEPARATOR = re.compile(rf"(?<![{_KEY_CHARS}])([{_KEY_CHARS}]+)\s*[=:]\s*")
# A quoted value ends at its closing quote, or at the end of the text when
# it has none, as in a truncated body: what follows its opening quote is
# still its value.
_DOUBLE_QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)(?:"|\\?\Z)')
_SINGLE_QUOTED = re.compile(r"'((?:[^'\\]|\\.)*)(?:'|\\?\Z)")
# A value in escaped quotes, as JSON inside a JSON string spells it: it ends
# at the escaped closing quote, at the enclosing string's own quote, or at
# the end of the text.
_ESCAPED_QUOTED = re.compile(r'\\"((?:[^"\\]|\\[^"])*)(?:\\"|"|\\?\Z)')
_BARE_VALUE = re.compile(r"[^\s&,;\"'(){}\[\]<>]+")
# Well-known credential shapes. No word boundary in front: a key glued to
# the text before it is still a key, and removing a little too much is the
# safe mistake. Each is linear: where a prefix's class covers the prefix
# itself (sk-, github_pat_, xox), a run of repeated prefixes is one match,
# and where it does not (hf_, gh*_), a failed attempt stops at the next _.
_SHAPES = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[abposr]-[A-Za-z0-9-]{10,}"),
)
_JWT_SEGMENT = re.compile(r"[A-Za-z0-9_-]*")


def _whole(pattern: re.Pattern[str]) -> Finder:
    return lambda text: (match.span() for match in pattern.finditer(text))


def _value(pattern: re.Pattern[str]) -> Finder:
    return lambda text: (match.span(1) for match in pattern.finditer(text))


def _bearer_spans(text: str) -> Iterator[Span]:
    """A Bearer or Basic token: anything but one all-lowercase word.

    Prose such as "the basic parameters" is left alone. A random token of
    20 letters and digits is all lower-case letters with a probability of
    about 3e-8, so real tokens are not missed.
    """
    for match in _BEARER.finditer(text):
        token = match.group(1)
        if not (token.isascii() and token.isalpha() and token.islower()):
            yield match.span(1)


def _url_spans(text: str) -> Iterator[Span]:
    """A URL's user information and query; its fragment is never sent.

    Each URL is found from its "://", so a scheme glued to the text before
    it is still found, and each run of scheme characters is read once. The
    user information runs to the last @ before the first / or whitespace,
    whatever it holds: a quote, <, >, # and ? can all appear in a password.
    """
    url_end = _NextMatch(text, _URL_END)
    question = _NextMatch(text, _QUESTION)
    hash_mark = _NextMatch(text, _HASH)
    for match in re.finditer("://", text):
        if not _has_scheme(text, match.start()):
            continue
        after = match.end()
        stop = _AUTHORITY_END.search(text, after)
        at = text.rfind("@", after, stop.start() if stop else len(text))
        if at > after:
            yield after, at
        host = max(at + 1, after)
        end = url_end.at_or_after(host)
        query = question.at_or_after(host)
        if query < end:
            fragment = min(hash_mark.at_or_after(query), end)
            if fragment > query + 1:
                yield query + 1, fragment


def _has_scheme(text: str, end: int) -> bool:
    """Whether the scheme characters before ``end`` include a letter to start one."""
    start = end
    while start > 0 and text[start - 1] in _SCHEME_CHARS:
        start -= 1
    return any(char.isalpha() for char in text[start:end])


class _NextMatch:
    """The next match of a pattern at or after a position, for rising positions.

    Several URLs in one long token would each search to the token's end;
    remembering the last match found keeps the total linear.
    """

    def __init__(self, text: str, pattern: re.Pattern[str]) -> None:
        self.text = text
        self.pattern = pattern
        self.found = -1

    def at_or_after(self, position: int) -> int:
        """The match's start, or the text's length when there is none."""
        if position > self.found:
            match = self.pattern.search(self.text, position)
            self.found = match.start() if match else len(self.text)
        return self.found


def _member_spans(text: str) -> Iterator[Span]:
    """The value of each member whose key, escapes decoded, is secret-like."""
    values = _Values(text)
    for match in _QUOTED_KEY.finditer(text):
        key = next(group for group in match.groups() if group is not None)
        if is_forbidden_key_name(_json_unescaped(key)):
            span = values.span(match.start(), match.end())
            if span is not None:
                yield span


def _json_unescaped(key: str) -> str:
    """A JSON string's content with its escapes decoded, or as given."""
    try:
        decoded = json.loads(f'"{key}"')
    except ValueError:
        return key
    return decoded if isinstance(decoded, str) else key


def _key_value_spans(text: str) -> Iterator[Span]:
    values = _Values(text)
    for match in _KEY_SEPARATOR.finditer(text):
        if is_forbidden_key_name(match.group(1)):
            span = values.span(match.start(), match.end())
            if span is not None:
                yield span


class _Values:
    """Read the value after a secret-like key, in linear time overall.

    A bare value ends where its run of value characters ends; in a chain
    such as "password=password=...", every key's value is a suffix of one
    run and all of them end where it ends, so that end is computed once
    per run. An array or object is redacted whole, and a key inside one
    already redacted is skipped, so nested values are not rescanned.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.run_start = self.run_end = -1
        self.covered = -1

    def span(self, key_start: int, position: int) -> Span | None:
        """The span to redact for a key at ``key_start`` whose value starts at ``position``."""
        if key_start < self.covered:
            return None
        text = self.text
        if text[position : position + 1] in ("[", "{"):
            end = _bracket_end(text, position)
            self.covered = end
            return position, end
        for quoted in (_DOUBLE_QUOTED, _SINGLE_QUOTED, _ESCAPED_QUOTED):
            match = quoted.match(text, position)
            if match:
                return match.span(1) if match.end(1) > match.start(1) else None
        end = self._bare_end(position)
        return (position, end) if end > position else None

    def _bare_end(self, position: int) -> int:
        if not self.run_start <= position < self.run_end:
            match = _BARE_VALUE.match(self.text, position)
            self.run_start = position
            self.run_end = match.end() if match else position
        return self.run_end


def _bracket_end(text: str, start: int) -> int:
    """Where the array or object opening at ``start`` closes, or the text's end.

    Brackets inside quoted strings, in either quote, do not count.
    """
    depth = 0
    quote = ""
    index = start
    while index < len(text):
        char = text[index]
        if quote:
            if char == "\\":
                index += 1
            elif char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return len(text)


def _jwt_spans(text: str) -> Iterator[Span]:
    """A three-part JWT, glued to the word before it or not.

    An "eyJ" inside the first segment of one that failed would fail the
    same way, so the search resumes after that segment: linear even on a
    run of "eyJ".
    """
    resume = 0
    for match in re.finditer("eyJ", text):
        start = match.start()
        if start < resume:
            continue
        end = _jwt_end(text, start + 3)
        if end is None:
            resume = _segment_end(text, start + 3)
            continue
        resume = end
        yield start, end


def _jwt_end(text: str, position: int) -> int | None:
    """Where a JWT whose first segment starts at ``position`` ends, if it is one."""
    for part in range(3):
        if part:
            if text[position : position + 1] != ".":
                return None
            position += 1
        end = _segment_end(text, position)
        if end - position < 8:
            return None
        position = end
    return position


def _segment_end(text: str, position: int) -> int:
    segment = _JWT_SEGMENT.match(text, position)
    return segment.end() if segment else position


_FINDERS: tuple[Finder, ...] = (
    _whole(_PRIVATE_KEY),
    _value(_AUTHORIZATION),
    _bearer_spans,
    _url_spans,
    _member_spans,
    _key_value_spans,
    *(_whole(shape) for shape in _SHAPES),
    _jwt_spans,
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
