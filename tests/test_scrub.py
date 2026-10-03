"""The shared scrubbing primitives in ``stormlog.scrub``."""

import base64
import json
import subprocess
import sys
from collections.abc import Iterator

import pytest

from stormlog import scrub
from stormlog.infer import cache_state
from stormlog.scrub import (
    SECRET_KEY_WORDS,
    KnownSecrets,
    is_forbidden_key_name,
    redact_url,
    scrub_text,
    truncate_utf8,
    url_secrets,
)


@pytest.mark.parametrize(
    ("url", "recorded"),
    [
        ("http://host:8000/reset_prefix_cache", "http://host:8000/reset_prefix_cache"),
        ("https://u:p@host/flush_cache?k=v", "https://host/flush_cache?<redacted>"),
        ("http://[::1]:8000/reset", "http://[::1]:8000/reset"),
        ("http://HOST:8000/v1", "http://host:8000/v1"),
        (None, None),
    ],
)
def test_redact_url_keeps_only_scheme_host_port_and_path(
    url: str | None, recorded: str | None
) -> None:
    assert redact_url(url) == recorded


def test_cache_state_still_exports_the_same_redact_url() -> None:
    assert cache_state.redact_url is scrub.redact_url


@pytest.mark.parametrize(
    ("url", "origin"),
    [
        ("https://u:p@host:8443/v1/sk-secret/chat?k=v", "https://host:8443"),
        ("http://[::1]:4318/v1/traces", "http://[::1]:4318"),
        ("http://host", "http://host"),
    ],
)
def test_redact_url_origin_only_drops_the_path_as_well(url: str, origin: str) -> None:
    assert redact_url(url, origin_only=True) == origin


def test_redact_url_origin_only_keeps_none_as_none() -> None:
    assert redact_url(None, origin_only=True) is None


# Each spelling below is written out by hand, not produced by the encoders
# the implementation might use, so a test cannot agree with a bug.
# A backslash, built here so no escape in this file can be mistaken for one.
BACKSLASH = chr(92)
U = BACKSLASH + "u"  # a JSON unicode escape
Q = BACKSLASH + '"'  # a JSON-escaped quote
S = BACKSLASH + "/"  # a JSON-escaped slash
SECRET = 'sk-Abc/123+"quoted" päss'
SPELLINGS = (
    'sk-Abc/123+"quoted" päss',
    # Percent-encoded: everything, with + for space, keeping "/", lowercase.
    "sk-Abc%2F123%2B%22quoted%22%20p%C3%A4ss",
    "sk-Abc%2F123%2B%22quoted%22+p%C3%A4ss",
    "sk-Abc/123%2B%22quoted%22%20p%C3%A4ss",
    "sk-Abc%2f123%2b%22quoted%22%20p%c3%a4ss",
    # A mix of encoded and plain characters.
    'sk-Abc%2f123+%22quoted" p%C3%A4ss',
    # JSON: ASCII-only, UTF-8, upper-case hex, an escaped slash, quotes as
    # unicode escapes, an escaped first letter.
    f"sk-Abc/123+{Q}quoted{Q} p{U}00e4ss",
    f"sk-Abc/123+{Q}quoted{Q} päss",
    f"sk-Abc/123+{Q}quoted{Q} p{U}00E4ss",
    f"sk-Abc{S}123+{Q}quoted{Q} p{U}00e4ss",
    f"sk-Abc/123+{U}0022quoted{U}0022 p{U}00e4ss",
    f"{U}0073k-Abc/123+{Q}quoted{Q} p{U}00e4ss",
    # Base64 of the UTF-8 bytes, padded and not.
    "c2stQWJjLzEyMysicXVvdGVkIiBww6Rzcw==",
    "c2stQWJjLzEyMysicXVvdGVkIiBww6Rzcw",
)


@pytest.mark.parametrize("form", SPELLINGS)
def test_known_secrets_redact_every_spelling_of_a_value(form: str) -> None:
    secrets = KnownSecrets([SECRET])
    text = f"before {form} after"
    assert secrets.found_in(text)
    assert secrets.redact(text) == "before <redacted> after"


@pytest.mark.parametrize(
    ("value", "form"),
    [
        ("abc/def+ghi", "abc%2fdef%2Bghi"),
        ("abc/def+ghi", f"abc{S}def+ghi"),
        ("abc/def+ghi", f"{U}0061bc/def+ghi"),
        # A character outside the BMP: JSON writes a surrogate pair.
        ("password\U0001f600", f"password{U}D83D{U}DE00"),
        ("password\U0001f600", f"password{U}d83d{U}de00"),
        ("password\U0001f600", "password%F0%9F%98%80"),
        ("password\U0001f600", "password%f0%9f%98%80"),
        # Base64's two alphabets differ for this value: + and / against - and _.
        ("z?>~?>~token-1", "ej8+fj8+fnRva2VuLTE="),
        ("z?>~?>~token-1", "ej8-fj8-fnRva2VuLTE="),
        ("z?>~?>~token-1", "ej8-fj8-fnRva2VuLTE"),
    ],
)
def test_known_secrets_redact_equivalent_escapes(value: str, form: str) -> None:
    secrets = KnownSecrets([value])
    assert secrets.redact(f"[{form}]") == "[<redacted>]"


def test_known_secrets_replace_a_longer_value_whole() -> None:
    # One value contains another; replacing the shorter one first would
    # leave the rest of the longer one behind.
    secrets = KnownSecrets(["abcdefgh", "abcdefgh-and-more"])
    assert secrets.redact("x abcdefgh-and-more y") == "x <redacted> y"


@pytest.mark.parametrize(
    ("values", "text", "redacted"),
    [
        # Two values that cross: replacing one must not uncover the other.
        (["abcdefgh", "efghijkl"], "abcdefghijkl", "<redacted>"),
        (["abcdefgh1234", "1234xyzwvuts"], "abcdefgh1234xyzwvuts", "<redacted>"),
        # A value overlapping itself.
        (["aaaaaaaa"], "x aaaaaaaaaa y", "x <redacted> y"),
        # Two values side by side stay two redactions.
        (["abcdefgh", "ijklmnop"], "abcdefgh ijklmnop", "<redacted> <redacted>"),
    ],
)
def test_known_secrets_merge_overlapping_matches(
    values: list[str], text: str, redacted: str
) -> None:
    assert KnownSecrets(values).redact(text) == redacted


def test_known_secrets_ignore_empty_and_skip_short_values() -> None:
    secrets = KnownSecrets([None, "", "short", "longer-secret"])
    assert secrets.skipped_short == 1
    assert secrets.redact("short longer-secret") == "short <redacted>"


def test_a_basic_header_is_redacted_through_its_user_password_pair() -> None:
    url = "https://alice:s3cret-pass@collector:4318/v1/traces"
    secrets = KnownSecrets(url_secrets(url))
    header = "Basic " + base64.b64encode(b"alice:s3cret-pass").decode()
    assert secrets.redact(header) == "Basic <redacted>"


@pytest.mark.parametrize(
    ("url", "found"),
    [
        (
            "https://u:p%40ss-word@h/x",
            ["u", "p@ss-word", "u:p@ss-word", "u:p%40ss-word"],
        ),
        # A user name is registered on its own, whatever the password.
        (
            "https://opaque-credential-value:@h/x",
            ["opaque-credential-value", "", "opaque-credential-value:"],
        ),
        (
            "https://opaque-credential-value:x@h/x",
            ["opaque-credential-value", "x", "opaque-credential-value:x"],
        ),
        ("https://ghp_tokenvalue1234@h/x", ["ghp_tokenvalue1234"]),
        ("https://h/x?api-key=abc12345&stream=true", ["abc12345", "true"]),
        ("https://h/x", []),
        (None, []),
    ],
)
def test_url_secrets_lists_the_values_that_may_be_credentials(
    url: str | None, found: list[str]
) -> None:
    assert url_secrets(url) == found


def test_a_user_name_with_an_empty_password_is_still_redacted() -> None:
    secrets = KnownSecrets(url_secrets("https://opaque-credential-value:@host/x"))
    assert secrets.redact("echo opaque-credential-value") == "echo <redacted>"


def test_a_basic_header_built_from_the_userinfo_as_written_is_redacted() -> None:
    # A client that did not decode the URL's %40 before building the header.
    secrets = KnownSecrets(url_secrets("https://alice:p%40ss-w0rd@h/x"))
    header = "Basic YWxpY2U6cCU0MHNzLXcwcmQ="
    assert secrets.redact(header) == "Basic <redacted>"


def test_short_url_values_are_not_used_for_redaction() -> None:
    secrets = KnownSecrets(url_secrets("https://h/x?api-key=abc12345&stream=true"))
    assert secrets.redact("stream=true key=abc12345") == "stream=true key=<redacted>"


@pytest.mark.parametrize(
    ("text", "max_bytes", "kept"),
    [
        ("hello", 10, "hello"),
        ("hello", 5, "hello"),
        ("hello", 3, "hel"),
        ("héllo", 2, "h"),  # é is two bytes; half of it is never kept
        ("héllo", 3, "hé"),
        ("日本", 4, "日"),
        ("x", 0, ""),
        ("a\ud800b", 10, "a?b"),
    ],
)
def test_truncate_utf8_never_splits_a_character(
    text: str, max_bytes: int, kept: str
) -> None:
    assert truncate_utf8(text, max_bytes) == kept


def test_truncate_utf8_refuses_a_negative_length() -> None:
    with pytest.raises(ValueError):
        truncate_utf8("x", -1)


@pytest.mark.parametrize(
    ("text", "leaked"),
    [
        ("Authorization: Bearer abcdefgh12345678", "abcdefgh12345678"),
        ("proxy-authorization=Basic dXNlcjpwYXNzd29yZA==", "dXNlcjpwYXNzd29yZA"),
        ("sent with bearer opaque.token-value_123", "opaque.token-value_123"),
        ("see https://alice:hunter22@host/x for details", "hunter22"),
        ("GET https://host/v1?api_key=QQQ-opaque&x=1 failed", "QQQ-opaque"),
        ('{"api_key": "opaque\\"quoted-value"}', "opaque"),
        ('{"X-Auth-Token":"zzz-opaque"}', "zzz-opaque"),
        ("client_secret=hidden-value-1 next", "hidden-value-1"),
        ("password: correct horse", "correct"),
        ("key sk-proj-abcdefghijklmnop1234 used", "abcdefghijklmnop1234"),
        ("hf_abcdefghijklmnopqrstuvwx in env", "hf_abcdefghijklmnopqrstuvwx"),
        ("aws AKIAABCDEFGHIJKLMNOP id", "AKIAABCDEFGHIJKLMNOP"),
        ("ghp_" + "a" * 36, "a" * 36),
        ("github_pat_" + "b" * 30, "b" * 30),
        ("slack xoxb-1234567890-abcdef", "1234567890-abcdef"),
        (
            "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJl here",
            "eyJzdWIiOiIxMjM0In0",
        ),
        (
            "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBg\n-----END PRIVATE KEY-----",
            "MIIEvQIBADANBg",
        ),
        ("-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQ cut here", "MIIEpAIBAAKCAQ"),
    ],
)
def test_scrub_text_removes_common_credential_shapes(text: str, leaked: str) -> None:
    scrubbed = scrub_text(text)
    assert leaked not in scrubbed
    assert "<redacted>" in scrubbed


@pytest.mark.parametrize(
    ("text", "scrubbed"),
    [
        ('password="opaque-credential-value"', 'password="<redacted>"'),
        ("password='opaque value with spaces' next", "password='<redacted>' next"),
        (
            'client_secret = "a' + chr(92) + '"b-c" end',
            'client_secret = "<redacted>" end',
        ),
        ('max_tokens="128"', 'max_tokens="<redacted>"'),
        ('model="Qwen/Qwen2.5-7B"', 'model="Qwen/Qwen2.5-7B"'),
    ],
)
def test_scrub_text_redacts_quoted_values_of_secret_keys(
    text: str, scrubbed: str
) -> None:
    assert scrub_text(text) == scrubbed


def test_scrub_text_leaves_ordinary_text_alone() -> None:
    text = "This model's maximum context length is 32768 tokens; max_tokens 128."
    assert scrub_text(text) == text


def test_scrub_text_removes_a_known_secret_no_pattern_recognises() -> None:
    secrets = KnownSecrets(["opaque-value-without-shape"])
    scrubbed = scrub_text("echo: opaque-value-without-shape", secrets=secrets)
    assert scrubbed == "echo: <redacted>"


@pytest.mark.parametrize(
    ("text", "secrets", "scrubbed"),
    [
        (
            "https://u:p@host/x?k=opaque-credential-value",
            [],
            "https://<redacted>@host/x?<redacted>",
        ),
        # A known secret in the path must not hide the query from the URL.
        (
            "https://host/abcdefgh?x=opaque-value",
            ["abcdefgh"],
            "https://host/<redacted>?<redacted>",
        ),
        (
            "see https://host/abcdefgh/v1?token=opaque#frag now",
            ["abcdefgh"],
            "see https://host/<redacted>/v1?<redacted>#frag now",
        ),
        # A known secret inside a pattern's value is one redaction.
        ("token=abc-opaque-12345xyz end", ["opaque-12345"], "token=<redacted> end"),
    ],
)
def test_scrub_text_finds_every_span_before_replacing_any(
    text: str, secrets: list[str], scrubbed: str
) -> None:
    assert scrub_text(text, secrets=KnownSecrets(secrets)) == scrubbed


def test_scrub_text_cuts_after_redacting_so_no_fragment_is_left() -> None:
    key = "sk-" + "k" * 30
    text = "x" * 95 + key + " tail"
    # The marker does not fit whole, so it is dropped, not left as "<reda".
    assert scrub_text(text, max_bytes=100) == "x" * 95
    assert scrub_text(text, max_bytes=105) == "x" * 95 + "<redacted>"


@pytest.mark.parametrize(
    ("tail", "secret"),
    [
        ("opaque-credential-value", "opaque-credential-value"),
        ("opaque-value-without-shape-1234567890 tail", "opaque-value-without-shape"),
        ("hf_" + "Q" * 40, None),
    ],
)
def test_text_past_the_kept_length_never_reaches_the_output(
    tail: str, secret: str | None
) -> None:
    # A long header line shrinks to a few characters when redacted. The
    # text after it was beyond the kept length in the input, and must stay
    # out of the output instead of moving up into the room the redaction
    # made: there, the input bound may have cut it before it could match.
    line = "Authorization: " + "a" * (100 + scrub.INPUT_MARGIN_CHARS - 50)
    text = line + "\n" + tail
    secrets = KnownSecrets([secret] if secret else [])
    assert (
        scrub_text(text, max_bytes=100, secrets=secrets) == "Authorization: <redacted>"
    )
    assert (
        scrub_text(text, max_bytes=64, secrets=secrets) == "Authorization: <redacted>"
    )


def test_scrub_text_bounds_its_input_before_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int] = []

    def recording(text: str) -> Iterator[tuple[int, int]]:
        seen.append(len(text))
        return iter(())

    monkeypatch.setattr(scrub, "_FINDERS", (recording,))
    assert scrub_text("a" * 10_000_000, max_bytes=16) == "a" * 16
    assert seen == [16 + scrub.INPUT_MARGIN_CHARS]


@pytest.mark.parametrize(
    "name",
    [
        "db.password",
        "api.key",
        "API_KEY",
        "x-apikey",
        "service.Api-Token",
        "client_secret",
        "Authorization",
        "aws.credentials",
        "session.cookie",
        "user_passwd",
        "pwd",
        "private_key",
        "request.signature",
        "bearer",
        "sessionid",
    ],
)
def test_is_forbidden_key_name_catches_credential_like_names(name: str) -> None:
    assert is_forbidden_key_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "service.name",
        "service.instance.id",
        "deployment.environment.name",
        "host.name",
        "k8s.pod.uid",
        "cloud.availability_zone",
        "container.id",
        "os.type",
    ],
)
def test_is_forbidden_key_name_admits_ordinary_resource_names(name: str) -> None:
    assert not is_forbidden_key_name(name)


def test_free_text_patterns_use_the_same_words_as_the_key_check() -> None:
    for word in SECRET_KEY_WORDS:
        assert scrub_text(f"x_{word}_y=opaque-value") == f"x_{word}_y=<redacted>"


# Inputs that made the earlier patterns backtrack polynomially (Astra F1,
# Fable P1): a long run of key-like tokens, scheme-like runs, many "://",
# a quote followed by key words, runs of "eyJ", many BEGIN lines.
ADVERSARIAL = (
    ('"key." * 1300', None),
    ('"key." * 1300', 1024),
    ('"key-" * 3000', 16),
    ('"key-" * 13000', None),
    ('"auth_" * 10000', None),
    ('"a." * 26000', None),
    ('"a-" * 26000', None),
    ('"x://" * 13000', None),
    ('"http://" * 7400', None),
    ("'\"' + 'key' * 17000", None),
    ("'\"' + ('key' + 'a' * 10) * 4000", None),
    ('"eyJ" * 17000', None),
    ('"eyJ" + "a" * 52000', None),
    ('"authorization " * 3700', None),
    ('"-----BEGIN PRIVATE KEY-----\\n" * 1800', None),
    # A string that never closes because every later quote is escaped.
    ("'\"' + ('a' + chr(92) + '\"') * 15000", None),
    ("'password=\"' * 5000", None),
    ('"password=\'" * 5000', None),
)


def test_scrub_text_stays_linear_on_adversarial_input() -> None:
    # In a child process, so a regression fails on a timeout instead of
    # hanging the suite.
    code = (
        "import json, sys, time\n"
        "from stormlog.scrub import scrub_text\n"
        f"cases = {ADVERSARIAL!r}\n"
        "worst = 0.0\n"
        "for expression, max_bytes in cases:\n"
        "    text = eval(expression)\n"
        "    started = time.perf_counter()\n"
        "    scrub_text(text, max_bytes=max_bytes)\n"
        "    worst = max(worst, time.perf_counter() - started)\n"
        "print(json.dumps(worst))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
    # Each input is up to 52,000 characters; linear matching takes milliseconds.
    assert json.loads(result.stdout) < 1.0
