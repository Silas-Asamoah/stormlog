"""The shared scrubbing primitives in ``stormlog.scrub``."""

import base64
import json
import urllib.parse

import pytest

from stormlog import scrub
from stormlog.infer import cache_state
from stormlog.scrub import KnownSecrets, redact_url, url_secrets


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


SECRET = 'sk-Abc/123+"quoted" päss'


@pytest.mark.parametrize(
    "form",
    [
        SECRET,
        urllib.parse.quote(SECRET, safe=""),
        urllib.parse.quote_plus(SECRET, safe=""),
        json.dumps(SECRET)[1:-1],
        json.dumps(SECRET, ensure_ascii=False)[1:-1],
        base64.b64encode(SECRET.encode()).decode(),
        base64.urlsafe_b64encode(SECRET.encode()).decode().rstrip("="),
    ],
)
def test_known_secrets_redact_every_travelling_form(form: str) -> None:
    secrets = KnownSecrets([SECRET])
    text = f"before {form} after"
    assert secrets.found_in(text)
    assert secrets.redact(text) == "before <redacted> after"


def test_known_secrets_replace_a_longer_value_whole() -> None:
    # One value contains another; replacing the shorter one first would
    # leave the rest of the longer one behind.
    secrets = KnownSecrets(["abcdefgh", "abcdefgh-and-more"])
    assert secrets.redact("x abcdefgh-and-more y") == "x <redacted> y"


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
        ("https://u:p%40ss-word@h/x", ["p@ss-word", "u:p@ss-word"]),
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


def test_short_url_values_are_not_used_for_redaction() -> None:
    secrets = KnownSecrets(url_secrets("https://h/x?api-key=abc12345&stream=true"))
    assert secrets.redact("stream=true key=abc12345") == "stream=true key=<redacted>"
