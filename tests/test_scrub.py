"""The shared scrubbing primitives in ``stormlog.scrub``."""

import pytest

from stormlog import scrub
from stormlog.infer import cache_state
from stormlog.scrub import redact_url


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
