"""How an MLX launcher's arguments are recorded."""

from __future__ import annotations

import pytest

from stormlog.infer.mlx_launchers import redacted_arguments
from stormlog.scrub import UNPARSABLE_URL


@pytest.mark.parametrize(
    "url",
    ["http://[::1:80/model", "http://127.0.0.1:99999/m", "http://h:eighty/m"],
)
@pytest.mark.parametrize("joined", [False, True])
def test_a_url_argument_urllib_cannot_parse_is_recorded_as_a_marker(
    url: str, joined: bool
) -> None:
    # A URL value of a known flag was redacted with urllib, which raised for
    # these: recording the launcher's arguments ended the capture.
    args = [f"--model={url}"] if joined else ["--model", url]
    recorded = redacted_arguments(["serve", *args, "--port", "8100"])
    model = f"--model={UNPARSABLE_URL}" if joined else UNPARSABLE_URL
    assert model in recorded
    assert recorded[-1] == "8100"


def test_a_url_argument_that_parses_is_still_redacted() -> None:
    recorded = redacted_arguments(["--model", "http://u:p@host:8000/m?token=v"])
    assert recorded == ["--model", "http://host:8000/m?<redacted>"]
