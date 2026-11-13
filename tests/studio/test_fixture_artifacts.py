"""The fixture artifacts themselves: what each holds, and that Stormlog
reads it as the rest of the Studio suite assumes."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import pytest

from tests.studio.fake_openai import CasePolicy, fake_openai


def _post(endpoint: str, request_id: str, body: dict[str, Any]) -> Any:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json", "X-Request-Id": request_id},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=10) as response:
        return response.read()


def test_the_fake_server_answers_by_case_and_phase() -> None:
    policies = {
        "c1_in8_out4": CasePolicy(all_fail=True),
        "c2_in8_out4": CasePolicy(usage=False),
    }
    with fake_openai(policies) as server:
        with pytest.raises(urllib.error.HTTPError):
            _post(server.endpoint, "stormlog-r-1-c1_in8_out4_warmup_0_0", {})
        quiet = json.loads(
            _post(server.endpoint, "stormlog-r-1-c2_in8_out4_measured_0_0", {})
        )
        normal = json.loads(
            _post(server.endpoint, "stormlog-r-1-c3_x_measured_0_0", {})
        )
        stream = _post(
            server.endpoint,
            "stormlog-r-1-c2_in8_out4_measured_0_1",
            {"stream": True, "stream_options": {"include_usage": True}},
        )

    assert "usage" not in quiet and "usage" in normal
    assert stream.endswith(b"data: [DONE]\n\n") and b"usage" not in stream
    assert server.state.failed == 1
    assert server.state.measured == {"c2_in8_out4": 2, "c3_x": 1}
