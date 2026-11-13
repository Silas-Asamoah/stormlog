"""The fixture artifacts themselves: what each holds, and that Stormlog
reads it as the rest of the Studio suite assumes."""

from __future__ import annotations

import contextlib
import json
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.events import InferenceRequestEvent
from tests.studio.artifacts import (
    CaseSpec,
    decoded_records,
    profile_artifact,
    profile_config,
    synthetic_artifact,
    with_execution_import,
)
from tests.studio.fake_openai import CasePolicy, fake_openai


def _lines(path: Path) -> list[dict[str, Any]]:
    """Every complete JSON line, as written (contexts not expanded)."""
    records = []
    for line in path.read_bytes().split(b"\n"):
        with contextlib.suppress(ValueError):
            records.append(json.loads(line))
    return records


def _client_requests(path: Path) -> list[dict[str, Any]]:
    return [
        record
        for record in _lines(path)
        if record.get("event_type") == "infer.request"
        and record.get("schema_version", 1) == 1
    ]


def _measured(path: Path) -> list[dict[str, Any]]:
    return [r for r in _client_requests(path) if r["phase"] == "measured"]


def test_the_execution_import_replaces_the_file_keeping_its_bytes_as_prefix(
    tmp_path: Path,
) -> None:
    path = profile_artifact(tmp_path, [CaseSpec(2, 8, 4, 6, fail_every=3)], warmup=0)
    before, inode = path.read_bytes(), path.stat().st_ino

    with_execution_import(path)

    assert path.stat().st_ino != inode
    assert path.read_bytes().startswith(before)
    kinds = Counter(r["event_type"] for r in decoded_records(path))
    assert kinds["infer.iteration"] == 8
    assert kinds["infer.membership"] == 8
    server_view = [
        r
        for r in decoded_records(path)
        if r["event_type"] == "infer.request" and "context" in r
    ]
    assert len(server_view) == 4


def test_synthetic_requests_equal_what_the_dataclass_writes(tmp_path: Path) -> None:
    path = synthetic_artifact(
        tmp_path, requests=40, gaps_per_request=5, fail_every=7, tokens="mixed"
    )
    names = {field.name for field in fields(InferenceRequestEvent)}

    for record in _measured(path):
        event = InferenceRequestEvent(**{k: v for k, v in record.items() if k in names})
        assert event.to_record() == record
        assert len(record["chunk_interarrival_ms"]) == (
            5 if record["status"] == "ok" else 0
        )


def test_a_seed_always_writes_the_same_requests(tmp_path: Path) -> None:
    def requests(directory: str, seed: int) -> list[dict[str, Any]]:
        (tmp_path / directory).mkdir()
        path = synthetic_artifact(
            tmp_path / directory, requests=50, gaps_per_request=3, seed=seed
        )
        return _measured(path)

    first, again, other = requests("a", 1), requests("b", 1), requests("c", 2)

    assert first == again
    assert first != other


def test_synthetic_requests_lie_inside_their_phase_window(tmp_path: Path) -> None:
    path = synthetic_artifact(tmp_path, requests=300, gaps_per_request=15)
    (window,) = [r for r in _lines(path) if r["event_type"] == "infer.phase_window"]
    requests = _measured(path)

    assert min(r["started_at_ns"] for r in requests) >= window["started_at_ns"]
    assert max(r["started_at_ns"] for r in requests) == window["window_ended_at_ns"]
    assert max(r["ended_at_ns"] for r in requests) == window["drained_at_ns"]


def test_the_fake_server_fails_every_nth_measured_request(tmp_path: Path) -> None:
    path = profile_artifact(tmp_path, [CaseSpec(3, 8, 4, 10, fail_every=3)], warmup=2)
    requests = _client_requests(path)

    assert Counter((r["phase"], r["status"]) for r in requests) == {
        ("warmup", "ok"): 2,
        ("measured", "ok"): 7,
        ("measured", "error"): 3,
    }


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


@pytest.mark.parametrize(
    "cases",
    [
        [CaseSpec(1, 8, 4, 3), CaseSpec(2, 8, 4, 4)],
        [CaseSpec(1, 8, 4, 3), CaseSpec(2, 8, 4, 3, stream=False)],
        [CaseSpec(1, 8, 4, 3), CaseSpec(2, 8, 16, 3)],
        [CaseSpec(1, 8, 4, 3), CaseSpec(1, 8, 4, 3)],
        [],
    ],
)
def test_one_run_refuses_cases_it_cannot_hold(
    cases: list[CaseSpec], tmp_path: Path
) -> None:
    with pytest.raises(ValueError):
        profile_config(
            cases, endpoint="http://x", output=tmp_path / "a", warmup=0, seed=0
        )
