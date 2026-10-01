"""The workload record: what traffic a run sent, so it can be repeated."""

import contextlib
import io
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.analysis import format_analysis_text
from stormlog.infer.cli import main as infer_main
from stormlog.infer.openai_client import (
    OpenAIChatCompletionsClient,
    validate_extra_body,
)
from stormlog.infer.tokens import TiktokenCounter, TransformersTokenCounter
from stormlog.infer.workload import tokenizer_identity
from tests.infer_workload_helpers import run_profile_with_fake_client


def _workload(tmp_path: Path, **changes: Any) -> dict[str, Any]:
    run_profile_with_fake_client(tmp_path, latency_seconds=0.0, **changes)
    for line in (tmp_path / "infer.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("event_type") == "infer.workload":
            return dict(record)
    raise AssertionError("no infer.workload record")


def test_the_digest_names_the_traffic_not_the_target(tmp_path: Path) -> None:
    base = _workload(tmp_path / "a", request_count=2)
    other_target = _workload(
        tmp_path / "b",
        request_count=2,
        endpoint="http://10.0.0.2:9000/v1/chat/completions",
        model="another-model",
    )
    assert base["workload_digest"] == other_target["workload_digest"]
    assert len(base["workload_digest"]) == 64
    assert base["generator"] == {"name": "stormlog.synthetic", "version": 2}
    assert base["chat_template"] == {
        "applied_by": "server",
        "client_template_digest": None,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"seed": 1},
        {"prompt_mode": "unique"},
        {"input_tokens": (16,)},
        {"warmup_requests": 1},
        {"extra_body": {"temperature": 0}},
        {"arrival_mode": "poisson", "rates": (2.0,)},
        {"cache_state": "cold"},
    ],
)
def test_the_digest_changes_with_anything_that_shapes_the_requests(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    base = _workload(tmp_path / "base", request_count=2)
    changed = _workload(tmp_path / "changed", request_count=2, **changes)
    assert changed["workload_digest"] != base["workload_digest"]


def test_the_record_lists_settings_and_never_the_api_key(tmp_path: Path) -> None:
    record = _workload(
        tmp_path,
        request_count=2,
        api_key="sk-secret-value",
        extra_body={"temperature": 0, "ignore_eos": True},
        warmup_requests=3,
        prompt_mode="unique",
    )
    assert "sk-secret-value" not in (tmp_path / "infer.jsonl").read_text()
    assert record["decoding"]["extra_body"] == {"temperature": 0, "ignore_eos": True}
    assert record["decoding"]["other_settings"] == "server defaults"
    assert record["warmup"] == {"requests": 3, "prompts": "separate"}
    assert record["tokenizer"]["source"] == "estimated"
    assert [case["case_id"] for case in record["cases"]] == ["c1_in8_out4"]
    assert record["measurement"]["request_count"] == 2


def test_the_report_carries_the_workload(tmp_path: Path) -> None:
    _requests, report, _client = run_profile_with_fake_client(
        tmp_path, latency_seconds=0.0, request_count=1, seed=7
    )
    workload = report["workload"]
    assert workload["seed"] == 7 and workload["prompts"] == {"mode": "repeat"}
    digest = workload["workload_digest"][:12]
    assert f"Workload: {digest}, seed 7, prompts repeat" in format_analysis_text(report)


class _Response:
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def test_extra_fields_reach_the_request_body(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, Any]] = []

    def urlopen(request: Any, timeout: float) -> _Response:
        sent.append(json.loads(request.data))
        return _Response({"choices": [{"message": {"content": "hi"}}]})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = OpenAIChatCompletionsClient(
        endpoint="http://127.0.0.1:1/v1/chat/completions",
        model="m",
        timeout_seconds=1,
        extra_body={"temperature": 0, "ignore_eos": True},
    )
    client.complete(
        prompt="p", output_tokens=4, stream=False, stream_include_usage=False
    )
    assert sent[0]["temperature"] == 0 and sent[0]["ignore_eos"] is True
    assert (sent[0]["model"], sent[0]["max_tokens"]) == ("m", 4)


@pytest.mark.parametrize("field", ["model", "messages", "stream", "max_tokens"])
def test_extra_fields_cannot_replace_what_stormlog_sets(field: str) -> None:
    with pytest.raises(ValueError, match=f"cannot set {field}"):
        validate_extra_body({field: 1}, "max_tokens")
    # The other output cap field is not one Stormlog sets for this run.
    assert validate_extra_body({"max_completion_tokens": 1}, "max_tokens") == {
        "max_completion_tokens": 1
    }


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("{temperature: 0}", "--extra-body is not valid JSON"),
        ("[1, 2]", "--extra-body must be a JSON object"),
        ('{"stream": false}', "cannot set stream"),
    ],
)
def test_cli_rejects_bad_extra_bodies_before_writing(
    tmp_path: Path, raw: str, message: str
) -> None:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        code = infer_main(
            [
                "profile",
                "--endpoint",
                "http://127.0.0.1:1/v1/chat/completions",
                "--model",
                "fake-model",
                "--extra-body",
                raw,
                "--output",
                str(tmp_path / "never.jsonl"),
            ]
        )
    assert code == 1
    assert message in stderr.getvalue()
    assert not (tmp_path / "never.jsonl").exists()


def test_tokenizer_identity_names_the_library_and_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoding = types.SimpleNamespace(name="o200k_base", encode=lambda text: [1])
    fake_tiktoken = types.SimpleNamespace(
        __version__="0.9.0", get_encoding=lambda name: encoding
    )
    monkeypatch.setitem(sys.modules, "tiktoken", fake_tiktoken)
    counter = TiktokenCounter(model=None, encoding_name="o200k_base")
    assert tokenizer_identity(counter) == {
        "source": "tiktoken",
        "exact": True,
        "name": "o200k_base",
        "library_version": "0.9.0",
    }

    tokenizer = types.SimpleNamespace(
        name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        init_kwargs={"_commit_hash": "abc123"},
        chat_template="{% for m in messages %}{{ m.content }}{% endfor %}",
        encode=lambda text, add_special_tokens: [1],
    )
    auto = types.SimpleNamespace(from_pretrained=lambda model: tokenizer)
    fake_transformers = types.SimpleNamespace(__version__="4.51.0", AutoTokenizer=auto)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    hf = TransformersTokenCounter(model="Qwen/Qwen2.5-0.5B-Instruct")
    assert tokenizer_identity(hf)["revision"] == "abc123"
    assert tokenizer_identity(hf)["library_version"] == "4.51.0"
    digest = hf.chat_template_digest()
    assert digest is not None and len(digest) == 16
