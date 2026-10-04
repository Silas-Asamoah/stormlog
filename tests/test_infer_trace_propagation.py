"""``infer profile --trace-context``: the traceparent sent, and the IDs recorded."""

import argparse
import contextlib
import json
import threading
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.config import ProfileConfig
from stormlog.infer.export_config import (
    ExportConfig,
    add_export_arguments,
    add_trace_context_arguments,
    export_config_from_args,
)
from stormlog.infer.profile import InferenceProfiler
from stormlog.infer.trace_context import (
    FOLLOW_SAMPLING,
    OFF,
    PRESERVE_ENGINE,
    parse_traceparent,
)
from tests.infer_workload_helpers import run_profile_with_fake_client
from tests.test_infer_profile import _FakeOpenAIHandler


class _HeaderRecordingHandler(_FakeOpenAIHandler):
    seen: list[tuple[str | None, str | None]] = []
    lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802
        with self.lock:
            self.seen.append(
                (self.headers.get("X-Request-Id"), self.headers.get("traceparent"))
            )
        super().do_POST()


@contextlib.contextmanager
def _recording_server() -> Iterator[str]:
    _HeaderRecordingHandler.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HeaderRecordingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _run(tmp_path: Path, export: ExportConfig) -> list[dict[str, Any]]:
    output = tmp_path / "infer.jsonl"
    with _recording_server() as endpoint:
        config = ProfileConfig(
            endpoint=endpoint,
            model="fake-model",
            concurrency=(2,),
            input_tokens=(8,),
            output_tokens=(4,),
            output_path=str(output),
            request_count=4,
            warmup_requests=1,
            stream=True,
            tokenizer="none",
            system_sampler="none",
            export=export,
        )
        InferenceProfiler(config).run()
    return [json.loads(line) for line in output.read_text().splitlines()]


def _requests(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r["event_type"] == "infer.request"]


def _session_config(records: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(records[0]["config"])


def test_preserve_engine_sends_a_sampled_parent_and_records_its_ids(
    tmp_path: Path,
) -> None:
    records = _run(
        tmp_path,
        ExportConfig(
            trace_context=PRESERVE_ENGINE,
            server_trace_sampler="parentbased_traceidratio:0.1",
        ),
    )
    requests = _requests(records)
    sent = dict(_HeaderRecordingHandler.seen)
    assert len(sent) == len(requests) == 5
    for record in requests:
        header = sent[record["x_request_id"]]
        assert header is not None and header.endswith("-01")
        ids = parse_traceparent(header)
        assert ids is not None
        assert (ids.trace_id, ids.span_id) == (record["trace_id"], record["span_id"])
    # Each request is its own trace.
    assert len({r["trace_id"] for r in requests}) == len(requests)
    assert _session_config(records)["trace_context"] == {
        "policy": PRESERVE_ENGINE,
        "sample_ratio": 1.0,
        "server_trace_sampler": "parentbased_traceidratio:0.1",
    }


def test_follow_sampling_sends_stormlogs_own_decision(tmp_path: Path) -> None:
    records = _run(
        tmp_path, ExportConfig(trace_context=FOLLOW_SAMPLING, sample_ratio=0.0)
    )
    requests = _requests(records)
    sent = dict(_HeaderRecordingHandler.seen)
    for record in requests:
        header = sent[record["x_request_id"]]
        assert header == f"00-{record['trace_id']}-{record['span_id']}-00"
    assert _session_config(records)["trace_context"]["sample_ratio"] == 0.0


def test_without_trace_context_nothing_is_sent_even_while_exporting(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    records = _run(tmp_path, ExportConfig(prometheus_textfile_dir=metrics))
    requests = _requests(records)
    assert len(_HeaderRecordingHandler.seen) == len(requests) == 5
    assert all(header is None for _, header in _HeaderRecordingHandler.seen)
    assert all(r["trace_id"] is None and r["span_id"] is None for r in requests)
    assert _session_config(records)["trace_context"] == {
        "policy": OFF,
        "sample_ratio": 1.0,
        "server_trace_sampler": None,
    }


def test_a_cancelled_request_keeps_the_ids_it_was_sent_with(tmp_path: Path) -> None:
    requests, _report, client = run_profile_with_fake_client(
        tmp_path,
        latency_seconds=0.5,
        arrival_mode="fixed-rate",
        rates=(100.0,),
        request_count=2,
        drain_timeout_seconds=0.1,
        export=ExportConfig(trace_context=PRESERVE_ENGINE),
    )
    assert [r["status"] for r in requests] == ["cancelled", "cancelled"]
    for record in requests:
        header = client.headers[record["x_request_id"]]["traceparent"]
        assert header == f"00-{record['trace_id']}-{record['span_id']}-01"


def test_a_request_never_sent_has_no_ids(tmp_path: Path) -> None:
    requests, _report, client = run_profile_with_fake_client(
        tmp_path,
        latency_seconds=0.3,
        arrival_mode="fixed-rate",
        rates=(100.0,),
        request_count=3,
        max_in_flight=1,
        overflow="drop",
        export=ExportConfig(trace_context=PRESERVE_ENGINE),
    )
    dropped = [r for r in requests if r["status"] == "dropped"]
    (sent,) = [r for r in requests if r["status"] == "ok"]
    assert len(dropped) == 2 and client.calls == 1
    assert all(r["trace_id"] is None and r["span_id"] is None for r in dropped)
    assert sent["trace_id"] is not None


def _parse(argv: list[str]) -> ExportConfig:
    parser = argparse.ArgumentParser()
    add_export_arguments(parser)
    add_trace_context_arguments(parser)
    return export_config_from_args(parser.parse_args(argv))


def test_the_flags_set_the_policy_ratio_and_declared_sampler() -> None:
    config = _parse(
        [
            "--trace-context",
            "follow-sampling",
            "--otlp-sample-ratio",
            "0.25",
            "--server-trace-sampler",
            "always_on",
        ]
    )
    assert (config.trace_context, config.sample_ratio) == (FOLLOW_SAMPLING, 0.25)
    assert config.server_trace_sampler == "always_on"
    assert _parse([]).trace_context == OFF
    # Trace context alone exports nothing.
    assert not _parse(["--trace-context", "preserve-engine"]).enabled


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"sample_ratio": 1.5}, "between 0 and 1"),
        ({"sample_ratio": -0.1}, "between 0 and 1"),
        ({"sample_ratio": 0.5}, "only applies with --otlp-endpoint"),
        (
            {"trace_context": PRESERVE_ENGINE, "sample_ratio": 0.5},
            "only applies with --otlp-endpoint",
        ),
        (
            {
                "trace_context": PRESERVE_ENGINE,
                "sample_ratio": 0.5,
                "otlp_endpoint": "http://127.0.0.1:4318",
            },
            "no effect with --trace-context preserve-engine",
        ),
        ({"trace_context": "always"}, "must be one of"),
    ],
)
def test_unusable_trace_settings_are_refused(
    settings: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ExportConfig(**settings).validate()


def test_the_ratio_also_samples_exported_spans() -> None:
    ExportConfig(sample_ratio=0.5, otlp_endpoint="http://127.0.0.1:4318").validate()
    ExportConfig(sample_ratio=0.5, trace_context=FOLLOW_SAMPLING).validate()


@pytest.mark.parametrize(
    "sampler",
    [
        "parent_based_always_on",  # not the SDK's name
        "parentbased_traceidratio:abc",
        "traceidratio:1.5",
        "traceidratio:nan",
        "parentbased_always_on:0.1",  # takes no ratio
    ],
)
def test_a_server_sampler_the_sdk_does_not_know_is_refused(sampler: str) -> None:
    # Arms of a comparison that declare it would otherwise differ silently.
    with pytest.raises(ValueError, match="--server-trace-sampler"):
        _parse(
            ["--trace-context", "preserve-engine", "--server-trace-sampler", sampler]
        )


@pytest.mark.parametrize(
    "sampler",
    [
        "always_on",
        "always_off",
        "traceidratio",
        "parentbased_traceidratio:0.1",
        "parentbased_always_off",
        "parentbased_jaeger_remote:endpoint=http://jaeger:14250",
        "xray",
    ],
)
def test_the_sdks_sampler_names_are_accepted(sampler: str) -> None:
    _parse(["--trace-context", "preserve-engine", "--server-trace-sampler", sampler])


def test_follow_sampling_takes_the_servers_declared_ratio() -> None:
    declared = ["--server-trace-sampler", "parentbased_traceidratio:0.1"]
    follow = ["--trace-context", "follow-sampling", *declared]
    assert _parse(follow).sample_ratio == 0.1
    assert _parse([*follow, "--otlp-sample-ratio", "0.05"]).sample_ratio == 0.05
    with pytest.raises(ValueError, match="above the 0.1"):
        _parse([*follow, "--otlp-sample-ratio", "0.5"])


@pytest.mark.parametrize("sampler", ["traceidratio:0.1", "always_off"])
def test_a_sampler_that_ignores_the_parent_leaves_the_ratio_alone(sampler: str) -> None:
    # Not parent-based: the server keeps its own share whatever the flag
    # says, so the flag's ratio neither follows nor is bounded by it.
    follow = ["--trace-context", "follow-sampling", "--server-trace-sampler", sampler]
    assert _parse(follow).sample_ratio == 1.0
    assert _parse([*follow, "--otlp-sample-ratio", "0.5"]).sample_ratio == 0.5


@pytest.mark.parametrize(
    ("sampler", "warned"),
    [
        ("parentbased_traceidratio:0.1", True),
        ("parentbased_always_off", True),
        ("parentbased_always_on", False),  # the SDK's default: no change
        ("traceidratio:0.1", False),  # not parent-based: ignores the flag
        (None, False),
    ],
)
def test_preserve_engine_warns_when_it_raises_the_servers_volume(
    tmp_path: Path, sampler: str | None, warned: bool
) -> None:
    warnings: list[str] = []
    config = ProfileConfig(
        endpoint="http://127.0.0.1:9/v1/chat/completions",
        model="m",
        concurrency=(1,),
        input_tokens=(8,),
        output_tokens=(4,),
        output_path=str(tmp_path / "infer.jsonl"),
        tokenizer="none",
        system_sampler="none",
        export=ExportConfig(
            trace_context=PRESERVE_ENGINE, server_trace_sampler=sampler
        ),
    )
    InferenceProfiler(config, on_warning=warnings.append)
    found = [w for w in warnings if "preserve-engine" in w]
    assert bool(found) is warned
    if warned:
        assert sampler is not None and sampler in found[0]


def test_a_watcher_refuses_trace_context() -> None:
    with pytest.raises(ValueError, match="infer profile only"):
        ExportConfig.from_mapping({"trace_context": PRESERVE_ENGINE}, "watch")
    ExportConfig.from_mapping({"trace_context": PRESERVE_ENGINE}, "profile")
    ExportConfig.from_mapping({"trace_context": OFF}, "watch")
