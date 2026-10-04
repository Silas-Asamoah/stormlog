"""Span export from a real ``infer profile`` run against fake servers."""

import json
import socket
import time
from pathlib import Path
from typing import Any

import pytest

from stormlog.exit_codes import ExitCode
from stormlog.infer.cli import main as infer_main
from stormlog.infer.config import ProfileConfig
from stormlog.infer.export_config import ExportConfig
from stormlog.infer.profile import InferenceProfiler
from stormlog.infer.trace_context import PRESERVE_ENGINE
from stormlog.infer.vllm_spans import read_span_file
from tests.export_conformance import check_exposition
from tests.fake_otlp_collector import FakeCollector, running
from tests.test_infer_profile import _fake_server

pytest.importorskip("opentelemetry.proto.collector.trace.v1.trace_service_pb2")


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _config(endpoint: str, output: Path, export: ExportConfig, **kw: Any) -> Any:
    settings: dict[str, Any] = {
        "endpoint": endpoint,
        "model": "fake-model",
        "concurrency": (1,),
        "input_tokens": (8,),
        "output_tokens": (4,),
        "output_path": str(output),
        "request_count": 3,
        "warmup_requests": 1,
        "stream": True,
        "tokenizer": "none",
        "system_sampler": "none",
        "export": export,
    }
    settings.update(kw)
    return ProfileConfig(**settings)


def _capability(records: list[dict[str, Any]], component: str) -> dict[str, Any]:
    return next(
        r
        for r in records
        if r["event_type"] == "infer.capabilities" and r["component"] == component
    )


def _balanced(spans: dict[str, Any]) -> bool:
    return bool(
        spans["offered"]
        == spans["exported"]
        + spans["rejected"]
        + sum(spans["refused"].values())
        + sum(spans["dropped"].values())
        + sum(spans["unknown"].values())
        + spans["queued"]
        + spans["in_flight"]
    )


def _run(
    tmp_path: Path, collector: FakeCollector, **export: Any
) -> list[dict[str, Any]]:
    output = tmp_path / "infer.jsonl"
    with _fake_server() as endpoint:
        config = _config(
            endpoint, output, ExportConfig(otlp_endpoint=collector.url, **export)
        )
        InferenceProfiler(config).run()
    return _records(output)


def test_a_run_exports_its_requests_phases_and_capture(tmp_path: Path) -> None:
    with running() as collector:
        records = _run(tmp_path, collector, trace_context=PRESERVE_ENGINE)
        spans = collector.stored_spans()
    requests = [r for r in records if r["event_type"] == "infer.request"]
    by_name: dict[str, list[Any]] = {}
    for span in spans:
        by_name.setdefault(span.name, []).append(span)
    assert len(by_name["stormlog.infer.request"]) == len(requests) == 4
    assert len(by_name["stormlog.infer.phase"]) == 2
    (capture,) = by_name["stormlog.infer.capture"]
    assert capture.attributes["stormlog.capture.outcome"] == "completed"
    for phase in by_name["stormlog.infer.phase"]:
        assert phase.parent_span_id == capture.span_id
    # The request spans carry the IDs sent in traceparent.
    sent = {(r["trace_id"], r["span_id"]) for r in requests}
    assert {(s.trace_id, s.span_id) for s in by_name["stormlog.infer.request"]} == sent
    assert {s.resource["service.name"] for s in spans} == {"stormlog"}
    assert {s.resource["service.instance.id"] for s in spans} == {
        requests[0]["session_id"]
    }
    capability = _capability(records, "export.otlp")
    summary = capability["metadata"]["summary"]
    assert capability["available"] and capability["collected"] == ["endpoint"]
    assert summary["spans"]["exported"] == summary["spans"]["offered"] == 7
    assert summary["spans"]["frozen"] and _balanced(summary["spans"])
    assert capability["metadata"]["trace_context"] == PRESERVE_ENGINE
    assert capability["metadata"]["destination"] == collector.url.rsplit("/v1", 1)[0]
    # No Prometheus capability without Prometheus.
    components = {r.get("component") for r in records}
    assert "export.prometheus" not in components
    assert records[-1]["event_type"] == "infer.session"


def test_spans_go_to_a_file_when_asked(tmp_path: Path) -> None:
    output = tmp_path / "infer.jsonl"
    spans_file = tmp_path / "spans.jsonl"
    with _fake_server() as endpoint:
        InferenceProfiler(
            _config(endpoint, output, ExportConfig(otlp_file=spans_file))
        ).run()
    _source, spans = read_span_file(spans_file)
    assert sorted({s.name for s in spans}) == [
        "stormlog.infer.capture",
        "stormlog.infer.phase",
        "stormlog.infer.request",
    ]
    capability = _capability(_records(output), "export.otlp")
    assert capability["metadata"]["encoding"] == "json"
    assert capability["metadata"]["summary"]["spans"]["exported"] == len(spans)


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_a_down_collector_never_fails_the_run(tmp_path: Path) -> None:
    output = tmp_path / "infer.jsonl"
    with _fake_server() as endpoint:
        config = _config(
            endpoint,
            output,
            ExportConfig(
                otlp_endpoint=f"http://127.0.0.1:{_closed_port()}",
                otlp_flush_timeout_seconds=1.0,
            ),
        )
        started = time.monotonic()
        InferenceProfiler(config).run()
        elapsed = time.monotonic() - started
    records = _records(output)
    assert records[-1]["status"] == "completed"
    statuses = {r["status"] for r in records if r["event_type"] == "infer.request"}
    assert statuses == {"ok"}
    spans = _capability(records, "export.otlp")["metadata"]["summary"]["spans"]
    assert spans["exported"] == 0 and _balanced(spans) and spans["in_flight"] == 0
    assert set(spans["dropped"]) <= {"connect_refused", "shutdown"}
    assert elapsed < 20


def test_prometheus_shows_the_span_accounting(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    with running() as collector:
        records = _run(tmp_path, collector, prometheus_textfile_dir=metrics_dir)
    spans = _capability(records, "export.otlp")["metadata"]["summary"]["spans"]
    exposition = check_exposition((metrics_dir / "stormlog-default.prom").read_text())
    assert exposition.value("stormlog_export_spans_exported_total") == spans["exported"]
    assert exposition.value("stormlog_export_spans_offered_total") == spans["offered"]
    assert exposition.value("stormlog_export_destination_up") == 1
    assert exposition.value("stormlog_export_requests_total", outcome="confirmed") >= 1


@pytest.mark.parametrize(
    "flags",
    [
        ["--otlp-endpoint", "grpc://127.0.0.1:4317"],
        ["--otlp-endpoint", "http://user:pw@127.0.0.1:4318"],
        ["--otlp-header", "x-api-key=abc"],
        ["--otlp-endpoint", "http://127.0.0.1:4318", "--otlp-file", "spans.jsonl"],
        ["--otlp-endpoint", "http://127.0.0.1:4318", "--export-content", "bodies"],
        [
            "--otlp-endpoint",
            "http://127.0.0.1:4318",
            "--otlp-resource-attribute-allow",
            "db.password",
        ],
        ["--otlp-endpoint", "http://127.0.0.1:4318", "--otlp-probe-interval", "60"],
        ["--otlp-endpoint", "http://127.0.0.1:4318", "--otlp-header", "bad name=x"],
    ],
)
def test_unusable_span_settings_exit_2_before_sending(
    tmp_path: Path, flags: list[str]
) -> None:
    output = tmp_path / "infer.jsonl"
    code = infer_main(
        [
            "profile",
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--model",
            "m",
            "--requests",
            "1",
            "--output",
            str(output),
            *flags,
        ]
    )
    assert code == ExitCode.USAGE
    assert not output.exists()
