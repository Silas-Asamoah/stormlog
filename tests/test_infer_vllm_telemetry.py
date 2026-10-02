"""vLLM scrape and span records: validation, schema, and round trips."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from stormlog.infer.vllm_metrics import compact_scrape, discover, parse_prometheus_text
from stormlog.infer.vllm_telemetry import (
    MARKER_INTERVAL,
    MARKER_PHASE_START,
    SCRAPE_ERROR,
    SCRAPE_OK,
    SPAN_SOURCE_JSONL,
    SPAN_SOURCE_RECEIVER,
    VllmScrapeRecord,
    VllmSpanRecord,
    load_vllm_records,
    request_id_from_span_id,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "vllm"
VALIDATOR = Draft202012Validator(
    json.loads((ROOT / "docs/schemas/inference_vllm_v1.schema.json").read_text())
)
CLOCK = "client-host/boot-1/unix_epoch_ns"


def _scrape(**changes: Any) -> VllmScrapeRecord:
    text = (FIXTURES / "q05_c08_metrics_post.txt").read_text(encoding="utf-8")
    compact = compact_scrape(parse_prometheus_text(text))
    values: dict[str, Any] = {
        "session_id": "session",
        "run_id": "run",
        "observed_at_ns": 1_790_000_000_000_000_000,
        "source_url": "http://127.0.0.1:8000/metrics",
        "marker": MARKER_PHASE_START,
        "interval_ms": 1000,
        "status": SCRAPE_OK,
        "clock_domain": CLOCK,
        "case_id": "c8_in512_out128",
        "phase": "measured",
        "duration_ms": 3.5,
        "http_status": 200,
        "content_digest": "ab" * 32,
        "content_bytes": len(text.encode()),
        "scrape": compact,
        "discovery": discover(compact),
    }
    values.update(changes)
    return VllmScrapeRecord(**values)


def _span(**changes: Any) -> VllmSpanRecord:
    values: dict[str, Any] = {
        "session_id": "session",
        "run_id": "run",
        "source": SPAN_SOURCE_RECEIVER,
        "name": "llm_request",
        "clock_domain": "server-host/unix_epoch_ns",
        "received_at_ns": 1_790_000_000_000_000_500,
        "trace_id": "0af7651916cd43dd8448eb211c80319c",
        "span_id": "b7ad6b7169203331",
        "kind": "SERVER",
        "start_unix_ns": 1_790_000_000_000_000_000,
        "end_unix_ns": 1_790_000_000_500_000_000,
        "attributes": {
            "gen_ai.request.id": "chatcmpl-stormlog-run-c8_measured_0_1",
            "gen_ai.latency.time_in_model_inference": 0.48,
        },
        "resource": {"service.name": "vllm"},
        "scope": {"name": "vllm.llm_engine"},
        "status": {"code": "UNSET"},
        "dropped": {"attributes": 0},
        "request_id": "stormlog-run-c8_measured_0_1",
    }
    values.update(changes)
    return VllmSpanRecord(**values)


class TestScrapeRecord:
    def test_ok_record_validates_and_round_trips(self) -> None:
        record = _scrape()
        payload = record.to_record()
        VALIDATOR.validate(payload)
        assert payload["observation_scope"] == "engine_aggregate"
        assert payload["timestamp_ns"] == record.observed_at_ns
        restored = VllmScrapeRecord.from_record(json.loads(json.dumps(payload)))
        assert restored == record

    def test_error_record_validates_and_round_trips(self) -> None:
        record = _scrape(
            status=SCRAPE_ERROR,
            error="URLError: connection refused",
            http_status=None,
            content_digest=None,
            content_bytes=None,
            scrape=None,
            discovery=None,
            marker=MARKER_INTERVAL,
        )
        payload = record.to_record()
        VALIDATOR.validate(payload)
        assert VllmScrapeRecord.from_record(payload) == record

    @pytest.mark.parametrize(
        "changes",
        [
            {"status": SCRAPE_OK, "scrape": None},
            {"status": SCRAPE_OK, "error": "but ok"},
            {"status": SCRAPE_ERROR, "scrape": None, "discovery": None},
            {"status": "weird"},
            {"marker": "sometimes"},
            {"observed_at_ns": 0},
            {"interval_ms": 0},
            {"duration_ms": -1.0},
            {"case_id": ""},
            {"http_status": -1},
            {"clock_domain": ""},
        ],
    )
    def test_invalid_scrape_records_are_rejected(self, changes: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            _scrape(**changes)

    def test_wrong_envelope_is_rejected(self) -> None:
        payload = _scrape().to_record()
        payload["schema_version"] = 2
        with pytest.raises(ValueError, match="unsupported"):
            VllmScrapeRecord.from_record(payload)
        payload = _scrape().to_record()
        payload["observation_scope"] = "per_request"
        with pytest.raises(ValueError, match="engine-aggregate"):
            VllmScrapeRecord.from_record(payload)


class TestSpanRecord:
    def test_span_validates_and_round_trips(self) -> None:
        record = _span()
        payload = record.to_record()
        VALIDATOR.validate(payload)
        assert payload["timestamp_ns"] == record.start_unix_ns
        assert VllmSpanRecord.from_record(json.loads(json.dumps(payload))) == record
        assert record.duration_ns == 500_000_000

    def test_minimal_span_from_a_file_has_no_times(self) -> None:
        record = _span(
            source=SPAN_SOURCE_JSONL,
            received_at_ns=None,
            trace_id=None,
            span_id=None,
            kind=None,
            start_unix_ns=None,
            end_unix_ns=None,
            status=None,
            dropped={},
            request_id=None,
        )
        VALIDATOR.validate(record.to_record())
        assert record.duration_ns is None

    @pytest.mark.parametrize(
        "changes",
        [
            {"source": "carrier_pigeon"},
            {"name": ""},
            {"start_unix_ns": 10, "end_unix_ns": 9},
            {"attributes": {1: "x"}},
            {"status": "ok"},
            {"request_id": ""},
            {"received_at_ns": -1},
        ],
    )
    def test_invalid_spans_are_rejected(self, changes: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            _span(**changes)

    def test_request_id_recovery_from_vllm_ids(self) -> None:
        # Chat completions carry the header as sent; text completions append
        # the per-prompt index, which is stripped.
        assert request_id_from_span_id("chatcmpl-stormlog-run-c8_measured_0_1") == (
            "stormlog-run-c8_measured_0_1"
        )
        assert request_id_from_span_id("cmpl-q05-u-c01-rep0-on-r0000-0") == (
            "q05-u-c01-rep0-on-r0000"
        )
        # An index is only stripped when it is one; a header without a dash
        # suffix stays whole.
        assert request_id_from_span_id("chatcmpl-abc") == "abc"
        assert request_id_from_span_id("chatcmpl-abc-def") == "abc-def"
        assert request_id_from_span_id("something-else") is None
        assert request_id_from_span_id(None) is None


class TestLoader:
    def test_loader_returns_sorted_scrapes_and_all_spans(self) -> None:
        later = _scrape(observed_at_ns=1_790_000_000_000_000_900)
        earlier = _scrape(observed_at_ns=1_790_000_000_000_000_100)
        records = [
            {"event_type": "infer.request", "request_id": "ignored"},
            later.to_record(),
            _span().to_record(),
            earlier.to_record(),
        ]
        scrapes, spans = load_vllm_records(records)
        assert [s.observed_at_ns for s in scrapes] == [
            earlier.observed_at_ns,
            later.observed_at_ns,
        ]
        assert len(spans) == 1

    def test_loader_names_the_broken_record(self) -> None:
        broken = _span().to_record()
        del broken["name"]
        with pytest.raises(ValueError, match="record 1"):
            load_vllm_records([_scrape().to_record(), broken])

    def test_fixture_span_lines_load_as_file_spans(self) -> None:
        lines = (FIXTURES / "q05_spans_sample.jsonl").read_text().splitlines()
        spans = []
        for line in lines:
            raw = json.loads(line)
            spans.append(
                VllmSpanRecord(
                    session_id="session",
                    run_id="run",
                    source=SPAN_SOURCE_JSONL,
                    name=raw["name"],
                    clock_domain="audit-host/unix_epoch_ns",
                    start_unix_ns=raw["start_unix_ns"],
                    end_unix_ns=raw["end_unix_ns"],
                    attributes=raw["attributes"],
                    request_id=request_id_from_span_id(
                        raw["attributes"].get("gen_ai.request.id")
                    ),
                )
            )
        for span in spans:
            VALIDATOR.validate(span.to_record())
        requests = [s for s in spans if s.name == "llm_request"]
        assert len(requests) == 12
        assert all(s.request_id is not None for s in requests)
        assert {s.request_id for s in spans if s.name != "llm_request"} == {None}
