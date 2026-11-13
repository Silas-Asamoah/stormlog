"""Summaries, cases, rows, request detail and plots (design §5.4, §3.4; C2).

Every number in these documents comes from an analyzer function, so the
tests check two things: that the documents hold what those functions
return (parity with ``infer analyze`` for every case of every standard
artifact, §3.4), and that the selection and presentation around them
(bins, points, samples, pages) follow §5.4 and §5.7.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import jsonschema  # type: ignore[import-untyped, unused-ignore]
import pytest
from referencing import Registry, Resource

from stormlog.infer.latency_report import CLIENT_METRICS, spread, streaming_summary
from stormlog.studio import summaries
from stormlog.studio.contracts import (
    LatencyRange,
    NotResident,
    RequestFilter,
    TimeWindow,
    UsageError,
    envelope,
    to_wire,
)
from stormlog.studio.selection import case_facts, page, parse_filter, select
from tests.studio.artifacts import MULTI_SESSION, STANDARD_ARTIFACTS
from tests.studio.conftest import StandardArtifacts
from tests.studio.test_index import Harness
from tests.studio.test_selection import (
    ORIGIN,
    SECOND,
    SESSION,
    closed_case,
    phase_window,
    request,
    session_record,
    snapshot_from_records,
    snapshot_of,
    workload,
)

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "docs" / "schemas"


def _registry() -> Registry[Any]:
    resources = []
    for path in sorted(SCHEMA_DIR.glob("studio_*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


REGISTRY = _registry()


def schema_errors(name: str, document: Mapping[str, Any]) -> list[str]:
    """Where ``document`` breaks ``docs/schemas/studio_<name>_v1.schema.json``."""
    schema = json.loads((SCHEMA_DIR / f"studio_{name}_v1.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema, registry=REGISTRY)
    return [
        f"{list(error.absolute_path)}: {error.message}"
        for error in validator.iter_errors(document)
    ]


def _wire(name: str, body: Mapping[str, Any], snapshot: Any) -> dict[str, Any]:
    document = to_wire(envelope(name, body, snapshot))
    assert isinstance(document, dict)
    return document


# Every standard artifact Studio opens: it refuses multi-session files.
STUDIO_ARTIFACTS = sorted(set(STANDARD_ARTIFACTS) - MULTI_SESSION)


def _summary(snapshot: Any, f: RequestFilter) -> dict[str, Any]:
    return summaries.summary(snapshot, select(snapshot, f))


# --- Summary rules -----------------------------------------------------------------


def test_a_whole_case_counts_its_cohort_and_any_other_selection_does_not() -> None:
    snapshot = snapshot_from_records(closed_case(12))
    whole = _summary(snapshot, RequestFilter())
    assert whole["counts"]["cohort_valid"] is True
    assert whole["selection"]["cohort"] == {"valid": True, "issues": [], "reason": None}
    assert whole["intervals"] is not None and whole["throughput"] is not None
    narrowed = _summary(snapshot, RequestFilter(statuses=("ok",)))
    assert narrowed["counts"]["cohort_valid"] is None
    assert narrowed["selection"]["cohort"]["reason"] == "not_a_cohort"
    assert narrowed["intervals"] is None and narrowed["throughput"] is None
    assert narrowed["counts"]["offered"] == 12


def test_server_evidence_is_not_indexed_never_a_bare_null() -> None:
    body = _summary(snapshot_from_records(closed_case()), RequestFilter())
    assert body["counts"]["server_evidence"] == "not_indexed"
    assert set(body["latency"]["metrics"]) == set(CLIENT_METRICS)


def test_an_empty_selection_has_the_same_shape_with_nothing_in_it() -> None:
    snapshot = snapshot_from_records(closed_case())
    body = _summary(snapshot, RequestFilter(cases=("nowhere",)))
    assert body["selection"]["size"] == 0
    assert body["failure"] == {
        "share": None,
        "numerator": 0,
        "denominator": 0,
        "reason": "empty_selection",
    }
    assert body["throughput"] is None
    assert body["client_memory"]["window"] is None
    assert (
        body["latency"]["metrics"]["client.e2e"]["successful"]["p50"]["value_ms"]
        is None
    )
    assert schema_errors("summary", _wire("summary", body, snapshot)) == []


def test_an_all_failed_case_keeps_its_zero_rate_and_its_elapsed_times() -> None:
    """Choice 21: the analyzer's 0.0 over a defined interval, labelled by the
    UI from counts.successful == 0."""
    snapshot = snapshot_from_records(
        closed_case(6, status="error", ttft_ms=None, e2e_latency_ms=None)
    )
    body = _summary(snapshot, RequestFilter())
    assert body["counts"]["successful"] == 0
    assert body["throughput"]["requests_per_second"] == 0.0
    assert body["failure"]["share"] == 1.0
    assert body["latency"]["unsuccessful"]["error"]["count"] == 6
    assert body["latency"]["unsuccessful"]["error"]["elapsed_ms"]["p50"] == 500.0


def test_a_planned_case_with_no_observations_gets_the_empty_summary() -> None:
    """Choice 16."""
    records = closed_case()
    records[1] = workload("c1", "c9")
    snapshot = snapshot_from_records(records)
    document = summaries.cases(snapshot)
    by_case = {entry["case_id"]: entry for entry in document["cases"]}
    assert set(by_case) == {"c1", "c9"}
    planned = by_case["c9"]
    assert planned["planned"] and not planned["observed"]
    assert planned["summary"]["selection"]["size"] == 0
    assert planned["summary"]["failure"]["reason"] == "empty_selection"
    assert by_case["c1"]["observed"]
    assert document["comparison_scope"] == "full_measured_case"
    assert schema_errors("cases", _wire("cases", document, snapshot)) == []


def test_non_streamed_responses_have_no_chunk_statistics() -> None:
    snapshot = snapshot_from_records(closed_case(stream=False, ttft_ms=None))
    streaming = _summary(snapshot, RequestFilter())["streaming"]
    assert streaming["responses"] == 0
    assert streaming["chunk_interarrival_ms"]["n"] == 0


def test_gaps_that_are_not_resident_make_streaming_unavailable() -> None:
    snapshot = replace(snapshot_from_records(closed_case()), gaps_resident=False)
    body = _summary(snapshot, RequestFilter())
    assert body["streaming"] == {"state": "unavailable", "reason": "gaps_not_resident"}
    assert schema_errors("summary", _wire("summary", body, snapshot)) == []


def test_missing_token_counts_show_as_partial_coverage() -> None:
    records = closed_case(4)
    records[3]["output_tokens"] = None
    body = _summary(snapshot_from_records(records), RequestFilter())
    assert body["tokens"]["output"]["known"] == 3
    assert body["tokens"]["output"]["partial"] is True


def _open_loop(count: int, endpoint_ns: int | None) -> list[dict[str, Any]]:
    spec = {"mode": "poisson", "rate_per_second": 10.0}
    load = workload("c1")
    load["cases"][0]["arrival"] = spec
    rows = [request(i, intended_at_ns=ORIGIN + i * SECOND // 10) for i in range(count)]
    extra: dict[str, Any] = {"arrival_mode": "poisson", "scheduled_arrivals": count}
    if endpoint_ns is not None:
        extra["scheduled_endpoint_offset_ns"] = endpoint_ns
    window = phase_window(
        "c1", ORIGIN, ORIGIN + count * SECOND // 10 + 2 * SECOND, **extra
    )
    return [session_record(), load, *rows, window]


def test_an_open_loop_case_with_a_covering_window_equals_no_window() -> None:
    snapshot = snapshot_from_records(_open_loop(20, 2 * SECOND))
    rate = case_facts(snapshot).populations["c1"].intervals.rate
    assert rate is not None and rate.kind == "scheduled_window"
    plain = _summary(snapshot, RequestFilter())
    covering = _summary(
        snapshot, RequestFilter(window=TimeWindow(ORIGIN - SECOND, ORIGIN + 9 * SECOND))
    )
    assert covering["throughput"] == plain["throughput"]
    assert covering["selection"]["whole_case"] == "c1"
    half = _summary(snapshot, RequestFilter(window=TimeWindow(ORIGIN, ORIGIN + SECOND)))
    assert half["selection"]["rate"]["rule"] == "segment"
    assert half["throughput"]["interval_seconds"] == 1.0
    assert half["throughput"]["requests_per_second"] == 10.0


def test_an_open_loop_case_without_an_endpoint_has_no_rate() -> None:
    records = _open_loop(20, None)
    records[1]["measurement"] = {}
    snapshot = snapshot_from_records(records)
    body = _summary(snapshot, RequestFilter())
    assert body["selection"]["rate"]["reason"] == "endpoint_undeclared"
    # A whole case keeps the analyzer's block: every rate null, no interval.
    assert body["throughput"]["requests_per_second"] is None
    assert body["throughput"]["interval_seconds"] is None
    windowed = _summary(
        snapshot, RequestFilter(window=TimeWindow(ORIGIN, ORIGIN + SECOND))
    )
    assert windowed["throughput"] is None


def test_a_kept_record_over_the_limit_is_disclosed_in_the_summary() -> None:
    """D-4: no cohort and no rate, with the reason, rather than wrong ones."""
    records = closed_case()
    records[1]["padding"] = "x" * 400
    snapshot = snapshot_from_records(records, kept_record_bytes=350)
    body = _summary(snapshot, RequestFilter())
    assert body["selection"]["cohort"] == {
        "valid": None,
        "issues": [],
        "reason": "kept_record_over_limit",
    }
    assert body["selection"]["rate"]["reason"] == "kept_record_over_limit"
    assert body["throughput"] is None and body["intervals"] is None


def test_client_memory_peaks_by_device_inside_the_successful_window() -> None:
    def sample(at: int, **fields: Any) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "event_type": "infer.system_sample",
            "session_id": SESSION,
            "timestamp_ns": at,
            "observation_scope": "client_local",
            **fields,
        }

    records = closed_case(4)
    records += [
        sample(ORIGIN + SECOND // 10, sampler="psutil", process_rss_bytes=100),
        sample(ORIGIN + SECOND // 5, sampler="psutil", process_rss_bytes=300),
        sample(ORIGIN + 50 * SECOND, sampler="psutil", process_rss_bytes=999),
        sample(ORIGIN + SECOND // 5, sampler="nvml", device_id=0, device_used_bytes=7),
    ]
    memory = _summary(snapshot_from_records(records), RequestFilter())["client_memory"]
    assert memory["window"] == {
        "started_at_ns": ORIGIN,
        "ended_at_ns": ORIGIN + 3 * SECOND // 10 + SECOND // 2,
    }
    assert [(d["sampler"], d["samples"]) for d in memory["by_device"]] == [
        ("nvml", 1),
        ("psutil", 2),
    ]
    assert memory["by_device"][1]["peak_process_rss_bytes"] == 300


def test_summary_documents_validate_for_every_standard_artifact(
    studio_artifacts: StandardArtifacts,
) -> None:
    filters = [
        RequestFilter(),
        RequestFilter(phase="all"),
        RequestFilter(statuses=("error",)),
        RequestFilter(latency=LatencyRange("client.e2e", 0.0, None)),
    ]
    for name in STUDIO_ARTIFACTS:
        snapshot = snapshot_of(studio_artifacts[name])
        for f in filters:
            assert (
                schema_errors(
                    "summary", _wire("summary", _summary(snapshot, f), snapshot)
                )
                == []
            ), name
        assert (
            schema_errors("cases", _wire("cases", summaries.cases(snapshot), snapshot))
            == []
        ), name


# --- Rows and request detail ----------------------------------------------------------


def test_requests_pages_hold_rows_without_messages_or_gaps(
    studio_artifacts: StandardArtifacts,
) -> None:
    snapshot = snapshot_of(studio_artifacts["mixed"])
    sel = select(snapshot, RequestFilter())
    got = page(snapshot, sel, sort="client.e2e", order="desc", cursor=None, limit=5)
    document = summaries.requests_page(snapshot, sel, got)
    assert schema_errors("requests", _wire("requests", document, snapshot)) == []
    assert len(document["rows"]) == 5 and document["total"] == len(sel.rows)
    row = document["rows"][0]
    assert "error_message" not in row and "chunk_interarrival_ms" not in row
    request_row = snapshot.requests[got.rows[0]]
    assert row["start_ms"] == (request_row["started_at_ns"] - snapshot.origin_ns) / 1e6
    assert row["chunk_count"] in (None, request_row.gap_count + 1)


def test_request_detail_gives_the_record_spread_links_and_membership() -> None:
    records = [
        session_record(),
        request(0),
        request(
            1,
            status="timeout",
            e2e_latency_ms=None,
            ttft_ms=float("nan"),
            error_message="slow",
            chunk_interarrival_ms=[5.0, "x", 1.0] + [2.0] * 200,
        ),
    ]
    records.append(
        {
            "schema_version": 1,
            "event_type": "infer.dispatch",
            "session_id": SESSION,
            "request_id": "r1",
        }
    )
    snapshot = snapshot_from_records(records)
    sel = select(snapshot, RequestFilter(statuses=("ok",)))
    detail = summaries.request_detail(snapshot, 1, sel)
    assert detail["in_selection"] is False
    assert detail["record"]["error_message"] == "slow"
    assert "chunk_interarrival_ms" not in detail["record"]
    assert detail["record_omitted"] == ["chunk_interarrival_ms"]
    assert detail["non_finite_fields"] == ["ttft_ms"]
    assert detail["elapsed_ms"] == 500.0
    numeric = [5.0, 1.0] + [2.0] * 200
    assert detail["gaps"] == {
        **spread(numeric),
        "n": 202,
        "gap_count": 203,
        "first": [5.0, None, 1.0] + [2.0] * 125,
    }
    assert detail["links"]["dispatch"]["line"] == 3
    assert detail["links"]["first_content"] is None
    assert summaries.request_detail(snapshot, 0, None)["elapsed_ms"] is None
    assert schema_errors("request", _wire("request", detail, snapshot)) == []


def test_request_detail_reads_the_message_through_its_loader() -> None:
    class Unresident(dict[str, Any]):
        pass

    snapshot = snapshot_from_records([request(0, error_message="boom")])
    row = snapshot.requests[0]
    original = type(row).__getitem__

    def guarded(self: Any, key: str) -> Any:
        if key in ("error_message", "chunk_interarrival_ms"):
            raise NotResident(key)
        return original(self, key)

    patched = type("Guarded", (type(row),), {"__getitem__": guarded})
    row.__class__ = patched
    detail = summaries.request_detail(snapshot, 0, None)
    assert detail["record"]["error_message"] == "boom"


def test_chunks_page_the_gaps_in_written_order() -> None:
    gaps = [float(i) for i in range(25)] + [float("inf")]
    snapshot = snapshot_from_records([request(0, chunk_interarrival_ms=gaps)])
    first = summaries.chunks(snapshot, 0, offset=0, limit=10)
    assert first["gaps_ms"] == gaps[:10] and first["total"] == 26
    last = summaries.chunks(snapshot, 0, offset=20, limit=10)
    assert last["gaps_ms"] == gaps[20:25] + [None]
    for offset, limit in ((-1, 5), (0, 0), (0, 10_001)):
        with pytest.raises(UsageError):
            summaries.chunks(snapshot, 0, offset=offset, limit=limit)


# --- Distribution, points, samples -------------------------------------------------


def _latency_snapshot() -> Any:
    rows = [request(i, e2e_latency_ms=float(100 + (i * 37) % 400)) for i in range(60)]
    rows.append(request(60, e2e_latency_ms=None, status="error"))
    return snapshot_from_records([session_record(), *rows])


def test_each_bin_carries_the_latency_parameter_that_selects_its_count() -> None:
    snapshot = _latency_snapshot()
    sel = select(snapshot, RequestFilter())
    document = summaries.distribution(
        snapshot, sel, metric="client.e2e", edges=None, bins=7
    )
    assert sum(b["count"] for b in document["bins"]) == 60
    assert document["denominator"] == {
        "params": "phase=measured",
        "size": 61,
        "with_value": 60,
        "without_value": 1,
    }
    largest = max(100 + (i * 37) % 400 for i in range(60))
    assert document["bins"][-1]["upper_ms"] == math.nextafter(largest, math.inf)
    for b in document["bins"]:
        query = parse_qs(f"latency={b['latency']}")
        brushed = select(snapshot, parse_filter(query, origin_ns=None))
        assert len(brushed.rows) == b["count"]


def test_given_edges_count_values_outside_as_below_and_above() -> None:
    snapshot = _latency_snapshot()
    narrowed = select(
        snapshot, RequestFilter(latency=LatencyRange("client.e2e", 200.0, 300.0))
    )
    document = summaries.distribution(
        snapshot, narrowed, metric="client.e2e", edges=[150.0, 250.0, 350.0], bins=None
    )
    assert document["denominator"]["size"] == 61  # the latency condition is dropped
    total = (
        sum(b["count"] for b in document["bins"])
        + document["below"]
        + document["above"]
    )
    assert total == 60
    assert document["selection"]["size"] == len(narrowed.rows)
    for edges in (
        [1.0],
        [2.0, 1.0],
        [0.0, float("nan")],
        [float(i) for i in range(202)],
    ):
        with pytest.raises(UsageError):
            summaries.distribution(
                snapshot, narrowed, metric="client.e2e", edges=edges, bins=None
            )
    with pytest.raises(UsageError):
        summaries.distribution(
            snapshot, narrowed, metric="server.e2e", edges=None, bins=3
        )
    with pytest.raises(UsageError):
        summaries.distribution(
            snapshot, narrowed, metric="client.e2e", edges=None, bins=201
        )


def test_one_value_gives_one_bin_that_holds_it() -> None:
    snapshot = snapshot_from_records([request(0)])
    document = summaries.distribution(
        snapshot,
        select(snapshot, RequestFilter()),
        metric="client.e2e",
        edges=None,
        bins=50,
    )
    assert [b["count"] for b in document["bins"]] == [1]


def test_points_draw_all_rows_up_to_the_bound_then_bucket_extremes() -> None:
    rows = [
        request(
            i,
            e2e_latency_ms=float((i * 7919) % 1000),
            status="error" if i % 97 == 0 else "ok",
        )
        for i in range(1200)
    ]
    snapshot = snapshot_from_records([session_record(), *rows])
    sel = select(snapshot, RequestFilter())
    small = summaries.points(
        snapshot, sel, x="start", y="client.e2e", max_points=5000, keep=[]
    )
    assert small["method"] == "all" and small["drawn"] == 1200
    bounded = summaries.points(
        snapshot, sel, x="arrival", y="client.e2e", max_points=100, keep=["r5"]
    )
    assert bounded["method"] == "bucket_extremes"
    ids = {p["request_id"] for p in bounded["points"]}
    assert "r5" in ids
    assert {f"r{i}" for i in range(0, 1200, 97)} <= ids
    drawn_ys = [p["y"] for p in bounded["points"]]
    assert min(drawn_ys) == 0.0 and max(drawn_ys) == 999.0
    assert bounded["drawn"] <= 100 + 1 + 500
    xs = [(p["x_ns"], p["line"]) for p in bounded["points"]]
    assert xs == sorted(xs)


@pytest.mark.parametrize(
    "gaps",
    [[5.0, "x", 1.0], [5.0, math.inf, 1.0], [5.0, None], [math.nan], [5.0, 1.0]],
)
@pytest.mark.parametrize("built", ["dict_rows", "index"])
def test_chunk_counts_count_numeric_gaps_as_the_analyzer_does(
    tmp_path: Path, gaps: list[Any], built: str
) -> None:
    """A gap that is not a finite number is no chunk to the analyzer, so
    the rows, the points and the request detail say what the summary and
    ``streaming_summary`` say (Codex: ``[5.0, "x", 1.0]`` gave 4, not 3)."""
    record = request(0, chunk_interarrival_ms=gaps)
    records = [session_record(), record]
    want = streaming_summary([record])["content_chunks_per_response"]["max"]
    if built == "index":
        (tmp_path / "run.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records)
        )
        snapshot = Harness(tmp_path).build("run.jsonl")
    else:
        snapshot = snapshot_from_records(records)
    sel = select(snapshot, RequestFilter())
    got = page(snapshot, sel, sort="artifact", order="asc", cursor=None, limit=5)
    [row] = summaries.requests_page(snapshot, sel, got)["rows"]
    [point] = summaries.points(
        snapshot, sel, x="end", y="chunk_count", max_points=10, keep=[]
    )["points"]
    detail = summaries.request_detail(snapshot, sel.rows[0], sel)
    streaming = summaries.summary(snapshot, sel)["streaming"]
    assert want == streaming["content_chunks_per_response"]["max"]
    assert row["chunk_count"] == point["y"] == detail["row"]["chunk_count"] == want
    assert detail["gaps"]["n"] + 1 == want and row["gap_count"] == len(gaps)


def test_points_count_rows_without_a_value() -> None:
    snapshot = snapshot_from_records(
        [request(0), request(1, ttft_ms=None), request(2, stream=False)]
    )
    document = summaries.points(
        snapshot,
        select(snapshot, RequestFilter()),
        x="end",
        y="chunk_count",
        max_points=10,
        keep=[],
    )
    assert document["total"] == 1 and document["without_value"] == 2
    assert [p["y"] for p in document["points"]] == [4]
    for x, y in (("middle", "client.e2e"), ("start", "server.e2e")):
        with pytest.raises(UsageError):
            summaries.points(
                snapshot,
                select(snapshot, RequestFilter()),
                x=x,
                y=y,
                max_points=10,
                keep=[],
            )


def test_samples_give_series_by_device_with_gaps_and_buckets() -> None:
    def sample(at: int, **fields: Any) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "event_type": "infer.system_sample",
            "session_id": SESSION,
            "timestamp_ns": at,
            **fields,
        }

    times = [ORIGIN + i * SECOND for i in range(30)] + [ORIGIN + 60 * SECOND]
    records = [session_record()] + [
        sample(t, sampler="psutil", process_rss_bytes=1000 + i)
        for i, t in enumerate(times)
    ]
    records.append(sample(ORIGIN, sampler="nvml", device_id=0, device_used_bytes=5))
    snapshot = snapshot_from_records(records)
    document = summaries.samples(snapshot, window=None, max_points=9)
    assert [s["sampler"] for s in document["series"]] == ["nvml", "psutil"]
    psutil = document["series"][1]
    assert psutil["gaps"] == [
        {"after_ns": ORIGIN + 29 * SECOND, "before_ns": ORIGIN + 60 * SECOND}
    ]
    rss = psutil["fields"]["process_rss_bytes"]
    assert rss["method"] == "bucket_min_max_last"
    assert sum(b["count"] for b in rss["buckets"]) == 31
    assert rss["buckets"][-1]["last"] == 1030
    assert psutil["peaks"]["peak_process_rss_bytes"] == 1030
    windowed = summaries.samples(
        snapshot, window=TimeWindow(ORIGIN, ORIGIN + 2 * SECOND), max_points=9
    )
    assert windowed["total"] == 3
