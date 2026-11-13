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
import shutil
import time
from array import array
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import jsonschema  # type: ignore[import-untyped, unused-ignore]
import pytest
from referencing import Registry, Resource

from stormlog.infer.analysis import analyze_inference_events
from stormlog.infer.events import InferenceSystemSample
from stormlog.infer.latency_report import CLIENT_METRICS, spread
from stormlog.studio import summaries
from stormlog.studio.contracts import (
    AnalysisState,
    LatencyRange,
    NotResident,
    RequestFilter,
    TimeWindow,
    UsageError,
    columnar,
    envelope,
    to_wire,
)
from stormlog.studio.metric_catalog import metric_catalog, panels
from stormlog.studio.selection import (
    case_facts,
    page,
    parse_filter,
    select,
    selection_of,
    warm,
)
from tests.studio.artifacts import (
    ANALYZE_FAILS,
    MULTI_SESSION,
    PROBLEM_LINES,
    STANDARD_ARTIFACTS,
    UNSUPPORTED,
)
from tests.studio.conftest import StandardArtifacts
from tests.studio.test_index import Harness
from tests.studio.test_selection import (
    ORIGIN,
    SECOND,
    SESSION,
    closed_case,
    column_snapshot_of,
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
    assert schema_errors("chunks", _wire("chunks", first, snapshot)) == []
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
    assert (
        schema_errors("distribution", _wire("distribution", document, snapshot)) == []
    )
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
    assert schema_errors("points", _wire("points", bounded, snapshot)) == []


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
    assert schema_errors("samples", _wire("samples", document, snapshot)) == []
    windowed = summaries.samples(
        snapshot, window=TimeWindow(ORIGIN, ORIGIN + 2 * SECOND), max_points=9
    )
    assert windowed["total"] == 3


# --- The example documents (design §10.2) ---------------------------------------------

EXAMPLE_DIR = SCHEMA_DIR / "examples"
EXAMPLE_PATH = "runs/tokens_mixed.jsonl"
PLANNED_CASE = "c8_in64_out16_planned"
# What differs between two builds of the same artifact on two checkouts: the
# session record names the Stormlog version, so the file's size and hash move.
VOLATILE = ("sha256", "size", "bytes_read")


def example_artifact(directory: Path) -> Path:
    """C5's tokens_mixed run with a planned case it never ran and psutil samples.

    The planned case shows choice 16, the samples the client memory and
    samples documents; the requests are C5's, untouched.
    """
    source = STANDARD_ARTIFACTS["tokens_mixed"](directory)
    lines = source.read_bytes().split(b"\n")
    starts = []
    for i, line in enumerate(lines):
        if b'"infer.workload"' in line:
            record = json.loads(line)
            planned = dict(record["cases"][0], case_id=PLANNED_CASE)
            record["cases"].append(planned)
            lines[i] = json.dumps(record).encode()
        if b'"infer.request"' in line:
            starts.append(int(json.loads(line)["started_at_ns"]))
    session = json.loads(lines[0])["session_id"]
    samples = [
        InferenceSystemSample(
            session_id=session,
            timestamp_ns=min(starts) + k * SECOND // 4,
            sampler="psutil",
            process_rss_bytes=180_000_000 + 65_536 * k * (k % 7),
        ).to_record()
        for k in range(40)
        if k != 30  # one missed sample: a gap
    ]
    body = b"\n".join(line for line in lines if line)
    extra = b"".join(json.dumps(s).encode() + b"\n" for s in samples)
    target = directory / "tokens_mixed.example.jsonl"
    target.write_bytes(body + b"\n" + extra)
    return target


def build_examples(directory: Path) -> dict[str, Any]:
    """Every C2 example document, from the real code over ``example_artifact``."""
    path = example_artifact(directory)
    snapshot = snapshot_of(path, rel=EXAMPLE_PATH, generation=3)
    case = sorted(case_facts(snapshot).by_case)[0]
    whole = select(snapshot, RequestFilter(cases=(case,)))
    statuses = snapshot.column("status")
    ok_row = next(i for i in whole.rows if statuses[i] == "ok")
    failed = next(i for i in whole.rows if statuses[i] != "ok")
    first = page(snapshot, whole, sort="client.e2e", order="desc", cursor=None, limit=4)
    report = analyze_inference_events(path)
    analysis = AnalysisState("ready", None, 0, report, snapshot.bytes_read)
    keep = [snapshot.requests[failed].request_id]
    bodies: dict[str, tuple[dict[str, Any], bool]] = {
        "summary": (summaries.summary(snapshot, whole), True),
        "requests": (summaries.requests_page(snapshot, whole, first), True),
        "request": (summaries.request_detail(snapshot, ok_row, whole), True),
        "cases": (summaries.cases(snapshot), True),
        "chunks": (summaries.chunks(snapshot, ok_row, offset=0, limit=8), True),
        "distribution": (
            summaries.distribution(
                snapshot, whole, metric="client.e2e", edges=None, bins=12
            ),
            True,
        ),
        "points": (
            summaries.points(
                snapshot, whole, x="start", y="client.e2e", max_points=16, keep=keep
            ),
            True,
        ),
        "samples": (summaries.samples(snapshot, window=None, max_points=12), True),
        "metric_catalog": (metric_catalog(), False),
        "artifact_capabilities": (panels(snapshot, analysis), True),
    }
    examples = {
        name: to_wire(envelope(name, body, snapshot if scoped else None))
        for name, (body, scoped) in bodies.items()
    }
    examples["selection_block"] = to_wire(summaries.selection_block(whole, snapshot))
    return examples


def _example_file(name: str) -> Path:
    return EXAMPLE_DIR / f"studio_{name}_v1.example.json"


def write_examples(directory: Path) -> list[Path]:
    """Regenerate the committed examples: ``python -m tests.studio.test_summaries``."""
    written = []
    for name, document in sorted(build_examples(directory).items()):
        path = _example_file(name)
        text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


def _same(left: Any, right: Any, where: str = "") -> list[str]:
    """Where two documents differ; floats within 1e-9 relative (F-8)."""
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return [f"{where}: keys {sorted(set(left) ^ set(right))}"]
        return [d for k in left for d in _same(left[k], right[k], f"{where}.{k}")]
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{where}: {len(left)} items, not {len(right)}"]
        pairs = enumerate(zip(left, right))
        return [d for i, (a, b) in pairs for d in _same(a, b, f"{where}[{i}]")]
    if isinstance(left, float) and isinstance(right, float):
        close = math.isclose(left, right, rel_tol=1e-9)
        return [] if close else [f"{where}: {left} != {right}"]
    return [] if left == right else [f"{where}: {left!r} != {right!r}"]


def _stable(document: Any) -> Any:
    copy = json.loads(json.dumps(document))
    for key in VOLATILE:
        copy.get("artifact", {}).pop(key, None)
    # The analysed prefix is the file's bytes_read, which moves with it.
    analysis = copy.get("panels", {}).get("analysis", {})
    (analysis.get("detail") or {}).pop("analysed_bytes", None)
    return copy


C2_EXAMPLES = (
    "selection_block",
    "summary",
    "requests",
    "request",
    "cases",
    "chunks",
    "distribution",
    "points",
    "samples",
    "metric_catalog",
    "artifact_capabilities",
)


def test_the_examples_are_what_the_code_writes(tmp_path: Path) -> None:
    built = build_examples(tmp_path)
    assert sorted(built) == sorted(C2_EXAMPLES)
    for name, document in built.items():
        assert schema_errors(name, document) == [], name
        committed = json.loads(_example_file(name).read_text(encoding="utf-8"))
        assert _same(_stable(document), _stable(committed)) == [], name


def test_the_cases_example_shows_a_planned_case_never_run() -> None:
    document = json.loads(_example_file("cases").read_text(encoding="utf-8"))
    planned = [c for c in document["cases"] if not c["observed"]]
    assert [c["case_id"] for c in planned] == [PLANNED_CASE]
    assert planned[0]["summary"]["selection"]["size"] == 0


# --- Columnar and row paths (Fable's F-2) ----------------------------------------------


def _documents(snapshot: Any, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The summary of the default filter, of each case and of the ok rows,
    by the columnar path and then by rows (``columnar`` giving None), as
    JSON text: floats by repr, so -0.0, NaN and 1 vs 1.0 all count."""
    filters = [RequestFilter(), RequestFilter(statuses=("ok",))]
    filters += [RequestFilter(cases=(c,)) for c in case_facts(snapshot).by_case]
    texts = []
    for by_rows in (False, True):
        with monkeypatch.context() as patch:
            if by_rows:
                patch.setattr(summaries, "columnar", lambda table: None)
            texts.append(
                json.dumps([_summary(snapshot, f) for f in filters], sort_keys=True)
            )
    return texts


def _indexed(path: Path, directory: Path) -> Any:
    """C1's real index of ``path`` (a copy in ``directory``)."""
    shutil.copyfile(path, directory / path.name)
    return Harness(directory).build(path.name)


@pytest.mark.parametrize("name", STUDIO_ARTIFACTS)
def test_the_columnar_summary_gives_the_row_summarys_floats(
    name: str,
    studio_artifacts: StandardArtifacts,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = studio_artifacts[name]
    stand_in = snapshot_of(path)
    assert columnar(stand_in.table) is not None
    columnar_text, rows_text = _documents(stand_in, monkeypatch)
    assert columnar_text == rows_text
    real = _indexed(path, tmp_path)
    if real.status in ("ready", "partial"):
        assert columnar(real.table) is not None
        columnar_text, rows_text = _documents(real, monkeypatch)
        assert columnar_text == rows_text


NAN, INF = float("nan"), float("inf")
BIG = 2**53 + 1
# One request per hostile value: latencies, gaps, token counts, flags and
# times that are NaN, infinite, -0.0, ints past 2**53 or 2**64, null,
# absent, booleans or text, each where a rule reads it.
HOSTILE: list[dict[str, Any]] = [
    {"ttft_ms": NAN, "e2e_latency_ms": INF},
    {"ttft_ms": -0.0, "first_chunk_latency_ms": -0.0, "e2e_latency_ms": 0.0},
    {"ttft_ms": 0.0, "e2e_latency_ms": -0.0},
    {"ttft_ms": BIG, "e2e_latency_ms": 2**64},
    {"ttft_ms": True, "e2e_latency_ms": "500"},
    {"chunk_interarrival_ms": [NAN, INF, -INF, -0.0, 0.0, 0, BIG, True, None]},
    {"chunk_interarrival_ms": [0.0, -0.0, "7", [], {}, 1e308, 5e-324]},
    {"chunk_interarrival_ms": "abc"},
    {"chunk_interarrival_ms": 5},
    {"chunk_interarrival_ms": None},
    {"chunk_interarrival_ms": []},
    {"output_tokens": INF, "total_tokens": NAN},
    {"output_tokens": NAN, "prompt_tokens": -INF},
    {"output_tokens": True, "prompt_tokens": False, "total_tokens": True},
    {"output_tokens": BIG, "total_tokens": 2**64 + 3},
    {"output_tokens": 7.9, "total_tokens": -0.0},
    {"output_tokens": "12", "total_tokens": None},
    {"output_tokens": None, "output_token_source": None},
    {"output_token_source": 5, "prompt_token_source": True},
    {"output_token_exact": 1, "prompt_token_exact": "true"},
    {"output_token_exact": None, "prompt_token_source": None},
    {"stream": 1},
    {"stream": "yes", "ttft_ms": 3},
    {"stream": None},
    {"started_at_ns": float(ORIGIN), "ended_at_ns": ORIGIN + 2**63},
    {"started_at_ns": True, "ended_at_ns": None},
    {"started_at_ns": NAN},
    {"status": "timeout", "e2e_latency_ms": -0.0},
    {"status": "timeout", "e2e_latency_ms": NAN},
    {"status": "cancelled", "e2e_latency_ms": None, "ended_at_ns": BIG},
    {"status": None},
    {"status": 5},
    {"status": True},
    {"status": "dropped", "started_at_ns": None},
]
# Keys left out of a request entirely, beside the same keys set to null.
ABSENT = ("output_token_source", "prompt_token_source", "output_tokens", "stream")


def _hostile_records() -> list[dict[str, Any]]:
    rows = [request(i, **values) for i, values in enumerate(HOSTILE)]
    for key in ABSENT:
        record = request(len(rows))
        del record[key]
        rows.append(record)
    rows += [request(len(rows) + i) for i in range(4)]
    return [session_record(), *rows]


def test_the_columnar_summary_gives_the_row_summarys_floats_on_hostile_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _hostile_records()
    stand_in = snapshot_from_records(records)
    columnar_text, rows_text = _documents(stand_in, monkeypatch)
    assert columnar_text == rows_text
    path = tmp_path / "hostile.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    real = Harness(tmp_path).build(path.name)
    assert real.status == "ready"
    assert len(real.table) == len(records) - 1
    columnar_text, rows_text = _documents(real, monkeypatch)
    assert columnar_text == rows_text
    # The values reach the summary: -0.0 stays signed, null and absent
    # sources stay apart.
    document = json.loads(columnar_text)[0]
    assert document["tokens"]["output"]["by_source"]["None"] == 1
    assert document["tokens"]["output"]["by_source"]["unknown"] == 1
    assert "-0.0" in columnar_text


# --- Parity with infer analyze (§3.4) --------------------------------------------------

# §3.4 compares every case of every single-session fixture Studio reads
# whole. Not compared: the fixtures Studio refuses (UNSUPPORTED: no text
# inference run to summarize), and those with lines Studio counts as
# problems and leaves out where the analyzer pools them or fails
# (PROBLEM_LINES: unsupported_version's v2 client request is a row to the
# analyzer only, Fable's F-8; truncated_mid_line's cut line fails it).
PARITY_ARTIFACTS = sorted(set(STUDIO_ARTIFACTS) - set(UNSUPPORTED) - set(PROBLEM_LINES))


def _analyzer_cases(path: Path, tmp_path: Path) -> dict[str, Any]:
    """The analyzer's cases; for a cut file, of its complete lines (decision 12(a))."""
    data = path.read_bytes()
    complete = data[: data.rfind(b"\n") + 1]
    if complete != data:
        prefix = tmp_path / f"{path.stem}.complete.jsonl"
        prefix.write_bytes(complete)
        path = prefix
    cases = analyze_inference_events(path)["cases"]
    assert isinstance(cases, dict)
    return cases


def _parity_view(body: Mapping[str, Any]) -> dict[str, Any]:
    counts = dict(body["counts"])
    for key in ("server_admitted", "server_evidence_coverage", "server_evidence"):
        counts.pop(key)
    latency = body["latency"]
    return {
        "counts": counts,
        "intervals": body["intervals"],
        "throughput": body["throughput"],
        "streaming": body["streaming"],
        "latency.rule": latency["rule"],
        "latency.unsuccessful": latency["unsuccessful"],
        **{f"latency.metrics.{k}": latency["metrics"][k] for k in CLIENT_METRICS},
        "tokens": body["tokens"],
    }


def _analyzer_view(case: Mapping[str, Any]) -> dict[str, Any]:
    population = dict(case["population"])
    for key in ("server_admitted", "server_evidence_coverage"):
        population.pop(key)
    latency = case["latency"]
    return {
        "counts": population,
        "intervals": case["intervals"],
        "throughput": case["throughput"],
        "streaming": case["streaming"],
        "latency.rule": latency["rule"],
        "latency.unsuccessful": latency["unsuccessful"],
        **{f"latency.metrics.{k}": latency["metrics"][k] for k in CLIENT_METRICS},
        "tokens": case["tokens"]["coverage"],
    }


@pytest.mark.parametrize("name", PARITY_ARTIFACTS)
def test_every_case_summary_equals_the_analyzers_case_block(
    name: str, studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    """§3.4: block by block, exactly; both run in this process, so F-8's
    1-ulp coverage difference across environments cannot arise."""
    path = studio_artifacts[name]
    snapshot = snapshot_of(path)
    analyzer = _analyzer_cases(path, tmp_path)
    studio = {
        case: _summary(snapshot, RequestFilter(cases=(case,))) for case in analyzer
    }
    assert sorted(case_facts(snapshot).by_case) == sorted(analyzer)
    for case, block in analyzer.items():
        body = studio[case]
        assert body["selection"]["whole_case"] == case
        mine, theirs = _parity_view(body), _analyzer_view(block)
        for key in theirs:
            assert json.dumps(mine[key], sort_keys=True) == json.dumps(
                theirs[key], sort_keys=True
            ), f"{name} {case} {key}"
        assert body["counts"]["server_evidence"] == "not_indexed"
        assert body["counts"]["server_admitted"] is None


def _cases_by_select(snapshot: Any) -> dict[str, Any]:
    """``summaries.cases`` as it was: one ``select`` per case."""
    facts = case_facts(snapshot)
    planned = set(facts.planned)
    entries = []
    for case_id in sorted(planned | set(facts.by_case)):
        sel = select(snapshot, RequestFilter(cases=(case_id,)))
        entries.append(
            {
                "case_id": case_id,
                "planned": case_id in planned,
                "observed": case_id in facts.by_case,
                "summary": summaries.summary(snapshot, sel),
            }
        )
    return {"comparison_scope": "full_measured_case", "cases": entries}


@pytest.mark.parametrize("name", STUDIO_ARTIFACTS)
@pytest.mark.parametrize("kind", ["rows", "columns"])
def test_cases_group_rows_once_and_equal_a_select_per_case(
    name: str,
    kind: str,
    studio_artifacts: StandardArtifacts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fable's P3-2: the case facts' grouping replaces a whole-table scan
    per case, and every fixture's cases document is the same JSON text."""
    path = studio_artifacts[name]
    snapshot = snapshot_of(path) if kind == "rows" else column_snapshot_of(path)
    expected = _cases_by_select(snapshot)
    facts = case_facts(snapshot)
    for case_id in summaries.case_ids(snapshot):
        f = RequestFilter(cases=(case_id,))
        rows = array("q", facts.by_case.get(case_id, ()))
        assert selection_of(snapshot, f, rows) == select(snapshot, f), case_id
    scans: list[RequestFilter] = []

    def counted(snapshot: Any, f: RequestFilter) -> Any:
        scans.append(f)
        return select(snapshot, f)

    monkeypatch.setattr(summaries, "select", counted)
    document = summaries.cases(snapshot)
    assert scans == []
    assert json.dumps(document, sort_keys=True) == json.dumps(expected, sort_keys=True)


def test_parity_covers_every_fixture_studio_reads_whole() -> None:
    # The fixtures C5 added later are compared (samples, open loop,
    # stopped, interrupted, timeouts, a NaN gap); none the analyzer refuses.
    assert {
        "samples",
        "samples_psutil",
        "open_loop",
        "stopped_early",
        "interrupted",
        "timeouts",
        "nan_gap",
    } <= set(PARITY_ARTIFACTS)
    assert ANALYZE_FAILS.isdisjoint(PARITY_ARTIFACTS)
    left_out = set(STUDIO_ARTIFACTS) - set(PARITY_ARTIFACTS)
    assert left_out == set(UNSUPPORTED) | set(PROBLEM_LINES)


# --- 500,000 requests ------------------------------------------------------------------


@pytest.fixture(scope="module")
def large_snapshot(large_studio_artifact: Path) -> Any:
    snapshot = column_snapshot_of(large_studio_artifact)
    case_facts(snapshot)  # warmed once per snapshot, as on the builder pool
    return snapshot


@pytest.mark.studio_large
def test_requests_page_takes_under_2_s_on_500000_requests(large_snapshot: Any) -> None:
    started = time.perf_counter()
    sel = select(large_snapshot, RequestFilter())
    got = page(
        large_snapshot, sel, sort="client.ttft", order="asc", cursor=None, limit=1000
    )
    summaries.requests_page(large_snapshot, sel, got)
    elapsed = time.perf_counter() - started
    print(json.dumps({"select + page + requests_page": elapsed}))
    assert elapsed < 2.0


@pytest.fixture(scope="module")
def large_index_snapshot(large_studio_artifact: Path) -> Any:
    """C1's real ArtifactIndexes over the large file (Fable's F-2: time the
    index Studio serves, not a stand-in)."""
    snapshot = Harness(large_studio_artifact.parent).build(large_studio_artifact.name)
    warm(snapshot)  # as C4's publish listener does, on the builder pool
    return snapshot


@pytest.mark.studio_large
def test_summary_takes_under_2_s_on_500000_requests(large_index_snapshot: Any) -> None:
    # The first summary of the warmed snapshot.
    snapshot = large_index_snapshot
    assert snapshot.status == "ready" and columnar(snapshot.table) is not None
    sel = select(snapshot, RequestFilter())
    started = time.perf_counter()
    summaries.summary(snapshot, sel)
    elapsed = time.perf_counter() - started
    print(json.dumps({"summary": elapsed}))
    assert elapsed < 2.0


if __name__ == "__main__":
    import tempfile

    for written in write_examples(Path(tempfile.mkdtemp())):
        print(written)
