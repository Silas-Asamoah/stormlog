"""The step-0 contract: wire helpers, rows and columns, errors, imports."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import math
import numbers
import re
import subprocess
import sys
from array import array
from collections.abc import Iterator, Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any, get_args

import jsonschema  # type: ignore[import-untyped, unused-ignore]
import pytest
from referencing import Registry, Resource

from stormlog.infer import artifact_lines
from stormlog.infer.diagnosis_inputs import record_id
from stormlog.infer.latency_report import CLIENT_METRICS
from stormlog.infer.slo import CriterionValue, client_values
from stormlog.studio import contracts
from stormlog.studio.contracts import (
    AnalysisCache,
    BusyError,
    ConflictError,
    DictRow,
    DictTable,
    FileIdentity,
    ForbiddenError,
    IndexRegistry,
    IndexSnapshot,
    InvalidInputError,
    LineProblem,
    LineRef,
    NotFoundError,
    NotReady,
    SessionInfo,
    StudioError,
    UnauthorizedError,
    UsageError,
    artifact_block,
    criterion_record,
    encode_json,
    envelope,
    finite_copy,
    selection_fingerprint,
    to_wire,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = REPO_ROOT / "stormlog" / "studio" / "contracts.py"
ARTIFACT_LINES = REPO_ROOT / "stormlog" / "infer" / "artifact_lines.py"
ALLOWED_STORMLOG_IMPORTS = {
    "stormlog.infer.artifact_lines",
    "stormlog.infer.populations",
    "stormlog.infer.slo",
}
SCHEMA_DIR = REPO_ROOT / "docs" / "schemas"
EXAMPLE_DIR = SCHEMA_DIR / "examples"
SCHEMA_BASE = "https://github.com/Silas-Asamoah/stormlog/docs/schemas/"
STUDIO_SCHEMAS = sorted(path.name for path in SCHEMA_DIR.glob("studio_*.schema.json"))
SESSION = "5f0c1d2e3a4b5c6d7e8f901234567890"
INDEX_ID = "9a3e5b7c1d2f4a6b8c0e2d4f6a8b0c1d"


def request_record(**overrides: Any) -> dict[str, Any]:
    """A client request as InferenceRequestEvent.to_record writes it."""
    record: dict[str, Any] = {
        "schema_version": 1,
        "event_type": "infer.request",
        "timestamp_ns": 1788609625123456789,
        "session_id": SESSION,
        "request_id": "measured_053",
        "case_id": "c8_in4096_out128",
        "phase": "measured",
        "started_at_ns": 1788609625123456789,
        "ended_at_ns": 1788609626022756789,
        "status": "ok",
        "stream": True,
        "e2e_latency_ms": 899.3,
        "ttft_ms": 628.8,
        "first_chunk_latency_ms": 620.8,
        "chunk_interarrival_ms": [18.1] * 15,
        "output_tokens": 128,
        "output_token_source": "server_usage",
        "output_token_exact": True,
        "error_message": None,
    }
    record.update(overrides)
    return record


def snapshot(**overrides: Any) -> IndexSnapshot:
    rows = [DictRow(request_record(), line=842)]
    values: dict[str, Any] = {
        "path": "runs/042.jsonl",
        "index_id": INDEX_ID,
        "generation": 7,
        "status": "ready",
        "failure": None,
        "identity": FileIdentity(device=2049, inode=131, size=1_048_576, mtime_ns=1),
        "bytes_read": 1_048_576,
        "pending_tail_bytes": 0,
        "prefix_sha256": "a" * 64,
        "sha256": "a" * 64,
        "growing": False,
        "origin_ns": 1788609600000000000,
        "session": SessionInfo(
            session_id=SESSION,
            status="completed",
            started_at_ns=1788609600000000000,
            ended_at_ns=1788609662000000000,
            terminal=True,
            config={},
        ),
        "requests": rows,
        "table": DictTable(rows),
        "by_request_id": {"measured_053": (0,)},
        "samples": (),
        "kept": {},
        "request_links": {},
        "refs": {},
        "inventory": {"infer.request": {1: 1}},
        "problems": (),
        "problem_count": 0,
        "limits_hit": frozenset({"samples", "gaps"}),
        "gaps_resident": True,
        "error_messages_resident": True,
    }
    values.update(overrides)
    return IndexSnapshot(**values)


# --- to_wire ---------------------------------------------------------------


def test_ns_integers_become_decimal_strings() -> None:
    wired = to_wire({"started_at_ns": 1788609625123456789, "line": 842})
    assert wired == {"started_at_ns": "1788609625123456789", "line": 842}


def test_ns_rule_covers_lists_and_nested_lists_under_the_key() -> None:
    wired = to_wire({"edges_ns": [1, [2, 3]], "x_ms": [1.5]})
    assert wired == {"edges_ns": ["1", ["2", "3"]], "x_ms": [1.5]}


def test_nearest_key_decides_inside_embedded_documents() -> None:
    document = {"report_ns": {"count": 3, "ended_at_ns": 5, "rows": [{"n": 4}]}}
    assert to_wire(document) == {
        "report_ns": {"count": 3, "ended_at_ns": "5", "rows": [{"n": 4}]}
    }


def test_unsafe_integers_become_strings_anywhere() -> None:
    limit = contracts.MAX_SAFE_INTEGER
    wired = to_wire({"a": limit, "b": limit + 1, "c": -limit - 1, "d": [-limit]})
    assert wired == {
        "a": limit,
        "b": str(limit + 1),
        "c": str(-limit - 1),
        "d": [-limit],
    }


def test_booleans_none_and_floats_pass_unchanged_under_ns_keys() -> None:
    wired = to_wire({"flag_ns": True, "none_ns": None, "seconds_ns": 1.5})
    assert wired == {"flag_ns": True, "none_ns": None, "seconds_ns": 1.5}


def test_containers_become_plain_json_types() -> None:
    wired = to_wire({"inventory": {1: 12}, "t": (1, 2), "a": array("d", [0.5])})
    assert wired == {"inventory": {"1": 12}, "t": [1, 2], "a": [0.5]}


def test_to_wire_does_not_mutate_its_input() -> None:
    document = {"started_at_ns": 5, "rows": [{"ended_at_ns": 6}]}
    to_wire(document)
    assert document == {"started_at_ns": 5, "rows": [{"ended_at_ns": 6}]}


class Tick:
    """An integral number that is not an int, as numpy's int64 is."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __int__(self) -> int:
        return self.value

    def __index__(self) -> int:
        return self.value


numbers.Integral.register(Tick)


class Grid:
    """An array-like with tolist, as a numpy array is."""

    def tolist(self) -> list[list[Any]]:
        return [[1, 2.5], [Fraction(1, 4)]]


class SelfList:
    """A tolist that answers itself must not recurse forever."""

    def tolist(self) -> SelfList:
        return self


def test_other_integral_and_real_numbers_become_int_and_float() -> None:
    wired = to_wire({"n": Tick(3), "at_ns": Tick(5), "share": Fraction(1, 4)})
    assert wired == {"n": 3, "at_ns": "5", "share": 0.25}
    assert type(wired["n"]) is int and type(wired["share"]) is float
    assert to_wire(Tick(contracts.MAX_SAFE_INTEGER + 1)) == str(2**53)


def test_objects_with_tolist_are_wired_through_it() -> None:
    assert to_wire({"grid_ns": Grid()}) == {"grid_ns": [["1", 2.5], [0.25]]}
    with pytest.raises(TypeError):
        to_wire(SelfList())


def test_numpy_values_are_wired_as_python_values() -> None:
    np: Any = pytest.importorskip("numpy")
    document = {
        "count": np.int64(120),
        "started_at_ns": np.int64(1788609625123456789),
        "p95_ms": np.float64(616.4),
        "ok": np.bool_(True),
        "edges_ns": np.array([1, 2], dtype=np.int64),
        "values_ms": np.array([[0.5, 1.5]]),
        "flags": np.array([True, False]),
        "n32": np.int32(7),
        "x32": np.float32(0.5),
    }
    wired = to_wire(document)
    assert wired == {
        "count": 120,
        "started_at_ns": "1788609625123456789",
        "p95_ms": 616.4,
        "ok": True,
        "edges_ns": ["1", "2"],
        "values_ms": [[0.5, 1.5]],
        "flags": [True, False],
        "n32": 7,
        "x32": 0.5,
    }
    assert type(wired["ok"]) is bool and type(wired["count"]) is int
    assert type(wired["x32"]) is float
    assert json.loads(encode_json(document)) == wired


@pytest.mark.parametrize("value", [b"bytes", {1.5: 1}, {"s": {1, 2}}, object()])
def test_to_wire_refuses_values_json_cannot_carry_exactly(value: Any) -> None:
    with pytest.raises(TypeError):
        to_wire(value)


@pytest.mark.parametrize(
    "value",
    [
        b"ab",
        bytearray(b"ab"),
        memoryview(b"ab"),
        memoryview(bytearray(b"ab")),
        memoryview(array("b", [1, 2])),
        memoryview(b"ab").cast("c"),
    ],
)
def test_bytes_and_byte_memoryviews_have_no_wire_form(value: Any) -> None:
    with pytest.raises(TypeError):
        to_wire(value)
    with pytest.raises(TypeError):
        to_wire({"payload": [value]})


def test_a_memoryview_of_numbers_is_a_list() -> None:
    assert to_wire(memoryview(array("q", [1, 2]))) == [1, 2]
    assert to_wire({"t_ns": memoryview(array("q", [3]))}) == {"t_ns": ["3"]}


# --- finite_copy and encode_json ---------------------------------------------


def test_finite_copy_nulls_nan_and_both_infinities_and_names_them() -> None:
    record: dict[str, Any] = {
        "ttft_ms": math.nan,
        "e2e_latency_ms": math.inf,
        "chunk_interarrival_ms": [1.0, -math.inf, 2.0],
        "usage": {"x": math.nan, "y": 1},
        "status": "ok",
    }
    copy, changed = finite_copy(record)
    assert copy == {
        "ttft_ms": None,
        "e2e_latency_ms": None,
        "chunk_interarrival_ms": [1.0, None, 2.0],
        "usage": {"x": None, "y": 1},
        "status": "ok",
    }
    assert changed == [
        "ttft_ms",
        "e2e_latency_ms",
        "chunk_interarrival_ms[1]",
        "usage.x",
    ]
    assert math.isnan(record["ttft_ms"])


def test_finite_copy_walks_arrays_and_memoryviews() -> None:
    gaps = array("d", [1.0, math.nan, 2.0])
    record: dict[str, Any] = {
        "chunk_interarrival_ms": gaps,
        "view_ms": memoryview(array("d", [math.inf, 3.0])),
        "nested": {"a": (array("f", [math.nan]),)},
        "text": "nan",
        "raw": b"\x00",
    }
    copy, changed = finite_copy(record)
    assert copy["chunk_interarrival_ms"] == [1.0, None, 2.0]
    assert copy["view_ms"] == [None, 3.0]
    assert copy["nested"] == {"a": [[None]]}
    assert copy["text"] == "nan" and copy["raw"] == b"\x00"
    assert changed == [
        "chunk_interarrival_ms[1]",
        "view_ms[0]",
        "nested.a[0][0]",
    ]
    assert math.isnan(gaps[1])
    encode_json({key: copy[key] for key in ("chunk_interarrival_ms", "view_ms")})


def test_finite_copy_of_a_finite_record_changes_nothing() -> None:
    record = request_record()
    copy, changed = finite_copy(record)
    assert copy == record and changed == []
    assert copy["chunk_interarrival_ms"] is not record["chunk_interarrival_ms"]


def test_encode_json_applies_to_wire_and_writes_utf8() -> None:
    body = encode_json({"started_at_ns": 1788609625123456789, "text": "café"})
    assert json.loads(body) == {"started_at_ns": "1788609625123456789", "text": "café"}
    assert "café".encode() in body


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_encode_json_refuses_non_finite_numbers(bad: float) -> None:
    with pytest.raises(ValueError):
        encode_json({"value_ms": bad})


# --- envelope and artifact_block ---------------------------------------------


def test_envelope_without_a_snapshot_has_no_artifact_block() -> None:
    document = envelope("health", {"ok": True, "instance_id": INDEX_ID})
    assert document == {
        "format": "stormlog.studio.health",
        "schema_version": 1,
        "ok": True,
        "instance_id": INDEX_ID,
    }
    assert list(document)[:2] == ["format", "schema_version"]


def test_envelope_with_a_snapshot_carries_its_artifact_block() -> None:
    snap = snapshot()
    document = envelope("summary", {"freshness": {}}, snap)
    assert document["format"] == "stormlog.studio.summary"
    assert document["schema_version"] == 1
    assert document["artifact"] == artifact_block(snap)
    assert list(document) == ["format", "schema_version", "artifact", "freshness"]


@pytest.mark.parametrize("key", ["format", "schema_version", "artifact"])
def test_envelope_refuses_a_body_that_sets_envelope_keys(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        envelope("summary", {key: 1})


@pytest.mark.parametrize("name", ["", "Summary", "studio.summary", "1x", "a-b"])
def test_envelope_refuses_bad_names(name: str) -> None:
    with pytest.raises(ValueError):
        envelope(name, {})


def test_artifact_block_lists_the_section_5_1_fields() -> None:
    block = artifact_block(snapshot())
    assert block == {
        "path": "runs/042.jsonl",
        "index_id": INDEX_ID,
        "generation": 7,
        "status": "ready",
        "size": 1_048_576,
        "bytes_read": 1_048_576,
        "sha256": "a" * 64,
        "growing": False,
        "session_id": SESSION,
        "session_status": "completed",
        "terminal": True,
        "origin_ns": 1788609600000000000,
        "gaps_resident": True,
        "error_messages_resident": True,
        "limits_hit": ["gaps", "samples"],
        "cut": None,
    }
    assert to_wire(block)["origin_ns"] == "1788609600000000000"


def test_a_snapshot_names_its_cut_and_line_count() -> None:
    plain = snapshot()
    assert (plain.cut, plain.line_count) == (None, None)
    ref = LineRef(line=7, offset=512, length=90, sha256="b" * 64)
    cut = snapshot(cut=ref, line_count=7, status="partial", limits_hit={"rows"})
    assert artifact_block(cut)["cut"] == {
        "line": 7,
        "offset": 512,
        "length": 90,
        "sha256": "b" * 64,
    }
    block = to_wire(artifact_block(cut))
    assert _errors("studio_artifact_block_v1.schema.json", block) == []
    bad = dict(block, cut={"line": 7})
    assert _errors("studio_artifact_block_v1.schema.json", bad)


def test_a_snapshot_counts_its_problems_by_code() -> None:
    assert dict(snapshot().problem_counts) == {}
    counted = snapshot(problem_count=3, problem_counts={"not_json": 2, "oversized": 1})
    assert sum(counted.problem_counts.values()) == counted.problem_count
    fields = {f.name for f in dataclasses.fields(IndexSnapshot)}
    assert {"problem_counts", "cut", "line_count"} <= fields


def test_contexts_is_a_limit_of_the_contract_and_the_schema() -> None:
    assert contracts.Limits().context_bytes == 16 * 2**20
    hits = set(get_args(contracts.LimitHit))
    assert "contexts" in hits
    schema = _load(SCHEMA_DIR / "studio_artifact_block_v1.schema.json")
    assert set(schema["properties"]["limits_hit"]["items"]["enum"]) == hits
    block = to_wire(artifact_block(snapshot(limits_hit=frozenset({"contexts"}))))
    assert _errors("studio_artifact_block_v1.schema.json", block) == []


def test_internal_is_a_failure_of_the_contract_and_the_schema() -> None:
    failures = set(get_args(contracts.IndexFailure))
    assert "internal" in failures
    schema = _load(SCHEMA_DIR / "studio_artifact_v1.schema.json")
    enum = schema["properties"]["index"]["properties"]["failure"]["enum"]
    assert set(enum) == failures | {None}


def test_artifact_block_without_a_session() -> None:
    block = artifact_block(snapshot(session=None, origin_ns=None, sha256=None))
    assert block["session_id"] is None
    assert block["session_status"] is None
    assert block["terminal"] is False
    assert block["origin_ns"] is None and block["sha256"] is None


# --- rows and columns --------------------------------------------------------


def test_dict_row_exposes_identity_client_values_and_gaps() -> None:
    record = request_record(chunk_interarrival_ms=[1.0, "x", 2])
    row = DictRow(record, line=842)
    assert dict(row) == record
    assert row.ref.line == 842 and row.ref.offset == 0
    assert (row.session_id, row.request_id) == (SESSION, "measured_053")
    assert (row.case_id, row.phase) == ("c8_in4096_out128", "measured")
    assert row.record_id == record_id(record) == "infer.request/measured_053"
    assert row.client == client_values(record)
    assert row.gap_count == 3
    gaps = row.load_gaps()
    assert gaps[0] == 1.0 and math.isnan(gaps[1]) and gaps[2] == 2.0
    assert row.load_error_message() is None


def test_dict_row_identity_rule_for_missing_fields() -> None:
    row = DictRow({"status": "error", "error_message": "boom"})
    assert (row.session_id, row.request_id) == ("unknown", "unknown")
    assert (row.case_id, row.phase) == ("unknown", "unknown")
    assert row.record_id is None and row.gap_count == 0
    assert row.load_gaps() == () and row.load_error_message() == "boom"


def test_dict_row_reads_a_null_identity_as_unknown() -> None:
    nulls = {"session_id": None, "request_id": None, "case_id": None, "phase": None}
    row = DictRow(request_record(**nulls))
    assert (row.session_id, row.request_id) == ("unknown", "unknown")
    assert (row.case_id, row.phase) == ("unknown", "unknown")
    assert contracts.identity_text(None) == "unknown"
    assert contracts.identity_text(0) == "0" and contracts.identity_text("") == ""


def test_dict_row_record_id_matches_diagnosis_inputs_for_v2_records() -> None:
    record = request_record(schema_version=2, event_id="evt-1")
    assert DictRow(record).record_id == record_id(record) == "evt-1"


def test_dict_row_builds_a_synthetic_ref_without_one() -> None:
    record = request_record()
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    assert DictRow(record, line=842).ref == LineRef(
        line=842,
        offset=0,
        length=len(encoded),
        sha256=hashlib.sha256(encoded).hexdigest(),
    )


def test_dict_row_uses_a_given_ref_as_is() -> None:
    real = LineRef(line=842, offset=1_048_000, length=611, sha256="b" * 64)
    row = DictRow(request_record(), line=3, ref=real)
    assert row.ref is real


def test_dict_row_is_read_only() -> None:
    row = DictRow(request_record())
    assert isinstance(row, Mapping)
    assert not hasattr(row, "__setitem__")


def test_columns_equal_row_access() -> None:
    rows = [
        DictRow(request_record(request_id="a", ttft_ms=10.0), line=0),
        DictRow(request_record(request_id="b", ttft_ms=None, status="timeout"), line=1),
        DictRow({"request_id": "c"}, line=2),
    ]
    snap = snapshot(requests=rows, table=DictTable(rows))
    assert len(snap.table) == 3
    for name in ("request_id", "status", "ttft_ms", "no_such_key"):
        assert list(snap.column(name)) == [row.get(name) for row in rows]
    assert list(snap.column("client.ttft")) == [10.0, None, None]
    with pytest.raises(KeyError):
        snap.column("client.no_such_metric")


def test_a_dict_table_gives_the_columnar_arrays_of_its_rows() -> None:
    rows = [
        DictRow(request_record(request_id="a", chunk_interarrival_ms=[1.0, "x"])),
        DictRow(request_record(request_id="b", ttft_ms=None, status="timeout")),
        DictRow({"request_id": "c", "chunk_interarrival_ms": None}, line=2),
        DictRow(request_record(request_id="d", chunk_interarrival_ms=[3]), line=3),
    ]
    table = contracts.columnar(DictTable(rows))
    assert table is not None and isinstance(table, contracts.ColumnarTable)
    for metric in CLIENT_METRICS:
        values = table.client_values(metric)
        assert len(values) == len(rows)
        for value, row in zip(values, rows):
            expected = row.client[metric].value_ms
            assert math.isnan(value) if expected is None else value == expected
        codes, reasons = table.client_reasons(metric)
        assert reasons[0] == (None, False)
        assert [reasons[code] for code in codes] == [
            (row.client[metric].reason, row.client[metric].not_applicable)
            for row in rows
        ]
    assert list(table.line_refs()) == [row.ref for row in rows]
    assert list(table.line_lengths()) == [row.ref.length for row in rows]
    assert list(table.gap_counts()) == [row.gap_count for row in rows] == [2, 15, 0, 1]
    pool = table.gap_pool()
    for i, row in enumerate(rows):
        start, count = pool.starts[i], pool.counts[i]
        gaps = list(pool.values[start : start + count])
        assert repr(gaps) == repr(list(row.load_gaps()))
    assert list(table.column("status")) == [row.get("status") for row in rows]


def test_a_table_without_the_arrays_is_not_columnar() -> None:
    class Bare:
        def __len__(self) -> int:
            return 0

        def column(self, name: str) -> list[Any]:
            return []

    assert contracts.columnar(Bare()) is None
    doc = _doc(contracts.ColumnarTable)
    for token in ("NaN", "milliseconds", "NotResident", "copy", "must not mutate"):
        assert token in doc, token


def test_the_registry_has_a_publish_listener_that_survives_replace() -> None:
    doc = _doc(contracts.PublishingRegistry.add_publish_listener)
    assert "builder pool" in doc and "id(snapshot.table)" in doc

    class Bare:
        def snapshot(self, rel: str) -> IndexSnapshot:
            return snapshot()

    class Hooked(Bare):
        def add_publish_listener(self, fn: Any) -> None:
            pass

    assert contracts.publish_hook(Bare()) is None  # type: ignore[arg-type]
    hooked: Any = Hooked()
    for name in ("wait", "read_line", "read_lines", "close"):
        setattr(hooked, name, lambda *args: None)
    assert contracts.publish_hook(hooked) is hooked


def test_not_resident_is_not_a_key_error() -> None:
    assert not issubclass(contracts.NotResident, KeyError)


# --- errors ------------------------------------------------------------------

ERROR_CLASSES: list[tuple[type[StudioError], int, str | None]] = [
    (StudioError, 500, "error"),
    (UsageError, 400, "usage"),
    (ConflictError, 409, "usage"),
    (NotFoundError, 404, "invalid_input"),
    (InvalidInputError, 422, "invalid_input"),
    (UnauthorizedError, 401, None),
    (ForbiddenError, 403, None),
    (BusyError, 503, None),
]


@pytest.mark.parametrize(("cls", "status", "verdict"), ERROR_CLASSES)
def test_error_classes_carry_the_section_5_2_status_pairs(
    cls: type[StudioError], status: int, verdict: str | None
) -> None:
    error = cls("some_code", "a message", field="status", details={"n": 1})
    assert error.http_status == status and error.verdict_status == verdict
    assert str(error) == "a message"
    assert error.document() == {
        "format": "stormlog.studio.error",
        "schema_version": 1,
        "error": {
            "code": "some_code",
            "message": "a message",
            "http_status": status,
            "verdict_status": verdict,
            "field": "status",
            "details": {"n": 1},
        },
    }


def test_error_document_defaults_and_independence() -> None:
    details = {"allowed": ["ok"]}
    error = UsageError("bad_parameter", "status: unknown value", details=details)
    details["allowed"] = []
    document = error.document()
    assert document["error"]["field"] is None
    assert document["error"]["details"] == {"allowed": ["ok"]}
    assert UsageError("x", "y").document()["error"]["details"] == {}


def test_not_ready_carries_the_enveloped_analysis_document() -> None:
    state = envelope("analysis", {"status": "running"}, snapshot())
    ready = NotReady(state, retry_after_s=2)
    assert ready.state == state and ready.retry_after_s == 2
    assert NotReady(state).retry_after_s == 1
    assert not isinstance(ready, StudioError)


@pytest.mark.parametrize(
    "state",
    [
        {"status": "building", "bytes_read": 10},
        envelope("artifact", {}, snapshot()),
        {},
    ],
)
def test_not_ready_refuses_anything_but_an_analysis_document(
    state: Mapping[str, Any],
) -> None:
    with pytest.raises(ValueError, match="stormlog.studio.analysis"):
        NotReady(state)


def test_not_ready_needs_the_artifact_block_and_a_header_safe_retry() -> None:
    bare = envelope("analysis", {"status": "running"})
    with pytest.raises(ValueError, match="artifact block"):
        NotReady(bare)
    state = envelope("analysis", {"status": "running"}, snapshot())
    assert NotReady(state, retry_after_s=0).retry_after_s == 0
    for bad in (-1, 1.5, True, "2"):
        with pytest.raises(ValueError, match="retry_after_s"):
            NotReady(state, retry_after_s=bad)  # type: ignore[arg-type]


def _doc(item: Any) -> str:
    return " ".join((item.__doc__ or "").split())


def _says(item: Any, *tokens: str) -> None:
    """Each decision's docstring names it; short tokens, so a reword passes."""
    doc = _doc(item).casefold()
    for token in tokens:
        assert token.casefold() in doc, token


def test_the_contract_states_who_raises_not_ready() -> None:
    _says(NotReady, "C4", "202")
    _says(IndexRegistry.snapshot, "Never raises", "NotReady", "building")
    _says(IndexRegistry.wait, "never raises", "timeout", "building")
    _says(AnalysisCache, "NotReady", "202")


def test_the_contract_states_kept_records_are_finite_at_index_time() -> None:
    _says(IndexSnapshot, "finite_copy", "index time")
    nan_config = finite_copy({"config": {"temperature": math.nan}})[0]["config"]
    session = SessionInfo(SESSION, "completed", None, None, True, nan_config)
    assert json.loads(encode_json({"config": session.config})) == {
        "config": {"temperature": None}
    }


def test_the_contract_states_unknown_is_never_a_second_session() -> None:
    _says(IndexSnapshot, "unknown", "second session")
    _says(vars(contracts.RequestRow)["session_id"], "unknown", "second session")


def test_the_artifact_document_leaves_sample_sufficiency_to_the_summary() -> None:
    name = "studio_artifact_v1.schema.json"
    document = _load(EXAMPLE_DIR / "studio_artifact_v1.example.json")
    assert all(w["code"] != "insufficient_samples" for w in document["warnings"])
    document["warnings"] = [{"code": "insufficient_samples", "message": "p95"}]
    assert _errors(name, document)
    document["warnings"] = [{"code": "growing", "message": "still written"}]
    assert _errors(name, document) == []


def test_a_planned_unobserved_case_still_has_a_summary_body() -> None:
    schema = _load(SCHEMA_DIR / "studio_cases_v1.schema.json")
    description = " ".join(schema["description"].split()).casefold()
    assert "planned" in description and "empty-selection" in description
    entry = schema["properties"]["cases"]["items"]
    assert "summary" in entry["required"]
    document = _load(EXAMPLE_DIR / "studio_cases_v1.example.json")
    document["cases"][0].update(observed=False, summary=None)
    assert _errors("studio_cases_v1.schema.json", document)


def test_the_artifact_document_is_never_a_202_body() -> None:
    description = _load(SCHEMA_DIR / "studio_artifact_v1.schema.json")["description"]
    assert "also the 202 body" not in description.lower()
    assert "200" in description


def test_a_zero_row_building_snapshot_gives_a_valid_artifact_document() -> None:
    building = snapshot(
        generation=0,
        status="building",
        identity=None,
        bytes_read=0,
        sha256=None,
        prefix_sha256=hashlib.sha256(b"").hexdigest(),
        growing=False,
        origin_ns=None,
        session=None,
        requests=(),
        table=DictTable(()),
        by_request_id={},
        inventory={},
        limits_hit=frozenset(),
    )
    body: dict[str, Any] = {
        "session": None,
        "index": {
            "status": "building",
            "failure": None,
            "bytes_read": 0,
            "size": None,
            "progress": None,
            "pending_tail_bytes": 0,
            "prefix_sha256": building.prefix_sha256,
            "requests": 0,
            "samples": 0,
        },
        "inventory": [],
        "problems": {"count": 0, "first": []},
        "cases": [],
        "warnings": [],
    }
    document = to_wire(envelope("artifact", body, building))
    assert document["artifact"]["status"] == "building"
    assert _errors("studio_artifact_v1.schema.json", document) == []


def test_line_types_are_frozen() -> None:
    ref = LineRef(line=0, offset=0, length=2, sha256="0" * 64)
    problem = LineProblem(ref=ref, code="not_json", detail="x")
    with pytest.raises(AttributeError):
        problem.detail = "y"  # type: ignore[misc]


# --- imports -----------------------------------------------------------------


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "contracts uses absolute imports only"
            names.append(node.module or "")
    return names


def test_contracts_imports_only_stdlib_populations_and_slo() -> None:
    stdlib = sys.stdlib_module_names
    for name in _imported_modules(CONTRACTS):
        allowed = name.split(".")[0] in stdlib or name in ALLOWED_STORMLOG_IMPORTS
        assert allowed, f"contracts imports {name}"


def test_the_line_types_live_in_infer_and_import_only_the_stdlib() -> None:
    stdlib = sys.stdlib_module_names
    for name in _imported_modules(ARTIFACT_LINES):
        assert name.split(".")[0] in stdlib, f"artifact_lines imports {name}"
    for name in ("LineRef", "ProblemCode", "LineProblem", "FileIdentity"):
        assert getattr(contracts, name) is getattr(artifact_lines, name), name
        assert name in contracts.__all__ and name in artifact_lines.__all__


def test_importing_contracts_loads_no_other_studio_module() -> None:
    probe = (
        "import sys, stormlog.studio.contracts; "
        "print(sorted(m for m in sys.modules if m.startswith('stormlog.studio')))"
    )
    output = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    ).stdout
    assert json.loads(output.replace("'", '"')) == [
        "stormlog.studio",
        "stormlog.studio.contracts",
    ]


def test_a_workspace_scanner_can_be_waited_on() -> None:
    wait = vars(contracts.WorkspaceScanner)["wait"]
    assert list(inspect.signature(wait).parameters) == ["self", "timeout_s"]
    assert inspect.signature(wait).return_annotation in (bool, "bool")
    assert "never raises on a timeout" in _doc(wait)


def test_every_public_name_is_exported() -> None:
    for name in contracts.__all__:
        assert hasattr(contracts, name), name


# --- schemas and examples (design §10.2) --------------------------------------


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _registry() -> Any:
    resources = []
    for name in STUDIO_SCHEMAS:
        schema = _load(SCHEMA_DIR / name)
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


def _validator(name: str) -> Any:
    schema = _load(SCHEMA_DIR / name)
    return jsonschema.Draft202012Validator(schema, registry=_registry())


def _errors(name: str, document: Any) -> list[str]:
    found = _validator(name).iter_errors(document)
    return [f"{list(error.absolute_path)}: {error.message}" for error in found]


def _example_name(schema_name: str) -> str:
    return schema_name.replace(".schema.json", ".example.json")


def _walk(node: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(node, Mapping):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def test_there_are_step_0_schemas() -> None:
    assert "studio_error_v1.schema.json" in STUDIO_SCHEMAS
    assert "studio_artifact_block_v1.schema.json" in STUDIO_SCHEMAS


@pytest.mark.parametrize("name", STUDIO_SCHEMAS)
def test_schema_is_valid_under_draft_2020_12(name: str) -> None:
    schema = _load(SCHEMA_DIR / name)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == SCHEMA_BASE + name
    jsonschema.Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("name", STUDIO_SCHEMAS)
def test_document_schema_names_its_envelope(name: str) -> None:
    schema = _load(SCHEMA_DIR / name)
    document = name.removeprefix("studio_").removesuffix("_v1.schema.json")
    if document.endswith("_block"):
        assert "format" not in schema["properties"]
        return
    properties = schema["properties"]
    assert properties["format"]["const"] == f"stormlog.studio.{document}"
    assert properties["schema_version"]["const"] == 1
    assert {"format", "schema_version"} <= set(schema["required"])
    if "artifact" in properties:
        assert properties["artifact"] == {
            "$ref": "studio_artifact_block_v1.schema.json"
        }
        assert "artifact" in schema["required"]


@pytest.mark.parametrize("name", STUDIO_SCHEMAS)
def test_ns_fields_are_decimal_strings_in_every_schema(name: str) -> None:
    schema = _load(SCHEMA_DIR / name)
    canonical = _load(SCHEMA_DIR / "studio_artifact_block_v1.schema.json")
    for node in _walk(schema):
        for key, value in dict(node.get("properties", {})).items():
            if key.endswith("_ns"):
                assert value.get("$ref") in {"#/$defs/ns", "#/$defs/ns_or_null"}, key
    defs = schema.get("$defs", {})
    if "ns" in defs:
        assert defs["ns"] == canonical["$defs"]["ns"]
        assert defs["ns"]["type"] == "string"


def test_every_schema_has_an_example_and_every_example_a_schema() -> None:
    examples = sorted(path.name for path in EXAMPLE_DIR.glob("studio_*.json"))
    assert examples == [_example_name(name) for name in STUDIO_SCHEMAS]


@pytest.mark.parametrize("name", STUDIO_SCHEMAS)
def test_example_validates_against_its_schema(name: str) -> None:
    example = _load(EXAMPLE_DIR / _example_name(name))
    assert _errors(name, example) == []
    assert to_wire(example) == example, "examples are already in wire form"


@pytest.mark.parametrize(("cls", "status", "verdict"), ERROR_CLASSES)
def test_error_documents_validate(
    cls: type[StudioError], status: int, verdict: str | None
) -> None:
    error = cls("cursor_mismatch", "the index was rebuilt", details={"at_ns": 5})
    assert _errors("studio_error_v1.schema.json", to_wire(error.document())) == []


def test_error_schema_pairs_http_and_verdict_status() -> None:
    document = UsageError("bad_parameter", "x").document()
    document["error"]["verdict_status"] = None
    assert _errors("studio_error_v1.schema.json", document)


def test_error_schema_names_the_route_and_media_type_codes() -> None:
    schema = _load(SCHEMA_DIR / "studio_error_v1.schema.json")
    codes = schema["properties"]["error"]["properties"]["code"]["description"]
    assert "route_not_found (404)" in codes
    assert "unsupported_media_type (415)" in codes
    examples = {doc["error"]["code"]: doc for doc in schema["examples"]}
    assert examples["route_not_found"]["error"]["http_status"] == 404
    assert examples["unsupported_media_type"]["error"]["http_status"] == 415
    for document in schema["examples"]:
        assert _errors("studio_error_v1.schema.json", document) == []


def test_error_schema_names_c3s_malformed_request() -> None:
    schema = _load(SCHEMA_DIR / "studio_error_v1.schema.json")
    codes = schema["properties"]["error"]["properties"]["code"]["description"]
    assert "malformed_request (400)" in codes
    assert "malformed_request" in schema["description"]
    examples = {doc["error"]["code"]: doc for doc in schema["examples"]}
    document = UsageError("malformed_request", "the Host header is repeated")
    assert examples["malformed_request"] == document.document()


class UnsupportedMediaTypeError(StudioError):
    """415, as C3's httpd will define it."""

    http_status = 415
    verdict_status = None


def test_a_415_error_validates_only_with_a_null_verdict() -> None:
    error = UnsupportedMediaTypeError("unsupported_media_type", "use JSON")
    document = to_wire(error.document())
    assert _errors("studio_error_v1.schema.json", document) == []
    document["error"]["verdict_status"] = "usage"
    assert _errors("studio_error_v1.schema.json", document)


def test_artifact_block_validates_with_and_without_a_session() -> None:
    name = "studio_artifact_block_v1.schema.json"
    assert _errors(name, to_wire(artifact_block(snapshot()))) == []
    bare = snapshot(session=None, origin_ns=None, sha256=None, status="building")
    assert _errors(name, to_wire(artifact_block(bare))) == []


def test_a_file_never_statted_has_no_identity_and_a_null_size() -> None:
    unreadable = snapshot(
        status="failed",
        failure="unreadable",
        identity=None,
        bytes_read=0,
        sha256=None,
        session=None,
        origin_ns=None,
        requests=(),
        table=DictTable(()),
    )
    block = artifact_block(unreadable)
    assert block["size"] is None and block["bytes_read"] == 0
    assert _errors("studio_artifact_block_v1.schema.json", to_wire(block)) == []
    document = _load(EXAMPLE_DIR / "studio_artifact_v1.example.json")
    document["artifact"] = to_wire(block)
    document["index"].update(
        status="failed", failure="unreadable", size=None, progress=None, bytes_read=0
    )
    assert _errors("studio_artifact_v1.schema.json", document) == []


def _ns_as_int(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: int(value) if key.endswith("_ns") and isinstance(value, str) else value
        for key, value in record.items()
    }


def test_criterion_record_is_the_wire_form_of_a_client_value() -> None:
    value = CriterionValue(value_ms=None, reason="not_streamed", not_applicable=True)
    assert criterion_record(value) == {
        "value_ms": None,
        "reason": "not_streamed",
        "not_applicable": True,
    }
    with pytest.raises(TypeError):
        to_wire(value)


def test_criterion_record_matches_every_example_row() -> None:
    rows = list(_load(EXAMPLE_DIR / "studio_requests_v1.example.json")["rows"])
    detail = _load(EXAMPLE_DIR / "studio_request_v1.example.json")
    rows.append(detail["row"])
    assert len(rows) > 1
    for row in rows:
        for entry in row["client"].values():
            assert criterion_record(CriterionValue(**entry)) == entry
    built = DictRow(_ns_as_int(detail["record"])).client
    assert {key: criterion_record(value) for key, value in built.items()} == (
        detail["row"]["client"]
    )


def test_selection_fingerprint_hashes_the_four_parts_newline_joined() -> None:
    params = "case=c8_in4096_out128&phase=measured"
    text = f"runs/042.jsonl\n{INDEX_ID}\n7\n{params}"
    expected = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert selection_fingerprint("runs/042.jsonl", INDEX_ID, 7, params) == expected


def test_selection_fingerprint_changes_with_each_part() -> None:
    base = ("runs/042.jsonl", INDEX_ID, 7, "phase=measured")
    variants = [
        ("runs/043.jsonl", INDEX_ID, 7, "phase=measured"),
        ("runs/042.jsonl", "0" * 32, 7, "phase=measured"),
        ("runs/042.jsonl", INDEX_ID, 8, "phase=measured"),
        ("runs/042.jsonl", INDEX_ID, 7, "phase=all"),
        ("runs/042.jsonl\n" + INDEX_ID, INDEX_ID, 7, "phase=measured"),
    ]
    prints = {selection_fingerprint(*parts) for parts in [base, *variants]}
    assert len(prints) == 1 + len(variants)
    assert all(re.fullmatch(r"sha256:[0-9a-f]{64}", value) for value in prints)


def test_every_example_selection_has_the_contract_fingerprint() -> None:
    checked = 0
    for path in sorted(EXAMPLE_DIR.glob("studio_*.example.json")):
        example = _load(path)
        artifact = example.get("artifact")
        for node in _walk(example):
            if artifact is None or "fingerprint" not in node:
                continue
            assert node["fingerprint"] == selection_fingerprint(
                artifact["path"],
                artifact["index_id"],
                artifact["generation"],
                node["params"],
            ), path.name
            checked += 1
    assert checked >= 4


def test_artifact_block_refuses_an_integer_ns_field() -> None:
    block = to_wire(artifact_block(snapshot()))
    block["origin_ns"] = 1788609600000000000
    assert _errors("studio_artifact_block_v1.schema.json", block)
