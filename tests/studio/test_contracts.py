"""The step-0 contract: wire helpers, rows and columns, errors, imports."""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from array import array
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.diagnosis_inputs import record_id
from stormlog.infer.slo import client_values
from stormlog.studio import contracts
from stormlog.studio.contracts import (
    BusyError,
    ConflictError,
    DictRow,
    DictTable,
    FileIdentity,
    ForbiddenError,
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
    encode_json,
    envelope,
    finite_copy,
    to_wire,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = REPO_ROOT / "stormlog" / "studio" / "contracts.py"
ALLOWED_STORMLOG_IMPORTS = {"stormlog.infer.populations", "stormlog.infer.slo"}
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


@pytest.mark.parametrize("value", [b"bytes", {1.5: 1}, {"s": {1, 2}}, object()])
def test_to_wire_refuses_values_json_cannot_carry_exactly(value: Any) -> None:
    with pytest.raises(TypeError):
        to_wire(value)


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
    }
    assert to_wire(block)["origin_ns"] == "1788609600000000000"


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


def test_dict_row_record_id_matches_diagnosis_inputs_for_v2_records() -> None:
    record = request_record(schema_version=2, event_id="evt-1")
    assert DictRow(record).record_id == record_id(record) == "evt-1"


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


def test_not_ready_keeps_its_state_and_retry_after() -> None:
    state = {"status": "building", "bytes_read": 10}
    ready = NotReady(state, retry_after_s=2)
    assert ready.state == state and ready.retry_after_s == 2
    assert NotReady({}).retry_after_s == 1


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


def test_every_public_name_is_exported() -> None:
    for name in contracts.__all__:
        assert hasattr(contracts, name), name
