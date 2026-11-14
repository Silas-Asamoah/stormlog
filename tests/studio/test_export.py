"""C4's raw export: the preview and the byte-exact, hash-checked stream.

The index is a fake IndexRegistry over a real artifact written by
JsonlEventWriter: its rows carry the real line references, and its
read_lines reads the real file by offset, as C1's will.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped, unused-ignore]
import pytest
from referencing import Registry, Resource

from stormlog.infer.events import (
    InferenceRequestEvent,
    InferenceSystemSample,
    JsonlEventWriter,
)
from stormlog.studio import export
from stormlog.studio.contracts import (
    ConflictError,
    DictRow,
    DictTable,
    FileIdentity,
    IndexRegistry,
    IndexSnapshot,
    LineRef,
    NotResident,
    RateScope,
    RequestFilter,
    RequestRow,
    Selection,
    SessionInfo,
    envelope,
    to_wire,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "docs" / "schemas"
SESSION = "5f0c1d2e3a4b5c6d7e8f901234567890"
INDEX_ID = "9a3e5b7c1d2f4a6b8c0e2d4f6a8b0c1d"
T0 = 1788609600000000000
REL = "runs/042.jsonl"


# --- helpers -------------------------------------------------------------------


def schema_errors(name: str, document: Any) -> list[str]:
    resources = []
    for path in SCHEMA_DIR.glob("studio_*.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        resources.append((schema["$id"], Resource.from_contents(schema)))
    schema = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(
        schema, registry=Registry().with_resources(resources)
    )
    return [
        f"{list(error.absolute_path)}: {error.message}"
        for error in validator.iter_errors(document)
    ]


class FileRow(DictRow):
    """A DictRow that carries its real line reference."""

    def __init__(self, record: dict[str, Any], ref: LineRef) -> None:
        self._file_ref = ref
        super().__init__(record, line=ref.line)

    @property  # type: ignore[override, unused-ignore]
    def ref(self) -> LineRef:
        return self._file_ref

    @ref.setter
    def ref(self, _synthetic: LineRef) -> None:
        """DictRow sets a synthetic ref; this row keeps the real one."""


class NoErrorMessages(DictTable):
    """A table whose error-message pool was dropped."""

    def column(self, name: str) -> Sequence[Any]:
        if name == "error_message":
            raise NotResident(name)
        return super().column(name)


def request(index: int, **overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "session_id": SESSION,
        "request_id": f"measured_{index:03d}",
        "case_id": "c1" if index % 2 == 0 else "c2",
        "phase": "measured",
        "started_at_ns": T0 + 123456789 + index * 10**7,
        "ended_at_ns": T0 + 123456789 + index * 10**7 + 899_300_000,
        "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "concurrency": 2,
        "target_input_tokens": 8,
        "target_output_tokens": 4,
        "stream": True,
        "status": "ok",
        "e2e_latency_ms": 899.3,
        "ttft_ms": 120.5,
        "first_chunk_latency_ms": 110.25,
        "chunk_interarrival_ms": [10.0, 12.5, 11.0],
        "output_tokens": 4,
        "output_token_source": "server_usage",
        "output_token_exact": True,
    }
    values.update(overrides)
    return InferenceRequestEvent(**values).to_record()


def write_artifact(path: Path, requests: int = 8) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with JsonlEventWriter(path) as writer:
        writer.append(
            {
                "schema_version": 1,
                "event_type": "infer.session",
                "session_id": SESSION,
                "timestamp_ns": T0,
                "status": "running",
                "config": {"endpoint": "http://127.0.0.1:8000/v1"},
            }
        )
        writer.append(request(90, phase="warmup", request_id="warmup_000"))
        for index in range(requests):
            overrides: dict[str, Any] = {}
            if index == 3:
                overrides = {
                    "status": "error",
                    "error_type": "http_error",
                    "error_message": "server said: boom (é)",
                    "http_status": 500,
                    "ttft_ms": None,
                    "output_tokens": None,
                }
            if index == 5:
                overrides = {"model": "/home/me/models/qwen-7b"}
            writer.append(request(index, **overrides))
            writer.append(
                InferenceSystemSample(
                    session_id=SESSION,
                    timestamp_ns=T0 + index * 10**7,
                    sampler="psutil",
                    process_rss_bytes=181403648,
                ).to_record()
            )
        writer.append(
            {
                "schema_version": 1,
                "event_type": "infer.session",
                "session_id": SESSION,
                "timestamp_ns": T0 + 10**9,
                "status": "completed",
            }
        )
    return path


def index_file(path: Path, rel: str = REL, **overrides: Any) -> IndexSnapshot:
    """What C1's index would publish for a whole, single-session file."""
    data = path.read_bytes()
    rows: list[RequestRow] = []
    inventory: dict[str, dict[int, int]] = {}
    offset = 0
    for number, line in enumerate(data.split(b"\n")[:-1]):
        ref = LineRef(number, offset, len(line), hashlib.sha256(line).hexdigest())
        offset += len(line) + 1
        record = json.loads(line)
        version = int(record.get("schema_version", 1))
        counts = inventory.setdefault(record["event_type"], {})
        counts[version] = counts.get(version, 0) + 1
        if record["event_type"] == "infer.request" and "context" not in record:
            rows.append(FileRow(record, ref))
    stat = path.stat()
    values: dict[str, Any] = {
        "path": rel,
        "index_id": INDEX_ID,
        "generation": 3,
        "status": "ready",
        "failure": None,
        "identity": FileIdentity(stat.st_dev, stat.st_ino, len(data), stat.st_mtime_ns),
        "bytes_read": len(data),
        "pending_tail_bytes": 0,
        "prefix_sha256": hashlib.sha256(data).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "growing": False,
        "origin_ns": T0,
        "session": SessionInfo(SESSION, "completed", T0, T0 + 10**9, True, {}),
        "requests": rows,
        "table": DictTable(rows),
        "by_request_id": {},
        "samples": (),
        "kept": {},
        "request_links": {},
        "refs": {},
        "inventory": inventory,
        "problems": (),
        "problem_count": 0,
        "limits_hit": frozenset(),
        "gaps_resident": True,
        "error_messages_resident": True,
    }
    values.update(overrides)
    return IndexSnapshot(**values)


class FileIndexes:
    """An IndexRegistry over one real file, reading lines by offset."""

    def __init__(self, path: Path, snapshot: IndexSnapshot) -> None:
        self.path = path
        self._snapshot = snapshot
        self.requested = 0

    def snapshot(self, rel: str) -> IndexSnapshot:
        return self._snapshot

    def wait(self, rel: str, timeout_s: float) -> IndexSnapshot:
        return self._snapshot

    def read_line(self, rel: str, ref: LineRef) -> bytes:
        with self.path.open("rb") as handle:
            handle.seek(ref.offset)
            return handle.read(ref.length)

    def read_lines(self, rel: str, refs: Iterable[LineRef]) -> Iterator[bytes]:
        with self.path.open("rb") as handle:
            for ref in refs:
                self.requested += 1
                handle.seek(ref.offset)
                yield handle.read(ref.length + 1)

    def close(self) -> None:
        pass


def select(
    snapshot: IndexSnapshot, rows: Sequence[int] | None = None, **overrides: Any
) -> Selection:
    """A selection of ``rows``; by default every measured request."""
    if rows is None:
        rows = [i for i, row in enumerate(snapshot.requests) if row.phase == "measured"]
    values: dict[str, Any] = {
        "filter": RequestFilter(),
        "params": "phase=measured",
        "fingerprint": "sha256:" + "1" * 64,
        "path": snapshot.path,
        "index_id": snapshot.index_id,
        "generation": snapshot.generation,
        "rows": tuple(rows),
        "cases": ("c1", "c2"),
        "whole_case": None,
        "rate": RateScope(False, None, None, "multiple_cases"),
        "searched_fields": (),
    }
    values.update(overrides)
    return Selection(**values)


def selection_block(sel: Selection) -> dict[str, Any]:
    """The step-0 example block, sized to the selection (C2 builds the real one)."""
    example = SCHEMA_DIR / "examples" / "studio_selection_block_v1.example.json"
    block: dict[str, Any] = json.loads(example.read_text(encoding="utf-8"))
    block.update(params=sel.params, fingerprint=sel.fingerprint, size=len(sel.rows))
    return block


def original_lines(path: Path, snapshot: IndexSnapshot, sel: Selection) -> bytes:
    lines = path.read_bytes().split(b"\n")
    return b"".join(lines[snapshot.requests[i].ref.line] + b"\n" for i in sel.rows)


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    return write_artifact(tmp_path / REL)


# --- stream -----------------------------------------------------------------------


def test_the_fake_is_an_index_registry(artifact: Path) -> None:
    registry: IndexRegistry = FileIndexes(artifact, index_file(artifact))
    assert registry.snapshot(REL).path == REL


def test_stream_yields_the_selected_lines_byte_for_byte(artifact: Path) -> None:
    snapshot = index_file(artifact)
    sel = select(snapshot)
    body = b"".join(export.stream(FileIndexes(artifact, snapshot), snapshot, sel))
    assert body == original_lines(artifact, snapshot, sel)
    assert body.count(b"\n") == 8
    # Integer nanoseconds stay exact: nothing was re-serialized.
    first = json.loads(body.split(b"\n")[0])
    assert first["started_at_ns"] == T0 + 123456789


def test_stream_keeps_file_order_for_a_subset(artifact: Path) -> None:
    snapshot = index_file(artifact)
    sel = select(snapshot, rows=[2, 4, 7])
    body = b"".join(export.stream(FileIndexes(artifact, snapshot), snapshot, sel))
    ids = [json.loads(line)["request_id"] for line in body.splitlines()]
    assert ids == ["measured_001", "measured_003", "measured_006"]


def test_an_empty_selection_streams_nothing(artifact: Path) -> None:
    snapshot = index_file(artifact)
    sel = select(snapshot, rows=[])
    assert list(export.stream(FileIndexes(artifact, snapshot), snapshot, sel)) == []


def test_stream_is_lazy_and_chunked(
    artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(export, "CHUNK_BYTES", 1)
    snapshot = index_file(artifact)
    indexes = FileIndexes(artifact, snapshot)
    chunks = export.stream(indexes, snapshot, select(snapshot))
    assert indexes.requested == 0
    first = next(chunks)
    assert first.count(b"\n") == 1
    assert indexes.requested == 1
    assert len(list(chunks)) == 7


def test_lines_are_joined_up_to_the_chunk_size(artifact: Path) -> None:
    snapshot = index_file(artifact)
    sel = select(snapshot)
    chunks = list(export.stream(FileIndexes(artifact, snapshot), snapshot, sel))
    assert len(chunks) == 1  # eight lines are well under 64 KiB


def test_a_line_changed_in_place_stops_the_stream_before_it(
    artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(export, "CHUNK_BYTES", 1)
    snapshot = index_file(artifact)
    sel = select(snapshot)
    second = snapshot.requests[sel.rows[1]].ref
    data = bytearray(artifact.read_bytes())
    position = data.index(b"measured_001", second.offset)
    data[position : position + 12] = b"measured_999"  # same length
    artifact.write_bytes(bytes(data))
    chunks = export.stream(FileIndexes(artifact, snapshot), snapshot, sel)
    assert b"measured_000" in next(chunks)
    with pytest.raises(ConflictError) as raised:
        next(chunks)
    assert raised.value.code == "stale_line"
    assert raised.value.details == {"line": second.line, "reason": "sha256"}


def test_a_truncated_file_is_stale(artifact: Path) -> None:
    snapshot = index_file(artifact)
    sel = select(snapshot)
    last = snapshot.requests[sel.rows[-1]].ref
    with artifact.open("r+b") as handle:
        handle.truncate(last.offset + 10)
    with pytest.raises(ConflictError) as raised:
        b"".join(export.stream(FileIndexes(artifact, snapshot), snapshot, sel))
    assert raised.value.details == {"line": last.line, "reason": "length"}


def test_a_line_that_lost_its_newline_is_stale(artifact: Path) -> None:
    snapshot = index_file(artifact)
    ref = snapshot.requests[1].ref
    data = bytearray(artifact.read_bytes())
    data[ref.offset + ref.length] = ord(" ")
    artifact.write_bytes(bytes(data))
    sel = select(snapshot, rows=[1])
    with pytest.raises(ConflictError) as raised:
        list(export.stream(FileIndexes(artifact, snapshot), snapshot, sel))
    assert raised.value.details["reason"] == "newline"


class ShortIndexes(FileIndexes):
    def read_lines(self, rel: str, refs: Iterable[LineRef]) -> Iterator[bytes]:
        for number, line in enumerate(super().read_lines(rel, refs)):
            if number == 2:
                return
            yield line


class LongIndexes(FileIndexes):
    def read_lines(self, rel: str, refs: Iterable[LineRef]) -> Iterator[bytes]:
        yield from super().read_lines(rel, refs)
        yield b"{}\n"


class StaleIndexes(FileIndexes):
    def read_lines(self, rel: str, refs: Iterable[LineRef]) -> Iterator[bytes]:
        for number, line in enumerate(super().read_lines(rel, refs)):
            if number == 1:
                raise ConflictError(
                    "stale_line", "the registry found it", details={"x": 1}
                )
            yield line


@pytest.mark.parametrize(
    ("indexes", "reason"),
    [(ShortIndexes, "missing"), (LongIndexes, "unexpected")],
)
def test_a_registry_that_skips_or_adds_lines_is_caught(
    artifact: Path, indexes: type[FileIndexes], reason: str
) -> None:
    snapshot = index_file(artifact)
    with pytest.raises(ConflictError) as raised:
        list(export.stream(indexes(artifact, snapshot), snapshot, select(snapshot)))
    assert raised.value.details["reason"] == reason


def test_the_registrys_own_stale_line_passes_through(artifact: Path) -> None:
    snapshot = index_file(artifact)
    with pytest.raises(ConflictError) as raised:
        list(
            export.stream(StaleIndexes(artifact, snapshot), snapshot, select(snapshot))
        )
    assert raised.value.details == {"x": 1}


@pytest.mark.parametrize(
    "overrides",
    [{"index_id": "f" * 32}, {"path": "other.jsonl"}, {"generation": 4}],
)
def test_a_selection_from_another_snapshot_is_refused(
    artifact: Path, overrides: dict[str, Any]
) -> None:
    snapshot = index_file(artifact)
    sel = select(snapshot, **overrides)
    with pytest.raises(ValueError):
        export.stream(FileIndexes(artifact, snapshot), snapshot, sel)
    with pytest.raises(ValueError):
        export.preview(snapshot, sel, selection_block=selection_block(sel))


def test_an_older_generation_of_the_same_index_is_accepted(artifact: Path) -> None:
    snapshot = index_file(artifact)
    sel = select(snapshot, generation=1)
    assert list(export.stream(FileIndexes(artifact, snapshot), snapshot, sel))


def test_the_export_is_valid_input_to_infer_analyze(
    artifact: Path, tmp_path: Path
) -> None:
    snapshot = index_file(artifact)
    exported = tmp_path / "export.jsonl"
    sel = select(snapshot)
    exported.write_bytes(
        b"".join(export.stream(FileIndexes(artifact, snapshot), snapshot, sel))
    )
    report_path = tmp_path / "report.json"
    command = [sys.executable, "-m", "stormlog", "infer", "analyze", str(exported)]
    command += ["--format", "json", "--output", str(report_path)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=240)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report["cases"]) == {"c1", "c2"}
    offered = sum(case["arrivals"]["offered"] for case in report["cases"].values())
    assert offered == 8
    # What the preview's no_phase_windows note says: the rate falls back to
    # the requests' span (design §5.6 said null; the analyzer does this).
    for case in report["cases"].values():
        assert case["intervals"]["rate_reason"] == "no_phase_window"
        assert case["throughput"]["interval_kind"] == "request_span"


# --- preview ----------------------------------------------------------------------


def test_preview_counts_lines_bytes_and_sensitive_fields(artifact: Path) -> None:
    snapshot = index_file(artifact)
    sel = select(snapshot)
    body = export.preview(snapshot, sel, selection_block=selection_block(sel))
    streamed = b"".join(export.stream(FileIndexes(artifact, snapshot), snapshot, sel))
    assert (body["count"], body["bytes"]) == (8, len(streamed))
    assert body["profile"] == "original_bytes"
    fields = {entry["field"]: entry for entry in body["sensitive_fields"]}
    assert (fields["error_message"]["count"], fields["error_message"]["reason"]) == (
        1,
        None,
    )
    assert (fields["endpoint"]["count"], fields["endpoint"]["distinct"]) == (8, 1)
    assert (fields["model"]["count"], fields["model"]["distinct"]) == (1, 1)
    codes = [note["code"] for note in body["limitations"]]
    assert codes == ["not_a_redaction", "no_phase_windows"]
    assert "#111" in body["limitations"][0]["message"]
    source = body["source"]
    assert source["sha256"] == snapshot.sha256
    assert source["client_requests"] == 9
    assert source["records"] == 1 + 1 + 8 + 8 + 1
    assert body["download"] == {
        "filename": "042.selection.jsonl",
        "content_type": "application/x-ndjson",
    }


def test_preview_counts_only_the_selected_rows(artifact: Path) -> None:
    snapshot = index_file(artifact)
    sel = select(snapshot, rows=[1, 2])
    body = export.preview(snapshot, sel, selection_block=selection_block(sel))
    fields = {entry["field"]: entry["count"] for entry in body["sensitive_fields"]}
    assert fields == {"error_message": 0, "endpoint": 2, "model": 0}


def test_preview_document_validates(artifact: Path) -> None:
    snapshot = index_file(artifact)
    sel = select(snapshot)
    body = export.preview(snapshot, sel, selection_block=selection_block(sel))
    document = to_wire(envelope("export_preview", body, snapshot))
    assert schema_errors("studio_export_preview_v1.schema.json", document) == []


def test_preview_says_when_error_messages_are_not_resident(artifact: Path) -> None:
    snapshot = index_file(artifact)
    snapshot = dataclasses.replace(
        snapshot,
        table=NoErrorMessages(snapshot.requests),
        error_messages_resident=False,
    )
    sel = select(snapshot)
    body = export.preview(snapshot, sel, selection_block=selection_block(sel))
    entry = body["sensitive_fields"][0]
    assert (entry["field"], entry["count"], entry["reason"]) == (
        "error_message",
        None,
        "not_resident",
    )
    document = to_wire(envelope("export_preview", body, snapshot))
    assert schema_errors("studio_export_preview_v1.schema.json", document) == []


def test_preview_of_a_growing_partial_index_says_so(artifact: Path) -> None:
    snapshot = index_file(artifact, sha256=None, growing=True, status="partial")
    sel = select(snapshot)
    body = export.preview(snapshot, sel, selection_block=selection_block(sel))
    codes = [note["code"] for note in body["limitations"]]
    assert codes[2:] == ["source_not_complete", "index_partial"]
    document = to_wire(envelope("export_preview", body, snapshot))
    assert schema_errors("studio_export_preview_v1.schema.json", document) == []


def test_download_headers_name_count_selection_and_source(artifact: Path) -> None:
    snapshot = index_file(artifact)
    sel = select(snapshot, params="case=c1&q=caf\u00e9")
    headers = dict(export.download_headers(snapshot, sel))
    assert headers == {
        "Content-Disposition": 'attachment; filename="042.selection.jsonl"',
        "X-Stormlog-Export-Profile": "original_bytes",
        "X-Stormlog-Export-Count": "8",
        "X-Stormlog-Selection": "case=c1&q=caf%C3%A9",
        "X-Stormlog-Source-Sha256": snapshot.sha256,
    }
    for name, value in headers.items():
        value.encode("ascii")


def test_download_headers_of_an_unfinished_file_name_the_prefix(
    artifact: Path,
) -> None:
    snapshot = index_file(artifact, sha256=None)
    headers = dict(export.download_headers(snapshot, select(snapshot)))
    assert "X-Stormlog-Source-Sha256" not in headers
    assert headers["X-Stormlog-Source-Prefix-Sha256"] == snapshot.prefix_sha256
    assert headers["X-Stormlog-Source-Prefix-Bytes"] == str(snapshot.bytes_read)


@pytest.mark.parametrize(
    ("path", "filename"),
    [
        ("runs/042.jsonl", "042.selection.jsonl"),
        ('we"ird name;.jsonl', "we_ird_name.selection.jsonl"),
        ("...jsonl", "artifact.selection.jsonl"),
    ],
)
def test_filenames_are_header_safe(path: str, filename: str) -> None:
    assert export._filename(path) == filename


@pytest.mark.parametrize(
    ("model", "local"),
    [
        ("/home/me/model", True),
        ("~/models/qwen", True),
        ("./model", True),
        ("../model", True),
        ("C:\\models\\qwen", True),
        ("\\\\server\\share", True),
        ("file:///models/qwen", True),
        ("Qwen/Qwen2.5-7B-Instruct", False),
        ("gpt-4o-mini", False),
        (None, False),
        (3, False),
    ],
)
def test_local_path_heuristic(model: Any, local: bool) -> None:
    assert export.looks_like_local_path(model) is local
