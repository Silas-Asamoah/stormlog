"""ArtifactIndexes: Studio's columnar index of an artifact (design §4.4, §8, §9.1).

Each test builds indexes with a pool the test runs by hand and a clock it
moves by hand, so a refresh happens exactly when the test says. The
fixtures are C5's standard artifacts; the few states they lack (system
samples, an unknown session, audio, a non-inference file, an unreadable
file) are made here from them, as noted on each helper.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
import time
import tracemalloc
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, BinaryIO, ParamSpec, TypeVar

import pytest

from stormlog.infer.correlation_codec import expand_inference_records
from stormlog.infer.latency_report import CLIENT_METRICS as LATENCY_METRICS
from stormlog.studio import index as index_module
from stormlog.studio.contracts import (
    ConflictError,
    DictRow,
    DictTable,
    ForbiddenError,
    IndexRegistry,
    IndexSnapshot,
    Limits,
    NotFoundError,
    NotResident,
    RequestRow,
    artifact_block,
    columnar,
    encode_json,
    publish_hook,
)
from stormlog.studio.index import (
    CLIENT_METRICS,
    ERROR_MESSAGE,
    GAPS,
    KEPT_EVENT_TYPES,
    ArtifactIndexes,
    ColumnRow,
)
from tests.studio.artifacts import (
    MIXED_CASES,
    MULTI_SESSION,
    NO_TERMINAL_RECORD,
    STANDARD_ARTIFACTS,
    GrowingCopy,
    rewrite_in_place,
    with_execution_import,
)
from tests.studio.conftest import StandardArtifacts

P = ParamSpec("P")
T = TypeVar("T")


# --- Test doubles ----------------------------------------------------------


class Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class ManualPool(Executor):
    """A builder pool that runs its tasks only when the test says."""

    def __init__(self) -> None:
        self.queue: list[tuple[Future[Any], Callable[[], Any]]] = []
        self.submitted = 0

    def submit(
        self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        future: Future[T] = Future()
        self.queue.append((future, lambda: fn(*args, **kwargs)))
        self.submitted += 1
        return future

    def run(self) -> None:
        while self.queue:
            future, task = self.queue.pop(0)
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(task())
            except BaseException as exc:  # handed to the future, as a pool does
                future.set_exception(exc)


class Root:
    """A WorkspaceRoot over one directory (C3's RootDirectory stands in)."""

    def __init__(self, path: Path, *, refuse_open: bool = False) -> None:
        self.path = path.resolve()
        self.refuse_open = refuse_open
        self.opened = 0

    def resolve(self, rel: str) -> Path:
        if rel.startswith("/") or ".." in rel.split("/"):
            raise ForbiddenError("path_not_allowed", f"{rel} is outside the root")
        target = self.path / rel
        if not target.is_file():
            raise NotFoundError("artifact_not_found", f"no artifact {rel}")
        return target

    def open(self, rel: str) -> BinaryIO:
        if self.refuse_open:
            raise PermissionError(13, "Permission denied", rel)
        self.opened += 1
        return self.resolve(rel).open("rb")

    def relative(self, path: Path) -> str:
        return path.relative_to(self.path).as_posix()


class Harness:
    """Indexes over one directory, a manual pool and a manual clock."""

    def __init__(self, directory: Path, limits: Limits | None = None, **options: Any):
        self.root = Root(directory)
        self.pool = ManualPool()
        self.clock = Clock()
        self.indexes = ArtifactIndexes(
            self.root, self.pool, limits or Limits(), clock=self.clock, **options
        )

    def build(self, rel: str) -> IndexSnapshot:
        self.indexes.snapshot(rel)
        self.pool.run()
        return self.indexes.snapshot(rel)

    def refresh(self, rel: str, seconds: float = 1.5) -> IndexSnapshot:
        self.clock.now += seconds
        return self.build(rel)


def harness_for(path: Path, limits: Limits | None = None, **options: Any) -> Harness:
    return Harness(path.parent, limits, **options)


# --- Expected values, from the file itself ---------------------------------


def complete_lines(path: Path) -> list[bytes]:
    data = path.read_bytes()
    return data[: data.rfind(b"\n") + 1].split(b"\n")[:-1]


def all_records(path: Path) -> list[dict[str, Any]]:
    """Every decoded record of the file's complete lines."""
    return expand_inference_records(
        json.loads(line) for line in complete_lines(path) if line.strip()
    )


def is_client(record: dict[str, Any]) -> bool:
    return (
        record.get("event_type") == "infer.request"
        and "context" not in record
        and record.get("schema_version", 1) == 1
    )


def client_lines(path: Path) -> list[tuple[int, bytes, dict[str, Any]]]:
    """(line number, bytes, record) of each client request line."""
    out = []
    for number, line in enumerate(complete_lines(path)):
        if b'"infer.request"' not in line:
            continue
        record = json.loads(line)
        if is_client(record):
            out.append((number, line, record))
    return out


def write_records(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def edited_copy(
    source: Path, target: Path, edit: Callable[[list[dict[str, Any]]], None]
) -> Path:
    """A copy of ``source``'s records after ``edit`` (contexts expanded)."""
    records = all_records(source)
    edit(records)
    return write_records(target, records)


def check_rows(snapshot: IndexSnapshot, path: Path) -> None:
    """Rows, row attributes and columns equal the file's client records."""
    expected = client_lines(path)
    assert len(snapshot.requests) == len(snapshot.table) == len(expected)
    for i, (number, line, record) in enumerate(expected):
        row = snapshot.requests[i]
        assert isinstance(row, ColumnRow) and isinstance(row, RequestRow)
        assert dict(row) == record and list(row) == list(record)
        assert row.index == i and row.ref.line == number
        assert row.ref.sha256 == hashlib.sha256(line).hexdigest()
        twin = DictRow(record, index=i, ref=row.ref)
        for name in ("session_id", "request_id", "record_id", "case_id", "phase"):
            assert getattr(row, name) == getattr(twin, name), name
        assert row.gap_count == twin.gap_count
        assert dict(row.client) == dict(twin.client)
        assert tuple(row.load_gaps()) == tuple(twin.load_gaps())
        assert row.load_error_message() == twin.load_error_message()
        assert i in snapshot.by_request_id[row.request_id]
    keys = {key for _, _, record in expected for key in record} | {"no_such_key"}
    for name in sorted(keys) + [f"client.{m[7:]}" for m in CLIENT_METRICS]:
        column = snapshot.column(name)
        assert len(column) == len(expected)
        for i, (_, _, record) in enumerate(expected):
            if name in CLIENT_METRICS:
                assert column[i] == snapshot.requests[i].client[name].value_ms
            else:
                assert column[i] == record.get(name), (name, i)


# --- Every standard artifact -----------------------------------------------


SINGLE_SESSION = sorted(set(STANDARD_ARTIFACTS) - MULTI_SESSION)


@pytest.mark.parametrize("name", SINGLE_SESSION)
def test_each_standard_artifact_indexes_its_client_requests(
    studio_artifacts: StandardArtifacts, name: str
) -> None:
    path = studio_artifacts[name]
    snapshot = harness_for(path).build(path.name)
    assert (snapshot.status, snapshot.failure) == ("ready", None)
    assert snapshot.path == path.name and len(snapshot.index_id) == 32
    check_rows(snapshot, path)


@pytest.mark.parametrize("name", SINGLE_SESSION)
def test_each_standard_artifact_keeps_its_records_and_counts(
    studio_artifacts: StandardArtifacts, name: str
) -> None:
    path = studio_artifacts[name]
    snapshot = harness_for(path).build(path.name)
    records = all_records(path)
    kept = {
        kind: tuple(r for r in records if r.get("event_type") == kind)
        for kind in KEPT_EVENT_TYPES
    }
    assert snapshot.kept == {kind: rows for kind, rows in kept.items() if rows}
    inventory: dict[str, dict[int, int]] = {}
    for record in records:
        counts = inventory.setdefault(str(record["event_type"]), {})
        version = record.get("schema_version", 1)
        counts[version] = counts.get(version, 0) + 1
    contexts = sum(1 for line in complete_lines(path) if b'"infer.context"' in line)
    if contexts:
        inventory["infer.context"] = {4: contexts}
    assert snapshot.inventory == inventory
    assert snapshot.problems == () and snapshot.problem_count == 0
    sessions = [r for r in records if r.get("event_type") == "infer.session"]
    assert snapshot.session is not None
    assert snapshot.session.session_id == sessions[0]["session_id"]
    assert snapshot.session.terminal is (name not in NO_TERMINAL_RECORD)
    assert snapshot.session.status == sessions[-1]["status"]
    assert snapshot.origin_ns == sessions[0]["timestamp_ns"]
    encode_json({"artifact": artifact_block(snapshot), "kept": snapshot.kept})


def test_the_line_count_is_every_complete_line_consumed(tmp_path: Path) -> None:
    lines = [
        json.dumps({"event_type": "infer.session", "session_id": "s"}),
        "",
        "not json",
        json.dumps({"event_type": "infer.request", "request_id": "a"}),
    ]
    path = tmp_path / "count.jsonl"
    path.write_text("\n".join(lines) + "\n" + '{"tail": ')
    harness = harness_for(path)
    snapshot = harness.build(path.name)
    assert snapshot.line_count == 4 and snapshot.cut is None
    assert harness.indexes.wait(path.name, 0).line_count == 4


@pytest.mark.parametrize("name", SINGLE_SESSION)
def test_each_complete_artifact_hashes_as_the_file_does(
    studio_artifacts: StandardArtifacts, name: str
) -> None:
    path = studio_artifacts[name]
    snapshot = harness_for(path).build(path.name)
    data = path.read_bytes()
    complete = data[: data.rfind(b"\n") + 1]
    assert snapshot.identity is not None and snapshot.identity.size == len(data)
    assert snapshot.bytes_read == len(complete)
    assert snapshot.prefix_sha256 == hashlib.sha256(complete).hexdigest()
    if name == "truncated_mid_line":
        assert snapshot.sha256 is None and snapshot.pending_tail_bytes > 0
    else:
        assert snapshot.sha256 == hashlib.sha256(data).hexdigest()
        assert snapshot.pending_tail_bytes == 0


def test_the_column_names_are_the_latency_reports() -> None:
    assert CLIENT_METRICS == tuple(LATENCY_METRICS)


def test_an_execution_import_counts_server_view_requests_but_never_rows(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["execution_import"]
    snapshot = harness_for(path).build(path.name)
    assert snapshot.inventory["infer.request"] == {1: 10, 2: 6}
    assert len(snapshot.requests) == 10
    assert all("context" not in row for row in snapshot.requests)


def test_every_request_links_its_dispatch_and_first_content(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["mixed"]
    harness = harness_for(path)
    snapshot = harness.build(path.name)
    assert set(snapshot.request_links) == set(snapshot.by_request_id)
    for row in snapshot.requests:
        dispatch, first = snapshot.request_links[row.request_id]
        assert dispatch is not None
        line = json.loads(harness.indexes.read_line(path.name, dispatch))
        assert (line["event_type"], line["request_id"]) == (
            "infer.dispatch",
            row.request_id,
        )
        if first is not None:
            line = json.loads(harness.indexes.read_line(path.name, first))
            assert line["event_type"] == "infer.first_content"
    assert sum(1 for p in snapshot.request_links.values() if p[1] is None) == 7
    with pytest.raises(KeyError):
        snapshot.request_links["nope"]


def test_the_refs_are_the_first_lines_of_each_type(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["synthetic"]
    snapshot = harness_for(path).build(path.name)
    refs = snapshot.refs["infer.request"]
    assert len(refs) == index_module.REFS_PER_TYPE
    assert [ref.line for ref in refs[:3]] == [r.ref.line for r in snapshot.requests[:3]]


# --- §8 states -------------------------------------------------------------


def test_a_multi_session_file_fails_with_no_rows(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["multi_session"]
    snapshot = harness_for(path).build(path.name)
    assert (snapshot.status, snapshot.failure) == ("failed", "multiple_sessions")
    assert len(snapshot.requests) == len(snapshot.table) == 0
    assert snapshot.column("request_id") == []


def test_an_unknown_session_never_counts_as_a_second_one(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    """Made here: one request without a session_id, one with null."""

    def drop_session(records: list[dict[str, Any]]) -> None:
        requests = [r for r in records if r.get("event_type") == "infer.request"]
        del requests[0]["session_id"]
        requests[1]["session_id"] = "unknown"

    path = edited_copy(studio_artifacts["mixed"], tmp_path / "u.jsonl", drop_session)
    snapshot = harness_for(path).build(path.name)
    assert snapshot.status == "ready"
    assert snapshot.requests[0].session_id == "unknown"
    check_rows(snapshot, path)


def test_a_null_identity_is_unknown_never_none(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    """Made here: a null session_id, request_id, case_id and phase (F-5)."""

    def null_identity(records: list[dict[str, Any]]) -> None:
        requests = [r for r in records if r.get("event_type") == "infer.request"]
        requests[0].update(session_id=None, request_id=None, case_id=None)
        requests[1]["phase"] = None
        records[:] = [r for r in records if r.get("event_type") != "infer.session"]

    for source in ("mixed", "warmup_measured"):
        target = tmp_path / f"{source}.null.jsonl"
        path = edited_copy(studio_artifacts[source], target, null_identity)
        snapshot = harness_for(path).build(path.name)
        assert (snapshot.status, snapshot.failure) == ("ready", None), source
        first, second = snapshot.requests[0], snapshot.requests[1]
        assert (first.session_id, first.request_id) == ("unknown", "unknown")
        assert (first.case_id, second.phase) == ("unknown", "unknown")
        assert "None" not in snapshot.by_request_id
        assert snapshot.by_request_id["unknown"] == (0,)
        check_rows(snapshot, path)


def test_a_second_session_record_alone_fails_the_index(tmp_path: Path) -> None:
    path = write_records(
        tmp_path / "s.jsonl",
        [
            {"event_type": "infer.session", "session_id": "a", "status": "running"},
            {"event_type": "infer.session", "session_id": "b", "status": "running"},
        ],
    )
    snapshot = harness_for(path).build(path.name)
    assert (snapshot.status, snapshot.failure) == ("failed", "multiple_sessions")


def test_an_all_failed_run_keeps_every_failure_and_message(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["all_failed"]
    snapshot = harness_for(path).build(path.name)
    assert snapshot.status == "ready" and len(snapshot.requests) == 10
    assert {row["status"] for row in snapshot.requests} == {"error"}
    assert all(isinstance(m, str) for m in snapshot.column(ERROR_MESSAGE))
    assert all(v is None for v in snapshot.column("client.ttft"))


def test_an_empty_run_is_ready_with_no_rows(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["empty"]
    snapshot = harness_for(path).build(path.name)
    assert snapshot.status == "ready" and len(snapshot.requests) == 0
    assert snapshot.session is not None and snapshot.session.terminal
    assert snapshot.column("client.e2e") == []


def test_a_cut_mid_line_reports_its_tail_once_still(
    fresh_artifact: Callable[[str], Path],
) -> None:
    path = fresh_artifact("truncated_mid_line")
    harness = harness_for(path)
    first = harness.build(path.name)
    assert first.status == "ready" and first.problems == ()
    assert first.session is not None and not first.session.terminal
    later = harness.refresh(path.name, seconds=6)
    assert later.generation == first.generation and later.index_id == first.index_id
    (problem,) = later.problems
    assert problem.code == "truncated_final_line" and later.problem_count == 1
    assert problem.ref.offset == later.bytes_read
    with path.open("ab") as handle:
        handle.write(b"\n")
    healed = harness.refresh(path.name)
    assert [p.code for p in healed.problems] == ["not_json"]
    assert healed.pending_tail_bytes == 0 and healed.sha256 is not None


def test_a_cut_after_a_newline_has_no_terminal_record(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["truncated_after_newline"]
    harness = harness_for(path)
    snapshot = harness.refresh(path.name, seconds=10)
    assert snapshot.status == "ready" and snapshot.problems == ()
    assert snapshot.session is not None and not snapshot.session.terminal


def test_a_growing_file_refreshes_under_the_same_index_id(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    source = studio_artifacts["mixed"]
    growing = GrowingCopy(source, tmp_path / "g.jsonl", lines=40)
    harness = harness_for(growing.target)
    first = harness.build("g.jsonl")
    assert first.status == "ready" and not first.growing
    assert harness.indexes.snapshot("g.jsonl") is first  # within 1 s: no refresh
    growing.append(30)
    growing.append_partial()
    grown = harness.refresh("g.jsonl")
    assert grown.index_id == first.index_id and grown.generation > first.generation
    assert grown.growing and grown.pending_tail_bytes > 0
    assert len(grown.requests) > len(first.requests)
    assert len(first.requests) == len(first.table.column("request_id"))
    growing.finish()
    done = harness.refresh("g.jsonl")
    assert (
        done.growing and done.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    check_rows(done, growing.target)
    still = harness.refresh("g.jsonl", seconds=6)
    assert not still.growing and still.generation == done.generation
    assert still.requests is done.requests  # republished, same rows and caches


def test_publish_listeners_see_each_complete_snapshot_once_current(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    source = studio_artifacts["mixed"]
    growing = GrowingCopy(source, tmp_path / "g.jsonl", lines=40)
    harness = harness_for(growing.target, batch_bytes=2_000)
    indexes = harness.indexes
    assert publish_hook(indexes) is indexes
    seen: list[IndexSnapshot] = []

    def listen(snapshot: IndexSnapshot) -> None:
        assert indexes.snapshot("g.jsonl") is snapshot  # current when called
        seen.append(snapshot)

    def broken(snapshot: IndexSnapshot) -> None:
        raise RuntimeError("a listener bug")

    indexes.add_publish_listener(broken)
    indexes.add_publish_listener(listen)
    first = harness.build("g.jsonl")
    assert seen == [first] and first.status == "ready"  # no per-batch calls
    growing.append(30)
    grown = harness.refresh("g.jsonl")
    assert seen[-1] is grown and grown.generation > first.generation
    assert all(s.status != "building" for s in seen)
    count = len(seen)
    still = harness.refresh("g.jsonl", seconds=6)  # growing flips: a replace
    assert len(seen) == count + 1 and seen[-1] is still
    assert not still.growing and still.table is grown.table  # id(table) survives


def test_the_growing_fixture_is_ready_and_growing_after_an_append(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    source = studio_artifacts["growing"]
    path = tmp_path / source.name
    path.write_bytes(source.read_bytes())
    harness = harness_for(path)
    first = harness.build(path.name)
    assert first.status == "ready" and len(first.requests) == 4
    assert first.session is not None and first.session.status == "running"
    with path.open("ab") as handle:
        handle.write(complete_lines(source)[-1] + b"\n")
    assert harness.refresh(path.name).growing


def test_a_rewrite_in_place_rebuilds_under_a_new_index_id(
    fresh_artifact: Callable[[str], Path],
) -> None:
    path = fresh_artifact("tokens_exact")
    harness = harness_for(path)
    first = harness.build(path.name)
    rewrite_in_place(path, cases=MIXED_CASES[:1])
    second = harness.refresh(path.name)
    assert second.index_id != first.index_id and second.status == "ready"
    assert second.generation >= 1
    check_rows(second, path)
    assert second.session is not None and first.session is not None
    assert second.session.session_id != first.session.session_id
    assert len(first.requests) == 10  # the old snapshot is untouched


def test_a_completed_build_trims_the_heap_once(
    fresh_artifact: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def trim() -> bool:
        calls.append(1)
        return True

    monkeypatch.setattr(index_module, "trim_heap", trim)
    path = fresh_artifact("tokens_exact")
    harness = harness_for(path, batch_bytes=1_000)
    harness.build(path.name)
    assert len(calls) == 1  # once per build, not per batch
    with path.open("ab") as handle:
        handle.write(complete_lines(path)[-1] + b"\n")
    harness.refresh(path.name)
    assert len(calls) == 1  # growth is not a build
    rewrite_in_place(path, cases=MIXED_CASES[:1])
    harness.refresh(path.name)
    assert len(calls) == 2  # a rebuild under a new index ID is


def test_trim_heap_is_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    assert index_module.trim_heap() is sys.platform.startswith("linux")

    def no_libc(name: Any) -> Any:
        raise OSError("no C library")

    monkeypatch.setattr(index_module.ctypes, "CDLL", no_libc)
    assert index_module.trim_heap() is False
    monkeypatch.setattr(index_module.sys, "platform", "darwin")
    assert index_module.trim_heap() is False


def test_an_import_replace_keeps_the_index_and_adds_its_lines(
    fresh_artifact: Callable[[str], Path],
) -> None:
    path = fresh_artifact("tokens_exact")
    harness = harness_for(path)
    first = harness.build(path.name)
    with_execution_import(path)
    second = harness.refresh(path.name)
    assert second.index_id == first.index_id
    assert second.generation > first.generation
    assert second.inventory["infer.request"][2] > 0
    assert len(second.requests) == len(first.requests)
    assert second.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_an_unreadable_file_fails_with_no_identity(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["mixed"]
    harness = harness_for(path)
    harness.root.refuse_open = True
    snapshot = harness.build(path.name)
    assert (snapshot.status, snapshot.failure) == ("failed", "unreadable")
    assert snapshot.identity is None and artifact_block(snapshot)["size"] is None
    harness.root.refuse_open = False
    assert harness.refresh(path.name).status == "ready"


def test_a_path_outside_the_root_or_missing_raises_at_once(tmp_path: Path) -> None:
    indexes = Harness(tmp_path).indexes
    with pytest.raises(ForbiddenError):
        indexes.snapshot("../x.jsonl")
    with pytest.raises(NotFoundError):
        indexes.snapshot("missing.jsonl")
    with pytest.raises(NotFoundError):
        indexes.wait("missing.jsonl", 0.01)


def test_a_file_without_inference_records_is_not_an_artifact(tmp_path: Path) -> None:
    path = tmp_path / "other.jsonl"
    path.write_bytes(b"")
    harness = harness_for(path)
    first = harness.build(path.name)
    assert (first.status, first.failure) == ("failed", "not_an_inference_artifact")
    write_records(path, [{"event_type": "other", "x": 1}])
    assert harness.refresh(path.name).failure == "not_an_inference_artifact"
    with path.open("a") as handle:
        handle.write(json.dumps({"event_type": "infer.session", "session_id": "s"}))
        handle.write("\n")
    assert harness.refresh(path.name).status == "ready"


def test_an_audio_artifact_is_refused(tmp_path: Path) -> None:
    """Made here: what profile-audio's records look like to the index."""
    path = write_records(
        tmp_path / "audio.jsonl",
        [
            {"event_type": "infer.session", "session_id": "s", "status": "running"},
            {"event_type": "infer.audio.request", "session_id": "s"},
        ],
    )
    snapshot = harness_for(path).build(path.name)
    assert (snapshot.status, snapshot.failure) == ("failed", "audio")


def test_newer_types_versions_and_bad_lines_never_fail_an_index(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    source = studio_artifacts["tokens_exact"]
    newer = [
        b'{"event_type": "infer.request", "schema_version": 2, "request_id": "v2"}',
        b'{"event_type": "infer.future", "schema_version": 9, "x": {"y": [1]}}',
        b'{"schema_version": 4, "event_type": "infer.future", "context_id": "zz"}',
        b"not json at all",
        b"[1, 2, 3]",
        b'{"event_type": 7}',
        b"",
    ]
    path = tmp_path / "newer.jsonl"
    path.write_bytes(source.read_bytes() + b"\n".join(newer) + b"\n")
    snapshot = harness_for(path).build(path.name)
    assert snapshot.status == "ready" and len(snapshot.requests) == 10
    codes = [problem.code for problem in snapshot.problems]
    assert codes == ["unsupported_version", "undecodable", "not_json", "not_object"]
    assert snapshot.problem_count == 4
    assert snapshot.inventory["infer.request"] == {1: 10, 2: 1}
    assert snapshot.inventory["infer.future"] == {9: 1}
    assert snapshot.inventory["unknown"] == {1: 1}
    assert "v2" not in snapshot.by_request_id


def test_problems_are_counted_by_code_apart_from_server_view_requests(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    """Made here: an execution import (server-view requests at version 2)
    with one client request at version 2 and bad lines (Fable's F-9)."""
    source = studio_artifacts["execution_import"]
    extra = [
        b'{"event_type": "infer.request", "schema_version": 2, "request_id": "v2"}',
        b"not json at all",
        b"also not json",
        b"[1]",
    ]
    path = tmp_path / "counts.jsonl"
    path.write_bytes(source.read_bytes() + b"\n".join(extra) + b"\n" + b'{"t')
    harness = harness_for(path, tail_grace_s=0.0)
    snapshot = harness.build(path.name)
    assert snapshot.inventory["infer.request"] == {1: 10, 2: 7}  # 6 server view
    assert dict(snapshot.problem_counts) == {
        "unsupported_version": 1,
        "not_json": 2,
        "not_object": 1,
        "truncated_final_line": 1,
    }
    assert sum(snapshot.problem_counts.values()) == snapshot.problem_count == 5
    clean = harness_for(source).build(source.name)
    assert dict(clean.problem_counts) == {} and clean.problem_count == 0


def test_kept_records_are_finite(tmp_path: Path) -> None:
    path = tmp_path / "nan.jsonl"
    path.write_text(
        '{"event_type": "infer.session", "session_id": "s", "status": "running",'
        ' "timestamp_ns": 5, "config": {"rate": NaN, "x": [Infinity, 1.0]}}\n'
        '{"event_type": "infer.workload", "value": -Infinity}\n'
    )
    snapshot = harness_for(path).build(path.name)
    assert snapshot.session is not None
    assert snapshot.session.config == {"rate": None, "x": [None, 1.0]}
    assert snapshot.kept["infer.workload"] == (
        {"event_type": "infer.workload", "value": None},
    )
    encode_json({"kept": snapshot.kept, "config": snapshot.session.config})
    assert snapshot.origin_ns == 5


def test_without_a_session_record_the_origin_is_the_first_request(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    def drop_sessions(records: list[dict[str, Any]]) -> None:
        records[:] = [r for r in records if r.get("event_type") != "infer.session"]

    path = edited_copy(studio_artifacts["mixed"], tmp_path / "n.jsonl", drop_sessions)
    snapshot = harness_for(path).build(path.name)
    assert snapshot.status == "ready" and snapshot.session is None
    assert snapshot.origin_ns == snapshot.requests[0]["started_at_ns"]
    assert artifact_block(snapshot)["terminal"] is False


def test_duplicate_request_ids_keep_every_row(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    def duplicate(records: list[dict[str, Any]]) -> None:
        requests = [r for r in records if r.get("event_type") == "infer.request"]
        records.append(dict(requests[0]))

    path = edited_copy(studio_artifacts["synthetic"], tmp_path / "d.jsonl", duplicate)
    snapshot = harness_for(path).build(path.name)
    first = snapshot.requests[0]
    assert snapshot.by_request_id[first.request_id] == (0, len(snapshot.requests) - 1)
    assert len(snapshot.by_request_id) == len(snapshot.requests) - 1
    with pytest.raises(KeyError):
        snapshot.by_request_id["nope"]


def test_odd_values_round_trip_exactly(tmp_path: Path) -> None:
    """Made here: values no column type holds go to the side table."""
    base = {"event_type": "infer.request", "session_id": "s", "case_id": "c"}
    rows: list[dict[str, Any]] = [
        {**base, "request_id": "a", "ttft_ms": 5, "x": True, "big": 2**70},
        {**base, "request_id": "b", "ttft_ms": 1.5, "x": "yes", "big": 3},
        {**base, "request_id": "c", "ttft_ms": None, "x": [1, {"y": None}], "big": -1},
        {**base, "request_id": "a" + chr(0xD800), "ttft_ms": float("nan"), "x": None},
        {**base, GAPS: "not a list", ERROR_MESSAGE: 7, "request_id": "e"},
        {**base, GAPS: [1, "x", None, True, 2.5], "request_id": "f"},
        {"event_type": "infer.request", "request_id": None},
    ]
    path = write_records(tmp_path / "odd.jsonl", rows)
    snapshot = harness_for(path).build(path.name)
    got = [dict(row) for row in snapshot.requests]
    assert math.isnan(got[3].pop("ttft_ms")) and math.isnan(rows[3]["ttft_ms"])
    rows[3].pop("ttft_ms")
    rows[5][GAPS] = [1.0, math.nan, math.nan, math.nan, 2.5]
    assert repr(got) == repr(json.loads(json.dumps(rows)))
    assert snapshot.requests[6].request_id == "unknown"  # F-5
    assert snapshot.requests[4].gap_count == 0 and snapshot.requests[5].gap_count == 5
    assert snapshot.column("big") == [2**70, 3, -1, None, None, None, None]
    assert snapshot.column("x")[:3] == [True, "yes", [1, {"y": None}]]
    assert snapshot.requests[4].load_gaps() == ()
    assert snapshot.requests[4].load_error_message() is None


# With their wrappers, 126 lists nest 128 deep (the limit) on line 0 and 127
# on the others; 127 lists nest 129 on line 0. 900 and 5000 are past it.
@pytest.mark.parametrize("depth", [126, 127, 900, 5000])
def test_a_deeply_nested_record_is_a_problem_never_a_failed_index(
    tmp_path: Path, depth: int
) -> None:
    """A line nested past MAX_NESTING is the reader's not_json problem on
    every Python (json.loads takes 50,000 levels from 3.14). One within it
    is kept whole, and its request's detail can be copied and written as a
    route does (F-4: finite_copy recursed past Python's limit and the
    request route answered 500). Either way the index is ready.
    """
    from stormlog.infer.diagnosis_inputs import MAX_NESTING
    from stormlog.studio.summaries import request_detail

    nested = "[" * depth + "]" * depth
    head = '{"event_type":"infer.session","session_id":"s1","timestamp_ns":1,'
    lines = [
        head + '"status":"running","config":{"d":' + nested + "}}",
        '{"event_type":"infer.workload","schema_version":1,"w":' + nested + "}",
        '{"event_type":"infer.request","session_id":"s1","request_id":"q",'
        '"status":"ok","weird":' + nested + "}",
        '{"event_type":"infer.system_sample","t":1,"weird":' + nested + "}",
        '{"event_type":"infer.request","session_id":"s1","request_id":"r"}',
    ]
    path = tmp_path / f"deep{depth}.jsonl"
    path.write_text("\n".join(lines) + "\n")
    snapshot = harness_for(path).build(path.name)
    assert (snapshot.status, snapshot.failure) == ("ready", None)
    problems = {p.ref.line: p.code for p in snapshot.problems}
    nesting = [depth + 2, depth + 1, depth + 1, depth + 1]
    deep = {line for line, n in enumerate(nesting) if n > MAX_NESTING}
    assert problems == dict.fromkeys(deep, "not_json")
    assert snapshot.problem_count == len(problems)
    ids = [row.request_id for row in snapshot.requests]
    assert ids == (["r"] if 2 in deep else ["q", "r"])
    for line, kind in ((0, "infer.session"), (1, "infer.workload")):
        kept = snapshot.kept.get(kind, ())
        assert len(kept) == (line not in deep), (depth, line)
        encode_json({"kept": kept})
    session = snapshot.session
    assert (session is None) == (0 in deep)
    assert session is None or session.config == json.loads('{"d":' + nested + "}")
    if 2 not in deep:
        assert snapshot.requests[0]["weird"] == json.loads(nested)
    if 3 not in deep:
        assert snapshot.samples[0]["weird"] == json.loads(nested)
    for row in range(len(snapshot.requests)):
        encode_json(request_detail(snapshot, row, None))
    encode_json({"artifact": artifact_block(snapshot)})


def test_many_distinct_strings_move_to_a_byte_pool(tmp_path: Path) -> None:
    rows = [
        {"event_type": "infer.request", "request_id": str(i), "tag": f"t{i}"}
        for i in range(70_000)
    ]
    path = write_records(tmp_path / "tags.jsonl", rows)
    snapshot = harness_for(path).build(path.name)
    assert snapshot.column("tag")[69_999] == "t69999"
    assert snapshot.requests[123]["tag"] == "t123"


# --- Budgets (§8.1): partial with the cut named ----------------------------


def test_past_the_row_budget_the_index_stops_at_a_named_cut(
    fresh_artifact: Callable[[str], Path],
) -> None:
    path = fresh_artifact("synthetic")
    whole = harness_for(path).build(path.name)
    harness = harness_for(path, Limits(row_bytes=200_000))
    cut = harness.build(path.name)
    assert (cut.status, cut.failure) == ("partial", None)
    assert cut.limits_hit == {"rows"}
    kept = len(cut.requests)
    assert 0 < kept < len(whole.requests)
    first_missing = whole.requests[kept].ref
    assert cut.bytes_read == first_missing.offset
    assert cut.cut == first_missing and whole.cut is None
    assert cut.line_count == first_missing.line
    assert whole.line_count == len(complete_lines(path))
    assert artifact_block(cut)["cut"] == {
        "line": first_missing.line,
        "offset": first_missing.offset,
        "length": first_missing.length,
        "sha256": first_missing.sha256,
    }
    data = path.read_bytes()
    assert cut.prefix_sha256 == hashlib.sha256(data[: cut.bytes_read]).hexdigest()
    assert cut.sha256 is None
    assert cut.inventory["infer.request"] == {1: kept}
    assert "infer.phase_window" not in cut.inventory  # it follows the cut
    entry = harness.indexes._entries[path.name]
    assert entry.state.requests.row_bytes() <= 200_000 + 256 * 600
    assert artifact_block(cut)["limits_hit"] == ["rows"]
    with path.open("ab") as handle:
        handle.write(complete_lines(path)[-1] + b"\n")
    later = harness.refresh(path.name)
    assert len(later.requests) == kept and later.bytes_read == cut.bytes_read
    assert later.growing and later.cut == first_missing


COLUMNAR = ("mixed", "synthetic", "warmup_measured", "non_streaming", "tokens_mixed")


def _same_floats(a: Sequence[float], b: Sequence[float]) -> bool:
    return len(a) == len(b) and all(
        x == y or (math.isnan(x) and math.isnan(y)) for x, y in zip(a, b)
    )


@pytest.mark.parametrize("name", COLUMNAR)
def test_the_columnar_arrays_equal_the_rows(
    studio_artifacts: StandardArtifacts, name: str
) -> None:
    path = studio_artifacts[name]
    snapshot = harness_for(path).build(path.name)
    table = columnar(snapshot.table)
    assert table is not None
    twin = DictTable(
        [DictRow(dict(row), index=row.index, ref=row.ref) for row in snapshot.requests]
    )
    for metric in CLIENT_METRICS:
        assert _same_floats(table.client_values(metric), twin.client_values(metric))
        codes, reasons = table.client_reasons(metric)
        twin_codes, twin_reasons = twin.client_reasons(metric)
        assert [reasons[c] for c in codes] == [twin_reasons[c] for c in twin_codes]
        assert reasons[0] == (None, False)
    assert list(table.line_refs()) == list(twin.line_refs())
    assert table.line_refs()[-1] == snapshot.requests[-1].ref
    assert list(table.line_lengths()) == list(twin.line_lengths())
    assert list(table.gap_counts()) == list(twin.gap_counts())
    pool, twin_pool = table.gap_pool(), twin.gap_pool()
    assert list(pool.counts) == list(twin_pool.counts)
    for i in range(len(snapshot.requests)):
        mine = pool.values[pool.starts[i] : pool.starts[i] + pool.counts[i]]
        theirs = twin_pool.values[
            twin_pool.starts[i] : twin_pool.starts[i] + twin_pool.counts[i]
        ]
        assert _same_floats(mine, theirs), i
    assert table.gap_pool() is pool  # cached per table
    with pytest.raises(KeyError):
        table.client_values("client.no_such_metric")


def test_columnar_arrays_are_copies_the_builder_never_changes(
    fresh_artifact: Callable[[str], Path],
) -> None:
    np = pytest.importorskip("numpy")
    path = fresh_artifact("mixed")
    lines = complete_lines(path)
    harness = harness_for(path)
    first = harness.build(path.name)
    table = columnar(first.table)
    assert table is not None
    pool = table.gap_pool()
    values = np.frombuffer(pool.values, dtype=np.float64)
    ttft = np.frombuffer(table.client_values(CLIENT_METRICS[0]), dtype=np.float64)
    before = (values.copy(), ttft.copy(), list(table.gap_counts()))
    with path.open("ab") as handle:  # the builder appends to its arrays
        handle.write(b"\n".join(lines) + b"\n")
    later = harness.refresh(path.name)
    assert len(later.requests) == 2 * len(first.requests)
    assert np.array_equal(values, before[0], equal_nan=True)
    assert np.array_equal(ttft, before[1], equal_nan=True)
    assert list(table.gap_counts()) == before[2]
    assert len(table.line_refs()) == len(first.requests)


def test_a_dropped_gap_pool_is_not_resident_in_the_columnar_view(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["synthetic"]
    snapshot = harness_for(path, Limits(gap_bytes=60_000)).build(path.name)
    table = columnar(snapshot.table)
    assert table is not None and not snapshot.gaps_resident
    with pytest.raises(NotResident):
        table.gap_pool()
    counts = table.gap_counts()
    assert list(counts) == [row.gap_count for row in snapshot.requests]


def test_context_definitions_are_bounded_and_counted_in_memory(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    """Made here: many distinct context definitions (Fable's F-3)."""
    from tests.studio.test_artifact_reader import context_lines

    session = complete_lines(studio_artifacts["mixed"])[0]
    lines = [session, *context_lines(studio_artifacts["execution_import"], 300)]
    path = tmp_path / "contexts.jsonl"
    path.write_bytes(b"\n".join(lines) + b"\n")
    whole = harness_for(path)
    ready = whole.build(path.name)
    assert ready.status == "ready" and ready.limits_hit == frozenset()
    entry = whole.indexes._entries[path.name]
    assert entry.reader.context_bytes > 300 * len(lines[1])
    assert whole.indexes.stats()[path.name]["memory_bytes"] >= (
        entry.reader.context_bytes
    )
    harness = harness_for(path, Limits(context_bytes=50_000))
    cut = harness.build(path.name)
    assert (cut.status, cut.failure) == ("partial", None)
    assert cut.limits_hit == {"contexts"}
    assert cut.problem_count > 0
    assert {p.code for p in cut.problems} == {"undecodable"}
    assert artifact_block(cut)["limits_hit"] == ["contexts"]
    limited = harness.indexes._entries[path.name].reader
    assert limited.context_bytes <= 50_000


def test_past_the_gap_budget_gaps_are_read_from_the_file(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["synthetic"]
    harness = harness_for(path, Limits(gap_bytes=60_000))
    snapshot = harness.build(path.name)
    assert snapshot.status == "ready" and snapshot.limits_hit == {"gaps"}
    assert not snapshot.gaps_resident and snapshot.error_messages_resident
    row = snapshot.requests[1500]
    with pytest.raises(NotResident):
        row[GAPS]
    with pytest.raises(NotResident):
        row.get(GAPS)
    with pytest.raises(NotResident):
        snapshot.column(GAPS)
    with pytest.raises(NotResident):
        dict(row)
    record = json.loads(harness.indexes.read_line(path.name, row.ref))
    assert row.load_gaps() == DictRow(record).load_gaps() and row.gap_count == 15
    assert GAPS in row and len(row) == len(record)


def test_past_the_error_budget_messages_are_read_from_the_file(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["all_failed"]
    snapshot = harness_for(path, Limits(error_message_bytes=100)).build(path.name)
    assert snapshot.limits_hit == {"error_messages"} and snapshot.status == "ready"
    assert not snapshot.error_messages_resident
    expected = [record for _, _, record in client_lines(path)]
    for row, record in zip(snapshot.requests, expected):
        with pytest.raises(NotResident):
            row.get(ERROR_MESSAGE)
        assert row.load_error_message() == record[ERROR_MESSAGE]
    with pytest.raises(NotResident):
        snapshot.column(ERROR_MESSAGE)


def test_a_null_error_message_needs_no_line_once_the_pool_is_dropped(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["mixed"]
    harness = harness_for(path, Limits(error_message_bytes=100))
    snapshot = harness.build(path.name)
    assert not snapshot.error_messages_resident
    expected = [record for _, _, record in client_lines(path)]
    nulls = [i for i, r in enumerate(expected) if r.get(ERROR_MESSAGE) is None]
    texts = [i for i, r in enumerate(expected) if isinstance(r.get(ERROR_MESSAGE), str)]
    assert nulls and texts
    opened = harness.root.opened
    for i in nulls:  # known without the pool (F-11)
        row = snapshot.requests[i]
        assert row.get(ERROR_MESSAGE) is None and row.load_error_message() is None
    assert harness.root.opened == opened
    for i in texts:
        with pytest.raises(NotResident):
            snapshot.requests[i].get(ERROR_MESSAGE)
        assert snapshot.requests[i].load_error_message() == expected[i][ERROR_MESSAGE]
    assert harness.root.opened == opened + len(texts)


def with_samples(source: Path, target: Path, count: int) -> Path:
    """Made here: C5's profiles use the noop sampler, so this appends
    psutil-shaped ``infer.system_sample`` records to a copy."""
    sample = {
        "schema_version": 1,
        "event_type": "infer.system_sample",
        "sampler": "psutil",
        "device_id": None,
        "observation_scope": "client_local",
        "process_rss_bytes": 1,
        "metadata": {"pid": 1, "note": "x" * 50},
    }
    lines = [
        json.dumps({**sample, "timestamp_ns": i, "process_rss_bytes": i})
        for i in range(count)
    ]
    target.write_bytes(source.read_bytes() + "\n".join(lines).encode() + b"\n")
    return target


def test_samples_are_kept_without_metadata(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    path = with_samples(studio_artifacts["tokens_exact"], tmp_path / "s.jsonl", 50)
    snapshot = harness_for(path).build(path.name)
    assert len(snapshot.samples) == 50 and snapshot.status == "ready"
    sample = snapshot.samples[-1]
    assert "metadata" not in sample and sample["process_rss_bytes"] == 49
    assert [s["timestamp_ns"] for s in snapshot.samples[:3]] == [0, 1, 2]
    assert snapshot.samples[0]["device_id"] is None


def test_past_the_sample_budget_the_index_is_partial(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    path = with_samples(studio_artifacts["tokens_exact"], tmp_path / "s.jsonl", 2000)
    snapshot = harness_for(path, Limits(sample_bytes=4096)).build(path.name)
    assert (snapshot.status, snapshot.limits_hit) == ("partial", frozenset({"samples"}))
    assert 0 < len(snapshot.samples) < 2000
    assert len(snapshot.requests) == 10
    assert snapshot.inventory["infer.system_sample"] == {1: 2000}


def test_an_oversized_kept_record_leaves_its_line_reference(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["tokens_exact"]
    snapshot = harness_for(path, Limits(kept_record_bytes=1000)).build(path.name)
    assert snapshot.status == "ready" and snapshot.limits_hit == {"kept_records"}
    sessions = snapshot.kept.get("infer.session", ())
    assert len(sessions) == 1 and "config" not in sessions[0]  # the short terminal one
    assert len(snapshot.refs["infer.session"]) == 2
    assert snapshot.session is not None and snapshot.session.config  # digested


def test_an_unexpected_error_fails_the_index_instead_of_building_forever(
    studio_artifacts: StandardArtifacts, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> int:
        raise RuntimeError("a bug")

    monkeypatch.setattr(index_module._Requests, "add", broken)
    path = studio_artifacts["mixed"]
    harness = harness_for(path)
    snapshot = harness.build(path.name)
    assert (snapshot.status, snapshot.failure) == ("failed", "internal")
    assert len(snapshot.requests) == 0 and snapshot.identity is not None
    assert harness.refresh(path.name).failure == "internal"  # not retried


def test_past_the_kept_budget_later_records_leave_references(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["mixed"]
    snapshot = harness_for(path, Limits(kept_bytes=3000)).build(path.name)
    assert snapshot.limits_hit == {"kept_records"}
    kept = sum(len(records) for records in snapshot.kept.values())
    assert (
        0
        < kept
        < sum(
            len(refs)
            for name, refs in snapshot.refs.items()
            if name in KEPT_EVENT_TYPES
        )
    )


# --- Lines by reference ----------------------------------------------------


def test_read_line_and_read_lines_give_the_original_bytes(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["mixed"]
    harness = harness_for(path)
    snapshot = harness.build(path.name)
    refs = [row.ref for row in snapshot.requests]
    expected = [line for _, line, _ in client_lines(path)]
    assert [harness.indexes.read_line(path.name, ref) for ref in refs] == expected
    stream = harness.indexes.read_lines(path.name, refs)
    assert b"".join(stream) == b"".join(line + b"\n" for line in expected)


def test_a_changed_line_is_a_stale_line_conflict(
    fresh_artifact: Callable[[str], Path],
) -> None:
    path = fresh_artifact("synthetic")
    harness = harness_for(path, Limits(gap_bytes=0))
    snapshot = harness.build(path.name)
    row = snapshot.requests[3]
    data = bytearray(path.read_bytes())
    data[row.ref.offset + 2] ^= 0x01  # same length, other bytes
    path.write_bytes(bytes(data))
    with pytest.raises(ConflictError) as raised:
        harness.indexes.read_line(path.name, row.ref)
    assert raised.value.code == "stale_line" and raised.value.http_status == 409
    with pytest.raises(ConflictError):
        row.load_gaps()
    lines: Iterator[bytes] = harness.indexes.read_lines(
        path.name, [snapshot.requests[0].ref, row.ref]
    )
    assert next(lines).startswith(b"{")
    with pytest.raises(ConflictError):
        next(lines)
    path.write_bytes(bytes(data[: row.ref.offset + 5]))
    with pytest.raises(ConflictError):
        harness.indexes.read_line(path.name, row.ref)


# --- Lifecycle and concurrency ---------------------------------------------


def test_snapshot_never_waits_and_never_raises_not_ready(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["mixed"]
    harness = harness_for(path)
    building = harness.indexes.snapshot(path.name)
    assert (building.status, building.generation, len(building.requests)) == (
        "building",
        0,
        0,
    )
    assert building.identity is None and building.bytes_read == 0
    assert artifact_block(building)["status"] == "building"
    assert harness.indexes.wait(path.name, 0.05).status == "building"
    assert harness.pool.submitted == 1
    harness.indexes.snapshot(path.name)
    assert harness.pool.submitted == 1  # one build at a time
    harness.pool.run()
    ready = harness.indexes.wait(path.name, 0.05)
    assert ready.status == "ready" and ready.index_id == building.index_id


def test_snapshots_stay_fixed_while_a_build_goes_on(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["synthetic"]
    with ThreadPoolExecutor(1) as pool:
        indexes = ArtifactIndexes(Root(path.parent), pool, Limits(), batch_bytes=16_384)
        seen: list[tuple[IndexSnapshot, int, list[Any]]] = []
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            snapshot = indexes.snapshot(path.name)
            seen.append(
                (snapshot, len(snapshot.requests), list(snapshot.column("request_id")))
            )
            if snapshot.status != "building":
                break
        final = indexes.wait(path.name, 60)
        indexes.close()
    assert final.status == "ready" and len(final.requests) == 2000
    assert len({s.generation for s, _, _ in seen}) > 2
    ids = final.column("request_id")
    for snapshot, count, column in seen:
        assert len(snapshot.requests) == count == len(column)
        assert column == list(ids[:count])
        assert snapshot.column("request_id") == column


def test_many_threads_share_one_build(studio_artifacts: StandardArtifacts) -> None:
    path = studio_artifacts["synthetic"]
    submits: list[int] = []

    class Counting(ThreadPoolExecutor):
        def submit(
            self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
        ) -> Future[Any]:
            submits.append(1)
            return super().submit(fn, *args, **kwargs)

    with Counting(1) as pool:
        indexes = ArtifactIndexes(Root(path.parent), pool, Limits(), batch_bytes=65_536)
        results: list[IndexSnapshot] = []

        def worker() -> None:
            for _ in range(50):
                results.append(indexes.snapshot(path.name))
            results.append(indexes.wait(path.name, 60))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        indexes.close()
    assert len(submits) == 1
    assert {s.index_id for s in results} == {results[-1].index_id}
    assert all(s.status == "ready" for s in results[-8:])


def test_the_least_recently_used_index_is_evicted(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    for name in ("mixed", "synthetic"):
        (tmp_path / f"{name}.jsonl").write_bytes(studio_artifacts[name].read_bytes())
    harness = Harness(tmp_path, Limits(open_indexes=1))
    first = harness.build("mixed.jsonl")
    harness.build("synthetic.jsonl")
    assert list(harness.indexes.stats()) == ["synthetic.jsonl"]
    again = harness.build("mixed.jsonl")
    assert again.index_id != first.index_id and again.status == "ready"
    assert len(first.requests) == 56  # an evicted snapshot still reads


def test_past_the_memory_budget_older_indexes_are_evicted(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    for name in ("mixed", "synthetic"):
        (tmp_path / f"{name}.jsonl").write_bytes(studio_artifacts[name].read_bytes())
    harness = Harness(tmp_path, Limits(index_memory_bytes=1_000_000))
    harness.build("mixed.jsonl")
    assert list(harness.indexes.stats()) == ["mixed.jsonl"]
    harness.build("synthetic.jsonl")
    stats = harness.indexes.stats()
    assert list(stats) == ["synthetic.jsonl"]
    assert stats["synthetic.jsonl"]["memory_bytes"] > 0


def test_a_building_index_is_never_evicted(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    for name in ("mixed", "synthetic"):
        (tmp_path / f"{name}.jsonl").write_bytes(studio_artifacts[name].read_bytes())
    harness = Harness(tmp_path, Limits(open_indexes=1))
    harness.indexes.snapshot("mixed.jsonl")
    harness.indexes.snapshot("synthetic.jsonl")
    assert sorted(harness.indexes.stats()) == ["mixed.jsonl", "synthetic.jsonl"]


def test_close_cancels_builds_and_drops_every_index(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["mixed"]
    harness = harness_for(path)
    building = harness.indexes.snapshot(path.name)
    harness.indexes.close()
    harness.pool.run()
    assert harness.indexes.stats() == {}
    assert harness.root.opened == 0 and building.status == "building"
    assert harness.indexes.wait(path.name, 1.0).status == "building"


def test_the_registry_satisfies_the_contract_protocol(tmp_path: Path) -> None:
    registry: IndexRegistry = Harness(tmp_path).indexes
    assert registry is not None


def test_row_sequences_index_like_lists(studio_artifacts: StandardArtifacts) -> None:
    path = studio_artifacts["mixed"]
    snapshot = harness_for(path).build(path.name)
    rows = snapshot.requests
    assert rows[-1].index == len(rows) - 1
    assert [row.index for row in rows[2:5]] == [2, 3, 4]
    with pytest.raises(IndexError):
        rows[len(rows)]
    with pytest.raises(KeyError):
        rows[0]["no_such_key"]
    assert rows[0].get("no_such_key") is None and "no_such_key" not in rows[0]
    with pytest.raises(KeyError):
        snapshot.column("client.nope")
    assert "ColumnRow(index=0" in repr(rows[0])


# --- The 500,000-request artifact (opt-in) ---------------------------------

# Bounds for the cloud box (4 vCPU). Measured there on 3.12: build 31 s,
# 409 accounted and 439 traced bytes per row, columns and filters < 0.1 s,
# a sort 0.5 s, read_lines 300,000 lines/s (reports/studio-c1.md).
LARGE_BUILD_S = 120.0
ROUTE_S = 2.0
ROW_BYTES_TARGET = 512
LINES_PER_S = 50_000


def timed(fn: Callable[[], Any]) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


@pytest.mark.studio_large
def test_the_large_artifact_builds_within_its_budgets(
    large_studio_artifact: Path,
) -> None:
    path = large_studio_artifact
    with ThreadPoolExecutor(1) as pool:
        indexes = ArtifactIndexes(Root(path.parent), pool, Limits())
        start = time.perf_counter()
        snapshot = indexes.wait(path.name, LARGE_BUILD_S)
        build = time.perf_counter() - start
        assert snapshot.status == "ready", f"built {build:.1f} s"
        assert build < LARGE_BUILD_S
        assert len(snapshot.requests) == 500_000 and snapshot.limits_hit == set()
        assert snapshot.gaps_resident and snapshot.error_messages_resident
        requests = indexes._entries[path.name].state.requests
        assert requests.row_bytes() / 500_000 <= ROW_BYTES_TARGET
        assert indexes.stats()[path.name]["memory_bytes"] < Limits().index_memory_bytes
        assert timed(lambda: indexes.snapshot(path.name)) < ROUTE_S
        for name in ("client.e2e", "status", "case_id", "request_id", "output_tokens"):
            assert timed(lambda: snapshot.column(name)) < ROUTE_S, name
        e2e = snapshot.column("client.e2e")
        status = snapshot.column("status")
        assert (
            timed(
                lambda: [
                    i for i, s in enumerate(status) if s == "ok" and e2e[i] >= 150.0
                ]
            )
            < ROUTE_S
        )
        assert (
            timed(
                lambda: sorted(
                    range(len(e2e)), key=lambda i: (e2e[i] is None, e2e[i] or 0.0, i)
                )
            )
            < ROUTE_S
        )
        refs = [snapshot.requests[i].ref for i in range(0, 500_000, 5)]
        elapsed = timed(lambda: sum(1 for _ in indexes.read_lines(path.name, refs)))
        assert len(refs) / elapsed > LINES_PER_S
        assert (
            timed(lambda: [dict(snapshot.requests[i]) for i in range(1000)]) < ROUTE_S
        )
        indexes.close()


@pytest.mark.studio_large
def test_rows_cost_at_most_512_traced_bytes(
    large_studio_artifact: Path, tmp_path: Path
) -> None:
    """§4.4's target, measured with tracemalloc on the first 100,000
    requests of the large artifact (the cost per row does not depend on
    how many rows there are; the whole file under tracemalloc takes
    minutes)."""
    with large_studio_artifact.open("rb") as handle:
        head = handle.read(160_000_000)
    part = tmp_path / "part.jsonl"
    part.write_bytes(head[: head.rfind(b"\n") + 1])
    harness = harness_for(part)
    tracemalloc.start()
    try:
        snapshot = harness.build(part.name)
        current, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    rows = len(snapshot.requests)
    assert rows > 100_000
    requests = harness.indexes._entries[part.name].state.requests
    pools = requests.gap_bytes() + requests.error_bytes()
    assert (current - pools) / rows <= ROW_BYTES_TARGET
