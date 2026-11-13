"""C4's workspace: classify, the confined scan, items paging, reports."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, BinaryIO

import jsonschema  # type: ignore[import-untyped, unused-ignore]
import pytest
from referencing import Registry, Resource

from stormlog.infer.events import InferenceRequestEvent, JsonlEventWriter
from stormlog.report import build_report
from stormlog.studio import workspace
from stormlog.studio.contracts import (
    BusyError,
    ConflictError,
    ForbiddenError,
    InvalidInputError,
    Limits,
    NotFoundError,
    UsageError,
    WorkspaceScanner,
    envelope,
    to_wire,
)
from stormlog.studio.workspace import (
    SNIFF_BYTES,
    ArtifactScanner,
    WorkspaceItem,
    classify,
    read_report,
)

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "docs" / "schemas"
SESSION = "5f0c1d2e3a4b5c6d7e8f901234567890"
T0 = 1788609600000000000


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


class Root:
    """A WorkspaceRoot for tests: realpath confinement, as §7.4 describes."""

    def __init__(self, path: Path) -> None:
        self.path = Path(os.path.realpath(path))

    def resolve(self, rel: str) -> Path:
        parts = rel.split("/")
        if not rel or rel.startswith("/") or "\\" in rel or ".." in parts:
            raise ForbiddenError("path_not_allowed", rel)
        real = Path(os.path.realpath(self.path / rel))
        if self.path not in real.parents:
            raise ForbiddenError("path_not_allowed", rel)
        if not real.is_file():
            raise NotFoundError("artifact_not_found", rel)
        return real

    def open(self, rel: str) -> BinaryIO:
        return self.resolve(rel).open("rb")

    def relative(self, path: Path) -> str:
        return path.relative_to(self.path).as_posix()


def session_record(status: str, offset_ns: int = 0) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event_type": "infer.session",
        "session_id": SESSION,
        "timestamp_ns": T0 + offset_ns,
        "status": status,
        "config": {"endpoint": "http://127.0.0.1:8000/v1/chat/completions"},
    }


def request_record(index: int, *, error: str | None = None) -> dict[str, Any]:
    return InferenceRequestEvent(
        session_id=SESSION,
        request_id=f"measured_{index:03d}",
        case_id="c1",
        phase="measured",
        started_at_ns=T0 + index * 10**7,
        ended_at_ns=T0 + index * 10**7 + 5 * 10**8,
        endpoint="http://127.0.0.1:8000/v1/chat/completions",
        model="m",
        concurrency=1,
        target_input_tokens=8,
        target_output_tokens=4,
        stream=True,
        status="ok" if error is None else "error",
        e2e_latency_ms=500.0,
        ttft_ms=100.0,
        first_chunk_latency_ms=90.0,
        chunk_interarrival_ms=[10.0, 12.0],
        output_tokens=4,
        error_message=error,
    ).to_record()


def write_artifact(
    path: Path, requests: int = 3, *, terminal: str | None = "completed"
) -> Path:
    with JsonlEventWriter(path) as writer:
        writer.append(session_record("running"))
        for index in range(requests):
            writer.append(request_record(index))
        if terminal is not None:
            writer.append(session_record(terminal, 10**9))
    return path


def write_report(path: Path, *, exit_code: int = 0, padding: int = 0) -> Path:
    report = build_report(
        report_kind="inference_diagnosis",
        tool_name="stormlog",
        command="infer diagnose",
        exit_code=exit_code,
        summary="ok",
        session_id=SESSION,
        payload={"padding": "x" * padding},
        generated_at_utc="2026-11-13T00:00:00Z",
        tool_version="0.0.0",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def write_envelope(path: Path) -> Path:
    envelope_payload = {
        "schema_version": 1,
        "format": "stormlog.run_envelope",
        "run_id": "run-1",
        "metadata": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope_payload), encoding="utf-8")
    return path


@pytest.fixture
def pool() -> Iterator[ThreadPoolExecutor]:
    executor = ThreadPoolExecutor(max_workers=1)
    yield executor
    executor.shutdown(wait=True)


def scanned(
    root: Path, pool: ThreadPoolExecutor, limits: Limits | None = None
) -> ArtifactScanner:
    scanner = ArtifactScanner(Root(root), pool, limits)
    scanner.scan()
    assert scanner.wait(30)
    return scanner


def listed(scanner: ArtifactScanner, **filters: Any) -> list[str]:
    page = scanner.items(
        kind=filters.get("kind"), q=filters.get("q"), cursor=None, limit=1000
    )
    return [item["path"] for item in page["items"]]


# --- classify --------------------------------------------------------------------


def test_classify_reads_a_profile_artifact(tmp_path: Path) -> None:
    path = write_artifact(tmp_path / "run.jsonl")
    item = classify(path, rel="run.jsonl")
    assert item == WorkspaceItem(
        path="run.jsonl",
        kind="inference_artifact",
        size=path.stat().st_size,
        mtime_ns=path.stat().st_mtime_ns,
        session_id=SESSION,
        status="completed",
        report_kind=None,
        verdict=None,
    )


def test_classify_of_a_running_artifact_reports_running(tmp_path: Path) -> None:
    path = write_artifact(tmp_path / "run.jsonl", terminal=None)
    item = classify(path)
    assert item is not None
    assert (item.status, item.path) == ("running", path.as_posix())


def test_a_large_artifact_takes_its_status_from_the_tail(tmp_path: Path) -> None:
    path = write_artifact(tmp_path / "big.jsonl", requests=400, terminal="interrupted")
    assert path.stat().st_size > 2 * SNIFF_BYTES
    item = classify(path)
    assert item is not None
    assert (item.kind, item.session_id, item.status) == (
        "inference_artifact",
        SESSION,
        "interrupted",
    )


def test_a_large_artifact_without_a_terminal_record_has_no_status(
    tmp_path: Path,
) -> None:
    path = write_artifact(tmp_path / "killed.jsonl", requests=400, terminal=None)
    item = classify(path)
    assert item is not None
    assert (item.kind, item.status) == ("inference_artifact", None)


def test_classify_reads_at_most_64_kib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_artifact(tmp_path / "big.jsonl", requests=2000)
    read: list[int] = []
    real_pread = os.pread

    def counting(fd: int, size: int, offset: int) -> bytes:
        data = real_pread(fd, size, offset)
        read.append(len(data))
        return data

    monkeypatch.setattr(workspace.os, "pread", counting)
    assert classify(path) is not None
    assert 0 < sum(read) <= SNIFF_BYTES


def test_a_request_only_artifact_takes_the_requests_session(tmp_path: Path) -> None:
    path = tmp_path / "requests.jsonl"
    with JsonlEventWriter(path) as writer:
        writer.append(request_record(0))
    item = classify(path)
    assert item is not None
    assert (item.kind, item.session_id, item.status) == (
        "inference_artifact",
        SESSION,
        None,
    )


def test_audio_artifacts_are_listed_as_audio(tmp_path: Path) -> None:
    path = tmp_path / "audio.jsonl"
    lines = [
        session_record("completed"),
        {"event_type": "infer.audio.request", "session_id": SESSION},
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    item = classify(path)
    assert item is not None
    assert item.kind == "audio_artifact"


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"not json\n",
        b'["infer.session"]\n',
        b'{"event_type": "server.sample"}\n',
        b'{"event_type": "infer.session"',  # no complete line
    ],
)
def test_other_jsonl_is_not_listed(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "other.jsonl"
    path.write_bytes(content)
    assert classify(path) is None


def test_classify_reads_report_envelopes(tmp_path: Path) -> None:
    path = write_report(tmp_path / "report.json", exit_code=3)
    item = classify(path)
    assert item is not None
    assert (item.kind, item.report_kind, item.verdict, item.session_id) == (
        "report",
        "inference_diagnosis",
        "findings",
        SESSION,
    )


def test_a_report_over_64_kib_is_still_classified(tmp_path: Path) -> None:
    path = write_report(tmp_path / "big.json", padding=200_000)
    item = classify(path)
    assert item is not None
    assert (item.kind, item.verdict) == ("report", "pass")


def test_classify_reads_run_envelopes(tmp_path: Path) -> None:
    item = classify(write_envelope(tmp_path / "stormlog_run.json"))
    assert item is not None
    assert item.kind == "run_envelope"


@pytest.mark.parametrize(
    "content",
    [
        "{}",
        "[]",
        '{"format": "stormlog.report", "schema_version": 2}',
        '{"format": "stormlog.report", "schema_version": true}',
        '{"format": "stormlog.rep',
        '{"schema_version": 1, "format": "stormlog.report"',  # cut before , or }
        "not json",
    ],
)
def test_other_json_is_not_listed(tmp_path: Path, content: str) -> None:
    path = tmp_path / "other.json"
    path.write_text(content)
    assert classify(path) is None


def test_leading_members_stop_at_a_cut_value() -> None:
    members = workspace._leading_members('{"a": 1, "b": [1, 2], "c": 12')
    assert members == {"a": 1, "b": [1, 2]}


@pytest.mark.parametrize("name", ["notes.txt", "run.jsonl.tmp", ".run.jsonl.tmp"])
def test_other_names_are_not_read(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    write_artifact(path)
    assert classify(path) is None


def test_import_temp_files_are_not_listed(tmp_path: Path) -> None:
    path = write_artifact(tmp_path / f".run.jsonl.{'a' * 32}.tmp")
    assert classify(path) is None


def test_classify_does_not_follow_a_symlink(tmp_path: Path) -> None:
    target = write_artifact(tmp_path / "run.jsonl")
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    assert classify(link) is None


def test_classify_skips_a_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe.jsonl"
    os.mkfifo(fifo)
    assert classify(fifo) is None


def test_classify_of_a_missing_file_is_none(tmp_path: Path) -> None:
    assert classify(tmp_path / "missing.jsonl") is None


# --- the scan -------------------------------------------------------------------


def test_artifact_scanner_implements_the_protocol(
    tmp_path: Path, pool: ThreadPoolExecutor
) -> None:
    scanner: WorkspaceScanner = ArtifactScanner(Root(tmp_path), pool)
    assert scanner.state()["scan"]["status"] == "not_started"


def test_a_scan_lists_every_kind_sorted_by_path(
    tmp_path: Path, pool: ThreadPoolExecutor
) -> None:
    write_artifact(tmp_path / "runs" / "b.jsonl")
    write_artifact(tmp_path / "runs" / "a.jsonl")
    write_report(tmp_path / "reports" / "diagnosis.json")
    write_envelope(tmp_path / "stormlog_run.json")
    (tmp_path / "notes.json").write_text('{"x": 1}')
    scanner = scanned(tmp_path, pool)
    assert listed(scanner) == [
        "reports/diagnosis.json",
        "runs/a.jsonl",
        "runs/b.jsonl",
        "stormlog_run.json",
    ]
    state = scanner.state()
    assert state["counts"] == {
        "inference_artifact": 2,
        "audio_artifact": 0,
        "report": 1,
        "run_envelope": 1,
        "total": 4,
    }
    assert state["scan"]["status"] == "complete"
    assert state["scan"]["files_read"] == 5


def test_nothing_outside_the_root_is_listed_or_followed(
    tmp_path: Path, pool: ThreadPoolExecutor
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    write_artifact(outside / "secret.jsonl")
    write_artifact(root / "inside.jsonl")
    (root / "linked_dir").symlink_to(outside, target_is_directory=True)
    (root / "linked_file.jsonl").symlink_to(outside / "secret.jsonl")
    (root / "inside_link.jsonl").symlink_to(root / "inside.jsonl")
    scanner = scanned(root, pool)
    assert listed(scanner) == ["inside.jsonl"]
    assert scanner.state()["scan"]["skipped"]["symlinks"] == 3


def test_a_directory_swapped_for_a_symlink_is_not_entered(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    write_artifact(outside / "secret.jsonl")
    (root / "runs").symlink_to(outside, target_is_directory=True)
    walk = workspace._Walk(Limits(), deadline=float("inf"), clock=lambda: 0.0)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        # As if scandir had seen a directory that became a link before the open.
        walk._subdirectory(fd, "runs", "", 0)
    finally:
        os.close(fd)
    assert walk.items == []
    assert walk.skipped["unreadable"] == 1


@pytest.mark.parametrize(
    "directory",
    [".git", "node_modules", "__pycache__", ".venv", "venv", "site-packages"],
)
def test_tool_directories_are_skipped(
    tmp_path: Path, pool: ThreadPoolExecutor, directory: str
) -> None:
    write_artifact(tmp_path / directory / "run.jsonl")
    scanner = scanned(tmp_path, pool)
    assert listed(scanner) == []
    assert scanner.state()["scan"]["skipped"]["excluded_dirs"] == 1


def test_a_virtualenv_by_marker_is_skipped(
    tmp_path: Path, pool: ThreadPoolExecutor
) -> None:
    write_artifact(tmp_path / "env" / "run.jsonl")
    (tmp_path / "env" / "pyvenv.cfg").write_text("home = /usr\n")
    assert listed(scanned(tmp_path, pool)) == []


def test_depth_is_bounded(tmp_path: Path, pool: ThreadPoolExecutor) -> None:
    write_artifact(tmp_path / "a" / "one.jsonl")
    write_artifact(tmp_path / "a" / "b" / "two.jsonl")
    scanner = scanned(tmp_path, pool, Limits(scan_depth=1))
    assert listed(scanner) == ["a/one.jsonl"]
    assert scanner.state()["scan"]["status"] == "partial"
    assert scanner.state()["scan"]["limits_hit"] == ["depth"]


def test_entries_are_bounded(tmp_path: Path, pool: ThreadPoolExecutor) -> None:
    for name in "abcdef":
        write_artifact(tmp_path / f"{name}.jsonl")
    scanner = scanned(tmp_path, pool, Limits(scan_entries=3))
    # Which entries a partial scan admits follows directory order.
    assert len(listed(scanner)) == 3
    scan = scanner.state()["scan"]
    assert (scan["status"], scan["limits_hit"], scan["entries"]) == (
        "partial",
        ["entries"],
        3,
    )


def test_time_is_bounded(tmp_path: Path, pool: ThreadPoolExecutor) -> None:
    for name in "abcdef":
        write_artifact(tmp_path / f"{name}.jsonl")
    ticks = iter(range(1000))
    scanner = ArtifactScanner(
        Root(tmp_path), pool, Limits(scan_seconds=3), clock=lambda: next(ticks)
    )
    scanner.scan()
    assert scanner.wait(30)
    scan = scanner.state()["scan"]
    assert (scan["status"], scan["limits_hit"]) == ("partial", ["time"])
    assert 0 < len(listed(scanner)) < 6


def test_a_missing_root_fails_the_scan(
    tmp_path: Path, pool: ThreadPoolExecutor
) -> None:
    root = Root(tmp_path / "gone")
    scanner = ArtifactScanner(root, pool)
    scanner.scan()
    assert scanner.wait(30)
    scan = scanner.state()["scan"]
    assert scan["status"] == "failed"
    assert "cannot read the root" in scan["reason"]


def test_one_scan_at_a_time(tmp_path: Path) -> None:
    class Held:
        """A pool that holds the submitted scan until released."""

        def __init__(self) -> None:
            self.queued: list[Callable[[], None]] = []

        def submit(self, task: Callable[[], None]) -> None:
            self.queued.append(task)

    held = Held()
    scanner = ArtifactScanner(Root(tmp_path), held)  # type: ignore[arg-type]
    scanner.scan()
    with pytest.raises(BusyError) as raised:
        scanner.scan()
    assert raised.value.code == "busy"
    assert scanner.state()["scan"]["status"] == "scanning"
    assert not scanner.wait(0)
    held.queued.pop()()
    assert scanner.wait(0)
    scanner.scan()  # the next scan is accepted


def test_a_shut_down_pool_is_busy_and_leaves_no_scan_running(
    tmp_path: Path,
) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    executor.shutdown()
    scanner = ArtifactScanner(Root(tmp_path), executor)
    with pytest.raises(BusyError):
        scanner.scan()
    assert scanner.state()["scan"]["running"] is False


def test_a_rescan_replaces_the_items(tmp_path: Path, pool: ThreadPoolExecutor) -> None:
    write_artifact(tmp_path / "a.jsonl")
    scanner = scanned(tmp_path, pool)
    write_artifact(tmp_path / "b.jsonl")
    (tmp_path / "a.jsonl").unlink()
    scanner.scan()
    assert scanner.wait(30)
    assert listed(scanner) == ["b.jsonl"]
    assert scanner.state()["scan"]["scans"] == 2


# --- items ------------------------------------------------------------------------


@pytest.fixture
def mixed(tmp_path: Path, pool: ThreadPoolExecutor) -> ArtifactScanner:
    for name in ("alpha", "beta", "gamma", "delta"):
        write_artifact(tmp_path / "runs" / f"{name}.jsonl")
    write_report(tmp_path / "runs" / "beta-report.json")
    return scanned(tmp_path, pool)


def test_items_filter_by_kind_and_text(mixed: ArtifactScanner) -> None:
    assert listed(mixed, kind="report") == ["runs/beta-report.json"]
    assert listed(mixed, q="BETA") == ["runs/beta-report.json", "runs/beta.jsonl"]
    assert listed(mixed, kind="inference_artifact", q="beta") == ["runs/beta.jsonl"]


def test_items_page_with_a_keyset_cursor(mixed: ArtifactScanner) -> None:
    seen: list[str] = []
    cursor = None
    pages = 0
    while True:
        page = mixed.items(kind="inference_artifact", q=None, cursor=cursor, limit=3)
        assert page["total"] == 4
        seen += [item["path"] for item in page["items"]]
        pages += 1
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert pages == 2
    assert seen == sorted(seen) and len(seen) == 4


def test_a_cursor_survives_a_rescan(tmp_path: Path, mixed: ArtifactScanner) -> None:
    page = mixed.items(kind=None, q=None, cursor=None, limit=2)
    write_artifact(tmp_path / "runs" / "aaa.jsonl")  # sorts before the cursor
    mixed.scan()
    assert mixed.wait(30)
    rest = mixed.items(kind=None, q=None, cursor=page["next_cursor"], limit=10)
    assert [item["path"] for item in rest["items"]] == [
        "runs/beta.jsonl",
        "runs/delta.jsonl",
        "runs/gamma.jsonl",
    ]


def test_a_cursor_from_another_filter_conflicts(mixed: ArtifactScanner) -> None:
    page = mixed.items(kind=None, q=None, cursor=None, limit=1)
    with pytest.raises(ConflictError) as raised:
        mixed.items(kind="report", q=None, cursor=page["next_cursor"], limit=1)
    assert raised.value.code == "cursor_mismatch"
    assert raised.value.details == {"reason": "other_filter"}


@pytest.mark.parametrize("cursor", ["", "!!!", "e30", "bm90IGpzb24"])
def test_a_malformed_cursor_is_a_bad_parameter(
    mixed: ArtifactScanner, cursor: str
) -> None:
    with pytest.raises(UsageError) as raised:
        mixed.items(kind=None, q=None, cursor=cursor, limit=1)
    assert (raised.value.code, raised.value.field) == ("bad_parameter", "cursor")


@pytest.mark.parametrize(
    ("kwargs", "code", "field"),
    [
        ({"kind": "pdf"}, "bad_parameter", "kind"),
        ({"q": "x" * 257}, "limit_exceeded", "q"),
        ({"limit": 0}, "bad_parameter", "limit"),
        ({"limit": 1001}, "limit_exceeded", "limit"),
    ],
)
def test_items_refuse_bad_parameters(
    mixed: ArtifactScanner, kwargs: dict[str, Any], code: str, field: str
) -> None:
    arguments: dict[str, Any] = {"kind": None, "q": None, "cursor": None, "limit": 10}
    arguments.update(kwargs)
    with pytest.raises(UsageError) as raised:
        mixed.items(**arguments)
    assert (raised.value.code, raised.value.field) == (code, field)


def test_items_before_any_scan_are_empty(
    tmp_path: Path, pool: ThreadPoolExecutor
) -> None:
    page = ArtifactScanner(Root(tmp_path), pool).items(
        kind=None, q=None, cursor=None, limit=10
    )
    assert (page["items"], page["total"], page["next_cursor"]) == ([], 0, None)


def test_workspace_and_items_documents_validate(mixed: ArtifactScanner) -> None:
    state = to_wire(envelope("workspace", mixed.state()))
    assert schema_errors("studio_workspace_v1.schema.json", state) == []
    page = mixed.items(kind=None, q="a", cursor=None, limit=2)
    items = to_wire(envelope("items", page))
    assert schema_errors("studio_items_v1.schema.json", items) == []
    assert isinstance(items["items"][0]["mtime_ns"], str)


def test_states_before_and_during_a_scan_validate(tmp_path: Path) -> None:
    class Never:
        def submit(self, task: Callable[[], None]) -> None:
            pass

    scanner = ArtifactScanner(Root(tmp_path), Never())  # type: ignore[arg-type]
    before = to_wire(envelope("workspace", scanner.state()))
    assert schema_errors("studio_workspace_v1.schema.json", before) == []
    scanner.scan()
    during = to_wire(envelope("workspace", scanner.state()))
    assert during["scan"]["status"] == "scanning"
    assert schema_errors("studio_workspace_v1.schema.json", during) == []


# --- reports ----------------------------------------------------------------------


def test_read_report_returns_the_validated_envelope(tmp_path: Path) -> None:
    path = write_report(tmp_path / "reports" / "r.json")
    body = read_report(Root(tmp_path), "reports/r.json", max_bytes=2**20)
    assert body["path"] == "reports/r.json"
    assert body["size"] == path.stat().st_size
    assert body["report"]["verdict"]["status"] == "pass"
    assert body["non_finite_fields"] == []
    document = to_wire(envelope("report", body))
    assert schema_errors("studio_report_v1.schema.json", document) == []


def test_read_report_nulls_non_finite_numbers(tmp_path: Path) -> None:
    path = write_report(tmp_path / "r.json")
    # validate_report refuses NaN in metrics; a payload is free-form.
    text = path.read_text().replace('"padding": ""', '"padding": NaN')
    path.write_text(text)
    body = read_report(Root(tmp_path), "r.json", max_bytes=2**20)
    assert body["report"]["payload"] == {"padding": None}
    assert body["non_finite_fields"] == ["payload.padding"]


def test_read_report_refuses_non_finite_metrics(tmp_path: Path) -> None:
    path = write_report(tmp_path / "r.json")
    text = path.read_text().replace('"metrics": {}', '"metrics": {"x": NaN}')
    path.write_text(text)
    with pytest.raises(InvalidInputError) as raised:
        read_report(Root(tmp_path), "r.json", max_bytes=2**20)
    assert "finite" in raised.value.message


def test_read_report_refuses_an_oversized_file(tmp_path: Path) -> None:
    write_report(tmp_path / "r.json", padding=5000)
    with pytest.raises(InvalidInputError) as raised:
        read_report(Root(tmp_path), "r.json", max_bytes=1000)
    assert raised.value.code == "report_too_large"
    assert raised.value.details["max_bytes"] == 1000


@pytest.mark.parametrize("content", ["[]", "{", '{"format": "stormlog.report"}'])
def test_read_report_refuses_what_is_not_a_report(tmp_path: Path, content: str) -> None:
    (tmp_path / "r.json").write_text(content)
    with pytest.raises(InvalidInputError) as raised:
        read_report(Root(tmp_path), "r.json", max_bytes=2**20)
    assert raised.value.code == "unreadable"
    assert raised.value.details == {"reason": "not_a_report"}


def test_read_report_goes_through_the_root(tmp_path: Path) -> None:
    write_report(tmp_path / "outside.json")
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ForbiddenError):
        read_report(Root(root), "../outside.json", max_bytes=2**20)
    with pytest.raises(NotFoundError):
        read_report(Root(root), "missing.json", max_bytes=2**20)
