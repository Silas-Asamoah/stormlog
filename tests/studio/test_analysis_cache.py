"""C4's analyzer runner: one subprocess, prefix copies, cache, timeout, close.

Most tests run the real ``stormlog infer analyze`` on small artifacts
written by JsonlEventWriter and the infer dataclasses. Timeouts, kills
and one-at-a-time use a scripted stand-in for the analyzer.
"""

from __future__ import annotations

import dataclasses
import json
import os
import stat
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

import jsonschema  # type: ignore[import-untyped, unused-ignore]
import pytest
from referencing import Registry, Resource

from stormlog.infer.events import InferenceRequestEvent, JsonlEventWriter
from stormlog.studio import analysis_cache
from stormlog.studio.analysis_cache import AnalyzerRunner
from stormlog.studio.contracts import (
    AnalysisCache,
    AnalysisState,
    DictTable,
    FileIdentity,
    IndexSnapshot,
    Limits,
    NotReady,
    SessionInfo,
    to_wire,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "docs" / "schemas"
SESSION = "5f0c1d2e3a4b5c6d7e8f901234567890"
INDEX_ID = "9a3e5b7c1d2f4a6b8c0e2d4f6a8b0c1d"
T0 = 1788609600000000000
DONE = ("ready", "unavailable", "failed")


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
    """A WorkspaceRoot for tests over one directory."""

    def __init__(self, path: Path) -> None:
        self.path = Path(os.path.realpath(path))

    def resolve(self, rel: str) -> Path:
        return self.path / rel

    def open(self, rel: str) -> BinaryIO:
        return self.resolve(rel).open("rb")

    def relative(self, path: Path) -> str:
        return path.relative_to(self.path).as_posix()


def request(index: int, case: str = "c1", status: str = "ok") -> dict[str, Any]:
    return InferenceRequestEvent(
        session_id=SESSION,
        request_id=f"measured_{index:03d}",
        case_id=case,
        phase="measured",
        started_at_ns=T0 + index * 10**7,
        ended_at_ns=T0 + index * 10**7 + 5 * 10**8,
        endpoint="http://127.0.0.1:8000/v1/chat/completions",
        model="m",
        concurrency=1,
        target_input_tokens=8,
        target_output_tokens=4,
        stream=True,
        status=status,
        e2e_latency_ms=500.0,
        ttft_ms=100.0 if status == "ok" else None,
        first_chunk_latency_ms=90.0 if status == "ok" else None,
        chunk_interarrival_ms=[10.0, 12.0] if status == "ok" else [],
        output_tokens=4 if status == "ok" else None,
        output_token_source="server_usage",
        output_token_exact=status == "ok",
    ).to_record()


def write_artifact(path: Path, requests: int = 5, *, terminal: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with JsonlEventWriter(path) as writer:
        writer.append(
            {
                "schema_version": 1,
                "event_type": "infer.session",
                "session_id": SESSION,
                "timestamp_ns": T0,
                "status": "running",
                "config": {},
            }
        )
        for index in range(requests):
            writer.append(request(index, status="error" if index == 2 else "ok"))
        if terminal:
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


def snapshot_of(path: Path, rel: str, **overrides: Any) -> IndexSnapshot:
    """A ready snapshot naming the file's current identity (no rows needed)."""
    info = path.stat()
    values: dict[str, Any] = {
        "path": rel,
        "index_id": INDEX_ID,
        "generation": 2,
        "status": "ready",
        "failure": None,
        "identity": FileIdentity(
            info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns
        ),
        "bytes_read": info.st_size,
        "pending_tail_bytes": 0,
        "prefix_sha256": "a" * 64,
        "sha256": None,
        "growing": False,
        "origin_ns": T0,
        "session": SessionInfo(SESSION, "completed", T0, T0 + 10**9, True, {}),
        "requests": (),
        "table": DictTable(()),
        "by_request_id": {},
        "samples": (),
        "kept": {},
        "request_links": {},
        "refs": {},
        "inventory": {},
        "problems": (),
        "problem_count": 0,
        "limits_hit": frozenset(),
        "gaps_resident": True,
        "error_messages_resident": True,
    }
    values.update(overrides)
    return IndexSnapshot(**values)


def settle(
    runner: AnalyzerRunner, rel: str, snapshot: IndexSnapshot, timeout_s: float = 120
) -> AnalysisState:
    deadline = time.monotonic() + timeout_s
    state = runner.get(rel, snapshot)
    while state.status not in DONE:
        assert time.monotonic() < deadline, f"still {state.status}"
        time.sleep(0.05)
        state = runner.get(rel, snapshot)
    return state


def wait_for(condition: Any, timeout_s: float = 30) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.02)


FAKE = """\
import json, os, sys, time
behaviour = json.loads({behaviour!r})
args = sys.argv[1:]
source, output = args[0], args[args.index("--output") + 1]
change = behaviour.get("change")
if change:  # a writer changes the artifact after the runner checked it
    target, how = change
    line = json.dumps({{"event_type": "note", "how": how}}).encode() + b"\\n"
    if how == "grow":
        with open(target, "ab") as handle:
            handle.write(line)
    elif how == "rewrite":
        with open(target, "r+b") as handle:
            handle.truncate(0)
            handle.write(line)
    else:
        with open(target + ".new", "wb") as handle:
            handle.write(line)
        os.replace(target + ".new", target)
log = behaviour.get("log")
if log:
    with open(log, "a") as handle:
        handle.write(json.dumps({{"pid": os.getpid(), "cwd": os.getcwd(),
            "source": source, "mode": os.stat(source).st_mode & 0o777,
            "bytes": open(source, "rb").read().decode("utf-8")}}) + "\\n")
lock = behaviour.get("lock")
if lock:
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        sys.exit(9)
time.sleep(behaviour.get("sleep", 0))
if lock:
    os.unlink(lock)
sys.stderr.write(behaviour.get("stderr", ""))
with open(output, "w") as handle:
    handle.write(behaviour.get("output", '{{"analysis_version": 2, "cases": {{}}}}'))
sys.exit(behaviour.get("exit", 0))
"""


def fake_analyzer(directory: Path, **behaviour: Any) -> tuple[str, ...]:
    script = directory / "fake_analyzer.py"
    script.write_text(FAKE.format(behaviour=json.dumps(behaviour)))
    return (sys.executable, str(script))


def log_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    return root


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    return tmp_path / "state"


@pytest.fixture
def runners() -> Iterator[list[AnalyzerRunner]]:
    made: list[AnalyzerRunner] = []
    yield made
    for runner in made:
        runner.close()


def make(
    runners: list[AnalyzerRunner],
    workspace: Path,
    state_dir: Path,
    limits: Limits | None = None,
    **kwargs: Any,
) -> AnalyzerRunner:
    runner = AnalyzerRunner(Root(workspace), state_dir, limits, **kwargs)
    runners.append(runner)
    return runner


# --- the real analyzer -------------------------------------------------------------


def test_analyzer_runner_implements_the_protocol(
    workspace: Path, state_dir: Path, runners: list[AnalyzerRunner]
) -> None:
    cache: AnalysisCache = make(runners, workspace, state_dir)
    path = write_artifact(workspace / "run.jsonl")
    assert cache.get("run.jsonl", snapshot_of(path, "run.jsonl")).status in (
        "pending",
        "running",
        "ready",
    )


def test_a_complete_artifact_is_analysed_as_it_is(
    workspace: Path, state_dir: Path, runners: list[AnalyzerRunner]
) -> None:
    runner = make(runners, workspace, state_dir)
    path = write_artifact(workspace / "runs" / "042.jsonl")
    snapshot = snapshot_of(path, "runs/042.jsonl")
    state = settle(runner, "runs/042.jsonl", snapshot)
    assert (state.status, state.reason, state.exit_code) == ("ready", None, 0)
    assert state.analysed_bytes == path.stat().st_size
    assert state.report is not None
    assert state.report["analysis_version"] == 2
    assert state.report["cases"]["c1"]["population"]["offered"] == 5
    document = runner.document("runs/042.jsonl", snapshot)
    assert document["input"] == {
        "kind": "file",
        "file_size": path.stat().st_size,
        "analysed_bytes": path.stat().st_size,
        "omitted_tail_bytes": 0,
    }
    assert document["retry_after_s"] is None
    assert document["elapsed_s"] is not None
    wired = to_wire(document)
    assert schema_errors("studio_analysis_v1.schema.json", wired) == []
    assert list((state_dir / "analysis").iterdir()) == []


def test_a_truncated_last_line_analyses_the_complete_lines(
    workspace: Path, state_dir: Path, runners: list[AnalyzerRunner]
) -> None:
    runner = make(runners, workspace, state_dir)
    path = write_artifact(workspace / "killed.jsonl", terminal=False)
    complete = path.stat().st_size
    tail = b'{"event_type": "infer.request", "request_id": "measured_0'
    with path.open("ab") as handle:
        handle.write(tail)
    original = path.read_bytes()
    snapshot = snapshot_of(path, "killed.jsonl")
    state = settle(runner, "killed.jsonl", snapshot)
    assert (state.status, state.reason) == ("ready", "truncated_final_line")
    assert state.analysed_bytes == complete
    assert state.report is not None
    assert state.report["cases"]["c1"]["population"]["offered"] == 5
    document = runner.document("killed.jsonl", snapshot)
    assert document["input"] == {
        "kind": "complete_line_prefix",
        "file_size": complete + len(tail),
        "analysed_bytes": complete,
        "omitted_tail_bytes": len(tail),
    }
    assert schema_errors("studio_analysis_v1.schema.json", to_wire(document)) == []
    assert path.read_bytes() == original
    assert list((state_dir / "analysis").iterdir()) == []


def test_without_the_copy_the_analyzer_fails_a_truncated_file(
    workspace: Path, state_dir: Path, runners: list[AnalyzerRunner]
) -> None:
    """Decision 12's reason: a bad line anywhere fails infer analyze (exit 5)."""
    runner = make(runners, workspace, state_dir)
    path = write_artifact(workspace / "bad.jsonl")
    lines = path.read_bytes().split(b"\n")
    lines.insert(2, b'{"event_type": "infer.req')
    path.write_bytes(b"\n".join(lines))
    state = settle(runner, "bad.jsonl", snapshot_of(path, "bad.jsonl"))
    assert (state.status, state.reason, state.exit_code) == (
        "failed",
        "nonzero_exit",
        5,
    )
    document = runner.document("bad.jsonl", snapshot_of(path, "bad.jsonl"))
    assert document["detail"] is not None and "line 3" in document["detail"]
    assert schema_errors("studio_analysis_v1.schema.json", to_wire(document)) == []


def test_results_are_cached_by_path_index_generation_and_identity(
    workspace: Path,
    state_dir: Path,
    tmp_path: Path,
    runners: list[AnalyzerRunner],
) -> None:
    log = tmp_path / "runs.log"
    runner = make(
        runners,
        workspace,
        state_dir,
        command=fake_analyzer(tmp_path, log=str(log)),
    )
    path = write_artifact(workspace / "run.jsonl")
    snapshot = snapshot_of(path, "run.jsonl")
    first = settle(runner, "run.jsonl", snapshot)
    again = settle(runner, "run.jsonl", snapshot)
    assert first is again
    assert len(log_lines(log)) == 1
    settle(runner, "run.jsonl", dataclasses.replace(snapshot, generation=3))
    settle(runner, "run.jsonl", dataclasses.replace(snapshot, index_id="b" * 32))
    assert len(log_lines(log)) == 3


def test_a_failure_is_cached_too(
    workspace: Path,
    state_dir: Path,
    tmp_path: Path,
    runners: list[AnalyzerRunner],
) -> None:
    log = tmp_path / "runs.log"
    command = fake_analyzer(tmp_path, log=str(log), exit=1, stderr="Error: boom\n")
    runner = make(runners, workspace, state_dir, command=command)
    path = write_artifact(workspace / "run.jsonl")
    snapshot = snapshot_of(path, "run.jsonl")
    state = settle(runner, "run.jsonl", snapshot)
    assert (state.status, state.reason, state.exit_code) == (
        "failed",
        "nonzero_exit",
        1,
    )
    assert runner.document("run.jsonl", snapshot)["detail"] == "Error: boom"
    settle(runner, "run.jsonl", snapshot)
    assert len(log_lines(log)) == 1


def test_one_analysis_at_a_time(
    workspace: Path,
    state_dir: Path,
    tmp_path: Path,
    runners: list[AnalyzerRunner],
) -> None:
    command = fake_analyzer(tmp_path, lock=str(tmp_path / "running"), sleep=0.2)
    runner = make(runners, workspace, state_dir, command=command)
    snapshots = {}
    for name in ("a", "b", "c"):
        path = write_artifact(workspace / f"{name}.jsonl")
        snapshots[f"{name}.jsonl"] = snapshot_of(path, f"{name}.jsonl")
    states = [runner.get(rel, snap) for rel, snap in snapshots.items()]
    assert [state.status for state in states].count("running") <= 1
    for rel, snap in snapshots.items():
        assert settle(runner, rel, snap).status == "ready"


def test_a_popen_that_raises_fails_its_entry_and_later_runs_complete(
    workspace: Path,
    state_dir: Path,
    tmp_path: Path,
    runners: list[AnalyzerRunner],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_popen = analysis_cache.subprocess.Popen
    calls: list[int] = []

    def popen_once_broken(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise FileNotFoundError(2, "No such file or directory", "/gone/python")
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(analysis_cache.subprocess, "Popen", popen_once_broken)
    runner = make(runners, workspace, state_dir, command=fake_analyzer(tmp_path))
    first = write_artifact(workspace / "first.jsonl")
    state = settle(runner, "first.jsonl", snapshot_of(first, "first.jsonl"), 30)
    assert (state.status, state.reason) == ("failed", "internal")
    detail = runner.document("first.jsonl", snapshot_of(first, "first.jsonl"))
    assert "No such file or directory" in detail["detail"]
    second = write_artifact(workspace / "second.jsonl")
    later = settle(runner, "second.jsonl", snapshot_of(second, "second.jsonl"), 30)
    assert (later.status, later.exit_code) == ("ready", 0)
    assert list((state_dir / "analysis").iterdir()) == []


def test_a_dead_worker_is_restarted_by_the_next_enqueue(
    workspace: Path,
    state_dir: Path,
    tmp_path: Path,
    runners: list[AnalyzerRunner],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make(runners, workspace, state_dir, command=fake_analyzer(tmp_path))
    first = write_artifact(workspace / "first.jsonl")
    # The first worker exits at once, as one that died would.
    monkeypatch.setattr(runner, "_work", lambda: None)
    runner.get("first.jsonl", snapshot_of(first, "first.jsonl"))
    thread = runner._thread
    assert thread is not None
    thread.join(timeout=10)
    monkeypatch.undo()
    second = write_artifact(workspace / "second.jsonl")
    later = settle(runner, "second.jsonl", snapshot_of(second, "second.jsonl"), 30)
    assert later.status == "ready"
    # The restarted worker also runs the entry the dead one left queued.
    earlier = settle(runner, "first.jsonl", snapshot_of(first, "first.jsonl"), 30)
    assert earlier.status == "ready"


def test_a_newer_snapshot_replaces_queued_work_for_its_path(
    workspace: Path,
    state_dir: Path,
    tmp_path: Path,
    runners: list[AnalyzerRunner],
) -> None:
    log = tmp_path / "runs.log"
    command = fake_analyzer(tmp_path, log=str(log), sleep=0.5)
    runner = make(runners, workspace, state_dir, command=command)
    busy = write_artifact(workspace / "busy.jsonl")
    runner.get("busy.jsonl", snapshot_of(busy, "busy.jsonl"))
    wait_for(lambda: len(log_lines(log)) == 1)
    path = write_artifact(workspace / "run.jsonl")
    old = snapshot_of(path, "run.jsonl", generation=1)
    new = snapshot_of(path, "run.jsonl", generation=2)
    assert runner.get("run.jsonl", old).reason == "queued"
    assert runner.get("run.jsonl", new).reason == "queued"
    assert settle(runner, "run.jsonl", new).status == "ready"
    assert len(log_lines(log)) == 2  # busy and the newer snapshot only


def test_the_subprocess_runs_in_the_private_analysis_directory(
    workspace: Path,
    state_dir: Path,
    tmp_path: Path,
    runners: list[AnalyzerRunner],
) -> None:
    log = tmp_path / "runs.log"
    runner = make(
        runners, workspace, state_dir, command=fake_analyzer(tmp_path, log=str(log))
    )
    path = write_artifact(workspace / "run.jsonl")
    settle(runner, "run.jsonl", snapshot_of(path, "run.jsonl"))
    [run] = log_lines(log)
    analysis_dir = state_dir / "analysis"
    assert os.path.realpath(run["cwd"]) == os.path.realpath(analysis_dir)
    assert stat.S_IMODE(analysis_dir.stat().st_mode) == 0o700
    # A private 0600 copy of the verified bytes, never the artifact's path.
    assert Path(run["source"]).parent == Path(run["cwd"])
    assert run["mode"] == 0o600 and run["bytes"] == path.read_text()


def _copies(state_dir: Path) -> list[Path]:
    return list((state_dir / "analysis").glob("*.jsonl"))


@pytest.mark.parametrize("how", ["grow", "rewrite", "replace"])
def test_the_analyzer_reads_the_verified_bytes_not_the_path(
    workspace: Path,
    state_dir: Path,
    tmp_path: Path,
    runners: list[AnalyzerRunner],
    how: str,
) -> None:
    """A writer that changes the artifact after the identity check must not
    get a report of its new bytes cached under the old snapshot's key
    (Codex's review of #345)."""
    log = tmp_path / "runs.log"
    path = write_artifact(workspace / "run.jsonl")
    verified = path.read_text()
    command = fake_analyzer(tmp_path, log=str(log), change=[str(path), how])
    runner = make(runners, workspace, state_dir, command=command)
    snapshot = snapshot_of(path, "run.jsonl")
    state = settle(runner, "run.jsonl", snapshot)
    [run] = log_lines(log)
    assert path.read_text() != verified  # the writer did change it
    assert state.status == "ready" and run["bytes"] == verified
    assert _copies(state_dir) == []


def test_an_artifact_changed_during_the_copy_is_not_analysed(
    workspace: Path,
    state_dir: Path,
    tmp_path: Path,
    runners: list[AnalyzerRunner],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = tmp_path / "runs.log"
    path = write_artifact(workspace / "run.jsonl")
    copy_prefix = analysis_cache._copy_prefix

    def copy_then_grow(handle: BinaryIO, target: Path, length: int) -> None:
        copy_prefix(handle, target, length)
        with path.open("ab") as grow:
            grow.write(b'{"event_type": "note"}\n')

    monkeypatch.setattr(analysis_cache, "_copy_prefix", copy_then_grow)
    command = fake_analyzer(tmp_path, log=str(log))
    runner = make(runners, workspace, state_dir, command=command)
    state = settle(runner, "run.jsonl", snapshot_of(path, "run.jsonl"))
    assert (state.status, state.reason) == ("unavailable", "changed")
    assert log_lines(log) == [] and _copies(state_dir) == []


def test_the_prefix_copy_is_private_and_exact(
    workspace: Path,
    state_dir: Path,
    tmp_path: Path,
    runners: list[AnalyzerRunner],
) -> None:
    log = tmp_path / "runs.log"
    runner = make(
        runners, workspace, state_dir, command=fake_analyzer(tmp_path, log=str(log))
    )
    path = write_artifact(workspace / "run.jsonl")
    complete = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(b'{"partial')
    settle(runner, "run.jsonl", snapshot_of(path, "run.jsonl"))
    [run] = log_lines(log)
    assert run["mode"] == 0o600
    assert Path(run["source"]).parent == state_dir / "analysis"
    assert run["bytes"].encode() == complete
    assert not Path(run["source"]).exists()


def test_a_timeout_kills_the_subprocess(
    workspace: Path,
    state_dir: Path,
    tmp_path: Path,
    runners: list[AnalyzerRunner],
) -> None:
    log = tmp_path / "runs.log"
    command = fake_analyzer(tmp_path, log=str(log), sleep=60)
    runner = make(
        runners, workspace, state_dir, Limits(analysis_seconds=0.5), command=command
    )
    path = write_artifact(workspace / "run.jsonl")
    snapshot = snapshot_of(path, "run.jsonl")
    started = time.monotonic()
    state = settle(runner, "run.jsonl", snapshot)
    assert time.monotonic() - started < 30
    assert (state.status, state.reason, state.exit_code) == ("failed", "timeout", None)
    [run] = log_lines(log)
    wait_for(lambda: not alive(run["pid"]))
    assert list((state_dir / "analysis").iterdir()) == []
    document = runner.document("run.jsonl", snapshot)
    assert document["limits"] == {"max_bytes": 128 * 2**20, "timeout_s": 0.5}
    assert schema_errors("studio_analysis_v1.schema.json", to_wire(document)) == []


def test_close_kills_the_running_subprocess(
    workspace: Path, state_dir: Path, tmp_path: Path
) -> None:
    log = tmp_path / "runs.log"
    command = fake_analyzer(tmp_path, log=str(log), sleep=60)
    runner = AnalyzerRunner(Root(workspace), state_dir, command=command)
    path = write_artifact(workspace / "run.jsonl")
    snapshot = snapshot_of(path, "run.jsonl")
    runner.get("run.jsonl", snapshot)
    wait_for(lambda: len(log_lines(log)) == 1)
    assert runner.get("run.jsonl", snapshot).status == "running"
    started = time.monotonic()
    runner.close()
    assert time.monotonic() - started < 15
    [run] = log_lines(log)
    assert not alive(run["pid"])
    assert runner.get("run.jsonl", snapshot) == AnalysisState(
        "unavailable", "closed", None, None, None
    )
    runner.close()  # idempotent


def test_close_before_any_run_is_harmless(workspace: Path, state_dir: Path) -> None:
    runner = AnalyzerRunner(Root(workspace), state_dir)
    runner.close()
    path = write_artifact(workspace / "run.jsonl")
    assert runner.get("run.jsonl", snapshot_of(path, "run.jsonl")).reason == "closed"


# --- states without a run ---------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "status", "reason"),
    [
        ({"status": "building"}, "pending", "index_building"),
        ({"identity": None}, "pending", "index_building"),
        ({"status": "failed", "failure": "audio"}, "unavailable", "audio"),
        ({"status": "failed", "failure": None}, "unavailable", "index_failed"),
        ({"growing": True}, "unavailable", "growing"),
    ],
)
def test_snapshot_states_that_run_nothing(
    workspace: Path,
    state_dir: Path,
    tmp_path: Path,
    runners: list[AnalyzerRunner],
    overrides: dict[str, Any],
    status: str,
    reason: str,
) -> None:
    log = tmp_path / "runs.log"
    runner = make(
        runners, workspace, state_dir, command=fake_analyzer(tmp_path, log=str(log))
    )
    path = write_artifact(workspace / "run.jsonl")
    state = runner.get("run.jsonl", snapshot_of(path, "run.jsonl", **overrides))
    assert (state.status, state.reason) == (status, reason)
    time.sleep(0.1)
    assert log_lines(log) == []


def test_artifacts_over_the_size_cap_are_not_analysed(
    workspace: Path, state_dir: Path, runners: list[AnalyzerRunner]
) -> None:
    path = write_artifact(workspace / "run.jsonl")
    runner = make(runners, workspace, state_dir, Limits(analysis_bytes=100))
    state = runner.get("run.jsonl", snapshot_of(path, "run.jsonl"))
    assert (state.status, state.reason) == ("unavailable", "over_size_cap")


def test_an_empty_file_is_not_analysed(
    workspace: Path, state_dir: Path, runners: list[AnalyzerRunner]
) -> None:
    path = workspace / "empty.jsonl"
    path.write_bytes(b"")
    runner = make(runners, workspace, state_dir)
    state = runner.get("empty.jsonl", snapshot_of(path, "empty.jsonl"))
    assert (state.status, state.reason) == ("unavailable", "empty")


def test_a_file_without_a_complete_line_is_not_analysed(
    workspace: Path, state_dir: Path, runners: list[AnalyzerRunner]
) -> None:
    path = workspace / "partial.jsonl"
    path.write_bytes(b'{"event_type": "infer.session"' * 5000)
    runner = make(runners, workspace, state_dir)
    state = settle(runner, "partial.jsonl", snapshot_of(path, "partial.jsonl"))
    assert (state.status, state.reason) == ("unavailable", "no_complete_lines")


def test_a_file_changed_since_the_snapshot_is_not_analysed(
    workspace: Path, state_dir: Path, runners: list[AnalyzerRunner]
) -> None:
    path = write_artifact(workspace / "run.jsonl")
    snapshot = snapshot_of(path, "run.jsonl")
    write_artifact(workspace / "run.jsonl", requests=7)  # a rerun, same path
    runner = make(runners, workspace, state_dir)
    state = settle(runner, "run.jsonl", snapshot)
    assert (state.status, state.reason) == ("unavailable", "changed")


def test_a_missing_file_fails_as_unreadable(
    workspace: Path, state_dir: Path, runners: list[AnalyzerRunner]
) -> None:
    path = write_artifact(workspace / "run.jsonl")
    snapshot = snapshot_of(path, "run.jsonl")
    path.unlink()
    runner = make(runners, workspace, state_dir)
    state = settle(runner, "run.jsonl", snapshot)
    assert (state.status, state.reason) == ("failed", "unreadable")


# --- output -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "limits", "reason"),
    [
        ("not json", Limits(), "bad_output"),
        ("[1, 2]", Limits(), "bad_output"),
        ('{"analysis_version": 2, "padding": "' + "x" * 200 + '"}', None, None),
    ],
)
def test_bad_or_oversized_output_fails(
    workspace: Path,
    state_dir: Path,
    tmp_path: Path,
    runners: list[AnalyzerRunner],
    output: str,
    limits: Limits | None,
    reason: str | None,
) -> None:
    command = fake_analyzer(tmp_path, output=output)
    runner = make(
        runners,
        workspace,
        state_dir,
        limits or Limits(report_bytes=100),
        command=command,
    )
    path = write_artifact(workspace / "run.jsonl")
    state = settle(runner, "run.jsonl", snapshot_of(path, "run.jsonl"))
    assert (state.status, state.reason) == ("failed", reason or "output_too_large")


def test_non_finite_numbers_in_the_report_become_null(
    workspace: Path,
    state_dir: Path,
    tmp_path: Path,
    runners: list[AnalyzerRunner],
) -> None:
    output = (
        '{"analysis_version": 2, "rate": NaN, "started_at_ns": 1788609600000000000}'
    )
    runner = make(
        runners, workspace, state_dir, command=fake_analyzer(tmp_path, output=output)
    )
    path = write_artifact(workspace / "run.jsonl")
    snapshot = snapshot_of(path, "run.jsonl")
    state = settle(runner, "run.jsonl", snapshot)
    assert state.report == {
        "analysis_version": 2,
        "rate": None,
        "started_at_ns": 1788609600000000000,
    }
    document = to_wire(runner.document("run.jsonl", snapshot))
    assert document["non_finite_fields"] == ["rate"]
    assert document["report"]["started_at_ns"] == "1788609600000000000"


# --- respond ----------------------------------------------------------------------


def test_respond_raises_not_ready_with_the_whole_document(
    workspace: Path,
    state_dir: Path,
    tmp_path: Path,
    runners: list[AnalyzerRunner],
) -> None:
    command = fake_analyzer(tmp_path, sleep=0.5)
    runner = make(runners, workspace, state_dir, command=command)
    path = write_artifact(workspace / "run.jsonl")
    snapshot = snapshot_of(path, "run.jsonl")
    with pytest.raises(NotReady) as raised:
        runner.respond("run.jsonl", snapshot)
    state = raised.value.state
    assert state["format"] == "stormlog.studio.analysis"
    assert state["status"] in ("pending", "running")
    assert state["retry_after_s"] == raised.value.retry_after_s == 2
    assert state["artifact"]["index_id"] == INDEX_ID
    assert schema_errors("studio_analysis_v1.schema.json", to_wire(state)) == []
    settle(runner, "run.jsonl", snapshot)
    document = runner.respond("run.jsonl", snapshot)
    assert document["status"] == "ready"


def test_respond_returns_unavailable_states_without_202(
    workspace: Path, state_dir: Path, runners: list[AnalyzerRunner]
) -> None:
    runner = make(runners, workspace, state_dir)
    path = write_artifact(workspace / "run.jsonl")
    snapshot = snapshot_of(path, "run.jsonl", growing=True)
    document = runner.respond("run.jsonl", snapshot)
    assert (document["status"], document["reason"]) == ("unavailable", "growing")
    assert schema_errors("studio_analysis_v1.schema.json", to_wire(document)) == []


def test_finished_results_are_bounded(
    workspace: Path,
    state_dir: Path,
    tmp_path: Path,
    runners: list[AnalyzerRunner],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analysis_cache, "KEPT_RESULTS", 2)
    log = tmp_path / "runs.log"
    runner = make(
        runners, workspace, state_dir, command=fake_analyzer(tmp_path, log=str(log))
    )
    path = write_artifact(workspace / "run.jsonl")
    for generation in range(4):
        snap = snapshot_of(path, "run.jsonl", generation=generation)
        settle(runner, "run.jsonl", snap)
    assert len(runner._entries) == 2
    settle(runner, "run.jsonl", snapshot_of(path, "run.jsonl", generation=0))
    assert len(log_lines(log)) == 5  # generation 0 was dropped, so it ran again
