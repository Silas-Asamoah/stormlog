"""Reading a vLLM server's processes from /proc."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from stormlog.infer.server_process import (
    API_SERVER,
    COMPILE_WORKER,
    ENGINE_CORE,
    OTHER,
    RESOURCE_TRACKER,
    WORKER,
    boot_time_s,
    classify_role,
    group_members,
    process_tree,
    read_environ,
    read_process,
    start_ns,
    still_running,
)


def _stat(
    pid: int, comm: str, ppid: int, pgid: int, sid: int, start: int, state: str
) -> str:
    # Fields 3..22 of /proc/<pid>/stat; starttime is the twentieth after comm.
    after = [state, ppid, pgid, sid] + [0] * 15 + [start] + [0] * 30
    return f"{pid} ({comm}) " + " ".join(str(field) for field in after) + "\n"


def _process(
    proc: Path,
    pid: int,
    *,
    ppid: int = 1,
    pgid: int | None = None,
    sid: int | None = None,
    start: int = 1000,
    comm: str = "python3",
    cmdline: tuple[str, ...] = ("python3",),
    cpus: str | None = "0-7",
    environ: dict[str, str] | None = None,
    state: str = "S",
) -> None:
    base = proc / str(pid)
    base.mkdir(parents=True)
    pgid = pid if pgid is None else pgid
    sid = pid if sid is None else sid
    (base / "stat").write_text(_stat(pid, comm, ppid, pgid, sid, start, state))
    (base / "cmdline").write_bytes("\0".join(cmdline).encode() + b"\0")
    status = f"Name:\t{comm}\n" + (f"Cpus_allowed_list:\t{cpus}\n" if cpus else "")
    (base / "status").write_text(status)
    if environ is not None:
        raw = "\0".join(f"{k}={v}" for k, v in environ.items()) + "\0"
        (base / "environ").write_bytes(raw.encode())


@pytest.fixture
def server(tmp_path: Path) -> Path:
    """A vLLM 0.30.0 server tree as /proc shows it, plus an unrelated process."""
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "stat").write_text("cpu 1 2 3\nbtime 1700000000\nprocesses 9\n")
    serve = ("/usr/bin/python3", "/usr/local/bin/vllm", "serve", "Qwen/Qwen2.5-0.5B")
    _process(proc, 100, start=500, cmdline=serve, environ={"VLLM_PORT": "8000"})
    _process(
        proc,
        101,
        ppid=100,
        pgid=100,
        sid=100,
        start=510,
        comm="VLLM::EngineCor",
        cmdline=("VLLM::EngineCore",),
    )
    _process(
        proc,
        102,
        ppid=101,
        pgid=100,
        sid=100,
        start=520,
        comm="VLLM::Worker_TP",
        cmdline=("VLLM::Worker_TP0",),
        cpus="0-3",
    )
    _process(
        proc,
        103,
        ppid=100,
        pgid=100,
        sid=100,
        start=505,
        cmdline=(
            "python3",
            "-c",
            "from multiprocessing.resource_tracker import main;main(9)",
        ),
    )
    _process(
        proc,
        104,
        ppid=100,
        pgid=100,
        sid=100,
        start=600,
        comm="pip",
        cmdline=("/usr/bin/python3", "-m", "pip", "list", "--format=freeze"),
    )
    _process(proc, 200, start=50, comm="bash", cmdline=("bash",))
    return proc


def test_a_process_is_read_with_its_group_session_and_start(server: Path) -> None:
    worker = read_process(102, server)
    assert worker is not None
    assert (worker.ppid, worker.pgid, worker.sid) == (101, 100, 100)
    assert worker.key == (102, 520)
    assert worker.role == WORKER
    assert worker.cpus_allowed_list == "0-3"
    record = worker.to_record(boot_time_s=boot_time_s(server))
    assert record["start_ns"] == start_ns(520, 1_700_000_000)


def test_a_command_name_with_spaces_and_parentheses_is_parsed(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _process(proc, 7, comm="a) (b c", ppid=3, pgid=4, sid=5, start=99)
    info = read_process(7, proc)
    assert info is not None
    assert (info.comm, info.ppid, info.pgid, info.sid, info.start_ticks) == (
        "a) (b c",
        3,
        4,
        5,
        99,
    )


def test_a_process_that_is_gone_reads_as_none(tmp_path: Path) -> None:
    assert read_process(12345, tmp_path) is None
    assert read_environ(12345, tmp_path) is None


@pytest.mark.parametrize(
    ("comm", "cmdline", "role"),
    [
        ("VLLM::EngineCor", ("VLLM::EngineCore",), ENGINE_CORE),
        ("VLLM::EngineCor", ("VLLM::EngineCore_DP0",), ENGINE_CORE),
        ("VLLM::Worker_TP", ("VLLM::Worker_TP1",), WORKER),
        ("VLLM::Worker", ("VLLM::Worker",), WORKER),
        ("MYPFX::Worker_P", ("MYPFX::Worker_PP0_TP0   ",), WORKER),
        ("vllm", ("/usr/bin/python3", "/usr/local/bin/vllm", "serve", "m"), API_SERVER),
        (
            "python3",
            ("python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "m"),
            API_SERVER,
        ),
        (
            "python3",
            ("python3", "-c", "from multiprocessing.resource_tracker import main"),
            RESOURCE_TRACKER,
        ),
        (
            "python3",
            ("python3", "-m", "torch._inductor.compile_worker", "--workers=4"),
            COMPILE_WORKER,
        ),
        ("nvidia-smi", ("nvidia-smi", "topo", "-m"), OTHER),
        ("sh", ("/bin/sh", "-c", "lsb_release -a"), OTHER),
        ("VLLM::Workers", ("VLLM::Workers",), OTHER),
    ],
)
def test_roles_come_from_vllms_titles_and_command_lines(
    comm: str, cmdline: tuple[str, ...], role: str
) -> None:
    assert classify_role(comm, cmdline) == role


def test_the_tree_holds_the_root_and_every_descendant(server: Path) -> None:
    tree = process_tree(100, server)
    assert tree[0].pid == 100
    assert sorted(info.pid for info in tree) == [100, 101, 102, 103, 104]
    roles = {info.pid: info.role for info in tree}
    assert roles == {
        100: API_SERVER,
        101: ENGINE_CORE,
        102: WORKER,
        103: RESOURCE_TRACKER,
        104: OTHER,
    }
    assert process_tree(999, server) == []


def test_group_members_include_the_session_when_asked(server: Path) -> None:
    _process(server, 105, ppid=1, pgid=105, sid=100, start=700)
    by_group = {info.pid for info in group_members(100, proc=server)}
    by_both = {info.pid for info in group_members(100, 100, proc=server)}
    assert by_group == {100, 101, 102, 103, 104}
    assert by_both == by_group | {105}


def test_still_running_matches_pid_and_start_time(server: Path) -> None:
    known = [(101, 510), (102, 999), (300, 1)]
    # PID 102 now belongs to a process that started later: a reused PID.
    assert [info.pid for info in still_running(known, server)] == [101]


def test_a_zombie_is_neither_a_member_nor_a_survivor(server: Path) -> None:
    _process(server, 106, ppid=100, pgid=100, sid=100, start=800, state="Z")
    assert 106 not in {info.pid for info in group_members(100, 100, proc=server)}
    assert still_running([(106, 800)], server) == []


def test_the_environment_is_read_as_names_and_values(server: Path) -> None:
    assert read_environ(100, server) == {"VLLM_PORT": "8000"}


# ------------------------------------------------------------- real processes

linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="reads the real /proc, which only Linux has"
)


def _wait_for(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


@pytest.fixture
def session() -> Iterator[subprocess.Popen[bytes]]:
    """A session leader with two children, one of which leaves the session."""
    escape = "import os, time; os.setsid(); time.sleep(60)"
    script = f'sleep 60 & {sys.executable} -c "{escape}" & wait'
    leader = subprocess.Popen(["/bin/sh", "-c", script], start_new_session=True)
    try:
        yield leader
    finally:
        for info in process_tree(leader.pid):
            try:
                os.kill(info.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        leader.wait(timeout=10)


@linux_only
def test_a_real_group_is_listed_and_empty_after_it_is_killed(
    session: subprocess.Popen[bytes],
) -> None:
    leader = session.pid
    assert _wait_for(lambda: len(process_tree(leader)) == 3)
    tree = process_tree(leader)
    escaped = [info for info in tree if info.sid != leader]
    assert len(escaped) == 1
    assert {info.pid for info in group_members(leader, leader)} == {
        info.pid for info in tree if info.sid == leader
    }

    os.killpg(leader, signal.SIGKILL)
    session.wait(timeout=10)
    # The group and session are empty once the killed processes are reaped,
    # but the child that called setsid survives: only the remembered tree
    # shows it.
    assert _wait_for(lambda: group_members(leader, leader) == [])
    survivors = still_running([info.key for info in tree])
    assert [info.pid for info in survivors] == [escaped[0].pid]


@linux_only
def test_the_real_proc_gives_this_process_its_start_and_environment() -> None:
    me = read_process(os.getpid())
    assert me is not None
    assert (me.pgid, me.sid) == (os.getpgid(0), os.getsid(0))
    assert me.cpus_allowed_list
    environ = read_environ(os.getpid())
    assert environ is not None and "PATH" in environ
    assert boot_time_s() is not None
