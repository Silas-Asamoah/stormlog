"""Starting, stopping and cleaning up after the processes an experiment runs."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.experiment_process import (
    identify,
    launch,
    parse_cpu_list,
    remembered_tree,
    run_step,
    still_there,
    stop,
    verify_cleanup,
    wait_for_file,
)

ESCAPE = "import os, time; os.setsid(); time.sleep(60)"


def test_cpu_lists_are_parsed() -> None:
    assert parse_cpu_list("0-3,8") == {0, 1, 2, 3, 8}
    for bad in ("", "3-1", "-1", "x"):
        with pytest.raises(ValueError):
            parse_cpu_list(bad)


def test_stopping_signals_the_whole_group_and_nothing_is_left(tmp_path: Path) -> None:
    launched = launch(
        "server",
        ["/bin/sh", "-c", "sleep 60 & sleep 60 & wait"],
        log_path=tmp_path / "server.log",
    )
    time.sleep(0.3)
    code = stop(launched, timeout_s=5)
    assert code is not None
    assert launched.stopped_by == "SIGTERM"
    cleanup = verify_cleanup(launched.pid, wait_s=5)
    assert cleanup.verified and cleanup.killed == ()


def test_a_process_that_left_the_group_is_found_and_killed(tmp_path: Path) -> None:
    launched = launch(
        "server",
        ["/bin/sh", "-c", f'{sys.executable} -c "{ESCAPE}" & wait'],
    )
    deadline = time.monotonic() + 10
    escapee = None
    while escapee is None and time.monotonic() < deadline:
        import psutil

        children = psutil.Process(launched.pid).children()
        escapee = next((c for c in children if os.getsid(c.pid) != launched.pid), None)
        time.sleep(0.05)
    assert escapee is not None
    remembered = remembered_tree(launched.pid) or [(escapee.pid, 0)]
    stop(launched, timeout_s=5)
    cleanup = verify_cleanup(launched.pid, remembered, wait_s=5)
    # The escapee survived the group's signal; it is killed by PID and listed.
    assert cleanup.verified
    assert escapee.pid in cleanup.killed


LATE_ESCAPE = (
    "import subprocess, sys, time; time.sleep(0.5); "
    "subprocess.Popen([sys.executable, '-c', "
    "'import os, time; os.setsid(); time.sleep(60)']); time.sleep(60)"
)


def test_a_process_forked_after_the_tree_was_remembered_is_found_by_its_mark(
    tmp_path: Path,
) -> None:
    # A collector forked late, then moved to its own session, is in no group,
    # no session and no remembered tree; only the mark it inherited finds it.
    launched = launch("server", [sys.executable, "-c", LATE_ESCAPE])
    remembered = remembered_tree(launched.pid)
    import psutil

    deadline = time.monotonic() + 10
    escapee = None
    while escapee is None and time.monotonic() < deadline:
        children = psutil.Process(launched.pid).children(recursive=True)
        escapee = next((c for c in children if os.getsid(c.pid) != launched.pid), None)
        time.sleep(0.05)
    assert escapee is not None
    stop(launched, timeout_s=5)
    cleanup = verify_cleanup(launched.pid, remembered, wait_s=5, mark=launched.mark)
    assert cleanup.verified
    assert escapee.pid in cleanup.killed
    assert not psutil.pid_exists(escapee.pid) or (
        psutil.Process(escapee.pid).status() == psutil.STATUS_ZOMBIE
    )


def test_every_launch_carries_its_own_mark() -> None:
    first = launch("a", [sys.executable, "-c", "pass"])
    second = launch("b", [sys.executable, "-c", "pass"])
    assert first.mark and second.mark and first.mark != second.mark
    for launched in (first, second):
        launched.process.wait(timeout=5)


def test_a_step_that_runs_too_long_is_stopped() -> None:
    launched, timed_out = run_step("slow", ["/bin/sleep", "30"], timeout_s=0.3)
    assert timed_out
    assert launched.exit_code is not None
    quick, timed_out = run_step("quick", ["/bin/sh", "-c", "exit 3"], timeout_s=10)
    assert not timed_out and quick.exit_code == 3


def test_a_ready_file_is_waited_for_while_its_writer_runs(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    writer = launch("writer", ["/bin/sh", "-c", f"sleep 0.2; touch {ready}; sleep 30"])
    assert wait_for_file(ready, 5, writer)
    stop(writer, timeout_s=5)
    gone = launch("gone", ["/bin/sh", "-c", "exit 1"])
    assert not wait_for_file(tmp_path / "never", 5, gone)


@pytest.mark.skipif(sys.platform != "linux", reason="CPU affinity is Linux-only")
def test_a_process_is_pinned_to_its_cpus() -> None:
    get_affinity = getattr(os, "sched_getaffinity")
    cpu = sorted(get_affinity(0))[0]
    launched = launch("pinned", ["/bin/sleep", "5"], cpu_affinity=str(cpu))
    try:
        assert launched.affinity_applied is True
        assert get_affinity(launched.pid) == {cpu}
    finally:
        stop(launched, timeout_s=2)


def test_the_mark_is_not_a_difference_between_runs() -> None:
    from stormlog.infer.config_classes import LABEL, field_class

    assert field_class("environ.STORMLOG_RUN_MARK") == LABEL


def test_a_survivor_is_known_by_its_start_time_as_well_as_its_pid() -> None:
    import subprocess

    process = subprocess.Popen(["/bin/sleep", "60"])
    try:
        survivor = identify(process.pid)
        assert survivor["pid"] == process.pid
        assert still_there(survivor)
        # The same PID, started at another time, is another process.
        other = {k: (v + 1 if k != "pid" else v) for k, v in survivor.items()}
        assert not still_there(other)
    finally:
        process.kill()
        process.wait()
    assert not still_there(survivor)
    # A survivor recorded without its start time cannot be told apart.
    assert not still_there({"pid": os.getpid()})


STUBBORN = (
    "import signal, time; "
    "signal.signal(signal.SIGINT, signal.SIG_IGN); "
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
)


def test_a_group_that_ignores_its_signals_is_killed() -> None:
    # rev-213-a's mutant r6: nothing failed without the SIGKILL escalation.
    launched = launch("server", [sys.executable, "-c", STUBBORN])
    time.sleep(0.5)
    code = stop(launched, signals=(2, 15), timeout_s=0.5)
    assert (code, launched.stopped_by) == (-9, "SIGKILL")


def test_a_survivor_that_outlives_its_kill_is_listed_and_not_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # rev-213-a's mutant r7: nothing failed when a survivor counted as clean.
    from stormlog.infer import experiment_process

    launched = launch("server", ["/bin/sleep", "60"])
    monkeypatch.setattr(experiment_process, "_kill", lambda pid: None)
    try:
        cleanup = verify_cleanup(launched.pid, wait_s=0.3, mark=launched.mark)
        assert not cleanup.verified
        assert launched.pid in [s["pid"] for s in cleanup.survivors]
        assert launched.pid in cleanup.killed
    finally:
        launched.process.kill()
        launched.process.wait()


def test_a_mark_search_that_could_not_read_an_environment_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # rev-213-a's E6: an environment psutil cannot read (macOS hides a
    # platform binary's) was skipped as unmarked, and the search called
    # complete.
    import psutil

    from stormlog.infer import experiment_process

    class Hidden:
        pid = 4242

        def environ(self) -> dict[str, str]:
            raise psutil.AccessDenied(4242)

    monkeypatch.setattr(experiment_process, "_linux", lambda: False)
    monkeypatch.setattr(psutil, "process_iter", lambda: iter([Hidden()]))
    launched = launch("server", [sys.executable, "-c", "pass"])
    launched.process.wait(timeout=5)
    cleanup = verify_cleanup(launched.pid, wait_s=0.3, mark=launched.mark)
    assert cleanup.to_record()["mark_search"] == {
        "complete": False,
        "unreadable": 1,
        "blind": [],
    }

    class Plain:
        pid = 4343

        def environ(self) -> dict[str, str]:
            return {}

    monkeypatch.setattr(psutil, "process_iter", lambda: iter([Plain()]))
    cleanup = verify_cleanup(launched.pid, wait_s=0.3, mark=launched.mark)
    assert cleanup.to_record()["mark_search"] == {
        "complete": True,
        "unreadable": 0,
        "blind": [],
    }
    # Without a mark there is no search to judge.
    assert verify_cleanup(launched.pid, wait_s=0.3).to_record()["mark_search"] is None


ORPHAN = (
    "import subprocess, sys; "
    "child = subprocess.Popen(['/bin/sleep', '60'], env={}, start_new_session=True); "
    "open(sys.argv[1], 'w').write(str(child.pid))"
)


def test_an_unreadable_process_that_may_be_the_launchs_fails_the_cleanup(
    tmp_path: Path,
) -> None:
    # The lead's ruling on rev-213-a's E6: a process whose environment
    # cannot be read, or was emptied, and that may be the launch's cannot be
    # shown unmarked, so the cleanup is not verified. The repro: the launch
    # leaves /bin/sleep, exec'd with an empty environment in a session of
    # its own, and exits, so init adopts it (on macOS psutil cannot read a
    # platform binary's environment either).
    pid_file = tmp_path / "orphan.pid"
    launched = launch("server", [sys.executable, "-c", ORPHAN, str(pid_file)])
    launched.process.wait(timeout=5)
    orphan = int(pid_file.read_text())
    try:
        time.sleep(0.2)
        cleanup = verify_cleanup(
            launched.pid, wait_s=0.5, mark=launched.mark, since=launched.identity
        )
        assert not cleanup.verified
        # Listed by PID and start time, so a resume can tell it still runs.
        (blind,) = [
            item
            for item in cleanup.to_record()["mark_search"]["blind"]
            if item["pid"] == orphan
        ]
        assert still_there(blind)
        # It may not be the launch's, so it is never killed.
        assert orphan not in cleanup.killed and still_there(blind)
    finally:
        os.kill(orphan, 9)


def _table(monkeypatch: pytest.MonkeyPatch, table: dict[int, Any]) -> None:
    from stormlog.infer import experiment_process as ep

    monkeypatch.setattr(ep, "_view", lambda pid, proc, method: table.get(pid))


def test_a_process_that_cannot_be_the_launchs_is_not_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Excluded: another user's (sudo's too), one older than the launch, and
    # any process with a parent other than init, the runner included. An
    # orphan may be the launch's, whatever it runs: rev-213-a's N6(a) found
    # an escapee running /usr/sbin/iostat cleared by a rule for launchd's
    # executables.
    from stormlog.infer import experiment_process as ep

    runner, uid = os.getpid(), os.getuid()
    launch_s = 1_000_000.0
    _table(
        monkeypatch,
        {
            10: ep._Process(launch_s + 1, 1, uid),
            11: ep._Process(launch_s + 1, runner, uid),
            12: ep._Process(launch_s + 1, 999_999, uid),
            13: ep._Process(launch_s - 10, 1, uid),
            14: ep._Process(launch_s + 1, 1, uid + 1),
            # Within the slack before the launch: may still be its.
            15: ep._Process(launch_s - 1, 1, uid),
        },
    )
    since = {"pid": 1, "create_time": launch_s}
    found = {
        pid: ep._may_be_launched(pid, since, Path("/proc"), "psutil")
        for pid in (10, 11, 12, 13, 14, 15, 16)
    }
    assert found == {
        10: True,
        11: False,
        12: False,
        13: False,
        14: False,
        15: True,
        16: False,
    }


def test_start_times_are_compared_in_the_processes_own_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # rev-213-a's N6(d): on Linux both starts are ticks since boot, so a
    # wall-clock step between the launch and the check changes nothing.
    from stormlog.infer import experiment_process as ep

    ticks = ep._clock_ticks()
    _table(
        monkeypatch,
        {
            20: ep._Process(5_000 * ticks, 1, os.getuid()),
            21: ep._Process(4_990 * ticks, 1, os.getuid()),
        },
    )
    since = {"pid": 1, "start_ticks": 5_000 * ticks - 1}
    assert ep._may_be_launched(20, since, Path("/proc"), "proc") is True
    assert ep._may_be_launched(21, since, Path("/proc"), "proc") is False


@pytest.mark.parametrize(
    ("environ", "unreadable"),
    [
        (b"", False),
        # py-setproctitle, which vLLM uses, overwrites it in place.
        (b"\0\0\0\0\0\0", False),
        (b"VLLM::EngineCore\0\0\0\0", False),
        (None, True),
    ],
)
def test_the_proc_search_cannot_judge_an_environment_it_cannot_read(
    tmp_path: Path, environ: bytes | None, unreadable: bool
) -> None:
    # rev-213-a's N9 and N6(b): the Linux branch, through a fake /proc.
    from stormlog.infer import experiment_process as ep

    if environ is None and os.geteuid() == 0:
        pytest.skip("root reads a file whatever its mode")
    pid = os.getpid()  # alive, as a real entry would be
    entry = tmp_path / str(pid)
    entry.mkdir()
    path = entry / "environ"
    path.write_bytes(b"HOME=/root\0" if environ is None else environ)
    if environ is None:
        path.chmod(0)
    try:
        search = ep._proc_marked(b"STORMLOG_RUN_MARK=m\0", tmp_path)
    finally:
        path.chmod(0o600)
    assert search.unclear == {pid} and search.found == set()
    assert search.unreadable == (1 if unreadable else 0)


def test_the_proc_search_finds_the_mark_and_passes_an_unmarked_environment(
    tmp_path: Path,
) -> None:
    from stormlog.infer import experiment_process as ep

    pid = os.getpid()
    (tmp_path / str(pid)).mkdir()
    environ = tmp_path / str(pid) / "environ"
    environ.write_bytes(b"HOME=/root\0PATH=/bin\0")
    search = ep._proc_marked(b"STORMLOG_RUN_MARK=m\0", tmp_path)
    assert (search.found, search.unclear, search.unreadable) == (set(), set(), 0)
    environ.write_bytes(b"HOME=/root\0STORMLOG_RUN_MARK=m\0")
    assert ep._proc_marked(b"STORMLOG_RUN_MARK=m\0", tmp_path).found == {pid}
