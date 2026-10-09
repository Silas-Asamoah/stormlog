"""Starting, stopping and cleaning up after the processes an experiment runs."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.experiment_process import (
    MARK_VARIABLE,
    Cleanup,
    Launched,
    clean_up_after,
    current_boot,
    end_journaled,
    identify,
    journaled,
    launch,
    parse_cpu_list,
    remembered_tree,
    run_step,
    still_there,
    stop,
    stop_journaled,
    verify_cleanup,
    wait_for_file,
)
from tests.infer_proc_helpers import fake_process

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
    process = subprocess.Popen(["/bin/sleep", "60"])
    try:
        survivor = identify(process.pid)
        assert survivor["pid"] == process.pid
        assert still_there(survivor)
        # The same PID, started at another time, is another process.
        start = "start_ticks" if "start_ticks" in survivor else "create_time"
        other = {**survivor, start: survivor[start] + 1}
        assert not still_there(other)
    finally:
        process.kill()
        process.wait()
    assert not still_there(survivor)
    # A survivor recorded without its start time cannot be told apart.
    assert not still_there({"pid": os.getpid()})


def test_a_survivor_recorded_in_another_boot_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Linux start ticks count from the boot, so after a reboot a process may
    # hold a recorded survivor's PID and start ticks; a resume then refused
    # and told the operator to stop it. A survivor is named by its boot too.
    from stormlog.infer import experiment_process

    process = subprocess.Popen(["/bin/sleep", "60"])
    try:
        survivor = identify(process.pid)
        assert survivor["boot_id"] == experiment_process.current_boot()
        assert still_there(survivor)
        assert not still_there({**survivor, "boot_id": "another-boot"})
        # Rebooted: what runs now has the PID and start the record names.
        monkeypatch.setattr(experiment_process, "current_boot", lambda: "next-boot")
        assert identify(process.pid)["boot_id"] == "next-boot"
        assert not still_there(survivor)
        # A record of no known boot is judged by its start alone.
        assert still_there({**survivor, "boot_id": None})
    finally:
        process.kill()
        process.wait()


STUBBORN = (
    "import pathlib, signal, sys, time; "
    "signal.signal(signal.SIGINT, signal.SIG_IGN); "
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "pathlib.Path(sys.argv[1]).touch(); time.sleep(60)"
)


def test_a_group_that_ignores_its_signals_is_killed(tmp_path: Path) -> None:
    # rev-213-a's mutant r6: nothing failed without the SIGKILL escalation.
    # It is signalled once it says its handlers are in place: under -n 4 a
    # fixed sleep let SIGINT arrive first (close-213-pr24-cloud's F6).
    ready = tmp_path / "ready"
    launched = launch("server", [sys.executable, "-c", STUBBORN, str(ready)])
    assert wait_for_file(ready, 10, launched)
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


def test_on_linux_an_orphan_a_subreaper_among_the_runners_ancestors_adopted_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # close-213-pr24-cloud's F4: a subreaper (systemd --user on a desktop,
    # here 777) adopts the launch's orphans in init's place, so only ppid 1
    # missed them. Any parent outside the runner's ancestry still shows
    # whose a process is; macOS has no subreapers.
    from stormlog.infer import experiment_process as ep

    ticks, uid, parent = ep._clock_ticks(), os.getuid(), os.getppid()
    start = 5_000 * ticks
    _table(
        monkeypatch,
        {
            parent: ep._Process(1, 777, uid),
            777: ep._Process(1, 1, uid),
            40: ep._Process(start, 777, uid),
            41: ep._Process(start, parent, uid),
            42: ep._Process(start, 888, uid),
            43: ep._Process(start, 1, uid),
        },
    )
    since = {"pid": 1, "start_ticks": start - 1}
    unclear = {40, 41, 42, 43}
    found = ep._blind(unclear, since, None, Path("/proc"), "proc")
    assert [item["pid"] for item in found] == [40, 41, 43]
    found = ep._blind(unclear, since, None, Path("/proc"), "psutil")
    assert [item["pid"] for item in found] == [43]


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


def test_a_process_started_after_the_launch_ended_is_not_its(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only a process the launch's tree was alive to start can be its: one
    # an orphan of another run started after this launch's leader exited is
    # another's (concurrent tests on a Mac made such orphans blind).
    from stormlog.infer import experiment_process as ep

    launch_s = 1_000_000.0
    _table(
        monkeypatch,
        {
            30: ep._Process(launch_s + 5, 1, os.getuid()),
            31: ep._Process(launch_s + 20, 1, os.getuid()),
        },
    )
    since = {"pid": 1, "create_time": launch_s}
    may = ep._may_be_launched
    assert may(30, since, Path("/proc"), "psutil", lasted_s=10.0) is True
    assert may(31, since, Path("/proc"), "psutil", lasted_s=10.0) is False
    # Still running, or not known: no bound.
    assert may(31, since, Path("/proc"), "psutil", lasted_s=None) is True


def test_on_linux_a_start_after_the_leader_exited_is_counted_in_ticks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # close-213-pr24-cloud's mutant n6_later_wallclock: on Linux the bound
    # is how long the leader ran, in ticks since boot, not in seconds.
    from stormlog.infer import experiment_process as ep

    ticks, launch_ticks = ep._clock_ticks(), 1_000_000
    _table(
        monkeypatch,
        {
            50: ep._Process(launch_ticks + 5 * ticks, 1, os.getuid()),
            51: ep._Process(launch_ticks + 20 * ticks, 1, os.getuid()),
        },
    )
    since = {"pid": 1, "start_ticks": launch_ticks}
    may = ep._may_be_launched
    assert may(50, since, Path("/proc"), "proc", lasted_s=10.0) is True
    assert may(51, since, Path("/proc"), "proc", lasted_s=10.0) is False


def test_a_launch_knows_how_long_its_leader_ran() -> None:
    launched = launch("step", [sys.executable, "-c", "import time; time.sleep(0.3)"])
    assert launched.lasted_s() is None
    launched.process.wait(timeout=5)
    launched.poll()
    lasted = launched.lasted_s()
    assert lasted is not None and 0.2 < lasted < 5


SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]


def _sleeper(env: dict[str, str] | None = None) -> subprocess.Popen[bytes]:
    """A process in a session and group of its own, as any shell or daemon is."""
    return subprocess.Popen(
        SLEEPER, env={**os.environ, **(env or {})}, start_new_session=True
    )


def _ended(launched: Launched) -> dict[str, Any]:
    """A journal entry for a launch whose leader has exited."""
    launched.process.wait(timeout=5)
    return {
        "name": "server",
        "pid": launched.pid,
        "pgid": launched.pid,
        "mark": launched.mark,
        "identity": launched.identity,
    }


def test_a_journaled_pid_another_process_holds_is_left_alone() -> None:
    # fable-213's lens a: an unrelated session leader holding a journaled
    # PID, started at another time, was counted as the launch's group on
    # resume and killed (verified: True, killed: (pid,)).
    other = _sleeper()
    try:
        identity = identify(other.pid)
        start = "start_ticks" if "start_ticks" in identity else "create_time"
        entry = {
            "name": "server",
            "pid": other.pid,
            "pgid": other.pid,
            "mark": secrets.token_hex(16),
            "identity": {**identity, start: identity[start] - 1},
        }
        cleanup = stop_journaled(entry, timeout_s=0.5)
        assert other.poll() is None
        assert other.pid not in cleanup.killed
        assert other.pid not in [s["pid"] for s in cleanup.survivors]
    finally:
        other.kill()
        other.wait()


def test_a_launch_journaled_in_another_boot_is_neither_signalled_nor_counted() -> None:
    # Its PID and start time name another process in this boot, whatever
    # runs under them now.
    launched = launch("server", SLEEPER)
    try:
        entry = {
            "name": "server",
            "pid": launched.pid,
            "pgid": launched.pid,
            "mark": launched.mark,
            "identity": {**launched.identity, "boot_id": "another-boot"},
        }
        cleanup = stop_journaled(entry, timeout_s=0.5)
        assert (cleanup.verified, cleanup.killed, cleanup.survivors) == (True, (), ())
        assert launched.poll() is None
    finally:
        stop(launched, timeout_s=2)


def test_once_its_leader_is_gone_only_its_exact_mark_ties_a_process_to_a_launch() -> (
    None
):
    # The lead's rule: with the leader gone its PID ties nothing, but a
    # readable environment holding the mark variable with exactly the
    # journal's 64-bit-or-more nonce is the launch's, in the journal's boot.
    entry = _ended(launch("server", [sys.executable, "-c", "pass"]))
    mark = entry["mark"]
    carrier = _sleeper({MARK_VARIABLE: mark})
    longer = _sleeper({MARK_VARIABLE: mark + "ff"})
    another = _sleeper({MARK_VARIABLE: secrets.token_hex(16)})
    short = _sleeper({MARK_VARIABLE: "ab12"})
    try:
        cleanup = stop_journaled(entry, timeout_s=2)
        assert cleanup.killed == (carrier.pid,)
        assert carrier.wait(timeout=5) == -9
        assert longer.poll() is None and another.poll() is None
        # A journaled mark too short to be a nonce ties nothing.
        cleanup = stop_journaled({**entry, "mark": "ab12"}, timeout_s=0.3)
        assert cleanup.killed == () and short.poll() is None
    finally:
        for process in (carrier, longer, another, short):
            process.kill()
            process.wait()


def test_a_launch_of_no_known_boot_lists_what_carries_its_mark_and_kills_nothing() -> (
    None
):
    entry = _ended(launch("server", [sys.executable, "-c", "pass"]))
    carrier = _sleeper({MARK_VARIABLE: entry["mark"]})
    try:
        unknown = {**entry, "identity": {**entry["identity"], "boot_id": None}}
        cleanup = stop_journaled(unknown, timeout_s=0.3)
        assert not cleanup.verified and cleanup.killed == ()
        assert [s["pid"] for s in cleanup.survivors] == [carrier.pid]
        assert carrier.poll() is None
    finally:
        carrier.kill()
        carrier.wait()


class _FakeLinux:
    """stop_journaled's Linux branch over a fake /proc: what it signals and
    kills, each kill or group signal ending the processes it reaches."""

    def __init__(self, proc: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from stormlog.infer import experiment_process as ep

        self.proc = proc
        self.signalled: list[tuple[int, int]] = []
        self.killed: list[int] = []
        monkeypatch.setattr(ep, "_linux", lambda: True)
        monkeypatch.setattr(ep, "current_boot", lambda: "boot-a")
        monkeypatch.setattr(ep, "_alive", lambda pid: (proc / str(pid)).is_dir())
        monkeypatch.setattr(ep, "_signal_group", self._signal_group)
        monkeypatch.setattr(ep, "_kill", self._kill)

    def _signal_group(self, pgid: int, signum: int) -> None:
        self.signalled.append((pgid, signum))
        shutil.rmtree(self.proc / str(pgid), ignore_errors=True)

    def _kill(self, pid: int) -> None:
        self.killed.append(pid)
        shutil.rmtree(self.proc / str(pid), ignore_errors=True)


MARK = "a1" * 16


def _journal_entry(pid: int, start: int, boot: str = "boot-a") -> dict[str, Any]:
    identity = {"pid": pid, "start_ticks": start, "boot_id": boot}
    return {
        "name": "server",
        "pid": pid,
        "pgid": pid,
        "mark": MARK,
        "identity": identity,
    }


def test_on_linux_a_journaled_pid_another_process_holds_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = tmp_path / "proc"
    fake_process(proc, 4000, start=900, environ={"HOME": "/root"})
    fake = _FakeLinux(proc, monkeypatch)
    # Another start time, in this boot: fable-213's repro.
    cleanup = stop_journaled(_journal_entry(4000, 500), timeout_s=0.2, proc=proc)
    assert (fake.signalled, fake.killed, cleanup.verified) == ([], [], True)
    # The same start ticks, in another boot.
    entry = _journal_entry(4000, 900, boot="boot-b")
    cleanup = stop_journaled(entry, timeout_s=0.2, proc=proc)
    assert (fake.signalled, fake.killed, cleanup.verified) == ([], [], True)


def test_on_linux_a_gone_leaders_group_is_judged_by_its_mark_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = tmp_path / "proc"
    marks = {4001: MARK, 4002: "b2" * 16, 4003: MARK + "ff", 4004: None}
    for pid, mark in marks.items():
        environ = None if mark is None else {MARK_VARIABLE: mark}
        fake_process(proc, pid, pgid=4000, sid=4000, start=600, environ=environ)
    # vLLM's titles overwrite an environment in place.
    (proc / "4004" / "environ").write_bytes(b"VLLM::EngineCore\0\0\0")
    fake = _FakeLinux(proc, monkeypatch)
    cleanup = stop_journaled(_journal_entry(4000, 500), timeout_s=0.3, proc=proc)
    assert (fake.signalled, fake.killed) == ([], [4001])
    assert not cleanup.verified
    assert [item["pid"] for item in cleanup.blind] == [4004]
    assert sorted(int(p.name) for p in proc.iterdir() if p.name.isdigit()) == [
        4002,
        4003,
        4004,
    ]


def test_on_linux_a_journaled_group_is_not_signalled_again_once_its_leader_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # close-213-pr24-cloud's mutant n1_stop_journaled_identity_off: once
    # the leader has left, its PID may be another's, so the group gets no
    # SIGKILL; what the group left is killed by PID, as the launch's.
    proc = tmp_path / "proc"
    fake_process(proc, 4000, start=500, environ={MARK_VARIABLE: MARK})
    fake_process(proc, 4001, pgid=4000, sid=4000, start=600, environ={})
    fake = _FakeLinux(proc, monkeypatch)
    cleanup = stop_journaled(_journal_entry(4000, 500), timeout_s=0.2, proc=proc)
    assert (fake.signalled, fake.killed) == ([(4000, signal.SIGTERM)], [4001])
    assert cleanup.verified


def test_on_linux_a_journaled_leader_still_running_has_its_group_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc = tmp_path / "proc"
    fake_process(proc, 4000, start=500, environ={MARK_VARIABLE: MARK})
    fake = _FakeLinux(proc, monkeypatch)
    cleanup = stop_journaled(_journal_entry(4000, 500), timeout_s=0.2, proc=proc)
    assert (fake.signalled, fake.killed) == ([(4000, signal.SIGTERM)], [])
    assert cleanup.verified


def test_a_launch_ends_in_its_journal_only_by_a_whole_line(tmp_path: Path) -> None:
    # A resume judges again only a launch whose cleanup did not verify. A
    # crash may tear the journal's last line: a torn end leaves its launch
    # to the resume, never ends it.
    journal = tmp_path / "launches.ndjson"
    launched = launch("step", [sys.executable, "-c", "pass"], journal=journal)
    launched.process.wait(timeout=5)
    launched.poll()
    (entry,) = journaled(journal)
    assert entry["mark"] == launched.mark
    assert entry["identity"]["boot_id"] == current_boot()
    whole = journal.read_text()
    end = json.dumps({"ended": launched.mark})
    for torn in (end[:-1], end, end[:20] + "\n"):
        journal.write_text(whole + torn)
        assert journaled(journal) == [entry], torn
    # An unverified cleanup ends nothing; a verified one ends the launch.
    journal.write_text(whole)
    end_journaled(launched, Cleanup(False, "proc"))
    assert journaled(journal) == [entry]
    end_journaled(launched, Cleanup(True, "proc"))
    assert journaled(journal) == []
    # gate-213-b23's G2: a later entry started on the torn line, joined it,
    # and so went unread.
    journal.write_text(whole + end[:-1])
    later = launch("step", [sys.executable, "-c", "pass"], journal=journal)
    later.process.wait(timeout=5)
    assert [item["mark"] for item in journaled(journal)] == [entry["mark"], later.mark]
    end_journaled(later, Cleanup(True, "proc"))
    assert journaled(journal) == [entry]
    # An end names its launch by the whole mark.
    journal.write_text(whole + json.dumps({"ended": launched.mark[:16]}) + "\n")
    assert journaled(journal) == [entry]


# An orphan with an emptied environment that waits for the launch's leader
# to exit, then 2.5 s, starts another one (its PID into argv[2]) and exits.
# The grandchild starts through env -i: Linux's dash exports PWD to its
# children, which would make the grandchild's environment readable.
LATE_GRANDCHILD = (
    "import subprocess, sys; "
    "script = 'while kill -0 {leader} 2>/dev/null; do /bin/sleep 0.1; done; "
    "/bin/sleep 2.5; /usr/bin/env -i /bin/sleep 60 & echo $! > {out}; exit 0'; "
    "subprocess.Popen(['/bin/sh', '-c', script.format(leader=sys.argv[1], "
    "out=sys.argv[2])], env={}, start_new_session=True)"
)


def test_an_orphan_started_late_by_a_blind_one_is_not_cleared_by_its_start(
    tmp_path: Path,
) -> None:
    # close-213-pr24-cloud's F3: a blind orphan, seen at the first poll,
    # started an emptied grandchild once the leader had been gone 2 s and
    # left; the grandchild's late start cleared it, and the check verified
    # while it ran on.
    out = tmp_path / "grandchild.pid"
    launched = launch(
        "server",
        [sys.executable, "-c", "import sys, time; time.sleep(0.3)", str(out)],
    )
    spawner = subprocess.Popen(
        [sys.executable, "-c", LATE_GRANDCHILD, str(launched.pid), str(out)],
        env={**os.environ, MARK_VARIABLE: launched.mark},
    )
    spawner.wait(timeout=5)
    launched.process.wait(timeout=5)
    launched.poll()
    try:
        cleanup = verify_cleanup(
            launched.pid,
            wait_s=4.5,
            mark=launched.mark,
            since=launched.identity,
            lasted_s=launched.lasted_s(),
        )
        assert not cleanup.verified
        assert int(out.read_text()) in [item["pid"] for item in cleanup.blind]
    finally:
        if out.exists():
            os.kill(int(out.read_text()), signal.SIGKILL)


def test_a_clean_poll_after_one_that_found_something_verifies_only_if_the_next_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The same escape within one poll: the blind orphan started its
    # grandchild and left while the poll read the process table, so the poll
    # listed neither. The next poll, which starts after it left, lists the
    # grandchild.
    from stormlog.infer import experiment_process as ep

    orphan = {"pid": 10, "start_ticks": 1, "boot_id": "b"}
    grandchild = {"pid": 11, "start_ticks": 2, "boot_id": "b"}

    def looks(*blinds: tuple[dict[str, Any], ...]) -> None:
        polls = iter([ep._Poll(set(), blind, 0) for blind in blinds])
        last = ep._Poll(set(), blinds[-1], 0)
        monkeypatch.setattr(ep, "_look", lambda *args: next(polls, last))

    looks((orphan,), (), (grandchild,))
    cleanup = verify_cleanup(4000, wait_s=0.3)
    assert (cleanup.verified, cleanup.blind) == (False, (grandchild,))
    looks((orphan,), (), ())
    assert verify_cleanup(4000, wait_s=0.3).verified
    # Nothing seen: the first clean poll verifies.
    looks(())
    assert verify_cleanup(4000, wait_s=0).verified
    # The deadline passes during the first clean poll: one more decides
    # (gate-213-final's mutant f3_deadline_ignores_clean).
    sequence = [(orphan,), (), ()]

    def slow(*args: Any) -> Any:
        blind = sequence.pop(0) if sequence else ()
        if not blind:
            time.sleep(0.3)
        return ep._Poll(set(), blind, 0)

    monkeypatch.setattr(ep, "_look", slow)
    assert verify_cleanup(4000, wait_s=0.2).verified


def test_a_late_start_clears_nothing_once_the_leaders_exit_was_seen_long_before(
    tmp_path: Path,
) -> None:
    # gate-213-final's NF3: a treatment whose leader exit was seen early is
    # checked only after another one's slow stop. An orphan of it started an
    # emptied grandchild over 2 s after that exit and left before the check,
    # so nothing of the launch was seen, and the grandchild's late start
    # cleared it.
    out = tmp_path / "grandchild.pid"
    launched = launch(
        "treatment", [sys.executable, "-c", "import time; time.sleep(0.3)"]
    )
    spawner = subprocess.Popen(
        [sys.executable, "-c", LATE_GRANDCHILD, str(launched.pid), str(out)],
        env={**os.environ, MARK_VARIABLE: launched.mark},
    )
    spawner.wait(timeout=5)
    launched.process.wait(timeout=5)
    launched.poll()  # the exit, seen at once
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not (
            out.exists() and out.read_text().strip()
        ):
            time.sleep(0.05)
        time.sleep(0.3)  # the orphan that started it has left
        assert launched.bound_s() is None
        cleanup = clean_up_after(launched, wait_s=0.5)
        assert int(out.read_text()) in [item["pid"] for item in cleanup.blind]
    finally:
        if out.exists() and out.read_text().strip():
            os.kill(int(out.read_text()), signal.SIGKILL)
