"""Starting, stopping and cleaning up after the processes an experiment runs."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from stormlog.infer.experiment_process import (
    launch,
    parse_cpu_list,
    remembered_tree,
    run_step,
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
