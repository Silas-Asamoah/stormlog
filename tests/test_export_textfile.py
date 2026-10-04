"""The textfile writer: one file and one lock per slot, written atomically."""

import json
import os
import socket
import threading
import time
from pathlib import Path

import psutil
import pytest

from stormlog._export import textfile
from stormlog._export.renders import RenderCache
from stormlog._export.textfile import (
    SlotInUse,
    TextfileWriter,
    slot_paths,
    validate_slot,
)
from tests.export_conformance import check_exposition

BODY = (
    b"# HELP stormlog_up Up.\n# TYPE stormlog_up gauge\n"
    b'stormlog_up{stormlog_producer="alpha"} 1\n'
)


def _writer(
    directory: Path,
    slot: str = "alpha",
    *,
    remove_on_exit: bool = False,
    forbidden: tuple[Path, ...] = (),
) -> TextfileWriter:
    return TextfileWriter(
        directory,
        slot,
        RenderCache(lambda: BODY),
        interval=0.05,
        remove_on_exit=remove_on_exit,
        forbidden=forbidden,
    )


def _wait_for(condition, timeout: float = 5.0) -> bool:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_the_slot_names_the_file_the_lock_and_the_label(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.start()
    try:
        assert _wait_for(lambda: writer.path.exists())
        assert writer.path == tmp_path / "stormlog-alpha.prom"
        assert writer.lock_path == tmp_path / "stormlog-alpha.lock"
        exposition = check_exposition(writer.path.read_text())
        assert exposition.value("stormlog_run_active", stormlog_producer="alpha") == 1
        updated = exposition.value(
            "stormlog_textfile_updated_timestamp_seconds", stormlog_producer="alpha"
        )
        assert abs(updated - time.time()) < 60
    finally:
        writer.close()


def test_the_final_write_says_the_run_ended_and_frees_the_slot(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.start()
    writer.close()
    exposition = check_exposition(writer.path.read_text())
    assert exposition.value("stormlog_run_active") == 0
    assert not writer.lock_path.exists()


def test_a_second_writer_of_the_same_slot_is_refused(tmp_path: Path) -> None:
    first = _writer(tmp_path)
    first.start()
    try:
        with pytest.raises(SlotInUse, match="--prometheus-slot"):
            _writer(tmp_path).start()
    finally:
        first.close()


def test_writers_of_different_slots_coexist(tmp_path: Path) -> None:
    alpha, beta = _writer(tmp_path, "alpha"), _writer(tmp_path, "beta")
    alpha.start()
    beta.start()
    try:
        assert _wait_for(lambda: alpha.path.exists() and beta.path.exists())
        assert alpha.path != beta.path and alpha.lock_path != beta.lock_path
    finally:
        alpha.close()
        beta.close()


def test_a_slot_is_reused_after_its_writer_exits(tmp_path: Path) -> None:
    first = _writer(tmp_path)
    first.start()
    first.close()
    second = TextfileWriter(
        tmp_path, "alpha", RenderCache(lambda: b"# TYPE stormlog_x gauge\n")
    )
    second.start()
    second.close()
    assert "stormlog_x" in second.path.read_text()


def test_a_lock_whose_process_is_gone_is_taken_over(tmp_path: Path) -> None:
    _, lock = slot_paths(tmp_path, "alpha")
    lock.write_text(
        json.dumps({"pid": 2**22 + 12345, "started": 1.0, "host": socket.gethostname()})
    )
    writer = _writer(tmp_path)
    writer.start()
    writer.close()
    assert writer.stats.writes_ok >= 1


def test_a_reused_pid_is_recognised_by_its_start_time(tmp_path: Path) -> None:
    _, lock = slot_paths(tmp_path, "alpha")
    lock.write_text(
        json.dumps({"pid": os.getpid(), "started": 1.0, "host": socket.gethostname()})
    )
    writer = _writer(tmp_path)
    writer.start()  # same pid, other start time: not the writer that locked
    writer.close()


def test_a_live_lock_from_this_process_is_respected(tmp_path: Path) -> None:
    _, lock = slot_paths(tmp_path, "alpha")
    lock.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "started": psutil.Process().create_time(),
                "host": socket.gethostname(),
            }
        )
    )
    with pytest.raises(SlotInUse):
        _writer(tmp_path).start()


def test_another_hosts_lock_is_never_taken_over(tmp_path: Path) -> None:
    _, lock = slot_paths(tmp_path, "alpha")
    lock.write_text(json.dumps({"pid": 1, "started": 1.0, "host": "elsewhere"}))
    with pytest.raises(SlotInUse):
        _writer(tmp_path).start()


def test_an_unreadable_lock_is_stale_only_once_old(tmp_path: Path) -> None:
    _, lock = slot_paths(tmp_path, "alpha")
    lock.write_text("")
    with pytest.raises(SlotInUse):
        _writer(tmp_path).start()
    old = time.time() - 60
    os.utime(lock, (old, old))
    writer = _writer(tmp_path)
    writer.start()
    writer.close()


def test_a_directory_holding_the_artifact_is_refused(tmp_path: Path) -> None:
    artifact = tmp_path / "infer.jsonl"
    artifact.write_text("")
    with pytest.raises(ValueError, match="directory of its own"):
        _writer(tmp_path, forbidden=(artifact,)).start()
    with pytest.raises(ValueError, match="does not exist"):
        _writer(tmp_path / "missing").start()


def test_a_failed_replace_keeps_the_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _writer(tmp_path)
    writer.start()
    assert _wait_for(lambda: writer.stats.writes_ok >= 1)

    def broken_replace(source: object, target: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(textfile.os, "replace", broken_replace)
    assert _wait_for(lambda: writer.stats.writes_failed >= 1)
    assert "stormlog_up" in writer.path.read_text()
    assert not list(tmp_path.glob(".stormlog-alpha.prom.*.tmp"))
    monkeypatch.undo()
    writer.close()


def test_remove_on_exit_leaves_no_file(tmp_path: Path) -> None:
    writer = _writer(tmp_path, remove_on_exit=True)
    writer.start()
    assert _wait_for(lambda: writer.path.exists())
    writer.close()
    assert not writer.path.exists() and not writer.lock_path.exists()


def test_a_writer_stuck_in_io_is_abandoned_with_its_lock_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()
    real_open = open

    def stuck_open(path: object, mode: str = "r", *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if str(path).endswith(".tmp"):
            release.wait(10)
        return real_open(path, mode, *args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(textfile, "open", stuck_open, raising=False)
    writer = _writer(tmp_path)
    writer.start()
    started = time.monotonic()
    writer.close(deadline=0.2)
    assert time.monotonic() - started < 2
    assert writer.stats.abandoned and writer.lock_path.exists()
    release.set()
    assert _wait_for(lambda: writer.path.exists())
    with pytest.raises(SlotInUse):  # the slot stays held while this process lives
        _writer(tmp_path).start()


@pytest.mark.parametrize("slot", ["", "a b", "x" * 65, "../up", "a/b"])
def test_slots_are_validated(slot: str) -> None:
    with pytest.raises(ValueError):
        validate_slot(slot)


def test_the_lock_can_be_taken_before_the_run_starts(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.acquire()
    assert writer.lock_path.exists() and not writer.path.exists()
    with pytest.raises(SlotInUse):
        _writer(tmp_path).acquire()
    writer.start()
    writer.close()
    assert not writer.lock_path.exists()


def test_closing_a_writer_that_never_started_releases_its_lock(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.acquire()
    writer.close()
    assert not writer.lock_path.exists() and not writer.path.exists()


def test_a_failed_render_is_counted_and_the_writer_carries_on(tmp_path: Path) -> None:
    calls = {"n": 0}

    def flaky() -> bytes:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("one bad render")
        return BODY

    clock = {"now": 0.0}
    cache = RenderCache(flaky, min_interval=0.0, clock=lambda: clock["now"])
    writer = TextfileWriter(tmp_path, "alpha", cache, interval=0.02)
    errors: list[BaseException | None] = []
    previous_hook = threading.excepthook
    threading.excepthook = lambda args: errors.append(args.exc_value)
    try:
        writer.start()
        assert _wait_for(lambda: writer.stats.writes_failed >= 1)
        assert _wait_for(lambda: writer.stats.writes_ok >= 2)
        writer.close()
    finally:
        threading.excepthook = previous_hook
    assert errors == []
    assert writer.stats.last_error == "RuntimeError: one bad render"
    exposition = check_exposition(writer.path.read_text())
    assert exposition.value("stormlog_run_active") == 0
