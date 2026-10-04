"""The textfile writer: one file and one lock per slot, written atomically."""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import psutil
import pytest

from stormlog._export import textfile
from stormlog._export.registry import FamilySpec, Registry, render
from stormlog._export.renders import RenderCache
from stormlog._export.textfile import (
    PRODUCER_LABEL,
    SlotInUse,
    TextfileWriter,
    slot_paths,
    validate_slot,
)
from tests.export_conformance import check_exposition

# No producer label: the writer cannot add one to a render it is given.
BODY = b"# HELP stormlog_up Up.\n# TYPE stormlog_up gauge\nstormlog_up 1\n"


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
        const_labels={PRODUCER_LABEL: slot},
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
    registry = Registry(const_labels={PRODUCER_LABEL: "alpha"})
    registry.add(FamilySpec("stormlog_up", "gauge", "Up.")).set((), 1)
    writer = TextfileWriter(
        tmp_path,
        "alpha",
        RenderCache(lambda: render(registry.snapshot())),
        const_labels=registry.const_labels,
        interval=0.05,
    )
    writer.start()
    try:
        assert _wait_for(lambda: writer.path.exists())
        assert writer.path == tmp_path / "stormlog-alpha.prom"
        assert writer.lock_path == tmp_path / "stormlog-alpha.lock"
        text = writer.path.read_text()
        samples = [line for line in text.splitlines() if not line.startswith("#")]
        assert samples and all('stormlog_producer="alpha"' in s for s in samples)
        exposition = check_exposition(text)
        assert exposition.value("stormlog_run_active", stormlog_producer="alpha") == 1
        updated = exposition.value(
            "stormlog_textfile_updated_timestamp_seconds", stormlog_producer="alpha"
        )
        assert abs(updated - time.time()) < 60
    finally:
        writer.close()


@pytest.mark.parametrize("labels", [{}, {PRODUCER_LABEL: "beta"}])
def test_a_render_not_labelled_with_the_slot_is_refused(
    tmp_path: Path, labels: dict[str, str]
) -> None:
    # Two slots in one directory would otherwise write the same series twice,
    # which node_exporter reports as duplicates.
    with pytest.raises(ValueError, match=PRODUCER_LABEL):
        TextfileWriter(
            tmp_path, "alpha", RenderCache(lambda: BODY), const_labels=labels
        )


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
        tmp_path,
        "alpha",
        RenderCache(lambda: b"# TYPE stormlog_x gauge\n"),
        const_labels={PRODUCER_LABEL: "alpha"},
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


@pytest.mark.usefixtures("no_flock")
def test_without_flock_a_reused_pid_is_recognised_by_its_start_time(
    tmp_path: Path,
) -> None:
    _, lock = slot_paths(tmp_path, "alpha")
    lock.write_text(
        json.dumps({"pid": os.getpid(), "started": 1.0, "host": socket.gethostname()})
    )
    writer = _writer(tmp_path)
    writer.start()  # same pid, other start time: not the writer that locked
    writer.close()


@pytest.fixture
def no_flock(monkeypatch: pytest.MonkeyPatch) -> None:
    """A system without flock: the lock is the file's existence."""
    monkeypatch.setattr(textfile, "_flock", None)


@pytest.mark.usefixtures("no_flock")
def test_without_flock_a_live_lock_from_this_process_is_respected(
    tmp_path: Path,
) -> None:
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


@pytest.mark.usefixtures("no_flock")
def test_without_flock_an_unreadable_lock_is_stale_only_once_old(
    tmp_path: Path,
) -> None:
    _, lock = slot_paths(tmp_path, "alpha")
    lock.write_text("")
    with pytest.raises(SlotInUse):
        _writer(tmp_path).start()
    old = time.time() - 60
    os.utime(lock, (old, old))
    writer = _writer(tmp_path)
    writer.start()
    writer.close()


def test_two_writers_taking_over_a_stale_lock_never_both_hold_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fable's order: B judges the dead holder's lock stale and pauses; A
    # takes the slot over; then B carries on with its takeover.
    _, lock = slot_paths(tmp_path, "alpha")
    lock.write_text(
        json.dumps({"pid": 2**22 + 12345, "started": 1.0, "host": socket.gethostname()})
    )
    b_judged, a_done = threading.Event(), threading.Event()
    real_stale: Callable[..., bool] | None = getattr(textfile, "_stale", None)

    def pausing_stale(*args: object) -> bool:
        assert real_stale is not None
        stale = real_stale(*args)
        if threading.current_thread().name == "B":
            b_judged.set()
            a_done.wait(5)
        return stale

    if real_stale is not None:
        monkeypatch.setattr(textfile, "_stale", pausing_stale)
    outcomes: dict[str, str] = {}

    def take(name: str) -> None:
        try:
            _writer(tmp_path).acquire()
            outcomes[name] = "holds"
        except SlotInUse:
            outcomes[name] = "refused"

    b = threading.Thread(target=take, args=("B",), name="B")
    b.start()
    while not b_judged.is_set() and b.is_alive():
        time.sleep(0.01)
    take("A")
    a_done.set()
    b.join(5)
    assert sorted(outcomes.values()) == ["holds", "refused"], outcomes


def test_a_lock_file_removed_while_being_taken_is_not_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # B opens the lock file A holds; before B locks it, A frees the slot
    # (removing the file) and C takes a new one. B's lock is then on a file
    # no longer in the directory, so B must not count it as the slot.
    a = _writer(tmp_path)
    a.acquire()
    c = _writer(tmp_path)
    real_flock = textfile._flock
    assert real_flock is not None
    interleaved: list[str] = []

    def flock_after_a_and_c(descriptor: int, operation: int) -> None:
        if threading.current_thread().name == "B" and not interleaved:
            interleaved.append("A frees, C takes")
            a.close()
            c.acquire()
        real_flock(descriptor, operation)

    monkeypatch.setattr(textfile, "_flock", flock_after_a_and_c)
    outcome: list[str] = []

    def take() -> None:
        try:
            _writer(tmp_path).acquire()
            outcome.append("holds")
        except SlotInUse:
            outcome.append("refused")

    b = threading.Thread(target=take, name="B")
    b.start()
    b.join(10)
    assert interleaved and outcome == ["refused"]
    c.close()


def test_a_lock_file_this_user_cannot_open_is_a_slot_in_use(tmp_path: Path) -> None:
    # Another user's lock in a shared node_exporter directory, or a
    # read-only one: refused as a slot in use, not as an I/O error.
    _, lock = slot_paths(tmp_path, "alpha")
    lock.write_text(json.dumps({"pid": 1, "host": "elsewhere"}))
    lock.chmod(0o444)
    try:
        with pytest.raises(SlotInUse, match="elsewhere"):
            _writer(tmp_path).acquire()
    finally:
        lock.chmod(0o644)


_HOLD_THE_SLOT = """
import sys, time
from pathlib import Path
from stormlog._export.renders import RenderCache
from stormlog._export.textfile import TextfileWriter

TextfileWriter(
    Path(sys.argv[1]),
    "alpha",
    RenderCache(lambda: b""),
    const_labels={"stormlog_producer": "alpha"},
).acquire()
print("held", flush=True)
time.sleep(60)
"""


def test_a_slot_held_by_a_live_process_is_taken_once_it_is_killed(
    tmp_path: Path,
) -> None:
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD_THE_SLOT, str(tmp_path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline() == "held\n"
        with pytest.raises(SlotInUse, match=f"pid {holder.pid} "):
            _writer(tmp_path).acquire()
    finally:
        holder.send_signal(signal.SIGKILL)
        holder.wait(10)
    writer = _writer(tmp_path)
    writer.acquire()  # no lock outlives its process on this host
    writer.close()


def test_a_lock_file_nobody_holds_is_taken_at_once(tmp_path: Path) -> None:
    # Left by a writer that was killed: whatever it says, nobody holds it.
    _, lock = slot_paths(tmp_path, "alpha")
    lock.write_text("")
    writer = _writer(tmp_path)
    writer.acquire()
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


def test_a_second_close_never_touches_the_next_writers_file(tmp_path: Path) -> None:
    first = _writer(tmp_path, remove_on_exit=True)
    first.start()
    first.close()
    second = _writer(tmp_path)
    second.start()
    try:
        assert _wait_for(lambda: second.path.exists())
        first.close()  # again: the slot is no longer the first writer's
        assert second.path.exists() and second.lock_path.exists()
    finally:
        second.close()


def test_close_returns_at_its_deadline_even_if_freeing_the_slot_is_stuck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _writer(tmp_path)
    writer.start()
    assert _wait_for(lambda: writer.stats.writes_ok >= 1)
    entered, proceed = threading.Event(), threading.Event()
    real_unlink = textfile._unlink

    def stuck_unlink(path: Path) -> None:
        entered.set()
        proceed.wait(10)
        real_unlink(path)

    monkeypatch.setattr(textfile, "_unlink", stuck_unlink)
    closer = threading.Thread(target=writer.close, kwargs={"deadline": 0.05})
    started = time.monotonic()
    closer.start()
    try:
        assert entered.wait(5)
        closer.join(1)
        assert not closer.is_alive() and time.monotonic() - started < 1
        started = time.monotonic()
        writer.close(deadline=5)  # a second close does not wait again
        assert time.monotonic() - started < 1
    finally:
        proceed.set()
        closer.join(5)
    assert _wait_for(lambda: not writer.lock_path.exists())


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _final_write(tmp_path: Path, *, readers_hold_renders: bool) -> TextfileWriter:
    """A run whose values change for good just before the writer closes."""
    values = {"up": 1}
    clock = _Clock()
    cache = RenderCache(
        lambda: f"stormlog_up {values['up']}\n".encode(),
        min_interval=60.0,
        clock=clock,
    )
    writer = TextfileWriter(
        tmp_path,
        "alpha",
        cache,
        const_labels={PRODUCER_LABEL: "alpha"},
        interval=3600,
    )
    writer.start()
    assert _wait_for(lambda: writer.stats.writes_ok == 1)
    held = []
    if readers_hold_renders:  # slow scrapes: the publication limit holds
        for tick in (60.0, 120.0, 180.0):
            clock.now = tick
            held.append(cache.acquire())
    values["up"] = 0
    cache.invalidate()
    writer.close()
    for generation in held:
        cache.release(generation)
    return writer


def test_a_final_write_from_a_render_out_of_date_says_so(tmp_path: Path) -> None:
    writer = _final_write(tmp_path, readers_hold_renders=True)
    assert writer.stats.final_stale
    assert "stormlog_up 1\n" in writer.path.read_text()


def test_a_final_write_from_a_fresh_render_is_not_flagged(tmp_path: Path) -> None:
    writer = _final_write(tmp_path, readers_hold_renders=False)
    assert not writer.stats.final_stale
    assert "stormlog_up 0\n" in writer.path.read_text()


def test_closing_leaves_a_lock_file_that_another_writer_put_there(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)
    writer.acquire()
    writer.lock_path.unlink()
    writer.lock_path.write_text(json.dumps({"pid": 1, "host": "other"}))
    writer.close()
    assert writer.lock_path.exists()


@pytest.mark.usefixtures("no_flock")
def test_without_flock_closing_leaves_another_writers_lock(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.acquire()
    writer.lock_path.write_text(json.dumps({"pid": 1, "host": "other"}))
    writer.close()
    assert writer.lock_path.exists()


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
    writer = TextfileWriter(
        tmp_path,
        "alpha",
        cache,
        const_labels={PRODUCER_LABEL: "alpha"},
        interval=0.02,
    )
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
