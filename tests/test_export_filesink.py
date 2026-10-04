"""The line file sink: whole lines, a cap, and a defined outcome for every error."""

import errno
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from stormlog._export import filesink
from stormlog._export.filesink import (
    FILE_DISABLED,
    FILE_ERROR,
    FILE_FULL,
    FILE_PARTIAL,
    WRITTEN,
    LineFileSink,
)


@pytest.fixture
def sink(tmp_path: Path) -> Iterator[LineFileSink]:
    line_sink = LineFileSink(tmp_path / "spans.jsonl", max_bytes=1000)
    line_sink.open()
    yield line_sink
    line_sink.close()


def _lines(path: Path) -> list[bytes]:
    return path.read_bytes().splitlines(keepends=True)


def test_lines_are_appended_whole(sink: LineFileSink) -> None:
    assert sink.write_line(b'{"a": 1}') == WRITTEN
    assert sink.write_line(b'{"b": 2}\n') == WRITTEN
    assert _lines(sink.path) == [b'{"a": 1}\n', b'{"b": 2}\n']
    assert (sink.stats.lines, sink.stats.bytes) == (2, 18)


def test_a_line_past_the_cap_is_refused_before_writing(tmp_path: Path) -> None:
    sink = LineFileSink(tmp_path / "f", max_bytes=10)
    sink.open()
    assert sink.write_line(b"12345") == WRITTEN
    assert sink.write_line(b"123456") == FILE_FULL
    assert _lines(sink.path) == [b"12345\n"] and sink.stats.full == 1
    sink.close()


def test_reopening_counts_what_the_file_already_holds(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_bytes(b"x" * 8)
    sink = LineFileSink(path, max_bytes=10)
    sink.open()
    assert sink.write_line(b"yy") == FILE_FULL
    sink.close()


# A run killed in the middle of a line: the second write puts half its line
# in the file, then the process exits at once, as SIGKILL or the OOM killer
# would end it.
_KILLED_MID_LINE = """
import os, sys
from pathlib import Path
from stormlog._export import filesink

real_write = filesink._write
writes = []

def write_half_then_exit(fd, view):
    writes.append(len(view))
    if len(writes) == 2:
        real_write(fd, bytes(view[: len(view) // 2]))
        os._exit(9)
    return real_write(fd, view)

filesink._write = write_half_then_exit
sink = filesink.LineFileSink(Path(sys.argv[1]), max_bytes=1 << 20)
sink.open()
sink.write_line(b'{"batch": 1}')
sink.write_line(b'{"batch": 2, "pad": "' + b"x" * 64 + b'"}')
"""


def test_a_line_written_after_a_killed_run_stays_readable(tmp_path: Path) -> None:
    path = tmp_path / "spans.jsonl"
    killed = subprocess.run(
        [sys.executable, "-c", _KILLED_MID_LINE, str(path)], timeout=60
    )
    assert killed.returncode == 9
    sink = LineFileSink(path, max_bytes=1 << 20)
    sink.open()
    assert sink.write_line(b'{"batch": 3}') == WRITTEN
    sink.close()
    readable = []
    for line in path.read_bytes().splitlines():
        try:
            readable.append(json.loads(line)["batch"])
        except ValueError:
            pass  # the killed run's partial line
    assert readable == [1, 3]
    assert sink.stats.ended_partial


def test_a_file_ending_in_a_newline_is_appended_to_as_it_is(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_bytes(b"old\n")
    sink = LineFileSink(path, max_bytes=100)
    sink.open()
    assert sink.write_line(b"new") == WRITTEN
    sink.close()
    assert path.read_bytes() == b"old\nnew\n"
    assert not sink.stats.ended_partial


def test_a_failed_write_is_cut_back_so_the_line_is_absent(
    sink: LineFileSink, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink.write_line(b"first")
    real_write = filesink._write
    calls = {"n": 0}

    def half_then_enospc(fd: int, data: memoryview) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            return real_write(fd, bytes(data[:3]))  # a partial write
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(filesink, "_write", half_then_enospc)
    assert sink.write_line(b"second-line") == FILE_ERROR
    monkeypatch.undo()
    assert _lines(sink.path) == [b"first\n"]
    assert sink.write_line(b"third") == WRITTEN
    assert _lines(sink.path) == [b"first\n", b"third\n"]


def test_a_failed_sync_is_cut_back_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sink = LineFileSink(tmp_path / "f", max_bytes=1000, fsync=True)
    sink.open()

    def eio(fd: int) -> None:
        raise OSError(errno.EIO, "I/O error")

    monkeypatch.setattr(filesink, "_sync", eio)
    assert sink.write_line(b"line") == FILE_ERROR
    assert sink.path.read_bytes() == b""
    sink.close()


def test_when_the_cut_back_fails_the_line_may_be_partial_and_the_sink_stops(
    sink: LineFileSink, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_write(fd: int, data: memoryview) -> int:
        raise OSError(errno.EIO, "I/O error")

    def failing_truncate(fd: int, length: int) -> None:
        raise OSError(errno.EIO, "I/O error")

    monkeypatch.setattr(filesink, "_write", failing_write)
    monkeypatch.setattr(filesink, "_truncate", failing_truncate)
    assert sink.write_line(b"line") == FILE_PARTIAL
    monkeypatch.undo()
    assert sink.stats.disabled
    assert sink.write_line(b"later") == FILE_DISABLED


def test_repeated_errors_disable_the_sink(
    sink: LineFileSink, monkeypatch: pytest.MonkeyPatch
) -> None:
    def enospc(fd: int, data: memoryview) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(filesink, "_write", enospc)
    assert [sink.write_line(b"x") for _ in range(3)] == [FILE_ERROR] * 3
    monkeypatch.undo()
    assert sink.stats.disabled and sink.stats.errors == 3
    assert sink.write_line(b"x") == FILE_DISABLED
    assert "No space left" in (sink.stats.last_error or "")


def test_a_success_resets_the_error_count(
    sink: LineFileSink, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write = filesink._write
    failures = iter([True, True, False, True, True, False])

    def sometimes(fd: int, data: memoryview) -> int:
        if next(failures):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_write(fd, data)

    monkeypatch.setattr(filesink, "_write", sometimes)
    outcomes = [sink.write_line(b"x") for _ in range(6)]
    assert outcomes == [
        FILE_ERROR,
        FILE_ERROR,
        WRITTEN,
        FILE_ERROR,
        FILE_ERROR,
        WRITTEN,
    ]
    assert not sink.stats.disabled


def test_an_unopened_or_closed_sink_writes_nothing(tmp_path: Path) -> None:
    sink = LineFileSink(tmp_path / "f", max_bytes=10)
    assert sink.write_line(b"x") == FILE_DISABLED
