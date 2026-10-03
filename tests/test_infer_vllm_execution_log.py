"""Reading the vLLM execution hook's raw log: segments, sequences, epochs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stormlog.infer.vllm_execution_log import (
    FORMAT,
    STATE_ALIVE,
    STATE_ENDED,
    STATE_GONE,
    read_execution_log,
)
from tests.vllm_execution_helpers import (
    BOOT,
    HOST,
    KEY,
    SECOND,
    WALL_OFFSET,
    engine_log,
    epoch_name,
    goodbye,
    heartbeat,
    hello,
    scheduled,
    write_epoch,
)

T0 = 1_000 * SECOND
NOW = T0 + WALL_OFFSET + 5 * SECOND


def _line(epoch: str, seq: int, **fields: object) -> str:
    return json.dumps({"format": FORMAT, "epoch": epoch, "seq": seq, **fields}) + "\n"


def test_reads_sealed_segments_and_complete_lines_of_the_open_one(
    tmp_path: Path,
) -> None:
    records = [scheduled(i, T0 + i * SECOND, []) for i in range(4)]
    # The open segment ends in the middle of a record the writer is still on.
    tail = _line(epoch_name("engine", 2600, 1), 99, kind="heartbeat")[:-10]
    directory = engine_log(tmp_path, records, sealed=2, open_tail=tail.encode())
    read = read_execution_log(tmp_path, now_ns=NOW)
    (epoch,) = read.epochs
    assert epoch.directory == directory
    assert (epoch.host, epoch.boot_id, epoch.role, epoch.pid) == (
        HOST,
        BOOT,
        "engine",
        2600,
    )
    assert [r.seq for r in epoch.records] == [0, 1, 2, 3, 4]
    assert [r.kind for r in epoch.records] == ["hello"] + ["scheduled"] * 4
    assert epoch.truncated is True
    assert epoch.last_seq == 4 and epoch.gaps == 0
    assert epoch.hello is not None and epoch.hello["pid"] == 2600
    assert epoch.key == KEY
    assert epoch.errors == []


def test_records_are_told_apart_by_sequence_not_by_file(tmp_path: Path) -> None:
    # hello is seq 0, the two scheduled records are seqs 1 and 2.
    directory = engine_log(tmp_path, [scheduled(0, T0, []), scheduled(1, T0, [])])
    name = directory.name
    # A second delivery of seq 1 with other content, in a later segment: the
    # first delivery is the one kept, whatever file either came from.
    with (directory / "000001.jsonl.part").open("a", encoding="utf-8") as handle:
        handle.write(_line(name, 1, kind="scheduled", iteration="other"))
        handle.write(_line(name, 3, kind="heartbeat", wall_ns=NOW - SECOND, last_seq=3))
    read = read_execution_log(tmp_path, now_ns=NOW)
    (epoch,) = read.epochs
    assert [r.seq for r in epoch.records] == [0, 1, 2, 3]
    assert [r.data["iteration"] for r in epoch.of_kind("scheduled")] == ["0", "1"]
    # A re-import with the previous high-water mark gets only the new records,
    # while the epoch's sequence facts still cover everything read.
    again = read_execution_log(tmp_path, high_water={name: 1}, now_ns=NOW)
    (epoch,) = again.epochs
    assert [r.seq for r in epoch.records] == [2, 3]
    assert epoch.high_water_before == 1 and epoch.last_seq == 3
    assert epoch.summary()["high_water_seq"] == 3


def test_gaps_and_the_status_file_extend_the_sequence_facts(tmp_path: Path) -> None:
    status = {
        "wall_ns": NOW - SECOND,
        "last_seq": 9,
        "dropped": {"scheduled": 2, "completed": 0, "alias": 0, "terminal": 0},
        "capped": True,
        "pending_iterations": 1,
        "range_misses": 0,
    }
    directory = engine_log(tmp_path, [scheduled(0, T0, [])], status=status)
    # Sequences 5 and 6 arrive after a hole where 2, 3 and 4 were dropped.
    (directory / "000001.jsonl").write_text(
        _line(directory.name, 5, kind="scheduled", iteration="5")
        + _line(directory.name, 6, kind="scheduled", iteration="6"),
        encoding="utf-8",
    )
    (epoch,) = read_execution_log(tmp_path, now_ns=NOW).epochs
    assert [r.seq for r in epoch.records] == [0, 1, 5, 6]
    assert epoch.gaps == 3
    assert epoch.last_seq == 9
    summary = epoch.summary()
    assert summary["dropped"] == {
        "scheduled": 2,
        "completed": 0,
        "alias": 0,
        "terminal": 0,
    }
    assert summary["capped"] is True and summary["pending_iterations"] == 1
    assert summary["state"] == STATE_ALIVE


MONO_NOW = NOW - WALL_OFFSET  # the engine's monotonic reading at NOW


@pytest.mark.parametrize(
    ("hello_mono_ns", "records", "state"),
    [
        (T0, [heartbeat(T0, 1), goodbye(T0 + SECOND, 2)], STATE_ENDED),
        (MONO_NOW - 20 * SECOND, [heartbeat(MONO_NOW - 10 * SECOND, 1)], STATE_ALIVE),
        (MONO_NOW - 50 * SECOND, [heartbeat(MONO_NOW - 40 * SECOND, 1)], STATE_GONE),
        (MONO_NOW - 5 * SECOND, [], STATE_ALIVE),  # the hello's own clock counts
        (MONO_NOW - 31 * SECOND, [], STATE_GONE),
    ],
)
def test_an_epoch_has_ended_gone_quiet_or_is_alive(
    tmp_path: Path, hello_mono_ns: int, records: list[dict[str, object]], state: str
) -> None:
    engine_log(tmp_path, records, hello_mono_ns=hello_mono_ns)
    (epoch,) = read_execution_log(tmp_path, now_ns=NOW).epochs
    assert epoch.state == state
    assert (epoch.goodbye is not None) == (state == STATE_ENDED)


def test_bad_lines_and_stray_entries_are_reported_not_fatal(tmp_path: Path) -> None:
    directory = engine_log(tmp_path, [scheduled(0, T0, [])], key=KEY[:8])
    name = directory.name
    (directory / "000001.jsonl").write_text(
        "not json\n"
        + json.dumps({"format": "other/9", "epoch": name, "seq": 7, "kind": "x"})
        + "\n"
        + _line("engine-1-1", 8, kind="scheduled")
        + json.dumps([1, 2])
        + "\n"
        + _line(name, 9, kind="scheduled", iteration="9")[:-1],  # sealed, mid-line
        encoding="utf-8",
    )
    (tmp_path / f"{HOST}-{BOOT}" / "notes").mkdir()
    (tmp_path / f"{HOST}-{BOOT}" / "README").write_text("x", encoding="utf-8")
    read = read_execution_log(tmp_path, now_ns=NOW)
    (epoch,) = read.epochs
    assert [r.seq for r in epoch.records] == [0, 1]
    assert epoch.key is None
    messages = " | ".join(epoch.errors)
    assert "000001.jsonl:1:" in messages
    assert "format 'other/9'" in messages
    assert "epoch 'engine-1-1'" in messages
    assert "not an object" in messages
    assert "ends mid-line" in messages
    assert "too short" in messages
    assert read.notes == [
        f"{tmp_path / f'{HOST}-{BOOT}' / 'notes'}: not an epoch directory"
    ]


def test_worker_and_engine_epochs_are_told_apart(tmp_path: Path) -> None:
    engine_log(tmp_path, [])
    write_epoch(
        tmp_path,
        "worker",
        2601,
        5,
        [hello("worker", 2601, 5, engine_pid=2600, rank={"tp": 0, "pp": 0, "dp": 0})],
        status={
            "wall_ns": NOW - SECOND,
            "last_seq": 0,
            "range_misses": 1,
            "startup_unranged": 9,
            "pending_samples": 2,
        },
    )
    read = read_execution_log(tmp_path, now_ns=NOW)
    assert [e.role for e in read.epochs] == ["engine", "worker"]
    assert [e.pid for e in read.workers()] == [2601]
    worker_hello = read.workers()[0].hello
    assert worker_hello is not None
    assert worker_hello["producer"].endswith(":2600:5")
    assert (read.workers()[0].host, read.workers()[0].boot_id) == (HOST, BOOT)
    summary = read.workers()[0].summary()
    assert (summary["range_misses"], summary["startup_unranged"]) == (1, 9)
    assert summary["pending_samples"] == 2


def test_a_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        read_execution_log(tmp_path / "nope")
