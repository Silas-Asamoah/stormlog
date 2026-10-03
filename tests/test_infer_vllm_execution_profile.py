"""The vLLM execution log at the end of ``stormlog infer profile``."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest

import stormlog.infer.profile as profile_module
from stormlog.infer.config import ProfileConfig
from stormlog.infer.profile import InferenceProfiler
from stormlog.infer.vllm_execution_import import flush_execution_log
from tests.test_infer_profile import _fake_server, _records
from tests.vllm_execution_helpers import (
    BOOT,
    FORMAT,
    HOST,
    SECOND,
    WALL_OFFSET,
    alias,
    completed,
    done,
    engine_log,
    epoch_name,
    goodbye,
    heartbeat,
    hello,
    member,
    scheduled,
    write_epoch,
)

PID, START = 2600, 1_790_000_000_000_000_000
EPOCH = epoch_name("engine", PID, START)
T0 = 1_000 * SECOND
NOW = T0 + WALL_OFFSET + 5 * SECOND


def _engine_dir(root: Path) -> Path:
    return root / f"{HOST}-{BOOT}" / EPOCH


def _append(directory: Path, seq: int, record: dict[str, Any]) -> None:
    line = json.dumps({"format": FORMAT, "epoch": directory.name, "seq": seq, **record})
    with (directory / "000000.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def test_flush_asks_live_epochs_and_waits_for_a_heartbeat(tmp_path: Path) -> None:
    engine_log(tmp_path, [heartbeat(T0 + SECOND, 1)])
    worker = write_epoch(
        tmp_path,
        "worker",
        2601,
        5,
        [hello("worker", 2601, 5, engine_pid=PID), goodbye(T0 + SECOND, 1)],
    )
    engine = _engine_dir(tmp_path)
    polls: list[float] = []

    def writer_heartbeats(seconds: float) -> None:
        polls.append(seconds)
        _append(engine, 2, heartbeat(T0 + 2 * SECOND, 2))

    result = flush_execution_log(
        tmp_path, timeout_seconds=10, now_ns=NOW, sleep=writer_heartbeats
    )

    assert (engine / "flush").is_file()
    assert not (worker / "flush").exists()  # ended: nothing to seal
    assert result == {
        "requested": [EPOCH],
        "flushed": [EPOCH],
        "timed_out": [],
        "errors": [],
    }
    assert polls == [0.25]


def test_flush_takes_a_removed_flush_file_as_sealed(tmp_path: Path) -> None:
    engine_log(tmp_path, [heartbeat(T0 + SECOND, 1)])
    engine = _engine_dir(tmp_path)

    def writer_seals(_seconds: float) -> None:
        (engine / "flush").unlink()

    result = flush_execution_log(tmp_path, now_ns=NOW, sleep=writer_seals)
    assert result["flushed"] == [EPOCH] and result["timed_out"] == []


def test_flush_times_out_without_a_writer(tmp_path: Path) -> None:
    engine_log(tmp_path, [heartbeat(T0 + SECOND, 1)])
    ticks = itertools.count(0.0, 1.0)
    polls: list[float] = []

    result = flush_execution_log(
        tmp_path,
        timeout_seconds=3.0,
        now_ns=NOW,
        sleep=polls.append,
        clock=lambda: next(ticks),
    )

    assert result["timed_out"] == [EPOCH] and result["flushed"] == []
    # The fake clock ticks a second per reading: the deadline passes after
    # two polls, and the flush request stays for the writer to honour.
    assert polls == [0.25, 0.25]
    assert (_engine_dir(tmp_path) / "flush").is_file()


def _config(endpoint: str, output: Path, hook: Path) -> ProfileConfig:
    return ProfileConfig(
        endpoint=endpoint,
        model="fake-model",
        concurrency=(1,),
        input_tokens=(8,),
        output_tokens=(4,),
        request_count=1,
        output_path=str(output),
        stream=False,
        system_sampler="none",
        tokenizer="none",
        run_id="run-1",
        vllm_execution_dir=hook,
    )


def _hook_log_for(config: ProfileConfig, hook: Path) -> str:
    """An ended engine epoch that ran the config's one measured request."""
    case_id = config.cases()[0].case_id
    request_id = f"{case_id}_measured_0_0"
    external = f"chatcmpl-stormlog-run-1-{request_id}"
    internal = f"{external}-0f3a9c1d"
    engine_log(
        hook,
        [
            alias(internal, external, T0 - 10),
            scheduled(0, T0, [member(internal, scheduled=8)]),
            completed(0, T0 + SECOND, [done(internal)]),
            goodbye(T0 + 2 * SECOND, 3),
        ],
    )
    return request_id


def test_profile_imports_the_execution_log_before_its_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(profile_module, "EXECUTION_FLUSH_TIMEOUT_SECONDS", 0.05)
    output, hook = tmp_path / "infer.jsonl", tmp_path / "hook"
    warnings: list[str] = []
    with _fake_server() as endpoint:
        config = _config(endpoint, output, hook)
        request_id = _hook_log_for(config, hook)
        report = InferenceProfiler(config, on_warning=warnings.append).run()

    # The report, written after the import, already covers it.
    execution = report["telemetry"]["execution"]
    assert execution["available"] is True
    assert execution["iterations"]["total"] == 1
    assert execution["requests"]["run_requests_bound"] == 1
    records = _records(output)
    types = [record["event_type"] for record in records]
    # The imported records precede the report, and the session ends the file.
    assert types.index("infer.iteration") < types.index("infer.summary")
    assert types[-1] == "infer.session"
    membership = next(r for r in records if r["event_type"] == "infer.membership")
    assert membership["request_ref"] == {"producer_id": "stormlog", "id": request_id}
    engine = next(
        r
        for r in records
        if r["event_type"] == "infer.capabilities"
        and r["component"] == "engine_adapter"
    )
    assert engine["collected"] == [
        "iterations",
        "memberships",
        "requests",
        "clock_alignment",
    ]
    assert engine["metadata"]["summary"]["execution"]["high_water"] == {EPOCH: 4}
    session = next(r for r in records if r["event_type"] == "infer.session")
    assert session["config"]["vllm_execution_dir"] == str(hook)
    assert [w for w in warnings if "execution log" in w] == []


def test_a_missing_execution_log_warns_and_records_a_partial_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(profile_module, "EXECUTION_FLUSH_TIMEOUT_SECONDS", 0.05)
    output, hook = tmp_path / "infer.jsonl", tmp_path / "nope"
    warnings: list[str] = []
    with _fake_server() as endpoint:
        InferenceProfiler(
            _config(endpoint, output, hook), on_warning=warnings.append
        ).run()

    assert any("could not be read" in w for w in warnings)
    assert any("was not imported" in w and "import-execution" in w for w in warnings)
    records = _records(output)
    assert records[-1]["event_type"] == "infer.session"
    assert records[-1]["status"] == "completed"
    engine = next(
        r
        for r in records
        if r["event_type"] == "infer.capabilities"
        and r["component"] == "engine_adapter"
    )
    assert engine["available"] is True and engine["collected"] == []
    assert engine["supported"] == [
        "iterations",
        "memberships",
        "requests",
        "clock_alignment",
    ]
    assert "nope" in engine["metadata"]["summary"]["execution"]["failed"]
