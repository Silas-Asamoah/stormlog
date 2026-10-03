"""Build synthetic vLLM execution-hook raw logs for tests.

The layout and records follow ``docs/vllm_execution.md`` exactly; the hook
itself is not involved, so these fixtures are the spec as the import sees it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FORMAT = "stormlog.vllm_hook/1"
HOST = "node-7"
BOOT = "boot-aaaa"
SECOND = 1_000_000_000
# One wall/monotonic pair for the engine: wall = mono + WALL_OFFSET.
WALL_OFFSET = 1_790_000_000_000_000_000
KEY = bytes(range(32))


def epoch_name(role: str, pid: int, start_ns: int) -> str:
    return f"{role}-{pid}-{start_ns}"


def producer(pid: int, start_ns: int) -> str:
    return f"vllm:{HOST}:{BOOT}:{pid}:{start_ns}"


def clock(mono_ns: int) -> dict[str, int]:
    return {"wall_ns": mono_ns + WALL_OFFSET, "mono_ns": mono_ns, "gap_ns": 1200}


def hello(
    role: str,
    pid: int,
    start_ns: int,
    *,
    mono_ns: int = 1_000 * SECOND,
    enabled: bool = True,
    refused: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": "hello",
        "role": role,
        "host": HOST,
        "boot_id": BOOT,
        "pid": pid,
        "start_ns": start_ns,
        "vllm_version": "0.30.0",
        "enabled": enabled,
        "refused": refused,
        "producer": producer(
            pid if role == "engine" else extra.pop("engine_pid", pid), start_ns
        ),
        "config": {
            "executor": "mp",
            "scheduler": "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
            "runner": None,
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "async_scheduling": True,
            "speculative": None,
            "max_num_batched_tokens": 2048,
            "request_id_randomization": True,
        },
        "clock": clock(mono_ns),
    }
    record.update(extra)
    return record


def alias(internal: str, external: str, mono_ns: int) -> dict[str, Any]:
    return {
        "kind": "alias",
        "internal": internal,
        "external": external,
        "wall_ns": mono_ns + WALL_OFFSET,
        "mono_ns": mono_ns,
    }


def member(
    internal: str,
    *,
    scheduled: int,
    computed_before: int = 0,
    prompt_tokens: int = 8,
    sighting: str = "first",
    prefill: int | None = None,
    decode: int | None = None,
    drafts: int = 0,
    cached: int = 0,
    recompute: bool = False,
    output_before: int = 0,
) -> dict[str, Any]:
    if prefill is None and decode is None:
        prefill = max(0, min(scheduled, prompt_tokens - computed_before))
        decode = scheduled - prefill
    return {
        "internal": internal,
        "sighting": sighting,
        "scheduled": scheduled,
        "computed_before": computed_before,
        "prompt_tokens": prompt_tokens,
        "prefill_scheduled": prefill or 0,
        "decode_scheduled": decode or 0,
        "drafts_scheduled": drafts,
        "cached_at_admission": cached,
        "recompute": recompute,
        "output_before": output_before,
    }


def scheduled(
    iteration: int,
    start_mono_ns: int,
    members: list[dict[str, Any]],
    *,
    duration_ns: int = 200_000,
    preempted: list[str] | None = None,
) -> dict[str, Any]:
    total = sum(int(item["scheduled"]) for item in members)
    return {
        "kind": "scheduled",
        "iteration": str(iteration),
        "start_wall_ns": start_mono_ns + WALL_OFFSET,
        "start_mono_ns": start_mono_ns,
        "end_wall_ns": start_mono_ns + duration_ns + WALL_OFFSET,
        "end_mono_ns": start_mono_ns + duration_ns,
        "total_tokens": total,
        "zero_token": total == 0,
        "preempted": list(preempted or []),
        "members": members,
    }


def done(
    internal: str,
    *,
    outcome: str = "kept",
    stale: bool = False,
    sampled: int = 1,
    accepted: int = 0,
    retained: int = 1,
    computed_after: int = 0,
) -> dict[str, Any]:
    return {
        "internal": internal,
        "outcome": outcome,
        "stale": stale,
        "sampled": sampled,
        "accepted_drafts": accepted,
        "retained": retained,
        "computed_after": computed_after,
    }


def completed(
    iteration: int, mono_ns: int, members: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "kind": "completed",
        "iteration": str(iteration),
        "wall_ns": mono_ns + WALL_OFFSET,
        "mono_ns": mono_ns,
        "members": members,
    }


def terminal(
    internal: str,
    mono_ns: int,
    *,
    status: str = "FINISHED_STOPPED",
    finish_reason: str = "stop",
    output_tokens: int = 4,
) -> dict[str, Any]:
    return {
        "kind": "terminal",
        "internal": internal,
        "status": status,
        "finish_reason": finish_reason,
        "output_tokens": output_tokens,
        "wall_ns": mono_ns + WALL_OFFSET,
        "mono_ns": mono_ns,
    }


def heartbeat(mono_ns: int, last_seq: int, **counters: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": "heartbeat",
        "wall_ns": mono_ns + WALL_OFFSET,
        "mono_ns": mono_ns,
        "last_seq": last_seq,
        "dropped": {"scheduled": 0, "completed": 0, "alias": 0, "terminal": 0},
        "errors": 0,
        "bytes": 1024,
        "capped": False,
        "pending_iterations": 0,
        "range_misses": 0,
    }
    record.update(counters)
    return record


def goodbye(mono_ns: int, last_seq: int) -> dict[str, Any]:
    return {
        "kind": "goodbye",
        "wall_ns": mono_ns + WALL_OFFSET,
        "mono_ns": mono_ns,
        "last_seq": last_seq,
    }


def write_epoch(
    root: Path,
    role: str,
    pid: int,
    start_ns: int,
    records: list[dict[str, Any]],
    *,
    sealed: int | None = None,
    open_tail: bytes = b"",
    status: dict[str, Any] | None = None,
    key: bytes | None = KEY,
    seq_start: int = 0,
    host_boot: str = f"{HOST}-{BOOT}",
) -> Path:
    """Write ``records`` as one epoch: the first ``sealed`` in a sealed segment,
    the rest in an open one (all sealed when ``sealed`` is None)."""
    name = epoch_name(role, pid, start_ns)
    directory = root / host_boot / name
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"format": FORMAT, "epoch": name, "seq": seq_start + i, **record})
        for i, record in enumerate(records)
    ]
    count = len(lines) if sealed is None else sealed
    if count:
        (directory / "000000.jsonl").write_text(
            "".join(line + "\n" for line in lines[:count]), encoding="utf-8"
        )
    if sealed is not None:
        body = "".join(line + "\n" for line in lines[count:]).encode("utf-8")
        (directory / "000001.jsonl.part").write_bytes(body + open_tail)
    if status is not None:
        (directory / "status.json").write_text(json.dumps(status), encoding="utf-8")
    if key is not None:
        (directory / "key").write_bytes(key)
    return directory


def engine_log(
    root: Path,
    records: list[dict[str, Any]],
    *,
    pid: int = 2600,
    start_ns: int = 1_790_000_000_000_000_000,
    hello_mono_ns: int = 1_000 * SECOND,
    **kwargs: Any,
) -> Path:
    """An engine epoch whose first record is a hello."""
    return write_epoch(
        root,
        "engine",
        pid,
        start_ns,
        [hello("engine", pid, start_ns, mono_ns=hello_mono_ns), *records],
        **kwargs,
    )
