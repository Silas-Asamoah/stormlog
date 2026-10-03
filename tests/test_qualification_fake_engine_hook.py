"""The fake engine's execution-hook raw log, read by Stormlog's own importer."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from examples.qualification.fake_engine import FakeEngine, FakeEngineConfig
from examples.qualification.fake_engine.hook_log import RUNNER, SCHEDULER
from tests.qualification_fake_engine_helpers import (
    chat,
    of_type,
    records,
    run_profile,
    wait_until,
    words,
)


def _engine(hook: Path, **changes: Any) -> FakeEngine:
    base = FakeEngineConfig(step_seconds=0.001, hook_dir=hook, hook_seal_seconds=0.5)
    return FakeEngine(replace(base, **changes))


def _engine_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    (capability,) = [
        item
        for item in of_type(items, "infer.capabilities")
        if item.get("component") == "engine_adapter"
    ]
    metadata = capability["metadata"]
    assert isinstance(metadata, dict)
    epochs = metadata["summary"]["execution"]["epochs"]
    (summary,) = [value for name, value in epochs.items() if name.startswith("engine-")]
    assert isinstance(summary, dict)
    return summary


def test_a_profile_imports_the_log_and_binds_every_request(tmp_path: Path) -> None:
    hook = tmp_path / "hook"
    output = tmp_path / "infer.jsonl"
    with _engine(hook) as engine:
        run_profile(engine, output, vllm_execution_dir=hook)
    items = records(output)
    client = {
        item["request_id"]
        for item in of_type(items, "infer.request")
        if item.get("schema_version", 1) == 1 and item.get("phase") == "measured"
    }
    kept: dict[str, int] = defaultdict(int)
    prefill: dict[str, int] = defaultdict(int)
    for membership in of_type(items, "infer.membership"):
        request = membership["request_ref"]["id"]
        kept[request] += membership["output_tokens"] or 0
        prefill[request] += membership["metadata"]["prefill_scheduled"]
    summary = _engine_summary(items)
    assert summary["executions"] == {"run": 4}
    assert (summary["gaps"], summary["dropped"]) == (0, {})
    assert summary["iterations_pending"] == 0
    assert set(kept) == client
    assert set(kept.values()) == {4}
    for item in of_type(items, "infer.request"):
        if item.get("schema_version") == 2:
            request = item["request_ref"]["id"]
            cached = item["metadata"]["cached_at_admission"]
            assert cached + prefill[request] == item["input_tokens"]


def test_preempted_requests_recompute_and_keep_every_token(tmp_path: Path) -> None:
    hook = tmp_path / "hook"
    output = tmp_path / "infer.jsonl"
    with _engine(hook, num_gpu_blocks=12, block_size=4, max_num_seqs=4) as engine:
        run_profile(
            engine,
            output,
            vllm_execution_dir=hook,
            concurrency=(4,),
            input_tokens=(12,),
            output_tokens=(10,),
            request_count=8,
        )
        preemptions = engine.engine.stats.preemptions
    items = records(output)
    memberships = of_type(items, "infer.membership")
    kept: dict[str, int] = defaultdict(int)
    for membership in memberships:
        kept[membership["request_ref"]["id"]] += membership["output_tokens"] or 0
    assert preemptions > 0
    assert any(m["metadata"]["recompute"] for m in memberships)
    assert set(kept.values()) == {10}


def test_segments_seal_promptly_and_the_epochs_end_with_goodbye(
    tmp_path: Path,
) -> None:
    hook = tmp_path / "hook"
    with _engine(hook, hook_seal_seconds=0.2) as engine:
        chat(engine, words(8, "a"), max_tokens=2)
        assert wait_until(lambda: any(hook.rglob("*.jsonl")), timeout=10)
    parts = list(hook.rglob("*.jsonl.part"))
    epochs = sorted(
        path.name.split("-")[0] for path in hook.glob("*/*") if path.is_dir()
    )
    last_lines = {
        directory.name.split("-")[0]: sorted(directory.glob("*.jsonl"))[-1]
        .read_text()
        .splitlines()[-1]
        for directory in hook.glob("*/*")
    }
    assert parts == []
    assert epochs == ["engine", "worker"]
    assert all('"kind":"goodbye"' in line for line in last_lines.values())


def test_each_epochs_hello_names_only_its_own_class(tmp_path: Path) -> None:
    # The hook's gate summarises the engine's config with the scheduler alone
    # and the worker's with the model runner alone (vllm_hook/__init__.py,
    # gate.check), so each hello leaves the other class null.
    hook = tmp_path / "hook"
    with _engine(hook):
        pass
    hellos = {}
    for path in sorted(hook.rglob("*.jsonl")):
        first = json.loads(path.read_text().splitlines()[0])
        if first["kind"] == "hello":
            hellos[path.parent.name.split("-")[0]] = first["config"]
    assert (hellos["engine"]["scheduler"], hellos["engine"]["runner"]) == (
        SCHEDULER,
        None,
    )
    assert (hellos["worker"]["scheduler"], hellos["worker"]["runner"]) == (
        None,
        RUNNER,
    )
