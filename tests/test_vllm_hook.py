"""The vLLM execution hook, against fakes shaped like vLLM 0.30.0's classes."""

from __future__ import annotations

import enum
import errno
import json
import os
import random
import stat
import subprocess
import sys
import threading
import time
import timeit
import tracemalloc
import types
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

import stormlog.infer.vllm_hook as hook
from stormlog.infer.vllm_hook import engine as engine_module
from stormlog.infer.vllm_hook import gate, process
from stormlog.infer.vllm_hook import writer as writer_module
from stormlog.infer.vllm_hook.engine import ITERATION_ATTRIBUTE, EngineRecorder
from stormlog.infer.vllm_hook.worker import RunnerRecorder
from stormlog.infer.vllm_hook.writer import EpochWriter, WriterLimits

# ---------------------------------------------------------------- vLLM fakes


class FinishReason(enum.IntEnum):
    """vLLM 0.30.0's enum: its name is upper case, its string lower case."""

    STOP = 0
    LENGTH = 1

    def __str__(self) -> str:
        return ("stop", "length")[self.value]


class PauseState(enum.Enum):
    UNPAUSED = 0
    PAUSED_NEW = 1
    PAUSED_ALL = 2


@dataclass
class FakeRequest:
    request_id: str
    num_prompt_tokens: int
    num_output_tokens: int = 0
    num_stale_output_tokens: int = 0
    drop_stale_output: bool = False
    finished: bool = False
    status: Any = None
    # Stop after this many output tokens; vLLM trims the sampled list there.
    max_tokens: int | None = None
    # A streaming-input request: on a stop its next input is appended instead.
    resumable: bool = False
    next_input: int = 0
    use_structured_output: bool = False

    def is_finished(self) -> bool:
        return self.finished

    def get_finished_reason(self) -> FinishReason | None:
        return FinishReason.LENGTH if self.finished else None


@dataclass
class NewRequestData:
    req_id: str
    num_computed_tokens: int
    prompt_token_ids: list[int]

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)


@dataclass
class CachedRequestData:
    req_ids: list[str] = field(default_factory=list)
    num_computed_tokens: list[int] = field(default_factory=list)
    num_output_tokens: list[int] = field(default_factory=list)
    context_phase: set[str] = field(default_factory=set)

    def is_context_phase(self, req_id: str) -> bool:
        return req_id in self.context_phase


@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: dict[str, list[int]] = field(default_factory=dict)
    preempted_req_ids: set[str] | None = None


@dataclass
class ModelRunnerOutput:
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]]


@dataclass
class EngineCoreOutput:
    request_id: str
    new_token_ids: list[int]
    finish_reason: FinishReason | None = None


@dataclass
class EngineCoreOutputs:
    outputs: list[EngineCoreOutput]


class Config(types.SimpleNamespace):
    pass


def vllm_config(**parallel: Any) -> Config:
    return Config(
        parallel_config=Config(
            distributed_executor_backend=parallel.get("executor", "uni"),
            tensor_parallel_size=parallel.get("tp", 1),
            pipeline_parallel_size=parallel.get("pp", 1),
            data_parallel_size=parallel.get("dp", 1),
        ),
        scheduler_config=Config(async_scheduling=True, max_num_batched_tokens=2048),
        speculative_config=parallel.get("speculative"),
        model_config=Config(runner_type="generate"),
        kv_transfer_config=None,
    )


def _fake_vllm() -> dict[str, types.ModuleType]:
    """Fresh fake vLLM classes, so each test patches its own."""

    class Scheduler:
        def __init__(self, vllm_config: Any, *args: Any, **kwargs: Any) -> None:
            self.vllm_config = vllm_config
            self.requests: dict[str, FakeRequest] = {}
            self.num_sampled_tokens_per_step = 1
            self.next_output: SchedulerOutput | None = None
            self.update_error: Exception | None = None
            self._pause_state = PauseState.UNPAUSED
            self.running: list[FakeRequest] = []
            self.reset_error: Exception | None = None

        @property
        def pause_state(self) -> PauseState:
            return self._pause_state

        def set_pause_state(self, pause_state: PauseState) -> None:
            self._pause_state = pause_state

        def add_request(self, request: Any) -> None:
            # A live ID's later add is a streaming-input request's next input.
            self.requests.setdefault(request.request_id, request)

        def reset_prefix_cache(
            self, reset_running_requests: bool = False, reset_connector: bool = False
        ) -> bool:
            """vLLM 0.30.0's order: preempt every running request, then reset."""
            if reset_running_requests:
                self.running.clear()
            if self.reset_error is not None:
                raise self.reset_error
            return not self.running

        def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
            assert self.next_output is not None
            return self.next_output

        def update_from_output(
            self, scheduler_output: Any, model_runner_output: Any
        ) -> dict[int, EngineCoreOutputs]:
            """vLLM 0.30.0's shape: stops trim in place and free inside."""
            if self.update_error is not None:
                raise self.update_error
            outputs = []
            for internal in scheduler_output.num_scheduled_tokens:
                request = self.requests.get(internal)
                if request is None or request.finished:
                    continue
                if request.num_stale_output_tokens and request.drop_stale_output:
                    continue
                index = model_runner_output.req_id_to_index[internal]
                new = model_runner_output.sampled_token_ids[index]
                limit = request.max_tokens
                if limit is not None and request.num_output_tokens + len(new) >= limit:
                    del new[limit - request.num_output_tokens :]
                    request.finished = True
                request.num_output_tokens += len(new)
                reason = request.get_finished_reason()
                outputs.append(EngineCoreOutput(internal, list(new), reason))
                if request.finished and request.resumable:
                    # The session resets: the output so far and the next input
                    # become the prompt of the request's next turn.
                    request.num_prompt_tokens += (
                        request.num_output_tokens + request.next_input
                    )
                    request.num_output_tokens = 0
                    request.finished = False
                    request.max_tokens = None
                elif request.finished:
                    self._free_request(request)
            return {0: EngineCoreOutputs(outputs)} if outputs else {}

        def _free_request(
            self, request: FakeRequest, delay_free_blocks: bool = False
        ) -> None:
            self.requests.pop(request.request_id, None)

    class AsyncScheduler(Scheduler):
        pass

    class EngineCore:
        def __init__(self, scheduler: Any) -> None:
            self.scheduler = scheduler

        def preprocess_add_request(self, request: Any) -> tuple[Any, int]:
            return request, 0

    class GPUModelRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []
            self.defer = True

        def execute_model(
            self,
            scheduler_output: Any,
            intermediate_tensors: Any = None,
            dummy_run: bool = False,
        ) -> Any:
            self.calls.append(("execute", dummy_run))
            if scheduler_output == "boom":
                raise RuntimeError("model failed")
            return None if self.defer and not dummy_run else "sampled"

        def sample_tokens(self, grammar_output: Any) -> str:
            self.calls.append(("sample", False))
            return "sampled"

        def _dummy_run(self, num_tokens: int) -> Any:
            return self.execute_model(None, dummy_run=True)

    class Worker:
        def __init__(self, vllm_config: Any) -> None:
            self.vllm_config = vllm_config
            self.rank = 0
            self.local_rank = 0
            self.device = types.SimpleNamespace(index=0)

        def init_device(self) -> None:
            self.model_runner = GPUModelRunner()

    for cls, module in (
        (Scheduler, "vllm.v1.core.sched.scheduler"),
        (AsyncScheduler, "vllm.v1.core.sched.async_scheduler"),
        (GPUModelRunner, "vllm.v1.worker.gpu.model_runner"),
    ):
        cls.__module__ = module
        cls.__qualname__ = cls.__name__
    envs = types.ModuleType("vllm.envs")
    setattr(envs, "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", False)
    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.envs": envs,
        "vllm.v1.core.sched.scheduler": types.ModuleType(
            "vllm.v1.core.sched.scheduler"
        ),
        "vllm.v1.core.sched.async_scheduler": types.ModuleType(
            "vllm.v1.core.sched.async_scheduler"
        ),
        "vllm.v1.engine.core": types.ModuleType("vllm.v1.engine.core"),
        "vllm.v1.worker.gpu_worker": types.ModuleType("vllm.v1.worker.gpu_worker"),
    }
    setattr(modules["vllm.v1.core.sched.scheduler"], "Scheduler", Scheduler)
    setattr(
        modules["vllm.v1.core.sched.async_scheduler"], "AsyncScheduler", AsyncScheduler
    )
    setattr(modules["vllm.v1.engine.core"], "EngineCore", EngineCore)
    setattr(modules["vllm.v1.worker.gpu_worker"], "Worker", Worker)
    return modules


@pytest.fixture
def vllm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[dict[str, Any]]:
    modules = _fake_vllm()
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(gate, "vllm_version", lambda: "0.30.0")
    monkeypatch.setattr(hook, "_PATCHED", False)
    monkeypatch.setattr(hook, "_WRITERS", {})
    root = tmp_path / "hook"
    assert hook.install({hook.ENV_DIR: str(root)}) == ["engine", "worker"]
    yield {
        "root": root,
        "Scheduler": getattr(modules["vllm.v1.core.sched.scheduler"], "Scheduler"),
        "AsyncScheduler": getattr(
            modules["vllm.v1.core.sched.async_scheduler"], "AsyncScheduler"
        ),
        "EngineCore": getattr(modules["vllm.v1.engine.core"], "EngineCore"),
        "Worker": getattr(modules["vllm.v1.worker.gpu_worker"], "Worker"),
        "envs": modules["vllm.envs"],
    }
    for writer in list(hook._WRITERS.values()):
        writer.close()


def _loads(line: str) -> dict[str, Any]:
    """One log line. A key written twice fails: a record's own fields follow
    the common ones on its line, so a field reusing a common name would."""

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        names = [name for name, _ in pairs]
        assert len(names) == len(set(names)), f"a key written twice: {names}"
        return dict(pairs)

    record: dict[str, Any] = json.loads(line, object_pairs_hook=unique)
    return record


def _records(root: Path, role: str) -> list[dict[str, Any]]:
    for writer in hook._WRITERS.values():
        if writer.role == role:
            writer.close()
    records = []
    for path in sorted(root.glob(f"*/{role}-*/*.jsonl")):
        records += [_loads(line) for line in path.read_text().splitlines()]
    return records


def _of(records: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [r for r in records if r["kind"] == kind]


def _new(internal: str, prompt: int, cached: int = 0) -> NewRequestData:
    return NewRequestData(internal, cached, list(range(prompt)))


# ---------------------------------------------------------------- engine side


def test_two_requests_share_iterations_and_finish(vllm: dict[str, Any]) -> None:
    scheduler = vllm["AsyncScheduler"](vllm_config())
    scheduler.requests = {
        "a-1": FakeRequest("a-1", 4),
        "b-1": FakeRequest("b-1", 6),
    }
    first = SchedulerOutput(
        [_new("a-1", 4), _new("b-1", 6, cached=2)],
        CachedRequestData(),
        {"a-1": 4, "b-1": 4},
        8,
    )
    scheduler.next_output = first
    assert scheduler.schedule() is first
    producer, iteration = getattr(first, ITERATION_ATTRIBUTE)
    assert producer.startswith("vllm:") and iteration == "0"
    scheduler.update_from_output(
        first, ModelRunnerOutput({"a-1": 0, "b-1": 1}, [[7], [9]])
    )
    second = SchedulerOutput(
        [], CachedRequestData(["a-1", "b-1"], [4, 6], [1, 1]), {"a-1": 1, "b-1": 1}, 2
    )
    scheduler.next_output = second
    scheduler.schedule()
    scheduler.requests["a-1"].finished = True  # finished before this output
    scheduler._free_request(scheduler.requests["a-1"])
    scheduler.requests["b-1"].max_tokens = 2  # its stop: freed inside the update
    scheduler.update_from_output(
        second, ModelRunnerOutput({"a-1": 0, "b-1": 1}, [[3], [4, 5]])
    )

    records = _records(vllm["root"], "engine")
    hello = _of(records, "hello")[0]
    assert (hello["enabled"], hello["refused"], hello["producer"]) == (
        True,
        None,
        producer,
    )
    scheduled = _of(records, "scheduled")
    first_members = {m["internal"]: m for m in scheduled[0]["members"]}
    assert first_members["b-1"]["sighting"] == "first"
    assert first_members["b-1"]["cached_at_admission"] == 2
    assert first_members["b-1"]["prefill_scheduled"] == 4
    second_members = {m["internal"]: m for m in scheduled[1]["members"]}
    assert second_members["a-1"]["sighting"] == "repeat"
    assert second_members["a-1"]["cached_at_admission"] is None
    assert second_members["a-1"]["past_prompt_scheduled"] == 1
    assert (first_members["a-1"]["phase"], second_members["a-1"]["phase"]) == (
        "context",
        "generation",
    )
    completed = _of(records, "completed")
    assert [m["outcome"] for m in completed[0]["members"]] == ["kept", "kept"]
    assert [m["retained"] for m in completed[0]["members"]] == [1, 1]
    assert completed[0]["members"][0]["computed_after"] == 4
    second_done = {m["internal"]: m for m in completed[1]["members"]}
    assert second_done["a-1"]["outcome"] == "discarded_finished"
    # b-1 sampled two tokens; its stop kept one, and vLLM freed it in the update.
    assert (second_done["b-1"]["outcome"], second_done["b-1"]["sampled"]) == (
        "kept",
        2,
    )
    assert second_done["b-1"]["retained"] == 1
    assert second_done["b-1"]["finish_reason"] == "length"
    terminals = {r["internal"]: r for r in _of(records, "terminal")}
    assert set(terminals) == {"a-1", "b-1"}
    assert terminals["b-1"]["output_tokens"] == 2
    assert terminals["b-1"]["finish_reason"] == "length"
    assert [r["seq"] for r in records] == list(range(len(records)))
    # Nothing outlives the requests: per-request state goes with _free_request.
    recorder = getattr(scheduler, hook.RECORDER_ATTRIBUTE)
    assert (recorder.committed, recorder.prompt_tokens, recorder.pending) == (
        {},
        {},
        {},
    )


def test_spec_decode_acceptance_and_stale_outputs(vllm: dict[str, Any]) -> None:
    scheduler = vllm["Scheduler"](
        vllm_config(
            speculative=Config(method="ngram", enable_adaptive_verification=False)
        )
    )
    scheduler.requests = {
        "s-1": FakeRequest("s-1", 3, num_output_tokens=2),
        "t-1": FakeRequest(
            "t-1",
            3,
            num_output_tokens=2,
            num_stale_output_tokens=1,
            drop_stale_output=True,
        ),
    }
    output = SchedulerOutput(
        [],
        CachedRequestData(["s-1", "t-1"], [5, 5], [2, 2]),
        {"s-1": 4, "t-1": 1},
        5,
        scheduled_spec_decode_tokens={"s-1": [1, 2, 3]},
    )
    scheduler.next_output = output
    scheduler.schedule()
    scheduler.update_from_output(
        output, ModelRunnerOutput({"s-1": 0, "t-1": 1}, [[1, 2, 9], [5]])
    )

    members = {
        m["internal"]: m
        for m in _of(_records(vllm["root"], "engine"), "completed")[0]["members"]
    }
    # Three drafts, three sampled with one bonus token: two accepted, one rejected.
    assert (members["s-1"]["sampled"], members["s-1"]["accepted_drafts"]) == (3, 2)
    assert members["s-1"]["computed_after"] == 5 + 4 - 1
    assert (members["t-1"]["outcome"], members["t-1"]["stale"]) == (
        "dropped_stale",
        True,
    )


def test_streaming_input_keeps_counts_whole(vllm: dict[str, Any]) -> None:
    scheduler = vllm["Scheduler"](vllm_config())
    request = FakeRequest("r-1", 4, max_tokens=2, resumable=True, next_input=3)
    scheduler.requests = {"r-1": request}
    steps = [
        SchedulerOutput([_new("r-1", 4)], CachedRequestData(), {"r-1": 4}, 4),
        SchedulerOutput([], CachedRequestData(["r-1"], [4], [1]), {"r-1": 1}, 1),
    ]
    for step, token in zip(steps, (7, 8)):
        scheduler.next_output = step
        scheduler.schedule()
        scheduler.update_from_output(step, ModelRunnerOutput({"r-1": 0}, [[token]]))
    # The turn ended: vLLM reset its output count and grew its prompt to 4 + 2 + 3.
    assert (request.num_prompt_tokens, request.num_output_tokens) == (9, 0)
    resumed = SchedulerOutput(
        [], CachedRequestData(["r-1"], [5], [0], {"r-1"}), {"r-1": 4}, 4
    )
    scheduler.next_output = resumed
    scheduler.schedule()

    records = _records(vllm["root"], "engine")
    completed = [c["members"][0] for c in _of(records, "completed")]
    assert [m["retained"] for m in completed] == [1, 1]
    assert completed[1]["finish_reason"] == "length"
    member = _of(records, "scheduled")[2]["members"][0]
    assert (member["prompt_tokens"], member["prefill_scheduled"]) == (9, 4)
    assert (member["phase"], member["resumable"], member["recompute"]) == (
        "context",
        True,
        False,
    )


def test_admission_writes_an_alias(vllm: dict[str, Any]) -> None:
    scheduler = vllm["Scheduler"](vllm_config())
    core = vllm["EngineCore"](scheduler)
    request = types.SimpleNamespace(
        request_id="chatcmpl-stormlog-r1-q0-0f3a9c1d",
        external_req_id="chatcmpl-stormlog-r1-q0",
    )

    assert core.preprocess_add_request(request) == (request, 0)

    alias = _of(_records(vllm["root"], "engine"), "alias")[0]
    assert (alias["internal"], alias["external"]) == (
        "chatcmpl-stormlog-r1-q0-0f3a9c1d",
        "chatcmpl-stormlog-r1-q0",
    )


def test_pause_transitions_and_the_state_each_step_saw(vllm: dict[str, Any]) -> None:
    scheduler = vllm["Scheduler"](vllm_config())
    scheduler.requests = {"a-1": FakeRequest("a-1", 2)}
    scheduler.set_pause_state(PauseState.PAUSED_NEW)
    output = SchedulerOutput([_new("a-1", 2)], CachedRequestData(), {"a-1": 2}, 2)
    scheduler.next_output = output
    scheduler.schedule()
    scheduler.set_pause_state(PauseState.PAUSED_ALL)
    scheduler.set_pause_state(PauseState.UNPAUSED)
    scheduler.set_pause_state(PauseState.UNPAUSED)  # a wake-up's resume: no change
    refused = vllm["Scheduler"](vllm_config(pp=2))
    refused.set_pause_state(PauseState.PAUSED_ALL)

    assert scheduler.pause_state is PauseState.UNPAUSED  # vLLM's own call ran
    assert refused.pause_state is PauseState.PAUSED_ALL
    records = _records(vllm["root"], "engine")
    hellos = _of(records, "hello")
    assert [hello["observes"] for hello in hellos] == [
        ["cache_reset", "enqueued", "pause"],
        [],
    ]
    pauses = _of(records, "pause")
    assert [(p["from"], p["to"]) for p in pauses] == [
        ("UNPAUSED", "PAUSED_NEW"),
        ("PAUSED_NEW", "PAUSED_ALL"),
        ("PAUSED_ALL", "UNPAUSED"),
    ]
    assert all(p["wall_ns"] <= p["wall_after_ns"] for p in pauses)
    assert _of(records, "scheduled")[0]["pause_state"] == "PAUSED_NEW"


def test_a_request_entering_the_scheduler_is_stamped_once(vllm: dict[str, Any]) -> None:
    scheduler = vllm["Scheduler"](vllm_config())
    scheduler.add_request(FakeRequest("s-1", 4, resumable=True))
    scheduler.add_request(FakeRequest("s-1", 6, resumable=True))  # its next input
    scheduler.add_request(FakeRequest("g-1", 3, use_structured_output=True))
    scheduler.add_request(types.SimpleNamespace(request_id="u-1"))  # flags unknown
    refused = vllm["Scheduler"](vllm_config(pp=2))
    refused.add_request(FakeRequest("r-1", 2))

    assert list(scheduler.requests) == ["s-1", "g-1", "u-1"]
    assert list(refused.requests) == ["r-1"]
    enqueued = _of(_records(vllm["root"], "engine"), "enqueued")
    assert [
        (e["internal"], e["structured_output"], e["resumable"]) for e in enqueued
    ] == [("s-1", False, True), ("g-1", True, False), ("u-1", None, None)]
    assert all(e["wall_ns"] <= e["wall_after_ns"] for e in enqueued)


def test_a_cache_reset_records_who_was_running_and_how_it_ended(
    vllm: dict[str, Any]
) -> None:
    scheduler = vllm["Scheduler"](vllm_config())
    scheduler.running = [FakeRequest("a-1", 2), FakeRequest("b-1", 2)]

    assert scheduler.reset_prefix_cache() is False  # requests hold the cache
    assert scheduler.reset_prefix_cache(True, reset_connector=True) is True
    scheduler.running = [FakeRequest("c-1", 2)]
    scheduler.reset_error = RuntimeError("remote KV transfer pending")
    with pytest.raises(RuntimeError, match="remote KV transfer"):
        scheduler.reset_prefix_cache(reset_running_requests=True)

    resets = _of(_records(vllm["root"], "engine"), "cache_reset")
    assert [
        (
            r["reset_running_requests"],
            r["reset_connector"],
            r["running"],
            r["succeeded"],
            r["raised"],
        )
        for r in resets
    ] == [
        (False, False, ["a-1", "b-1"], False, False),
        (True, True, ["a-1", "b-1"], True, False),
        # vLLM preempts before it fails, so its running requests still count.
        (True, False, ["c-1"], None, True),
    ]
    for reset in resets:
        assert reset["start_wall_ns"] <= reset["start_wall_after_ns"]
        assert reset["start_mono_ns"] <= reset["end_mono_ns"]
        assert reset["end_wall_ns"] <= reset["end_wall_after_ns"]


def test_a_method_vllm_lacks_is_neither_patched_nor_claimed() -> None:
    class OldScheduler:
        pass

    hook._patch_optional(OldScheduler, "set_pause_state", "pause", lambda f: f)

    assert not hasattr(OldScheduler, "set_pause_state")
    assert not hasattr(OldScheduler, hook.OBSERVES_ATTRIBUTE)


def test_every_stamp_is_bracketed_by_two_wall_reads(vllm: dict[str, Any]) -> None:
    scheduler = vllm["Scheduler"](vllm_config())
    vllm["EngineCore"](scheduler).preprocess_add_request(
        types.SimpleNamespace(request_id="a-1", external_req_id="chatcmpl-a")
    )
    scheduler.requests = {"a-1": FakeRequest("a-1", 2, max_tokens=1)}
    output = SchedulerOutput([_new("a-1", 2)], CachedRequestData(), {"a-1": 2}, 2)
    scheduler.next_output = output
    scheduler.schedule()
    scheduler.update_from_output(output, ModelRunnerOutput({"a-1": 0}, [[5]]))

    records = _records(vllm["root"], "engine")
    status = json.loads(next(vllm["root"].glob("*/engine-*/status.json")).read_text())
    stamps = [("hello", _of(records, "hello")[0]["clock"]), ("status", status)]
    for record in records:
        for prefix in ("", "start_", "end_"):
            if f"{prefix}wall_ns" in record:
                names = ("wall_ns", "mono_ns", "wall_after_ns")
                stamps.append(
                    (record["kind"], {name: record[prefix + name] for name in names})
                )
    assert {kind for kind, _ in stamps} == {
        "hello",
        "status",
        "alias",
        "scheduled",
        "completed",
        "terminal",
        "goodbye",
    }
    for _kind, read in stamps:
        assert read["wall_ns"] <= read["wall_after_ns"]
    clock = stamps[0][1]
    assert clock["gap_ns"] == clock["wall_after_ns"] - clock["wall_ns"]


def test_a_stamp_reads_the_wall_clock_again_after_the_monotonic_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    walls = iter([10, 30])
    monkeypatch.setattr(writer_module.time, "time_ns", lambda: next(walls))
    monkeypatch.setattr(writer_module.time, "monotonic_ns", lambda: 20)
    assert writer_module.stamp() == {"wall_ns": 10, "mono_ns": 20, "wall_after_ns": 30}


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        (vllm_config(executor="ray"), "executor ray is not supported"),
        (vllm_config(pp=2), "pipeline parallelism"),
        (vllm_config(dp=2), "data parallelism"),
        (vllm_config(speculative=Config(method="eagle")), "speculative method eagle"),
    ],
)
def test_an_unsupported_configuration_is_refused_and_left_alone(
    vllm: dict[str, Any], config: Config, reason: str
) -> None:
    scheduler = vllm["Scheduler"](config)
    output = SchedulerOutput([_new("a-1", 2)], CachedRequestData(), {"a-1": 2}, 2)
    scheduler.next_output = output

    scheduler.schedule()

    assert not hasattr(output, ITERATION_ATTRIBUTE)
    records = _records(vllm["root"], "engine")
    assert reason in _of(records, "hello")[0]["refused"]
    assert _of(records, "scheduled") == []


def test_another_vllm_version_is_refused(
    vllm: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate, "vllm_version", lambda: "0.31.0")

    vllm["Scheduler"](vllm_config())

    hello = _of(_records(vllm["root"], "engine"), "hello")[0]
    assert hello["refused"] == "vLLM 0.31.0 is not supported"


@pytest.mark.parametrize("disabled", [False, True])
def test_hello_says_whether_request_ids_are_randomized(
    vllm: dict[str, Any], disabled: bool
) -> None:
    setattr(vllm["envs"], "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", disabled)

    vllm["Scheduler"](vllm_config())

    hello = _of(_records(vllm["root"], "engine"), "hello")[0]
    assert hello["config"]["request_id_randomization"] is (not disabled)


class CUDAGraphMode(enum.Enum):
    FULL_AND_PIECEWISE = (2, 1)


PROFILER = {
    "profiler": "torch",
    "torch_profiler_dir": "/traces",
    # The .gz name and streaming write a truncated trace is read through.
    "torch_profiler_use_gzip": True,
    "torch_profiler_with_stack": False,
    # Each of these changes what a window costs and how long its stop pauses.
    "torch_profiler_record_shapes": False,
    "torch_profiler_with_memory": False,
    "torch_profiler_with_flops": False,
    "torch_profiler_dump_cuda_time_total": False,
    "capture_torch_profiler": False,
    "ignore_frontend": True,
    "max_iterations": 40,
    "delay_iterations": 0,
    "warmup_iterations": 0,
    "active_iterations": 5,
    "wait_iterations": 0,
}


def _sized_config() -> Config:
    config = vllm_config()
    config.scheduler_config.max_num_seqs = 256
    config.cache_config = Config(
        num_gpu_blocks=9000, block_size=32, gpu_memory_utilization=0.9
    )
    config.compilation_config = Config(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE)
    config.model_config.enable_cumem_allocator = False
    config.model_config.enable_sleep_mode = False
    config.profiler_config = Config(**PROFILER)
    return config


def test_hello_records_the_layout_the_profiler_and_the_process(
    vllm: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process, "start_ticks", lambda pid: pid * 10)
    monkeypatch.setattr(process, "start_ns", lambda pid: pid * 1_000)
    scheduler_class = vllm["Scheduler"]

    class SizedScheduler(scheduler_class):  # type: ignore[misc, valid-type]
        def __init__(self, vllm_config: Any) -> None:
            # vLLM resolves the scheduler's block size and KV groups itself.
            self.kv_cache_config = Config(kv_cache_groups=[object(), object()])
            self.block_size = 16
            super().__init__(vllm_config)

    SizedScheduler.__module__ = scheduler_class.__module__
    SizedScheduler.__qualname__ = scheduler_class.__qualname__  # passes the gate
    SizedScheduler(_sized_config())
    worker = vllm["Worker"](_sized_config())
    worker.init_device()

    engine = _of(_records(vllm["root"], "engine"), "hello")[0]
    assert {
        key: engine["config"][key]
        for key in (
            "max_num_seqs",
            "num_gpu_blocks",
            "kv_cache_groups",
            "block_size",
            "cudagraph_mode",
            "gpu_memory_utilization",
            "enable_cumem_allocator",
            "enable_sleep_mode",
            "profiler",
        )
    } == {
        "max_num_seqs": 256,
        "num_gpu_blocks": 9000,
        "kv_cache_groups": 2,
        "block_size": 16,
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "gpu_memory_utilization": 0.9,
        "enable_cumem_allocator": False,
        "enable_sleep_mode": False,
        "profiler": PROFILER,
    }
    pid, parent = os.getpid(), os.getppid()
    for hello in (engine, _of(_records(vllm["root"], "worker"), "hello")[0]):
        assert {
            key: hello[key]
            for key in (
                "process_start_ns",
                "process_start_ticks",
                "parent_pid",
                "parent_process_start_ticks",
                "parent_process_start_ns",
            )
        } == {
            "process_start_ns": pid * 1_000,
            "process_start_ticks": pid * 10,
            "parent_pid": parent,
            "parent_process_start_ticks": parent * 10,
            "parent_process_start_ns": parent * 1_000,
        }
    assert (engine["enabled"], engine["observes"]) == (
        True,
        ["cache_reset", "enqueued", "pause"],
    )
    assert _of(_records(vllm["root"], "worker"), "hello")[0]["observes"] == []
    worker_config = _of(_records(vllm["root"], "worker"), "hello")[0]["config"]
    # A worker has no scheduler: its layout is the configured one.
    assert (worker_config["kv_cache_groups"], worker_config["block_size"]) == (None, 32)


def test_a_hello_without_these_settings_records_them_as_null(
    vllm: dict[str, Any]
) -> None:
    vllm["Scheduler"](vllm_config())

    config = _of(_records(vllm["root"], "engine"), "hello")[0]["config"]
    assert config["profiler"] is None
    assert config["num_gpu_blocks"] is None and config["cudagraph_mode"] is None


def test_process_start_is_read_like_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    # A command name may hold spaces and parentheses.
    fields = " ".join(str(n) for n in range(5, 22))  # fields 5 to 21
    stat = f"4242 (vllm (x) y) S 1 {fields} 777 9"
    assert process.ticks_from_stat(stat) == 777
    assert process.ticks_from_stat("4242 (short) S 1") is None

    monkeypatch.setitem(sys.modules, "psutil", None)  # psutil missing
    monkeypatch.setattr(process, "start_ticks", lambda pid: 250)
    monkeypatch.setattr(process, "_boot_seconds", lambda: 1_700_000_000)
    monkeypatch.setattr(process.os, "sysconf", lambda name: 100)
    assert process.start_ns(4242) == 1_700_000_002_500_000_000
    monkeypatch.setattr(process, "_boot_seconds", lambda: None)
    assert process.start_ns(4242) is None


def test_vllm_errors_pass_through_and_telemetry_errors_do_not(
    vllm: dict[str, Any]
) -> None:
    scheduler = vllm["Scheduler"](vllm_config())
    scheduler.requests = {"a-1": FakeRequest("a-1", 2)}
    output = SchedulerOutput([_new("a-1", 2)], CachedRequestData(), {"a-1": 2}, 2)
    scheduler.next_output = output
    scheduler.schedule()
    scheduler.update_error = ValueError("vLLM's own error")

    with pytest.raises(ValueError, match="vLLM's own error"):
        scheduler.update_from_output(output, ModelRunnerOutput({"a-1": 0}, [[1]]))
    # An output the hook cannot read breaks only the telemetry.
    unreadable = types.SimpleNamespace(total_num_scheduled_tokens=1)
    scheduler.next_output = unreadable
    assert scheduler.schedule() is unreadable

    records = _records(vllm["root"], "engine")
    # Recorded although vLLM raised, and as unknown: the step's fate is not seen.
    (completed,) = _of(records, "completed")
    assert completed["update_failed"] is True
    (member,) = completed["members"]
    assert (member["outcome"], member["retained"], member["computed_after"]) == (
        "unknown",
        None,
        None,
    )
    status = json.loads(next(vllm["root"].glob("*/engine-*/status.json")).read_text())
    assert status["errors"] >= 1


def test_no_record_reuses_a_common_field_name(vllm: dict[str, Any]) -> None:
    scheduler = vllm["Scheduler"](vllm_config())
    vllm["EngineCore"](scheduler).preprocess_add_request(FakeRequest("a-1", 2))
    scheduler.requests = {"a-1": FakeRequest("a-1", 2, max_tokens=1)}
    output = SchedulerOutput([_new("a-1", 2)], CachedRequestData(), {"a-1": 2}, 2)
    scheduler.next_output = output
    scheduler.schedule()
    scheduler.update_from_output(output, ModelRunnerOutput({"a-1": 0}, [[1]]))
    scheduler.schedule()
    scheduler.update_error = ValueError("vLLM's own error")
    with pytest.raises(ValueError):
        scheduler.update_from_output(output, ModelRunnerOutput({}, []))
    vllm["Worker"](vllm_config()).init_device()
    # A heartbeat's own fields are the status, the worker's included.
    statuses = [writer._status() for writer in hook._WRITERS.values()]

    # _loads fails on a key written twice, in every record of every kind.
    records = _records(vllm["root"], "engine") + _records(vllm["root"], "worker")
    assert {record["kind"] for record in records} >= {
        "hello",
        "alias",
        "scheduled",
        "completed",
        "terminal",
        "goodbye",
    }
    assert any(record.get("update_failed") for record in records)
    assert {"range_misses", "pending_samples"} <= set(statuses[-1])
    for status in statuses:
        assert not {"format", "kind", "epoch", "seq"} & set(status)


# ---------------------------------------------------------------- worker side


def test_worker_ranges_pair_sampling_and_skip_dummy_runs(tmp_path: Path) -> None:
    opened: list[tuple[str, str]] = []
    writer = EpochWriter(tmp_path, "worker")
    recorder = RunnerRecorder(writer)
    recorder.range_factory = lambda producer, iteration, nvtx: _Range(
        opened, producer, iteration
    )
    runner = _runner_class()()
    recorder.wrap(runner)
    step = SchedulerOutput([], CachedRequestData(), {}, 0)
    setattr(step, ITERATION_ATTRIBUTE, ("vllm:h:b:1:2", "7"))
    runner._dummy_run(1)  # V2 warm-up and graph capture: execute_model(dummy_run)
    runner.execute_model(SchedulerOutput([], CachedRequestData(), {}, 0))  # warm-up

    assert runner.execute_model(step) is None
    assert runner.sample_tokens(None) == "sampled"
    runner._dummy_run(1)  # once serving, neither start-up nor a miss
    runner.execute_model(SchedulerOutput([], CachedRequestData(), {}, 0))  # no identity
    with pytest.raises(RuntimeError, match="model failed"):
        runner.execute_model("boom")
    writer.close()

    # The step and its sampling share one range; the dummy run reached vLLM unranged.
    assert opened == [("vllm:h:b:1:2", "7"), ("vllm:h:b:1:2", "7")]
    assert runner.calls.count(("execute", True)) == 2
    assert (recorder.startup_unranged, recorder.range_misses) == (2, 2)
    assert len(recorder.pending) == 0


def _runner_class() -> Any:
    modules = _fake_vllm()
    worker = getattr(modules["vllm.v1.worker.gpu_worker"], "Worker")(vllm_config())
    worker.init_device()
    return type(worker.model_runner)


def test_a_range_that_fails_to_open_still_runs_vllm_once(tmp_path: Path) -> None:
    writer = EpochWriter(tmp_path, "worker")
    recorder = RunnerRecorder(writer)

    def broken(producer: str, iteration: str, nvtx: bool) -> Any:
        raise RuntimeError("profiler unavailable")

    recorder.range_factory = broken
    runner = _runner_class()()
    runner.defer = False
    recorder.wrap(runner)
    step = SchedulerOutput([], CachedRequestData(), {}, 0)
    setattr(step, ITERATION_ATTRIBUTE, ("p", "0"))

    assert runner.execute_model(step) == "sampled"
    writer.close()
    assert runner.calls == [("execute", False)]
    status = json.loads((writer.directory / "status.json").read_text())
    assert status["errors"] == 1


def test_a_worker_hello_names_its_rank_and_is_gated(vllm: dict[str, Any]) -> None:
    worker = vllm["Worker"](vllm_config(executor="mp", tp=2))
    worker.init_device()

    hello = _of(_records(vllm["root"], "worker"), "hello")[0]
    assert (hello["role"], hello["enabled"]) == ("worker", True)
    assert hello["rank"]["global"] == 0 and hello["cuda_ordinal"] == 0
    assert hello["config"]["runner"] == "vllm.v1.worker.gpu.model_runner.GPUModelRunner"
    assert hello["config"]["request_id_randomization"] is True


class _Range:
    def __init__(
        self, opened: list[tuple[str, str]], producer: str, iteration: str
    ) -> None:
        self.opened = opened
        self.identity = (producer, iteration)

    def __enter__(self) -> None:
        self.opened.append(self.identity)

    def __exit__(self, *args: object) -> None:
        return None


# ---------------------------------------------------------------- writer


def test_the_writer_seals_segments_and_keeps_status(tmp_path: Path) -> None:
    writer = EpochWriter(
        tmp_path, "engine", limits=WriterLimits(heartbeat_seconds=0.05)
    )
    writer.emit("alias", {"internal": "x"})
    (writer.directory / "flush").touch()
    deadline = time.monotonic() + 5
    while not list(writer.directory.glob("*.jsonl")) and time.monotonic() < deadline:
        time.sleep(0.02)
    writer.close()

    assert not (writer.directory / "flush").exists()
    assert list(writer.directory.glob("*.part")) == []
    records = [
        _loads(line)
        for path in sorted(writer.directory.glob("*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    assert records[0]["kind"] == "alias" and records[-1]["kind"] == "goodbye"
    assert [r["seq"] for r in records] == list(range(len(records)))
    status = json.loads((writer.directory / "status.json").read_text())
    assert status["last_seq"] == records[-1]["seq"]
    assert stat.S_IMODE((writer.directory / "key").stat().st_mode) == 0o600
    assert stat.S_IMODE(writer.directory.stat().st_mode) == 0o700


def test_the_disk_cap_stops_records_but_not_the_status(tmp_path: Path) -> None:
    writer = EpochWriter(
        tmp_path, "engine", limits=WriterLimits(max_bytes=400, heartbeat_seconds=0.05)
    )
    for index in range(20):
        writer.emit("alias", {"internal": f"request-{index}"})
    writer.close()

    status = json.loads((writer.directory / "status.json").read_text())
    assert status["capped"] is True
    assert status["dropped"]["alias"] > 0


def test_the_queue_counts_bytes_by_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = threading.Event(), threading.Event()
    write = writer_module._Segment.write

    def stalled(segment: Any, line: bytes) -> bool:
        entered.set()
        release.wait(5)
        return write(segment, line)

    monkeypatch.setattr(writer_module._Segment, "write", stalled)
    writer = EpochWriter(
        tmp_path, "engine", limits=WriterLimits(queue_bytes=4096, record_bytes=2048)
    )

    def burst(tag: str) -> None:
        for index in range(20):
            writer.emit("alias", {"internal": f"{tag}{index:03d}" + "x" * 1000})

    with writer._condition:  # the writer thread cannot take any yet
        burst("a")
    assert entered.wait(5)  # it took them as one batch, and is stuck writing it
    burst("b")
    writer.emit("alias", {"internal": "y" * 4000})
    release.set()
    _wait(lambda: _queued_bytes(writer) == 0)  # written, and released
    burst("c")
    writer.close()

    # Each 1 KB ID counts as about 1 KB, so four fit in 4 KB. A batch being
    # written still counts, so the second burst is dropped whole; once
    # written, it no longer counts, so the third fits again.
    written = Counter(
        record["internal"][0]
        for record in _epoch_records(writer.directory)
        if record["kind"] == "alias"
    )
    assert (written["a"], written["b"]) == (4, 0)
    assert written["c"] >= 4
    status = json.loads((writer.directory / "status.json").read_text())
    assert status["dropped"]["alias_oversized"] == 1
    assert status["dropped"]["alias"] == 60 - sum(written.values())


# Escapes, control characters, a lone surrogate, and text beyond the BMP.
_TEXT = ["a", "Z", "0", " ", '"', "\\", "/", "\x00", "\x1f", "\x7f", "\u00e9"]
_TEXT += ["\u4e2d", "\u2028", "\ud800", "\ufeff", "\U0001f600"]


def _random_text(rng: random.Random) -> str:
    return "".join(rng.choices(_TEXT, k=rng.randrange(40)))


def _random_value(rng: random.Random, depth: int = 0) -> Any:
    kind = rng.randrange(6 if depth < 4 else 4)
    if kind == 0:
        return _random_text(rng)
    if kind == 1:
        return rng.choice([0, -1, 2**63, -(10**30), rng.randrange(-(10**6), 10**6)])
    if kind == 2:
        return rng.choice([0.1, -1e300, 5e-324, float("nan"), float("inf")])
    if kind == 3:
        return rng.choice([True, False, None, "chatcmpl-abc-123"])
    if kind == 4:
        return [_random_value(rng, depth + 1) for _ in range(rng.randrange(5))]
    # json writes int, float, bool and None keys as text.
    keys = [_random_text(rng), rng.randrange(-9, 9), 1.5, True, None]
    return {
        rng.choice(keys): _random_value(rng, depth + 1) for _ in range(rng.randrange(5))
    }


def _line_as_before(epoch: str, kind: str, seq: int, fields: dict[str, Any]) -> bytes:
    """The line the writer wrote when its own thread serialized each record."""
    record = {"format": writer_module.FORMAT, "kind": kind, "epoch": epoch, "seq": seq}
    record.update(fields)
    return json.dumps(record, separators=(",", ":")).encode()


def test_a_record_counts_its_exact_json_and_is_written_unchanged(
    tmp_path: Path,
) -> None:
    rng = random.Random(217)
    # Top-level names never reuse the common fields' names.
    records = [
        {f"field_{index}": _random_value(rng) for index in range(rng.randrange(6))}
        for _ in range(400)
    ]
    records += [  # step-shaped: what the floor reads before serializing
        {
            "iteration": _random_text(rng),
            "preempted": [_random_text(rng) for _ in range(rng.randrange(3))],
            "members": [
                {"internal": _random_text(rng), "other": _random_value(rng)}
                for _ in range(rng.randrange(6))
            ],
        }
        for _ in range(100)
    ]
    limits = WriterLimits(heartbeat_seconds=3600, queue_bytes=1 << 30)
    writer = EpochWriter(tmp_path, "engine", limits=limits)
    with writer._condition:  # the writer thread cannot take any yet
        for fields in records:
            before = writer._queued_bytes
            writer.emit("alias", fields)
            json_bytes = len(json.dumps(fields, separators=(",", ":")).encode())
            assert writer._queued_bytes - before == json_bytes
            assert writer_module._floor(fields) <= json_bytes
    writer.close()

    lines = [
        line
        for path in sorted(writer.directory.glob("*.jsonl"))
        for line in path.read_bytes().splitlines()
    ]
    assert lines[:-1] == [
        _line_as_before(writer.epoch, "alias", seq, fields)
        for seq, fields in enumerate(records)
    ]
    assert json.loads(lines[-1])["kind"] == "goodbye"
    assert _queued_bytes(writer) == 0


def test_a_record_is_oversized_by_its_exact_json(tmp_path: Path) -> None:
    writer = EpochWriter(tmp_path, "engine", limits=WriterLimits(record_bytes=4096))
    # {"internal":""} is 15 bytes, and each escaped "\u00e9" 6: 4,096 in all.
    writer.emit("alias", {"internal": "\u00e9" * 680 + "x"})
    writer.emit("alias", {"internal": "\u00e9" * 680 + "xy"})  # one byte over
    writer.close()

    records = _epoch_records(writer.directory)
    assert [len(r["internal"]) for r in records if r["kind"] == "alias"] == [681]
    status = json.loads((writer.directory / "status.json").read_text())
    assert status["dropped"] == {"alias_oversized": 1}


def test_a_record_json_cannot_serialize_is_an_error_and_takes_no_number(
    tmp_path: Path,
) -> None:
    circular: list[Any] = []
    circular.append(circular)
    deep: list[Any] = []
    for _ in range(100_000):
        deep = [deep]

    class Changing(dict[str, Any]):
        """A value another thread changes while json reads it."""

        def items(self) -> Any:
            raise RuntimeError("dictionary changed size during iteration")

    writer = EpochWriter(tmp_path, "engine")
    for value in (object(), circular, deep, Changing(a=1)):
        writer.emit("alias", {"internal": value})  # never raises
    writer.emit("alias", {"internal": "x"})
    writer.close()

    records = _epoch_records(writer.directory)
    assert [(r["kind"], r["seq"]) for r in records] == [("alias", 0), ("goodbye", 1)]
    status = json.loads((writer.directory / "status.json").read_text())
    assert (status["errors"], status["dropped"]) == (4, {})


def test_fields_that_are_not_one_object_are_an_error(tmp_path: Path) -> None:
    writer = EpochWriter(tmp_path, "engine")
    for fields in (["a", "list"], "text", None):
        writer.emit("alias", cast(Any, fields))  # never a malformed line
    writer.emit("alias", {"internal": "x"})
    writer.close()

    records = _epoch_records(writer.directory)
    assert [(r["kind"], r["seq"]) for r in records] == [("alias", 0), ("goodbye", 1)]
    status = json.loads((writer.directory / "status.json").read_text())
    assert (status["errors"], status["dropped"]) == (3, {})


def _scheduled(iteration: int, members: int) -> dict[str, Any]:
    member = {
        "sighting": "repeat",
        "phase": "generation",
        "scheduled": 1,
        "computed_before": 600,
        "prompt_tokens": 512,
        "prefill_scheduled": 0,
        "past_prompt_scheduled": 1,
        "drafts_scheduled": 0,
        "cached_at_admission": None,
        "recompute": False,
        "output_before": 88,
        "resumable": False,
    }
    return {
        "iteration": str(iteration),
        "members": [
            {"internal": f"chatcmpl-stormlog-{iteration}-{n}-0f3a9c1d", **member}
            for n in range(members)
        ],
    }


def _queue_memory(root: str) -> str:
    """Fill a held writer's queue; what tracemalloc saw it keep, as JSON."""
    limits = WriterLimits(queue_records=100_000, queue_bytes=1 << 20)
    writer = EpochWriter(Path(root), "engine", limits=limits)
    tracemalloc.start()
    with writer._condition:  # the writer thread cannot take any
        before = tracemalloc.get_traced_memory()[0]
        for iteration in range(1000):
            writer.emit("scheduled", _scheduled(iteration, 32))
        held = tracemalloc.get_traced_memory()[0] - before
        queued = len(writer._queue)
    tracemalloc.stop()
    writer.close()
    dropped = writer._counters.dropped["scheduled"]
    limit = limits.queue_bytes
    return json.dumps(
        {"held": held, "queued": queued, "dropped": dropped, "limit": limit}
    )


def test_the_queue_bytes_bound_its_memory(tmp_path: Path) -> None:
    # In a fresh process: tracemalloc counts every thread's allocations, and
    # other tests leave threads behind.
    code = f"from tests.test_vllm_hook import _queue_memory as m; print(m({str(tmp_path)!r}))"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    memory = json.loads(result.stdout.splitlines()[-1])

    assert memory["dropped"] > 0  # more was offered than fits
    # A queued record holds its JSON text and a small fixed overhead, no more.
    assert memory["held"] <= memory["limit"] + 200 * memory["queued"]


def _counting_bodies(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record every set of fields the writer serializes."""
    serialized: list[object] = []
    body = EpochWriter._body

    def counting(writer: EpochWriter, fields: dict[str, Any]) -> str | None:
        serialized.append(fields)
        return body(writer, fields)

    monkeypatch.setattr(EpochWriter, "_body", counting)
    return serialized


def test_a_record_the_queue_cannot_take_is_never_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serialized = _counting_bodies(monkeypatch)
    limits = WriterLimits(queue_records=2, heartbeat_seconds=3600)
    writer = EpochWriter(tmp_path, "engine", limits=limits)
    with writer._condition:  # the writer thread cannot take any
        for index in range(5):
            writer.emit("alias", {"internal": f"r{index}"})
    writer.close()
    writer.emit("alias", {"internal": "after close"})

    # Only the two records that fit were serialized; the rest cost nothing.
    assert serialized == [{"internal": "r0"}, {"internal": "r1"}]
    assert writer._status()["dropped"] == {"alias": 4}


def test_a_record_oversized_by_its_ids_alone_is_never_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serialized = _counting_bodies(monkeypatch)
    writer = EpochWriter(
        tmp_path, "engine", limits=WriterLimits(heartbeat_seconds=3600)
    )

    def step(members: int, id_length: int) -> dict[str, Any]:
        internal = f"chatcmpl-{'x' * id_length}"
        return {
            "iteration": "1",
            "members": [{"internal": f"{internal}-{n}"} for n in range(members)],
        }

    # Clients choose request IDs: 4 and 16 MiB of them in one step.
    writer.emit("scheduled", step(256, 16384))
    writer.emit("completed", step(1024, 16384))
    writer.emit("scheduled", {"iteration": "2", "preempted": ["y" * (5 << 20)]})
    writer.emit("alias", {"internal": "z" * (5 << 20)})
    kept = step(256, 4096)  # 1 MiB: it fits, so it is serialized and written
    writer.emit("scheduled", kept)
    writer.close()

    assert serialized == [kept]
    records = _epoch_records(writer.directory)
    assert [(r["kind"], len(r.get("members", ()))) for r in records] == [
        ("scheduled", 256),
        ("goodbye", 0),
    ]
    status = json.loads((writer.directory / "status.json").read_text())
    assert status["dropped"] == {
        "scheduled_oversized": 2,
        "completed_oversized": 1,
        "alias_oversized": 1,
    }


def test_a_record_after_the_disk_cap_is_never_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serialized = _counting_bodies(monkeypatch)
    writer = EpochWriter(tmp_path, "engine", limits=WriterLimits(max_bytes=100))
    writer.emit("alias", {"internal": "x" * 200})  # its line passes the cap
    _wait(lambda: writer._status()["capped"])
    writer.emit("alias", {"internal": "y"})
    writer.close()

    assert serialized == [{"internal": "x" * 200}]
    status = json.loads((writer.directory / "status.json").read_text())
    assert status["dropped"]["alias"] == 2


def _engine_step(writer: EpochWriter, members: int) -> Callable[[], None]:
    """One scheduler step of ``members`` decoding requests, through the recorder."""
    ids = [
        f"chatcmpl-stormlog-run-1-c1_measured_0_{n}-0f3a9c1d" for n in range(members)
    ]
    scheduler = types.SimpleNamespace(
        requests={internal: FakeRequest(internal, 512) for internal in ids},
        num_sampled_tokens_per_step=1,
    )
    output = SchedulerOutput(
        [], CachedRequestData(ids, [600] * members), dict.fromkeys(ids, 1), members
    )
    sampled = ModelRunnerOutput(
        {internal: index for index, internal in enumerate(ids)}, [[7] for _ in ids]
    )
    result = {0: EngineCoreOutputs([EngineCoreOutput(i, [7]) for i in ids])}
    recorder = EngineRecorder(writer, "vllm:h:b:1:1")

    def step() -> None:
        recorder.on_schedule(scheduler, output, writer_module.stamp())
        before = recorder.before_update(scheduler, output, sampled)
        recorder.after_update(scheduler, output, before, result=result)

    return step


@pytest.mark.benchmark
def test_the_engine_step_cost_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Report the engine thread's cost per scheduler step; never assert it.

    CI timing is too noisy for a bound, so it runs only when selected:
    ``pytest -m benchmark -rP`` prints the figures, and the JUnit XML keeps
    them as properties. ``emit`` is the time spent in ``EpochWriter.emit``,
    queueing the step's two records.
    """
    writer = EpochWriter(tmp_path, "engine")
    silent = EpochWriter(tmp_path, "worker")
    monkeypatch.setattr(silent, "emit", lambda kind, fields: None)
    report = []
    for members in (8, 64, 256):
        step = _best_seconds(_engine_step(writer, members))
        emit = step - _best_seconds(_engine_step(silent, members))
        record_property(f"step_us_{members}_members", round(step * 1e6, 1))
        record_property(f"emit_us_{members}_members", round(emit * 1e6, 1))
        report.append(f"{members} members {step * 1e6:.0f} us (emit {emit * 1e6:.0f})")
    writer.close()
    silent.close()
    print("engine thread per step:", "; ".join(report))

    status = json.loads((writer.directory / "status.json").read_text())
    assert (status["dropped"], status["errors"]) == ({}, 0)


def _best_seconds(step: Callable[[], None]) -> float:
    return min(timeit.repeat(step, number=3, repeat=10)) / 3


def test_an_escaped_record_over_the_limit_is_dropped(tmp_path: Path) -> None:
    # 400 emoji are 400 characters but 4,800 bytes of JSON escapes.
    writer = EpochWriter(tmp_path, "engine", limits=WriterLimits(record_bytes=4096))
    writer.emit("alias", {"internal": "\U0001f600" * 400})
    writer.close()

    status = json.loads((writer.directory / "status.json").read_text())
    assert status["dropped"] == {"alias_oversized": 1}


def test_a_status_field_json_cannot_write_is_an_error(tmp_path: Path) -> None:
    writer = EpochWriter(
        tmp_path,
        "worker",
        limits=WriterLimits(heartbeat_seconds=0.02),
        status_fields=lambda: {"bad": object()},
    )
    # Each heartbeat counts two errors: its record and its status file.
    _wait(lambda: writer._status()["errors"] >= 4)
    writer.emit("alias", {"internal": "x"})
    writer.close()

    # The writer thread lived on: the record after those heartbeats is written.
    kinds = [record["kind"] for record in _epoch_records(writer.directory)]
    assert kinds == ["alias", "goodbye"]
    assert not (writer.directory / "status.json").exists()


def test_goodbye_is_the_last_record(tmp_path: Path) -> None:
    # Every pass of the writer is past a zero heartbeat interval.
    writer = EpochWriter(tmp_path, "engine", limits=WriterLimits(heartbeat_seconds=0))
    for index in range(3):
        writer.emit("alias", {"internal": f"r{index}"})
    writer.close()

    records = _epoch_records(writer.directory)
    kinds = [record["kind"] for record in records]
    assert kinds.count("goodbye") == 1 and kinds[-1] == "goodbye"
    # goodbye names its own number, the epoch's last.
    assert records[-1]["last_seq"] == records[-1]["seq"] == len(records) - 1


def test_a_backlog_does_not_hold_up_the_status_or_sealing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fast = threading.Event()
    write = writer_module._Segment.write

    def slow(segment: Any, line: bytes) -> bool:
        if not fast.is_set():
            time.sleep(0.02)
        return write(segment, line)

    monkeypatch.setattr(writer_module._Segment, "write", slow)
    writer = EpochWriter(
        tmp_path,
        "engine",
        limits=WriterLimits(heartbeat_seconds=0.05, segment_bytes=1024),
    )
    for index in range(200):
        writer.emit("alias", {"internal": f"request-{index}"})
    # Writing the backlog takes 4 s; the status is due after 0.05 s.
    _wait((writer.directory / "status.json").exists, seconds=2)
    fast.set()
    writer.close()

    segments = sorted(writer.directory.glob("*.jsonl"))
    longest = max(
        len(line) for path in segments for line in path.read_bytes().splitlines()
    )
    assert max(path.stat().st_size for path in segments) <= 1024 + longest + 1


def test_a_heartbeat_counts_the_records_it_is_written_ahead_of(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a slow disk a heartbeat is written while records accepted before
    its stamp still wait; it says how many, so a reader knows the span it
    closes reaches past them."""
    write = writer_module._Segment.write

    def slow(segment: Any, line: bytes) -> bool:
        if b'"kind":"heartbeat"' not in line:
            time.sleep(0.02)
        return write(segment, line)

    monkeypatch.setattr(writer_module._Segment, "write", slow)
    writer = EpochWriter(
        tmp_path, "engine", limits=WriterLimits(heartbeat_seconds=0.05)
    )
    for index in range(10):
        writer.emit("filler", {"index": index})
    accepted = writer_module.stamp()
    writer.emit("pause", {"from": "UNPAUSED", "to": "PAUSED_ALL", **accepted})
    for index in range(10):
        writer.emit("filler", {"index": 10 + index})
    writer.close()

    records = _epoch_records(writer.directory)
    pause = next(r for r in records if r["kind"] == "pause")
    ahead = [
        r
        for r in records
        if r["kind"] == "heartbeat"
        and r["mono_ns"] > accepted["mono_ns"]
        and r["seq"] < pause["seq"]
    ]
    assert ahead  # written after the pause was accepted, before it was
    for beat in ahead:
        assert beat["seq"] + beat["pending"] >= pause["seq"]


def test_a_heartbeat_counts_a_record_still_being_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record's stamps are taken before emit, which serializes it outside
    the lock (milliseconds for a large step): a heartbeat stamped meanwhile
    is written ahead of it and must count it as pending."""
    dumps = writer_module._dumps
    sizing, release = threading.Event(), threading.Event()

    def slow(value: Any) -> str:
        if isinstance(value, dict) and value.get("to") == "PAUSED_ALL":
            sizing.set()
            release.wait(5)
        return dumps(value)

    monkeypatch.setattr(writer_module, "_dumps", slow)
    writer = EpochWriter(tmp_path, "engine")
    accepted = writer_module.stamp()
    emitting = threading.Thread(
        target=writer.emit,
        args=("pause", {"from": "UNPAUSED", "to": "PAUSED_ALL", **accepted}),
    )
    emitting.start()
    assert sizing.wait(5)
    try:
        assert writer._status()["pending"] == 1
    finally:
        release.set()
        emitting.join(5)
    writer.close()

    assert [r["kind"] for r in _epoch_records(writer.directory)].count("pause") == 1


def test_a_record_is_pending_from_before_its_last_stamp(
    vllm: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A span vouches for the instants before its closing heartbeat's stamp,
    and a record is matched by its last stamp, so the record must count as
    pending from before that stamp is taken, not only once emit takes the
    lock: a heartbeat stamped in between (a GIL handoff, or vLLM's own
    add_request after an enqueue's stamp) would be written ahead of it with
    nothing pending."""
    scheduler = vllm["Scheduler"](vllm_config())
    core = vllm["EngineCore"](scheduler)
    writer = getattr(scheduler, hook.RECORDER_ATTRIBUTE).writer
    _wait(lambda: writer._status()["pending"] == 0)  # the hello is written
    # From here the writer thread writes nothing, so pending only grows.
    written = threading.Event()
    write = writer_module._Segment.write

    def held(segment: Any, line: bytes) -> bool:
        written.wait(10)
        return write(segment, line)

    monkeypatch.setattr(writer_module._Segment, "write", held)
    pending_at: dict[int, int] = {}

    def stamped() -> dict[str, int]:
        value = writer_module.stamp()
        pending_at[value["mono_ns"]] = writer._status()["pending"]
        return value

    monkeypatch.setattr(engine_module, "stamp", stamped)
    monkeypatch.setattr(hook, "stamp", stamped)
    output = SchedulerOutput([_new("a-1", 2)], CachedRequestData(), {"a-1": 2}, 2)
    scheduler.next_output = output
    request = FakeRequest("a-1", 2)
    steps: list[Callable[[], object]] = [
        lambda: core.preprocess_add_request(
            types.SimpleNamespace(request_id="a-1", external_req_id="chatcmpl-a")
        ),
        lambda: scheduler.add_request(request),
        scheduler.schedule,
        lambda: scheduler.update_from_output(
            output, ModelRunnerOutput({"a-1": 0}, [[5]])
        ),
        lambda: scheduler.set_pause_state(PauseState.PAUSED_ALL),
        scheduler.reset_prefix_cache,
        lambda: scheduler._free_request(request),
    ]
    before = []
    for step in steps:
        before.append(writer._status()["pending"])
        step()
    written.set()

    records = _records(vllm["root"], "engine")
    unstamped = ("hello", "heartbeat", "goodbye")
    stamped_records = [r for r in records if r["kind"] not in unstamped]
    assert [r["kind"] for r in stamped_records] == [
        "alias",
        "enqueued",
        "scheduled",
        "completed",
        "pause",
        "cache_reset",
        "terminal",
    ]
    for record, pending in zip(stamped_records, before):
        last = record.get("end_mono_ns", record.get("mono_ns"))
        assert pending_at[last] == pending + 1, record["kind"]


def test_emit_never_lets_go_of_a_record_it_has_not_queued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record stops counting as pending in the same hold of the lock that
    queues or drops it: whenever the emitting thread lets the lock go, the
    record is still reserved, queued or dropped, so a heartbeat stamped at
    any moment counts it one way or another."""
    emitting = threading.Event()
    unaccounted: list[str] = []

    class Checked(threading.Condition):
        def __exit__(self, *args: Any) -> None:
            if emitting.is_set() and threading.current_thread() is caller:
                if not (
                    writer._reserved
                    or any(kind == "probe" for kind, _ in writer._queue)
                    or writer._counters.dropped["probe"]
                ):
                    unaccounted.append("lock released with the record uncounted")
            super().__exit__(*args)

    caller = threading.current_thread()
    with monkeypatch.context() as patched:
        patched.setattr(writer_module.threading, "Condition", Checked)
        writer = EpochWriter(tmp_path, "engine")
    emitting.set()
    writer.emit("probe", {"index": 0})
    emitting.clear()
    writer.close()

    assert unaccounted == []
    assert [r["kind"] for r in _epoch_records(writer.directory)] == [
        "probe",
        "goodbye",
    ]


def test_a_short_write_leaves_only_whole_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write = os.write
    disk: dict[str, int] = {}

    def full_disk(fd: int, data: Any) -> int:
        # 20 bytes of the marked line land, then the disk is full exactly once.
        if disk.get("full") == fd:
            disk["full"] = -1
            raise OSError(errno.ENOSPC, "No space left on device")
        if "full" not in disk and b"short-write" in bytes(data):
            disk["full"] = fd
            return write(fd, bytes(data)[:20])
        return write(fd, data)

    monkeypatch.setattr(os, "write", full_disk)
    writer = EpochWriter(tmp_path, "engine")
    writer.emit("alias", {"internal": "short-write"})
    writer.emit("alias", {"internal": "after"})
    writer.close()

    records = _epoch_records(writer.directory)
    assert [(r["kind"], r["seq"]) for r in records] == [("alias", 0), ("goodbye", 1)]
    assert records[0]["internal"] == "after"
    status = json.loads((writer.directory / "status.json").read_text())
    assert (status["errors"], status["dropped"]) == (1, {"alias": 1})


def test_a_failed_seal_is_counted_retried_and_not_acknowledged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replace = os.replace
    refused: list[str] = []
    refusing = threading.Event()
    refusing.set()

    def flaky(source: Any, target: Any) -> None:
        if str(source).endswith(".jsonl.part") and refusing.is_set():
            refused.append(str(source))
            raise PermissionError("rename refused")
        replace(source, target)

    monkeypatch.setattr(os, "replace", flaky)
    writer = EpochWriter(
        tmp_path, "engine", limits=WriterLimits(heartbeat_seconds=0.05)
    )
    writer.emit("alias", {"internal": "x"})
    flush = writer.directory / "flush"
    flush.touch()
    _wait(lambda: len(refused) >= 2)
    assert flush.exists() and list(writer.directory.glob("*.part"))

    refusing.clear()
    _wait(lambda: not flush.exists())
    writer.close()

    assert list(writer.directory.glob("*.part")) == []
    records = _epoch_records(writer.directory)
    assert [r["seq"] for r in records] == list(range(len(records)))
    assert [r["kind"] for r in records if r["kind"] != "heartbeat"] == [
        "alias",
        "goodbye",
    ]
    status = json.loads((writer.directory / "status.json").read_text())
    assert status["errors"] >= 2


def test_a_forked_worker_finalizes_its_log_on_a_normal_exit(tmp_path: Path) -> None:
    # multiprocessing ends a forked child with os._exit, which skips atexit.
    script = (
        "import multiprocessing, pathlib, sys\n"
        "from stormlog.infer.vllm_hook.writer import EpochWriter\n"
        "def child(root):\n"
        "    EpochWriter(pathlib.Path(root), 'worker').emit('alias', {'internal': 'x'})\n"
        "process = multiprocessing.get_context('fork').Process(\n"
        "    target=child, args=(sys.argv[1],))\n"
        "process.start()\n"
        "process.join(30)\n"
        "sys.exit(process.exitcode)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    (epoch,) = tmp_path.glob("*/worker-*")
    assert list(epoch.glob("*.part")) == []
    kinds = [r["kind"] for r in _epoch_records(epoch) if r["kind"] != "heartbeat"]
    assert kinds == ["alias", "goodbye"]
    assert (epoch / "status.json").exists()


def _epoch_records(directory: Path) -> list[dict[str, Any]]:
    return [
        _loads(line)
        for path in sorted(directory.glob("*.jsonl"))
        for line in path.read_text().splitlines()
    ]


def _queued_bytes(writer: EpochWriter) -> int:
    with writer._condition:
        return writer._queued_bytes


def _wait(condition: Callable[[], bool], seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_a_forked_process_gets_its_own_writer(
    vllm: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = hook._Settings(
        vllm["root"], nvtx=False, retain_hours=24, max_bytes=1 << 20
    )
    parent = hook.writer_for(settings, "engine")
    monkeypatch.setattr(os, "getpid", lambda: parent.pid + 1)

    child = hook.writer_for(settings, "engine")

    assert child is not parent and child.epoch != parent.epoch


# ---------------------------------------------------------------- entry point


def test_the_entry_point_is_inert_and_light_without_its_variable() -> None:
    env = {k: v for k, v in os.environ.items() if k != hook.ENV_DIR}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, stormlog.vllm_hook as h; h.register(); "
            "print('stormlog.infer.vllm_hook' in sys.modules, 'torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )

    assert result.stdout.split() == ["False", "False"], result.stderr


def test_pyproject_registers_the_vllm_plugin() -> None:
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()

    assert '[project.entry-points."vllm.general_plugins"]' in text
    assert 'stormlog = "stormlog.vllm_hook:register"' in text
