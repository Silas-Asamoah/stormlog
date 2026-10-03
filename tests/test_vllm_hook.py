"""The vLLM execution hook, against fakes shaped like vLLM 0.30.0's classes."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import stormlog.infer.vllm_hook as hook
from stormlog.infer.vllm_hook import gate
from stormlog.infer.vllm_hook.engine import ITERATION_ATTRIBUTE
from stormlog.infer.vllm_hook.worker import RunnerRecorder
from stormlog.infer.vllm_hook.writer import EpochWriter, WriterLimits

# ---------------------------------------------------------------- vLLM fakes


@dataclass
class FakeRequest:
    request_id: str
    num_prompt_tokens: int
    num_output_tokens: int = 0
    num_stale_output_tokens: int = 0
    drop_stale_output: bool = False
    finished: bool = False
    status: Any = None

    def is_finished(self) -> bool:
        return self.finished

    def get_finished_reason(self) -> str | None:
        return "stop" if self.finished else None


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

        def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
            assert self.next_output is not None
            return self.next_output

        def update_from_output(
            self, scheduler_output: Any, model_runner_output: Any
        ) -> str:
            for internal in scheduler_output.num_scheduled_tokens:
                request = self.requests.get(internal)
                if request is None or request.finished:
                    continue
                index = model_runner_output.req_id_to_index[internal]
                request.num_output_tokens += len(
                    model_runner_output.sampled_token_ids[index]
                )
            if self.update_error is not None:
                raise self.update_error
            return "outputs"

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
    modules = {
        "vllm": types.ModuleType("vllm"),
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
    }
    for writer in list(hook._WRITERS.values()):
        writer.close()


def _records(root: Path, role: str) -> list[dict[str, Any]]:
    for writer in hook._WRITERS.values():
        if writer.role == role:
            writer.close()
    records = []
    for path in sorted(root.glob(f"*/{role}-*/*.jsonl")):
        records += [json.loads(line) for line in path.read_text().splitlines()]
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
    scheduler.update_from_output(
        second, ModelRunnerOutput({"a-1": 0, "b-1": 1}, [[3], [4]])
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
    assert second_members["a-1"]["decode_scheduled"] == 1
    completed = _of(records, "completed")
    assert [m["outcome"] for m in completed[0]["members"]] == ["kept", "kept"]
    assert [m["retained"] for m in completed[0]["members"]] == [1, 1]
    assert completed[0]["members"][0]["computed_after"] == 4
    outcomes = {m["internal"]: m["outcome"] for m in completed[1]["members"]}
    assert outcomes == {"a-1": "discarded_finished", "b-1": "kept"}
    assert _of(records, "terminal")[0]["internal"] == "a-1"
    assert [r["seq"] for r in records] == list(range(len(records)))


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
    assert len(_of(records, "completed")) == 1  # recorded although vLLM raised
    status = json.loads(next(vllm["root"].glob("*/engine-*/status.json")).read_text())
    assert status["errors"] >= 1


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
    runner.execute_model(SchedulerOutput([], CachedRequestData(), {}, 0))  # warm-up

    assert runner.execute_model(step) is None
    assert runner.sample_tokens(None) == "sampled"
    runner._dummy_run(1)
    runner.execute_model(SchedulerOutput([], CachedRequestData(), {}, 0))  # no identity
    with pytest.raises(RuntimeError, match="model failed"):
        runner.execute_model("boom")
    writer.close()

    # The step and its sampling share one range; the dummy run reached vLLM unranged.
    assert opened == [("vllm:h:b:1:2", "7"), ("vllm:h:b:1:2", "7")]
    assert ("execute", True) in runner.calls
    assert (recorder.startup_unranged, recorder.range_misses) == (1, 2)
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
        json.loads(line)
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


def test_a_full_queue_drops_and_counts(tmp_path: Path) -> None:
    writer = EpochWriter(tmp_path, "engine", limits=WriterLimits(queue_bytes=300))
    for _ in range(50):
        writer.emit("scheduled", {}, size_hint=200)
    writer.close()

    status = json.loads((writer.directory / "status.json").read_text())
    assert status["dropped"].get("scheduled", 0) > 0


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
