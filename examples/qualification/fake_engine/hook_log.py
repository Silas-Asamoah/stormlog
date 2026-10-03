"""The execution hook's raw log, written as the real hook writes it.

Records go through Stormlog's own ``EpochWriter`` in the
``stormlog.vllm_hook/1`` format (``docs/vllm_execution.md``): an engine epoch
and a worker epoch from one process, as vLLM's ``uni`` executor runs them.
Fields that later hook versions add come here in the change that adds them to
the hook, so the fake never describes a format the hook does not write.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from stormlog.infer.vllm_hook import producer_name
from stormlog.infer.vllm_hook.writer import EpochWriter, WriterLimits

from .config import VLLM_VERSION, FakeEngineConfig
from .engine import EngineObserver, FakeRequest, ScheduledMember, Step

SCHEDULER = "vllm.v1.core.sched.scheduler.Scheduler"
RUNNER = "vllm.v1.worker.gpu.model_runner.GPUModelRunner"


class HookLog(EngineObserver):
    """One engine epoch and one worker epoch for the fake engine."""

    def __init__(self, root: Path, config: FakeEngineConfig) -> None:
        self.config = config
        limits = WriterLimits(seal_seconds=config.hook_seal_seconds)
        self.engine_writer = EpochWriter(root, "engine", limits=limits)
        self.producer = producer_name(self.engine_writer)
        self.engine_writer.emit(
            "hello",
            self._hello(
                self.engine_writer, self.producer, scheduler=SCHEDULER, runner=None
            ),
        )
        self.worker_writer = EpochWriter(
            root, "worker", limits=limits, status_fields=_worker_status
        )
        # Each side's gate sees only its own class, so the other stays null.
        hello = self._hello(self.worker_writer, None, scheduler=None, runner=RUNNER)
        hello.update(
            {
                "rank": {"global": 0, "tp": 0, "pp": 0, "dp": 0},
                "local_rank": 0,
                "cuda_ordinal": 0,
                "device_uuid": config.device_uuid,
                "trace_rank_suffix": "rank0",
            }
        )
        self.worker_writer.emit("hello", hello)

    def close(self) -> None:
        self.engine_writer.close()
        self.worker_writer.close()

    # ------------------------------------------------------------ records

    def on_admit(self, request: FakeRequest) -> None:
        self.engine_writer.emit(
            "alias",
            {
                "internal": request.internal_id,
                "external": request.external_id,
                **_stamp(),
            },
        )

    def on_scheduled(self, step: Step) -> None:
        self.engine_writer.emit(
            "scheduled",
            {
                "iteration": str(step.iteration),
                "start_wall_ns": step.start_wall_ns,
                "start_mono_ns": step.start_mono_ns,
                "end_wall_ns": step.end_wall_ns,
                "end_mono_ns": step.end_mono_ns,
                "total_tokens": step.total_tokens,
                "zero_token": step.total_tokens == 0,
                "preempted": sorted(step.preempted),
                "members": [_scheduled_member(member) for member in step.members],
            },
        )

    def on_completed(self, step: Step) -> None:
        self.engine_writer.emit(
            "completed",
            {
                "iteration": str(step.iteration),
                **_stamp(),
                "members": [_completed_member(member) for member in step.members],
            },
        )

    def on_free(self, request: FakeRequest) -> None:
        self.engine_writer.emit(
            "terminal",
            {
                "internal": request.internal_id,
                "status": request.status,
                "finish_reason": request.finish_reason,
                "output_tokens": request.output_tokens,
                **_stamp(),
            },
        )

    # ------------------------------------------------------------ helpers

    def _hello(
        self,
        writer: EpochWriter,
        producer: str | None,
        *,
        scheduler: str | None,
        runner: str | None,
    ) -> dict[str, Any]:
        wall = time.time_ns()
        mono = time.monotonic_ns()
        wall_after = time.time_ns()
        return {
            "role": writer.role,
            "host": writer.host,
            "boot_id": writer.boot_id,
            "pid": writer.pid,
            "start_ns": writer.start_ns,
            "vllm_version": VLLM_VERSION,
            "enabled": True,
            "refused": None,
            "producer": producer,
            "config": {
                "vllm_version": VLLM_VERSION,
                "executor": "uni",
                "tp": 1,
                "pp": 1,
                "dp": 1,
                "async_scheduling": False,
                "max_num_batched_tokens": self.config.max_num_batched_tokens,
                "speculative": None,
                "v2_model_runner": True,
                "request_id_randomization": self.config.request_id_randomization,
                "scheduler": scheduler,
                "runner": runner,
            },
            "clock": {"wall_ns": wall, "mono_ns": mono, "gap_ns": wall_after - wall},
        }


def _scheduled_member(member: ScheduledMember) -> dict[str, Any]:
    request = member.request
    prompt = request.prompt_len
    prefill = max(0, min(member.tokens, prompt - member.computed_before))
    return {
        "internal": request.internal_id,
        "sighting": "first" if member.first_sighting else "repeat",
        "phase": "context" if member.context else "generation",
        "scheduled": member.tokens,
        "computed_before": member.computed_before,
        "prompt_tokens": prompt,
        "prefill_scheduled": prefill,
        "past_prompt_scheduled": member.tokens - prefill,
        "drafts_scheduled": 0,
        "cached_at_admission": (
            member.computed_before if member.first_sighting else None
        ),
        "recompute": member.recompute,
        "output_before": member.output_before,
        "resumable": False,
    }


def _completed_member(member: ScheduledMember) -> dict[str, Any]:
    kept = member.outcome == "kept"
    return {
        "internal": member.request.internal_id,
        "outcome": member.outcome,
        "stale": member.outcome == "dropped_stale",
        "sampled": member.sampled,
        "accepted_drafts": 0,
        "retained": member.sampled if kept else None,
        "finish_reason": member.finish_reason if kept else None,
        "computed_after": (member.computed_before + member.tokens) if kept else None,
    }


def _worker_status() -> dict[str, Any]:
    return {"range_misses": 0, "startup_unranged": 0, "pending_samples": 0}


def _stamp() -> dict[str, int]:
    return {"wall_ns": time.time_ns(), "mono_ns": time.monotonic_ns()}


__all__ = ["RUNNER", "SCHEDULER", "HookLog"]
