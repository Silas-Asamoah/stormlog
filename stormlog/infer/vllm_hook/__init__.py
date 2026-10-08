"""Install the vLLM execution hook's patches; see ``docs/vllm_execution.md``.

``install`` patches classes once per process: the scheduler, the engine core's
admission, and the GPU worker's device setup. Whatever the process role, a
multiproc executor that forks its workers hands them the already patched
classes, while a spawned worker runs the plugin again. Each patched call
records only for an instance whose configuration passed the gate, runs vLLM's
own code exactly once, and lets its exceptions through unchanged; telemetry
failures are counted and never raised.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..host_clock import host_boot_id
from . import gate
from .engine import EngineRecorder
from .process import process_fields
from .worker import RunnerRecorder, worker_identity
from .writer import EpochWriter, WriterLimits, remove_old_epochs

ENV_DIR = "STORMLOG_VLLM_HOOK_DIR"
ENV_NVTX = "STORMLOG_VLLM_HOOK_NVTX"
ENV_RETAIN_HOURS = "STORMLOG_VLLM_HOOK_RETAIN_HOURS"
ENV_MAX_BYTES = "STORMLOG_VLLM_HOOK_MAX_BYTES"
RECORDER_ATTRIBUTE = "_stormlog_recorder"
# The optional record kinds a patched class emits; only methods the installed
# vLLM has are patched, so only their kinds are claimed.
OBSERVES_ATTRIBUTE = "_stormlog_observes"

_LOCK = threading.Lock()
_PATCHED = False
_WRITERS: dict[tuple[int, str], EpochWriter] = {}


def install(environ: Mapping[str, str] = os.environ) -> list[str]:
    """Patch whichever vLLM classes are importable; return what was patched."""
    global _PATCHED
    with _LOCK:
        if _PATCHED:
            return []
        _PATCHED = True
    settings = _Settings.from_environ(environ)
    patched = []
    for name, patch in (("engine", _patch_engine), ("worker", _patch_worker)):
        try:
            patch(settings)
            patched.append(name)
        except ImportError:
            continue
    return patched


class _Settings:
    def __init__(self, root: Path, *, nvtx: bool, retain_hours: float, max_bytes: int):
        self.root = root
        self.nvtx = nvtx
        self.retain_hours = retain_hours
        self.limits = WriterLimits(max_bytes=max_bytes)

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> _Settings:
        return cls(
            Path(environ[ENV_DIR]),
            nvtx=environ.get(ENV_NVTX) == "1",
            retain_hours=float(environ.get(ENV_RETAIN_HOURS) or 24),
            max_bytes=int(environ.get(ENV_MAX_BYTES) or 256 * 1024 * 1024),
        )


def writer_for(
    settings: _Settings,
    role: str,
    status_fields: Callable[[], dict[str, Any]] | None = None,
) -> EpochWriter:
    """One writer per process and role; a forked child gets its own."""
    key = (os.getpid(), role)
    with _LOCK:
        writer = _WRITERS.get(key)
        if writer is None:
            if not any(pid == key[0] for pid, _ in _WRITERS):
                remove_old_epochs(settings.root, settings.retain_hours)
            writer = EpochWriter(
                settings.root, role, limits=settings.limits, status_fields=status_fields
            )
            _WRITERS[key] = writer
        return writer


def producer_name(writer: EpochWriter) -> str:
    """Names one scheduler's iterations: unique per host boot and process."""
    boot = host_boot_id() or "noboot"
    return f"vllm:{socket.gethostname()}:{boot}:{writer.pid}:{writer.start_ns}"


# ---------------------------------------------------------------- engine side


def _patch_engine(settings: _Settings) -> None:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCore

    init = Scheduler.__init__
    schedule = Scheduler.schedule
    update = Scheduler.update_from_output
    free = Scheduler._free_request
    admit = EngineCore.preprocess_add_request

    def scheduler_init(self: Any, vllm_config: Any, *args: Any, **kwargs: Any) -> None:
        init(self, vllm_config, *args, **kwargs)
        _guard(None, lambda: _enable_engine(self, vllm_config, settings))

    def scheduler_schedule(self: Any, *args: Any, **kwargs: Any) -> Any:
        recorder = getattr(self, RECORDER_ATTRIBUTE, None)
        start = (time.time_ns(), time.monotonic_ns())
        output = schedule(self, *args, **kwargs)
        if recorder is not None:
            _guard(recorder, lambda: recorder.on_schedule(self, output, start))
        return output

    def scheduler_update(
        self: Any,
        scheduler_output: Any,
        model_runner_output: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        recorder = getattr(self, RECORDER_ATTRIBUTE, None)
        if recorder is None:
            return update(self, scheduler_output, model_runner_output, *args, **kwargs)
        before = _guard(
            recorder,
            lambda: recorder.before_update(self, scheduler_output, model_runner_output),
        )
        try:
            result = update(
                self, scheduler_output, model_runner_output, *args, **kwargs
            )
        except BaseException:
            _guard(
                recorder,
                lambda: recorder.after_update(
                    self, scheduler_output, before or {}, failed=True
                ),
            )
            raise
        _guard(
            recorder,
            lambda: recorder.after_update(
                self, scheduler_output, before or {}, result=result
            ),
        )
        return result

    def scheduler_free(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        recorder = getattr(self, RECORDER_ATTRIBUTE, None)
        if recorder is not None:
            _guard(recorder, lambda: recorder.on_free(request))
        return free(self, request, *args, **kwargs)

    def engine_admit(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        recorder = getattr(getattr(self, "scheduler", None), RECORDER_ATTRIBUTE, None)
        if recorder is not None:
            _guard(recorder, lambda: recorder.on_admit(request))
        return admit(self, request, *args, **kwargs)

    setattr(Scheduler, "__init__", scheduler_init)
    setattr(Scheduler, "schedule", scheduler_schedule)
    setattr(Scheduler, "update_from_output", scheduler_update)
    setattr(Scheduler, "_free_request", scheduler_free)
    setattr(EngineCore, "preprocess_add_request", engine_admit)


def _enable_engine(scheduler: Any, vllm_config: Any, settings: _Settings) -> None:
    result = gate.check(vllm_config, scheduler=scheduler)
    writer = writer_for(settings, "engine")
    producer = producer_name(writer)
    observes = (
        getattr(type(scheduler), OBSERVES_ATTRIBUTE, ()) if result.enabled else ()
    )
    writer.emit("hello", _hello(writer, result, producer=producer, observes=observes))
    if result.enabled:
        recorder = EngineRecorder(writer, producer)
        setattr(scheduler, RECORDER_ATTRIBUTE, recorder)


# ---------------------------------------------------------------- worker side


def _patch_worker(settings: _Settings) -> None:
    from vllm.v1.worker.gpu_worker import Worker

    init_device = Worker.init_device

    def worker_init_device(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = init_device(self, *args, **kwargs)
        _guard(None, lambda: _enable_worker(self, settings))
        return result

    setattr(Worker, "init_device", worker_init_device)


def _enable_worker(worker: Any, settings: _Settings) -> None:
    runner = getattr(worker, "model_runner", None)
    result = gate.check(worker.vllm_config, runner=runner)
    recorders: list[RunnerRecorder] = []
    writer = writer_for(
        settings,
        "worker",
        lambda: recorders[0].status_fields() if recorders else {},
    )
    hello = _hello(writer, result, producer=None)
    hello.update(worker_identity(worker))
    writer.emit("hello", hello)
    if result.enabled and runner is not None:
        recorder = RunnerRecorder(writer, nvtx=settings.nvtx)
        recorder.wrap(runner)
        recorders.append(recorder)


# ---------------------------------------------------------------- shared


def _hello(
    writer: EpochWriter,
    result: gate.GateResult,
    *,
    producer: str | None,
    observes: tuple[str, ...] = (),
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
        **process_fields(),
        "vllm_version": result.config.get("vllm_version"),
        "enabled": result.enabled,
        "refused": result.refused,
        "producer": producer,
        "observes": sorted(observes),
        "config": result.config,
        "clock": {"wall_ns": wall, "mono_ns": mono, "gap_ns": wall_after - wall},
    }


def _guard(recorder: Any, action: Callable[[], Any]) -> Any:
    """Run telemetry; count a failure on the recorder's writer, never raise."""
    try:
        return action()
    except Exception:
        writer = getattr(recorder, "writer", None)
        if writer is not None:
            try:
                writer.count_error()
            except Exception:
                pass
        return None


__all__ = ["ENV_DIR", "install", "producer_name", "writer_for"]
