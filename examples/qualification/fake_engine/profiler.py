"""vLLM's profiler routes: a window opened and closed between steps.

``/start_profile`` and ``/stop_profile`` run on the step loop, as vLLM runs its
utility calls, so the stop's pause (vLLM writes the trace inside the call)
holds serving. The trace is a small gzipped Kineto file, one iteration range
per profiled step with a launch and a kernel inside it; when the hook log is
on, its ranges name the hook's own iterations, so an import links the work
to the steps. Like torch's export, the write streams into the trace's final
name, so a reader can find it present but truncated. As vLLM does by default,
each stop then writes ``profiler_out_0.txt``, the profiler's kernel table.
"""

from __future__ import annotations

import gzip
import json
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stormlog.infer.trace_ranges import iteration_range_name

from .config import Controls, FakeEngineConfig
from .engine import Engine, EngineObserver, Step

NOT_CONFIGURED = 404
# Pieces a timed trace write is streamed in.
WRITE_PIECES = 8


@dataclass(eq=False)
class _DelayedWrite:
    """A trace written later; whoever claims it first writes it."""

    steps: list[Step]
    timer: threading.Timer | None = None
    claimed: bool = field(default=False)


class FakeProfiler(EngineObserver):
    def __init__(
        self,
        engine: Engine,
        config: FakeEngineConfig,
        controls: Controls,
        producer: str,
    ) -> None:
        self.engine = engine
        self.config = config
        self.controls = controls
        self.producer = producer
        self.active = False
        # Each stop's span on the loop, and each trace with the time its write
        # finished, in ``time.time_ns()``: tests time a pause by these.
        self.stops: list[tuple[int, int]] = []
        self.written: list[tuple[Path, int]] = []
        self._steps: list[Step] = []
        self._correlation = 0
        self._delayed: list[_DelayedWrite] = []
        self._delayed_lock = threading.Lock()

    # ------------------------------------------------------------ routes

    def start(self) -> int:
        status = self._status()
        if status != 200:
            return status
        result: int = self.engine.call_in_loop(self._start_in_loop)
        return result

    def stop(self) -> int:
        status = self._status()
        if status != 200:
            return status
        result: int = self.engine.call_in_loop(self._stop_in_loop)
        return result

    def drop_foreign_trace(self) -> Path:
        """A worker trace that appears outside any window Stormlog opened."""
        return self._write([])

    def _status(self) -> int:
        if self.config.trace_dir is None:
            return NOT_CONFIGURED
        return self.controls.profiler_status

    def _start_in_loop(self) -> int:
        # A second start while one runs is a no-op that answers 200, as in vLLM.
        if not self.active:
            self.active = True
            self._steps = []
            time.sleep(self.controls.start_pause_seconds)
        return 200

    def _stop_in_loop(self) -> int:
        # A stop with nothing running also answers 200.
        started = time.time_ns()
        if self.active:
            self._close_window()
        self.stops.append((started, time.time_ns()))
        return 200

    def shutdown(self) -> None:
        """As vLLM's worker shuts its profiler down: stop an open window and
        write its trace; then finish every delayed write, so no trace appears
        after the engine has stopped. Call it once the step loop has stopped."""
        if self.active and self.config.trace_dir is not None:
            self._close_window()
        with self._delayed_lock:
            delayed, self._delayed = self._delayed, []
        for write in delayed:
            if self._claim(write):
                assert write.timer is not None
                write.timer.cancel()
                self._write(write.steps)
            elif write.timer is not None:
                write.timer.join()

    # ------------------------------------------------------------ the loop

    def on_executed(self, step: Step) -> None:
        if not self.active:
            return
        self._steps.append(step)
        limit = self.controls.profiler_max_iterations
        if limit is not None and len(self._steps) >= limit:
            # (c) The worker stops itself, pause included, and nobody calls stop.
            self._close_window()

    def _close_window(self) -> None:
        self.active = False
        steps, self._steps = self._steps, []
        time.sleep(self.controls.stop_pause_seconds)
        self._write_or_schedule(steps)
        if self.config.torch_profiler_dump_cuda_time_total:
            self._write_table(steps)

    def _write_or_schedule(self, steps: list[Step]) -> None:
        if not self.controls.stop_writes_trace:
            return
        delay = self.controls.trace_write_delay_seconds
        if delay <= 0:
            self._write(steps)
            return
        write = _DelayedWrite(steps)
        write.timer = threading.Timer(delay, self._write_delayed, args=(write,))
        write.timer.daemon = True
        with self._delayed_lock:
            self._delayed.append(write)
        write.timer.start()

    def _write_delayed(self, write: _DelayedWrite) -> None:
        if self._claim(write):
            self._write(write.steps)

    def _claim(self, write: _DelayedWrite) -> bool:
        with self._delayed_lock:
            claimed, write.claimed = write.claimed, True
        return not claimed

    # ------------------------------------------------------------ the trace

    def _write(self, steps: list[Step]) -> Path:
        directory = self.config.trace_dir
        assert directory is not None
        directory.mkdir(parents=True, exist_ok=True)
        host = socket.gethostname()
        # Nanoseconds, as torch's trace handler names them, so two traces
        # written in the same millisecond don't collide.
        name = f"rank0.{host}_{os.getpid()}.{time.time_ns()}"
        path = directory / f"{name}.pt.trace.json.gz"
        data = json.dumps(self._document(steps, host)).encode()
        seconds = self.controls.trace_write_seconds
        pieces = WRITE_PIECES if seconds > 0 else 1
        size = -(-len(data) // pieces)
        with gzip.open(path, "wb") as handle:
            for start in range(0, len(data), size):
                handle.write(data[start : start + size])
                handle.flush()
                time.sleep(seconds / pieces)
        self.written.append((path, time.time_ns()))
        return path

    def _write_table(self, steps: list[Step]) -> None:
        """vLLM's profiler_out_<rank>.txt: torch's key_averages table sorted
        by self CUDA time, cut down to the fake kernel, rewritten each stop."""
        directory = self.config.trace_dir
        assert directory is not None
        directory.mkdir(parents=True, exist_ok=True)
        worked = [step for step in steps if step.total_tokens]
        cuda_us = sum(_kernel_us(step) for step in worked)
        rule = "-" * 24 + "  " + "  ".join(["-" * 12] * 3)
        lines = [
            rule,
            f"{'Name':>24}  {'Self CUDA':>12}  {'Self CUDA %':>12}  {'# of Calls':>12}",
            rule,
            f"{'fake_decode_kernel':>24}  {cuda_us / 1000:>10.3f}ms  "
            f"{100.0 if worked else 0.0:>11.2f}%  {len(worked):>12}",
            rule,
            f"Self CPU time total: {len(worked) * 0.001:.3f}ms",
            f"Self CUDA time total: {cuda_us / 1000:.3f}ms",
        ]
        (directory / "profiler_out_0.txt").write_text("\n".join(lines) + "\n\n")

    def _document(self, steps: list[Step], host: str) -> dict[str, Any]:
        base_ns = steps[0].exec_start_ns if steps else time.time_ns()
        events: list[dict[str, Any]] = []
        for step in steps:
            events.extend(self._step_events(step, base_ns))
        return {
            "schemaVersion": 1,
            "deviceProperties": [{"id": 0, "name": "Fake GPU"}],
            "distributedInfo": {"rank": 0, "world_size": 1},
            "baseTimeNanoseconds": base_ns,
            "host_name": host,
            "traceEvents": events,
        }

    def _step_events(self, step: Step, base_ns: int) -> list[dict[str, Any]]:
        pid = os.getpid()
        tid = self.engine.loop_thread_id or 0
        start_us = (step.exec_start_ns - base_ns) / 1000.0
        span_us = _span_us(step)
        ranged = {
            "ph": "X",
            "cat": "user_annotation",
            "name": iteration_range_name(self.producer, str(step.iteration)),
            "pid": pid,
            "tid": tid,
            "ts": start_us,
            "dur": span_us,
        }
        if not step.total_tokens:
            # vLLM's model runner returns before any forward pass.
            return [ranged]
        self._correlation += 1
        correlation = self._correlation
        return [
            ranged,
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "pid": pid,
                "tid": tid,
                "ts": start_us + 1.0,
                "dur": 1.0,
                "args": {"correlation": correlation},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "fake_decode_kernel",
                "pid": 0,
                "tid": 7,
                "ts": start_us + 2.0,
                "dur": _kernel_us(step),
                "args": {"device": 0, "stream": 7, "correlation": correlation},
            },
        ]


def _span_us(step: Step) -> float:
    return max((step.exec_end_ns - step.exec_start_ns) / 1000.0, 4.0)


def _kernel_us(step: Step) -> float:
    return max(_span_us(step) - 3.0, 1.0)


__all__ = ["NOT_CONFIGURED", "FakeProfiler"]
