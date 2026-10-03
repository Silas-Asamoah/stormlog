"""vLLM's profiler routes: a window opened and closed between steps.

``/start_profile`` and ``/stop_profile`` run on the step loop, as vLLM runs its
utility calls, so the stop's pause (vLLM writes the trace inside the call)
holds serving. The trace is a small gzipped Kineto file, one iteration range
per profiled step with a launch and a kernel inside it; when the hook log is
on, its ranges name the hook's own iterations, so an import links the work
to the steps.
"""

from __future__ import annotations

import gzip
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

from stormlog.infer.trace_ranges import iteration_range_name

from .config import Controls, FakeEngineConfig
from .engine import Engine, EngineObserver, Step

NOT_CONFIGURED = 404


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
        if not self.controls.stop_writes_trace:
            return
        delay = self.controls.trace_write_delay_seconds
        if delay <= 0:
            self._write(steps)
            return
        timer = threading.Timer(delay, self._write, args=(steps,))
        timer.daemon = True
        timer.start()

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
        document = self._document(steps, host)
        partial = path.with_name(path.name + ".tmp")
        with gzip.open(partial, "wt", encoding="utf-8") as handle:
            json.dump(document, handle)
        partial.replace(path)
        self.written.append((path, time.time_ns()))
        return path

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
        span_us = max((step.exec_end_ns - step.exec_start_ns) / 1000.0, 4.0)
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
                "dur": max(span_us - 3.0, 1.0),
                "args": {"device": 0, "stream": 7, "correlation": correlation},
            },
        ]


__all__ = ["NOT_CONFIGURED", "FakeProfiler"]
