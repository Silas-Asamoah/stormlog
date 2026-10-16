"""Explicit qualification-only server launcher with read-only model checkpoints.

This example wraps model calls to prove the serving PID owns native counters.
It neither evaluates arrays nor synchronizes/resets the allocator. Use identical
checkpoint settings in both native sampling off/on runs. Client scenarios never
invoke this launcher automatically.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psutil

from stormlog.infer.mlx_launchers import launch, prepare_launcher, redacted_arguments
from stormlog.infer.mlx_server import run_mlx_server
from stormlog.mlx.artifacts import CaptureOptions
from stormlog.mlx.runtime import MLXRuntime, Runtime


class ReadOnlyCheckpoints:
    def __init__(self, output: Path, runtime: Runtime, maximum: int = 64) -> None:
        self.output, self.runtime, self.maximum = output, runtime, maximum
        self.lock = threading.Lock()
        self.count = 0
        self.classes: set[tuple[type, str]] = set()

    def record(self, model_type: str, boundary: str) -> None:
        with self.lock:
            if self.count >= self.maximum:
                return
            self.count += 1
            start = time.time_ns()
            counters = {
                key: self.runtime.read_bytes(getter)
                for key, getter in (
                    ("active_bytes", "get_active_memory"),
                    ("cache_bytes", "get_cache_memory"),
                    ("runtime_peak_bytes", "get_peak_memory"),
                )
            }
            record = dict(
                schema_version=1,
                pid=os.getpid(),
                thread_id=threading.get_ident(),
                model_type=model_type,
                boundary=boundary,
                started_at_ns=start,
                ended_at_ns=time.time_ns(),
                scope="sequential_process_allocator_getters",
                process_rss_bytes=psutil.Process().memory_info().rss,
                **counters,
            )
            with self.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")

    def instrument(self, model: Any, method: str) -> None:
        cls = type(model)
        if (cls, method) in self.classes:
            return
        self.classes.add((cls, method))
        original = getattr(cls, method)
        label = cls.__module__ + "." + cls.__qualname__ + "." + method
        observer = self

        def observed(instance: Any, *args: Any, **kwargs: Any) -> Any:
            observer.record(label, "before_call")
            result = original(instance, *args, **kwargs)
            if isinstance(result, Iterator):
                return observer.observe_iterator(result, label)
            observer.record(label, "after_call")
            return result

        setattr(cls, method, observed)

    def observe_iterator(self, result: Iterator[Any], label: str) -> Iterator[Any]:
        try:
            for value in result:
                self.record(label, "generator_yield")
                yield value
            self.record(label, "generator_complete")
        finally:
            close = getattr(result, "close", None)
            if callable(close):
                close()


def install_observer(engine: str, observer: ReadOnlyCheckpoints) -> None:
    if engine == "mlx-audio":
        module = importlib.import_module("mlx_audio.server")
        original = module.load_model

        def audio_load(*args: Any, **kwargs: Any) -> Any:
            model = original(*args, **kwargs)
            observer.instrument(model, "generate")
            return model

        setattr(module, "load_model", audio_load)
    else:
        module = importlib.import_module("mlx_lm.utils")
        original = module.load_model

        def text_load(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            observer.instrument(result[0], "__call__")
            return result

        setattr(module, "load_model", text_load)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", choices=["omlx", "mlx-lm", "mlx-audio"], required=True
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--native-capture", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--max-checkpoints", type=int, default=64)
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    launcher, version, actual = prepare_launcher(args.engine, args.server_args)
    options = CaptureOptions(interval=args.interval)
    if not 2 <= args.max_checkpoints <= 1024:
        parser.error("max-checkpoints must be in [2,1024]")
    runtime = MLXRuntime()
    args.output.mkdir(parents=True, exist_ok=False)
    observer = ReadOnlyCheckpoints(
        args.output / "checkpoints.jsonl", runtime, args.max_checkpoints
    )
    install_observer(args.engine, observer)
    metadata = dict(
        schema_version=1,
        engine=args.engine,
        server_version=version,
        mlx_version=runtime.version,
        run_id=args.run_id,
        pid=os.getpid(),
        native_capture=args.native_capture,
        interval=args.interval,
        max_checkpoints=args.max_checkpoints,
        argv=redacted_arguments(actual),
        observer_scope="qualification_model_call_boundaries_not_request_spans",
    )
    (args.output / "configuration.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    if args.native_capture:
        _path, outcome = run_mlx_server(
            engine=args.engine,
            run_id=args.run_id,
            output=args.output / "native",
            args=args.server_args,
            options=options,
            runtime=runtime,
        )
        return outcome.exit_code
    try:
        launch(launcher, actual)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
