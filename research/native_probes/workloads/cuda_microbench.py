"""Controlled CUDA/HIP workloads for native probe correctness experiments."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import nullcontext
from typing import Any, Iterator, Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run a selected GPU microbenchmark and emit one JSON result line."""
    arguments = _parser().parse_args(argv)
    torch = _load_torch()
    _require_accelerator(torch)
    torch.manual_seed(arguments.seed)
    device = torch.device("cuda")
    context = _profiler_context(torch, arguments.torch_trace)
    with context as profiler:
        metrics, ground_truth = _run(torch, device, arguments)
        if profiler is not None:
            profiler.step()
    if profiler is not None:
        profiler.export_chrome_trace(arguments.torch_trace)
    result = {
        "schema_version": 1,
        "artifact_kind": "workload_result",
        "workload_id": arguments.workload,
        "seed": arguments.seed,
        "warmup_iterations": arguments.warmup,
        "measured_iterations": arguments.iterations,
        "device": str(torch.cuda.get_device_name(device)),
        "runtime": "hip" if torch.version.hip else "cuda",
        "metrics": metrics,
        "ground_truth": ground_truth,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        required=True,
        choices=("w1-eager", "w2-overlap", "w2-serialized", "w3-graph", "w4-stress"),
    )
    parser.add_argument("--warmup", type=_positive_int, default=100)
    parser.add_argument("--iterations", type=_positive_int, default=1_000)
    parser.add_argument("--matrix-size", type=_positive_int, default=512)
    parser.add_argument("--seed", type=int, default=118)
    parser.add_argument("--torch-trace")
    return parser


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise SystemExit("PyTorch is required for this workload") from error
    return torch


def _require_accelerator(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("UNTESTED - CUDA/HIP ACCELERATOR UNAVAILABLE")


def _profiler_context(torch: Any, path: str | None) -> Any:
    if path is None:
        return nullcontext(None)
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )


def _run(torch: Any, device: Any, arguments: argparse.Namespace) -> tuple[Any, Any]:
    if arguments.workload == "w1-eager":
        return _w1_eager(torch, device, arguments)
    if arguments.workload in {"w2-overlap", "w2-serialized"}:
        return _w2_streams(torch, device, arguments)
    if arguments.workload == "w3-graph":
        return _w3_graph(torch, device, arguments)
    return _w4_stress(torch, device, arguments)


def _w1_eager(
    torch: Any, device: Any, arguments: argparse.Namespace
) -> tuple[Any, Any]:
    left, right = _matrices(torch, device, arguments.matrix_size)

    def iteration() -> None:
        output = torch.mm(left, right)
        output.add_(1.0)

    metrics = _measure(torch, iteration, arguments.warmup, arguments.iterations)
    return metrics, {
        "expected_iterations": arguments.iterations,
        "expected_operations_per_iteration": ["mm", "add_"],
        "cpu_api_interval_is_device_interval": False,
    }


def _w2_streams(
    torch: Any, device: Any, arguments: argparse.Namespace
) -> tuple[Any, Any]:
    left, right = _matrices(torch, device, arguments.matrix_size)
    streams = [torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)]
    serialized = arguments.workload == "w2-serialized"

    def iteration() -> None:
        for index, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                torch.mm(left, right)
            if serialized and index == 0:
                stream.synchronize()

    metrics = _measure(torch, iteration, arguments.warmup, arguments.iterations)
    return metrics, {
        "expected_iterations": arguments.iterations,
        "stream_count": 2,
        "expected_overlap": not serialized,
        "serialized_control": serialized,
    }


def _w3_graph(
    torch: Any, device: Any, arguments: argparse.Namespace
) -> tuple[Any, Any]:
    if torch.version.hip:
        raise SystemExit("UNTESTED - HIP GRAPH WORKLOAD NOT QUALIFIED")
    left, right = _matrices(torch, device, arguments.matrix_size)
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            static_output = torch.mm(left, right)
        capture_stream.synchronize()
        with torch.cuda.graph(graph):
            static_output = torch.mm(left, right)
            static_output.add_(1.0)
    torch.cuda.current_stream(device).wait_stream(capture_stream)
    metrics = _measure(torch, graph.replay, arguments.warmup, arguments.iterations)
    return metrics, {
        "capture_count": 1,
        "expected_replays": arguments.iterations,
        "expected_operations_per_replay": ["mm", "add_"],
    }


def _w4_stress(
    torch: Any, device: Any, arguments: argparse.Namespace
) -> tuple[Any, Any]:
    value = torch.ones(256, device=device)
    launches_per_iteration = 100

    def iteration() -> None:
        nonlocal value
        for _ in range(launches_per_iteration):
            value = value + 1.0

    metrics = _measure(torch, iteration, arguments.warmup, arguments.iterations)
    return metrics, {
        "expected_iterations": arguments.iterations,
        "expected_launches": arguments.iterations * launches_per_iteration,
        "launches_per_iteration": launches_per_iteration,
    }


def _matrices(torch: Any, device: Any, size: int) -> tuple[Any, Any]:
    return (
        torch.randn((size, size), device=device),
        torch.randn((size, size), device=device),
    )


def _measure(
    torch: Any, operation: Any, warmup: int, iterations: int
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples: list[float] = []
    device_start = torch.cuda.Event(enable_timing=True)
    device_end = torch.cuda.Event(enable_timing=True)
    device_start.record()
    host_started = time.perf_counter_ns()
    for _ in _range_with_markers(torch, iterations):
        iteration_started = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - iteration_started) / 1_000_000)
    host_enqueue_ms = (time.perf_counter_ns() - host_started) / 1_000_000
    device_end.record()
    torch.cuda.synchronize()
    host_complete_ms = (time.perf_counter_ns() - host_started) / 1_000_000
    return {
        "host_enqueue_ms": host_enqueue_ms,
        "host_complete_ms": host_complete_ms,
        "device_elapsed_ms": float(device_start.elapsed_time(device_end)),
        "host_iteration_median_ms": statistics.median(samples),
        "host_iteration_p95_ms": _percentile(samples, 0.95),
        "host_iteration_p99_ms": _percentile(samples, 0.99),
    }


def _range_with_markers(torch: Any, iterations: int) -> Iterator[int]:
    torch.cuda.nvtx.range_push("stormlog-native-probe-measured")
    try:
        yield from range(iterations)
    finally:
        torch.cuda.nvtx.range_pop()


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = int((len(ordered) - 1) * quantile)
    return ordered[index]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
