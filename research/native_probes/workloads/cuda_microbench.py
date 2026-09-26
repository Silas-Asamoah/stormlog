"""Controlled CUDA/HIP workloads for native probe correctness experiments."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import nullcontext
from typing import Any, Iterator, Sequence

from ..models import WorkloadId


def main(argv: Sequence[str] | None = None) -> int:
    """Run a selected GPU microbenchmark and emit one JSON result line."""
    arguments = _parser().parse_args(argv)
    torch = _load_torch()
    _require_accelerator(torch)
    torch.manual_seed(arguments.seed)
    device = torch.device("cuda")
    context = _profiler_context(torch, arguments.torch_trace)
    with context as profiler:
        metrics, ground_truth, measurement_window = _run(torch, device, arguments)
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
        "measurement_window": measurement_window,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        required=True,
        choices=tuple(item.value for item in WorkloadId if item is not WorkloadId.VLLM),
    )
    parser.add_argument("--warmup", type=_positive_int, default=100)
    parser.add_argument("--iterations", type=_positive_int, default=1_000)
    parser.add_argument("--matrix-size", type=_positive_int, default=512)
    parser.add_argument("--seed", type=int, default=118)
    parser.add_argument("--torch-trace")
    parser.add_argument("--launches-per-iteration", type=_positive_int, default=100)
    parser.add_argument("--overlap-elements", type=_positive_int, default=1_048_576)
    parser.add_argument("--overlap-operations", type=_positive_int, default=8)
    parser.add_argument(
        "--measurement-range-id", default="stormlog-native-probe-measured"
    )
    parser.add_argument(
        "--producer-buffer-bytes", type=_positive_int, default=8 * 1024 * 1024
    )
    parser.add_argument(
        "--transport-buffer-bytes", type=_positive_int, default=8 * 1024 * 1024
    )
    parser.add_argument(
        "--output-byte-bound", type=_positive_int, default=256 * 1024 * 1024
    )
    parser.add_argument("--consumer-delay-ms", type=_nonnegative_float, default=0.0)
    parser.add_argument("--flush-interval-ms", type=_positive_float, default=100.0)
    parser.add_argument(
        "--postprocessor-delay-ms", type=_nonnegative_float, default=0.0
    )
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


def _run(
    torch: Any, device: Any, arguments: argparse.Namespace
) -> tuple[Any, Any, Any]:
    if arguments.workload == "w1-eager":
        return _w1_eager(torch, device, arguments)
    if arguments.workload in {"w2-overlap", "w2-serialized"}:
        return _w2_streams(torch, device, arguments)
    if arguments.workload == "w3-graph":
        return _w3_graph(torch, device, arguments)
    return _w4_stress(torch, device, arguments)


def _w1_eager(
    torch: Any, device: Any, arguments: argparse.Namespace
) -> tuple[Any, Any, Any]:
    left, right = _matrices(torch, device, arguments.matrix_size)

    def iteration() -> None:
        output = torch.mm(left, right)
        output.add_(1.0)

    metrics, window = _measure(torch, iteration, arguments)
    return (
        metrics,
        {
            "expected_iterations": arguments.iterations,
            "expected_operations_per_iteration": ["mm", "add_"],
            "cpu_api_interval_is_device_interval": False,
        },
        window,
    )


def _w2_streams(
    torch: Any, device: Any, arguments: argparse.Namespace
) -> tuple[Any, Any, Any]:
    values = [
        torch.ones(arguments.overlap_elements, device=device),
        torch.ones(arguments.overlap_elements, device=device),
    ]
    streams = [torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)]
    serialized = arguments.workload == "w2-serialized"

    def iteration() -> None:
        for index, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                for _ in range(arguments.overlap_operations):
                    values[index].mul_(1.000001).add_(0.000001)
            if serialized and index == 0:
                stream.synchronize()

    metrics, window = _measure(torch, iteration, arguments)
    contract = w2_contract(
        serialized=serialized,
        stream_ids=[int(stream.cuda_stream) for stream in streams],
        operations_per_stream=arguments.overlap_operations,
        elements=arguments.overlap_elements,
    )
    contract["expected_iterations"] = arguments.iterations
    return metrics, contract, window


def w2_contract(
    *,
    serialized: bool,
    stream_ids: Sequence[int],
    operations_per_stream: int,
    elements: int,
) -> dict[str, Any]:
    """Describe W2 design intent without claiming observed device concurrency."""
    if len(stream_ids) != 2 or len(set(stream_ids)) != 2:
        raise ValueError("W2 requires two distinct stream identifiers")
    return {
        "stream_count": 2,
        "stream_ids": list(stream_ids),
        "overlap_eligible_by_design": not serialized,
        "overlap_expected_by_design": not serialized,
        "overlap_observed_in_trusted_trace": None,
        "overlap_preserved_by_candidate": None,
        "device_makespan_ns": None,
        "concurrent_interval_ns": None,
        "ordering_edges": [[stream_ids[0], stream_ids[1]]] if serialized else [],
        "serialized_control": serialized,
        "operations_per_stream": operations_per_stream,
        "elements_per_operation": elements,
    }


def _w3_graph(
    torch: Any, device: Any, arguments: argparse.Namespace
) -> tuple[Any, Any, Any]:
    if not hasattr(torch.cuda, "CUDAGraph") or not hasattr(torch.cuda, "graph"):
        raise SystemExit("UNSUPPORTED - PYTORCH GRAPH API UNAVAILABLE")
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
    metrics, window = _measure(torch, graph.replay, arguments)
    return (
        metrics,
        {
            "capture_count": 1,
            "expected_replays": arguments.iterations,
            "expected_operations_per_replay": ["mm", "add_"],
            "runtime_graph_api": "HIP" if torch.version.hip else "CUDA",
        },
        window,
    )


def _w4_stress(
    torch: Any, device: Any, arguments: argparse.Namespace
) -> tuple[Any, Any, Any]:
    value = torch.ones(256, device=device)
    launches_per_iteration = arguments.launches_per_iteration

    def iteration() -> None:
        nonlocal value
        for _ in range(launches_per_iteration):
            value = value + 1.0

    metrics, window = _measure(torch, iteration, arguments)
    return (
        metrics,
        {
            "expected_iterations": arguments.iterations,
            "expected_launches": arguments.iterations * launches_per_iteration,
            "launches_per_iteration": launches_per_iteration,
            "pressure_controls": {
                "launches_per_iteration": launches_per_iteration,
                "producer_buffer_bytes": arguments.producer_buffer_bytes,
                "transport_buffer_bytes": arguments.transport_buffer_bytes,
                "output_byte_bound": arguments.output_byte_bound,
                "consumer_delay_ms": arguments.consumer_delay_ms,
                "flush_interval_ms": arguments.flush_interval_ms,
                "postprocessor_delay_ms": arguments.postprocessor_delay_ms,
            },
        },
        window,
    )


def _matrices(torch: Any, device: Any, size: int) -> tuple[Any, Any]:
    return (
        torch.randn((size, size), device=device),
        torch.randn((size, size), device=device),
    )


def _measure(
    torch: Any, operation: Any, arguments: argparse.Namespace
) -> tuple[dict[str, float], dict[str, Any]]:
    for _ in range(arguments.warmup):
        operation()
    torch.cuda.synchronize()
    samples: list[float] = []
    device_start = torch.cuda.Event(enable_timing=True)
    device_end = torch.cuda.Event(enable_timing=True)
    device_start.record()
    host_started = time.perf_counter_ns()
    capture_started_ns = time.time_ns()
    for _ in _range_with_markers(
        torch, arguments.iterations, arguments.measurement_range_id
    ):
        iteration_started = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - iteration_started) / 1_000_000)
    host_enqueue_ms = (time.perf_counter_ns() - host_started) / 1_000_000
    device_end.record()
    torch.cuda.synchronize()
    host_complete_ms = (time.perf_counter_ns() - host_started) / 1_000_000
    capture_finished_ns = time.time_ns()
    return {
        "host_enqueue_ms": host_enqueue_ms,
        "host_complete_ms": host_complete_ms,
        "device_elapsed_ms": float(device_start.elapsed_time(device_end)),
        "host_iteration_median_ms": statistics.median(samples),
        "host_iteration_p95_ms": _percentile(samples, 0.95),
        "host_iteration_p99_ms": _percentile(samples, 0.99),
    }, {
        "range_id": arguments.measurement_range_id,
        "marker": "nvtx-range",
        "warmup_iterations": arguments.warmup,
        "measured_iterations": arguments.iterations,
        "host_started_ns": capture_started_ns,
        "host_finished_ns": capture_finished_ns,
        "clock": "CLOCK_REALTIME",
        "flush_completed": True,
    }


def _range_with_markers(torch: Any, iterations: int, range_id: str) -> Iterator[int]:
    torch.cuda.nvtx.range_push(range_id)
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


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
