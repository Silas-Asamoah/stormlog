"""Check Kineto trace import on a CUDA workload with overlapping streams and a graph.

Each iteration launches work on two streams that overlap, then replays a
captured CUDA graph, inside a ``stormlog.iteration/...`` range. The scenario
profiles a bounded window with the PyTorch profiler, imports the exported
trace, and checks that:

- every GPU event launched inside an iteration is linked to it through its
  launch call, and the one copy made after the loop (outside any iteration)
  stays unresolved instead of being charged to an iteration;
- the graph replay's kernels are joined through the one ``cudaGraphLaunch``;
- overlapping streams are counted once in device busy time;
- results are identical with the profiler on and off.

It also reports the profiler's throughput cost. Run on a CUDA host:

    python -m examples.scenarios.trace_import_scenario --steps 200

With ``--external`` it only runs the workload, with NVTX iteration ranges and
no PyTorch profiler, for capture by Nsight Systems (two CUPTI clients cannot
run at once):

    nsys profile --trace=cuda,nvtx --cuda-graph-trace=node -o run \\
        python -m examples.scenarios.trace_import_scenario --external
    nsys export --type sqlite run.nsys-rep
    stormlog infer import-trace infer.jsonl run.sqlite
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from stormlog.infer.trace_kineto import import_kineto_trace
from stormlog.infer.trace_ranges import iteration_range

PRODUCER = "scenario"


class Workload:
    """Two overlapping streams and one CUDA graph replay per iteration."""

    def __init__(self, size: int) -> None:
        torch.manual_seed(0)
        self.a = torch.randn(size, size, device="cuda")
        self.b = torch.randn(size, size, device="cuda")
        self.side = torch.cuda.Stream()
        self.graph_in = torch.randn(size, size, device="cuda")
        self.graph = torch.cuda.CUDAGraph()
        warmup = torch.cuda.Stream()
        with torch.cuda.stream(warmup):
            self._graph_body()
        torch.cuda.current_stream().wait_stream(warmup)
        with torch.cuda.graph(self.graph):
            self.graph_out = self._graph_body()

    def _graph_body(self) -> torch.Tensor:
        x = self.graph_in
        for _ in range(4):
            x = torch.relu(x @ self.graph_in)
        return x

    def step(self) -> torch.Tensor:
        main = torch.cuda.current_stream()
        self.side.wait_stream(main)
        left = self.a @ self.b
        with torch.cuda.stream(self.side):
            right = self.b @ self.a
        main.wait_stream(self.side)
        self.graph.replay()
        return left + right + self.graph_out


def run(
    workload: Workload, steps: int, *, label: str = "step", nvtx: bool = False
) -> tuple[float, torch.Tensor]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = []
    for index in range(steps):
        with iteration_range(PRODUCER, f"{label}-{index}", nvtx=nvtx):
            outputs.append(workload.step())
    torch.cuda.synchronize()
    return time.perf_counter() - start, outputs[-1].clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--external",
        action="store_true",
        help="Run with NVTX ranges and no PyTorch profiler, for Nsight Systems",
    )
    args = parser.parse_args()
    if args.external:
        workload = Workload(args.size)
        # Nsight records the warmup too; its own IDs keep iterations unique.
        run(workload, 20, label="warmup", nvtx=True)
        seconds, _ = run(workload, args.steps, nvtx=True)
        print(json.dumps({"steps": args.steps, "seconds": seconds}))
        return

    workload = Workload(args.size)
    run(workload, 20)
    off = [run(workload, args.steps) for _ in range(args.repeats)]
    on: list[tuple[float, torch.Tensor]] = []
    trace_dir = Path(tempfile.mkdtemp(prefix="stormlog-trace-"))
    for repeat in range(args.repeats):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as profiler:
            on.append(run(workload, args.steps))
        if repeat == 0:
            profiler.export_chrome_trace(str(trace_dir / "scenario.pt.trace.json"))

    uuid = f"GPU-{torch.cuda.get_device_properties(0).uuid}"
    capture = import_kineto_trace(
        trace_dir / "scenario.pt.trace.json",
        run_id="scenario",
        session_id="scenario",
        device_uuids={0: uuid},
    )
    summary = capture.summary or {}
    device = summary["devices"]["0"]
    result: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0),
        "steps": args.steps,
        "repeats": args.repeats,
        "gpu_events": summary["gpu_events"],
        "graph_gpu_events": summary["graph_gpu_events"],
        "linked_gpu_events": summary["linked_gpu_events"],
        "unresolved_gpu_events": summary["unresolved_gpu_events"],
        "activity_records": summary["activity_records"],
        "busy_ms": device["busy_ns"] / 1e6,
        "launch_span_ms": device["launch_span_ns"] / 1e6,
        "summed_ms": device["summed_ns"] / 1e6,
        "steps_per_s_off": [args.steps / seconds for seconds, _ in off],
        "steps_per_s_on": [args.steps / seconds for seconds, _ in on],
        "identical_outputs": all(
            torch.equal(off[0][1], tensor) for _, tensor in off[1:] + on
        ),
    }
    result["checks"] = {
        "iteration_work_linked": result["linked_gpu_events"]
        == result["gpu_events"] - 1,
        "work_outside_iterations_unresolved": result["unresolved_gpu_events"]
        == {"launch_outside_iteration_range": 1},
        "graph_replay_joined": result["graph_gpu_events"] > 0,
        "overlap_counted_once": result["busy_ms"] < result["summed_ms"],
        "identical_outputs": result["identical_outputs"],
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
