"""Small evaluated MLX workload: snapshots, profiling, context and tracking."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from stormlog.mlx import MemoryTracker, MLXMemoryProfiler

    profiler = MLXMemoryProfiler()
    import mlx.core as mx

    baseline = profiler.capture_snapshot("baseline")
    x = mx.ones((128, 128))
    mx.eval(x)  # input preparation precedes profiling
    output = profiler.profile_function(lambda: x @ x.T, name="matmul")
    assert output[0, 0].item() == 128
    with profiler.profile_context("explicit_roots") as region:
        output = x @ x.T
        region.evaluate(output)
    tracker = MemoryTracker(sampling_interval=0.01, max_history=5)
    with tracker.tracking():
        with tracker.phase("evaluated_loop"):
            for _ in range(10):
                output = x @ x.T
                mx.eval(output)
                time.sleep(0.005)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        profiler.export(str(args.output / "profiles.json"))
        tracker.export(args.output / "tracking.json")
    print(
        json.dumps(
            {
                "baseline": baseline.to_dict(),
                "profiles": [p.to_dict() for p in profiler.get_results()],
                "tracking_samples": tracker.get_results().total_samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
