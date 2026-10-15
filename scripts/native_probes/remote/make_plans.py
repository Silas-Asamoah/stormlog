"""Write every seeded protocol v2 trial plan that was executed.

smoke, scaling and graph were planned before data; devbuf diagnosed the CUPTI
timestamp loss found in the smoke; ablation and graph_perturbation are the
post-hoc plans listed in protocol_v2.json.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[3] / "benchmarks/native_probes/plans_v2"
SEED = 118
WINDOWS = [50, 200, 800, 2200]
MODES = ["kineto", "cupti-min", "cupti-full"]
REPS = 3
SMOKE_LOAD = ("--warmup", "20", "--measured", "100")


def trial(
    trial_id: str,
    mode: str,
    window: int,
    *extra: str,
    keep: bool = False,
    timeout: int = 2400,
) -> dict:
    return {
        "trial_id": trial_id,
        "args": ["--mode", mode, "--window", str(window), *extra],
        "keep_traces": keep,
        "timeout": timeout,
    }


def _shuffled(trials: list[dict]) -> list[dict]:
    random.Random(SEED).shuffle(trials)
    return trials


def plans() -> dict[str, dict]:
    smoke = [
        trial(f"smoke-{mode}-w50", mode, 50, *SMOKE_LOAD, keep=True)
        for mode in ["off", "kineto", "cupti-min", "cupti-full", "nsys"]
    ]
    scaling = _shuffled(
        [
            trial(f"scale-{mode}-w{window}-r{rep}", mode, window)
            for mode in MODES
            for window in WINDOWS
            for rep in range(1, REPS + 1)
        ]
        + [trial(f"scale-off-w2200-r{rep}", "off", 2200) for rep in range(1, REPS + 1)]
    )
    graph = [
        trial(f"graph-{mode}-w200", mode, 200, "--cuda-graphs", keep=True)
        for mode in ["cupti-min", "nsys"]
    ] + [
        trial(f"eager-{mode}-w200", mode, 200, keep=True)
        for mode in ["cupti-min", "nsys"]
    ]
    devbuf_variants: dict[str, list[str]] = {
        "default": [],
        "flush100": ["--flush-period-ms", "100"],
        "devbuf32m": ["--device-buffer-bytes", str(32 * 2**20)],
        "all": [
            "--device-buffer-bytes",
            str(32 * 2**20),
            "--device-buffer-pool-limit",
            "16",
            "--flush-period-ms",
            "100",
        ],
    }
    devbuf = [
        trial(
            f"devbuf-{name}",
            "cupti-min",
            50,
            *SMOKE_LOAD,
            *extra,
            keep=True,
            timeout=1200,
        )
        for name, extra in devbuf_variants.items()
    ]
    ablation = _shuffled(
        [
            trial(f"ablate-{mode}-w200-r{rep}", mode, 200)
            for mode in ("cupti-kernel", "cupti-discard")
            for rep in (1, 2, 3)
        ]
    )
    graph_perturbation = _shuffled(
        [
            trial(f"graphperf-{mode}-w200-r{rep}", mode, 200, "--cuda-graphs")
            for mode in ("off", "cupti-min", "cupti-kernel")
            for rep in (1, 2, 3)
        ]
    )
    result = {
        name: {"plan_id": name, "seed": SEED, "trials": trials}
        for name, trials in (
            ("smoke", smoke),
            ("scaling", scaling),
            ("graph", graph),
            ("devbuf", devbuf),
            ("ablation", ablation),
        )
    }
    result["graph_perturbation"] = {
        "plan_id": "graph_perturbation",
        "seed": SEED,
        "posthoc": True,
        "trials": graph_perturbation,
    }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, plan in plans().items():
        (OUT / f"{name}.json").write_text(json.dumps(plan, indent=1) + "\n")


if __name__ == "__main__":
    main()
