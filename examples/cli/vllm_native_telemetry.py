"""Compare vLLM's own telemetry across three controlled workloads.

Runs ``stormlog infer profile`` three times against one vLLM server: unique
prompts, shared-prefix prompts, and a concurrency high enough to saturate the
scheduler. Each run scrapes ``/metrics`` around its phases and receives the
request spans. The script then prints, side by side, what the engine
reported: prefix-cache hits, queue depth and queue time, KV occupancy, token
rates and the span latencies.

The differences are what the engine reports under each workload. The script
does not claim that one caused the other, and nothing it prints attributes
GPU time to a request.

Start vLLM first, with spans pointed at the receiver this script runs::

    vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000 \\
        --otlp-traces-endpoint http://127.0.0.1:4318/v1/traces

Then::

    python -m examples.cli.vllm_native_telemetry \\
        --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts" / "examples" / "vllm_native_telemetry"

WORKLOADS: tuple[tuple[str, list[str]], ...] = (
    ("unique", ["--prompt-mode", "unique"]),
    (
        "shared_prefix",
        [
            "--prompt-mode",
            "shared-prefix",
            "--shared-prefix-ratio",
            "0.75",
            "--prefix-groups",
            "2",
        ],
    ),
    ("saturated", ["--prompt-mode", "unique"]),
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", required=True, help="vLLM /v1 base URL")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--concurrency", type=int, default=4, help="for the first two runs"
    )
    parser.add_argument(
        "--saturate-concurrency", type=int, default=64, help="for the saturated run"
    )
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--spans-listen", default="127.0.0.1:4318")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    return parser.parse_args(argv)


def _profile_command(
    args: argparse.Namespace, name: str, flags: list[str]
) -> list[str]:
    concurrency = args.saturate_concurrency if name == "saturated" else args.concurrency
    requests = max(args.requests, 4 * concurrency)
    return [
        sys.executable,
        "-m",
        "stormlog.entrypoint",
        "infer",
        "profile",
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--concurrency",
        str(concurrency),
        "--input-tokens",
        str(args.input_tokens),
        "--output-tokens",
        str(args.output_tokens),
        "--requests",
        str(requests),
        "--warmup-requests",
        str(concurrency),
        "--tokenizer",
        "none",
        "--system-sampler",
        "none",
        "--extra-body",
        '{"temperature": 0, "ignore_eos": true}',
        "--vllm-metrics",
        "--vllm-metrics-interval",
        "0.5",
        "--vllm-spans-listen",
        args.spans_listen,
        "--output",
        str(args.artifacts / f"{name}.jsonl"),
        *flags,
    ]


def _analyze(artifact: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "stormlog.entrypoint",
            "infer",
            "analyze",
            str(artifact),
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report: dict[str, Any] = json.loads(completed.stdout)
    return report


def _engine_rows(report: dict[str, Any]) -> list[tuple[str, str]]:
    """Flatten the first case's first engine block into printable rows."""
    vllm = report.get("telemetry", {}).get("vllm", {})
    if vllm.get("status") != "collected":
        return [("vllm telemetry", "absent")]
    rows: list[tuple[str, str]] = []
    for case_id, case in vllm.get("cases", {}).items():
        if case.get("state") != "resolved":
            rows.append(
                (f"{case_id} state", f"{case['state']} ({', '.join(case['reasons'])})")
            )
        for engine, block in case.get("engines", {}).items():
            rows.extend(_rows_for_engine(f"{case_id} engine {engine}", block))
        spans = case.get("spans", {})
        inference = spans.get("latency", {}).get("time_in_model_inference", {})
        rows.append(
            (
                f"{case_id} spans",
                f"{spans.get('requests_with_span')} of {spans.get('requests')} requests, "
                f"inference p50 {_num(inference.get('p50_ms'))} ms (residency)",
            )
        )
    return rows


def _rows_for_engine(prefix: str, block: dict[str, Any]) -> list[tuple[str, str]]:
    gauges = block.get("gauges", {})
    derived = block.get("derived", {})
    waiting = gauges.get("queue_depth", {}).get("stats", {}).get("_", {})
    kv = derived.get("kv_cache", {})
    prefix_cache = derived.get("prefix_cache", {})
    rates = derived.get("rates", {})
    queue = block.get("histograms", {}).get("queue_time", {})
    return [
        (
            f"{prefix} waiting max/mean",
            f"{_num(waiting.get('max'))} / {_num(waiting.get('mean'))}",
        ),
        (
            f"{prefix} queue time mean",
            f"{_num(_ms(queue.get('mean')))} ms over {_num(queue.get('count'))} requests",
        ),
        (f"{prefix} kv usage max", _pct(kv.get("max_usage_fraction"))),
        (
            f"{prefix} prefix cache",
            f"{_num(prefix_cache.get('hits'))} hits of {_num(prefix_cache.get('queries'))} "
            f"tokens ({_pct(prefix_cache.get('hit_ratio'))})",
        ),
        (
            f"{prefix} tokens/s",
            f"{_num(rates.get('prompt_tokens_per_second'))} prompt, "
            f"{_num(rates.get('generation_tokens_per_second'))} generated",
        ),
    ]


def _num(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}" if abs(value) < 1000 else f"{value:,.0f}"
    return str(value)


def _ms(value: Any) -> float | None:
    return value * 1000.0 if isinstance(value, (int, float)) else None


def _pct(value: Any) -> str:
    return f"{value * 100:.1f}%" if isinstance(value, (int, float)) else "n/a"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, Any]] = {}
    for name, flags in WORKLOADS:
        print(f"== {name}", flush=True)
        completed = subprocess.run(_profile_command(args, name, flags), check=False)
        if completed.returncode not in (0, 3):
            print(f"profile {name} failed with exit code {completed.returncode}")
            return completed.returncode
        reports[name] = _analyze(args.artifacts / f"{name}.jsonl")
    print()
    print(
        "What vLLM reported for each workload (engine-aggregate; no per-request GPU time):"
    )
    for name, report in reports.items():
        print(f"\n[{name}]")
        for label, value in _engine_rows(report):
            print(f"  {label:<48} {value}")
    print(f"\nArtifacts and reports: {args.artifacts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
