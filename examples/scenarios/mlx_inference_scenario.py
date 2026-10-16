"""Qualify explicit running endpoints; never start, download or reset a server.

Run separately for each engine and telemetry state with pre-provisioned models.
Saved client artifacts can be joined to finalized native evidence later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from stormlog.infer.analysis import analyze_inference_events
from stormlog.infer.arrivals import FIXED_RATE
from stormlog.infer.audio_config import AudioProfileConfig
from stormlog.infer.audio_profile import AudioProfiler
from stormlog.infer.config import ProfileConfig, resolve_endpoint
from stormlog.infer.profile import InferenceProfiler
from stormlog.scrub import redact_url


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--engine", choices=["omlx", "mlx-lm", "mlx-audio"], required=True
    )
    result.add_argument("--base-url", required=True)
    result.add_argument("--server-version", required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--model", help="Pre-provisioned text model ID")
    result.add_argument("--tts-model")
    result.add_argument("--stt-model")
    result.add_argument("--tts-manifest")
    result.add_argument("--stt-manifest")
    result.add_argument("--requests", type=int, default=10)
    result.add_argument("--warmup", type=int, default=2)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--rate", type=float, default=100)
    result.add_argument("--timeout", type=float, default=60)
    result.add_argument("--api-key-env")
    result.add_argument("--omlx-stats-url")
    result.add_argument("--omlx-stats-api-key-env")
    result.add_argument("--omlx-stats-cookie-env")
    result.add_argument(
        "--telemetry-state",
        choices=["off", "on", "unknown"],
        required=True,
        help="Operator declaration of native capture state; not inferred from endpoint",
    )
    return result


def _env_key(name: str | None) -> str | None:
    if name is None:
        return None
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Required credential environment variable {name} is unset")
    return value


def _arrival(args: argparse.Namespace, rate: bool) -> dict[str, Any]:
    if not rate:
        return {"concurrency": (1, 2)}
    return dict(
        concurrency=(1,),
        arrival_mode=FIXED_RATE,
        rates=(args.rate,),
        max_in_flight=1,
        overflow="drop",
        drain_timeout_seconds=args.timeout,
    )


def _text(
    args: argparse.Namespace,
    task: str,
    stream: bool,
    rate: bool,
    stats: bool,
    usage: bool,
    path: Path,
) -> dict[str, Any]:
    config = ProfileConfig(
        endpoint=resolve_endpoint(endpoint=None, base_url=args.base_url, task=task),
        engine=args.engine,
        task=task,
        server_version=args.server_version,
        model=args.model,
        input_tokens=(16,),
        output_tokens=(8,),
        output_path=str(path),
        request_count=args.requests,
        warmup_requests=args.warmup,
        seed=args.seed,
        stream=stream,
        stream_include_usage=usage,
        tokenizer="none",
        system_sampler="none",
        extra_body={"temperature": 0},
        timeout_seconds=args.timeout,
        run_id=args.run_id,
        api_key=_env_key(args.api_key_env),
        omlx_stats_url=args.omlx_stats_url if stats else None,
        omlx_stats_api_key=_env_key(args.omlx_stats_api_key_env) if stats else None,
        omlx_stats_cookie=_env_key(args.omlx_stats_cookie_env) if stats else None,
        omlx_stats_interval_seconds=0.1,
        **_arrival(args, rate),
    )
    return InferenceProfiler(config).run()


def _audio(
    args: argparse.Namespace, task: str, stream: bool, rate: bool, path: Path
) -> dict[str, Any]:
    config = AudioProfileConfig(
        endpoint=resolve_endpoint(endpoint=None, base_url=args.base_url, task=task),
        engine=args.engine,
        task=task,
        server_version=args.server_version,
        model=getattr(args, f"{task}_model"),
        manifest_path=getattr(args, f"{task}_manifest"),
        output_path=str(path),
        request_count=args.requests,
        warmup_requests=args.warmup,
        stream=stream,
        seed=args.seed,
        timeout_seconds=args.timeout,
        run_id=args.run_id,
        api_key=_env_key(args.api_key_env),
        **_arrival(args, rate),
    )
    return AudioProfiler(config).run()


def _matrix(args: argparse.Namespace) -> list[tuple[str, str, bool, bool, bool, bool]]:
    tasks = ["tts", "stt"] if args.engine == "mlx-audio" else ["chat", "completion"]
    rows = [
        (f"{task}-{mode}", task, stream, False, False, True)
        for task in tasks
        for mode, stream in [("buffered", False), ("stream", True)]
    ]
    rows += [(f"{task}-rate", task, True, True, False, True) for task in tasks]
    if args.engine != "mlx-audio":
        rows += [
            (f"{task}-stream-no-usage", task, True, False, False, False)
            for task in tasks
        ]
    if args.omlx_stats_url:
        rows += [
            (f"{task}-stream-stats", task, True, False, True, True) for task in tasks
        ]
    return rows


def _validate(args: argparse.Namespace) -> None:
    if args.requests < 2 or args.warmup < 1:
        raise ValueError("Qualification requires requests >= 2 and warmup >= 1")
    if args.engine == "mlx-audio":
        fields = ["tts_model", "stt_model", "tts_manifest", "stt_manifest"]
    else:
        fields = ["model"]
    if any(not getattr(args, field) for field in fields):
        raise ValueError(
            "Provide all models/manifests explicitly for the selected engine"
        )
    if args.omlx_stats_url and args.engine != "omlx":
        raise ValueError("Only oMLX has the qualified read-only statistics endpoint")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate(args)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for name, task, stream, rate, stats, usage in _matrix(args):
        path = args.output / (name + ".jsonl")
        print(f"Qualifying {args.engine} {name}", flush=True)
        if args.engine == "mlx-audio":
            report = _audio(args, task, stream, rate, path)
        else:
            report = _text(args, task, stream, rate, stats, usage, path)
        offline = analyze_inference_events(path)
        # The established text profiler computes its report before appending
        # the terminal session record. Compare all metrics after that status
        # transition; audio already finalizes before analysis.
        report["summary"]["session_status"] = offline["summary"]["session_status"]
        if report != offline:
            raise RuntimeError(f"Offline round-trip changed report for {name}")
        (args.output / (name + "-report.json")).write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        results.append(
            {
                "name": name,
                "artifact": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "summary": report["summary"],
                "cases": report["cases"],
                "telemetry": report["telemetry"],
            }
        )
    config = vars(args).copy()
    config["output"] = str(args.output)
    config["base_url"] = redact_url(args.base_url)
    config["omlx_stats_url"] = redact_url(args.omlx_stats_url)
    payload = {
        "schema_version": 1,
        "configuration": config,
        "results": results,
        "scope": "client HTTP timings; external native telemetry state is operator declared",
        "cache_state": "no resets; dedicated server model/cache state must be recorded separately",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "endpoint_protocol_qualified": all(_row_qualified(row) for row in results),
    }
    (args.output / "qualification.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    return payload


def _row_qualified(row: dict[str, Any]) -> bool:
    if row["name"].endswith("-rate"):
        return bool(row["summary"]["successful_requests"] > 0)
    if row["summary"]["failed_requests"]:
        return False
    if row["name"].endswith("-stats"):
        stats = row["telemetry"]["mlx"]["omlx_stats"]
        return bool(
            stats["samples"] >= 2
            and stats["failed_samples"] == 0
            and any(metric["valid_samples"] > 0 for metric in stats["metrics"].values())
        )
    return True


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = run(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Qualification failed: {exc}")
        return 1
    return int(not result["endpoint_protocol_qualified"])


if __name__ == "__main__":
    raise SystemExit(main())
