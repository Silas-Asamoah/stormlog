"""Audio CLI registration kept separate from the established text options."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from typing import Any

from ..exit_codes import ExitCode
from .analysis import format_analysis_text
from .audio_config import AudioProfileConfig
from .audio_profile import AudioProfiler
from .config import parse_float_list, parse_int_list, resolve_endpoint


def add_audio_parser(
    subparsers: Any, add_arrivals: Callable[[argparse.ArgumentParser], None]
) -> None:
    parser = subparsers.add_parser(
        "profile-audio",
        help="Profile MLX-Audio TTS/STT from a validated fixture manifest",
    )
    endpoint = parser.add_mutually_exclusive_group(required=True)
    endpoint.add_argument("--endpoint")
    endpoint.add_argument("--base-url")
    parser.add_argument("--engine", choices=["mlx-audio"], default="mlx-audio")
    parser.add_argument("--task", choices=["tts", "stt"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-version")
    parser.add_argument("--run-id")
    parser.add_argument("--api-key", help="Bearer token; defaults to OPENAI_API_KEY")
    parser.add_argument("--concurrency")
    bound = parser.add_mutually_exclusive_group()
    bound.add_argument("--requests", type=int)
    bound.add_argument("--duration", type=float)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--stream", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--response-format", choices=["wav", "json", "ndjson"])
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-response-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--max-fixture-bytes", type=int, default=32 * 1024**2)
    parser.add_argument("--max-manifest-bytes", type=int, default=8 * 1024**2)
    parser.add_argument("--max-total-fixture-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--max-audio-seconds", type=float, default=300)
    parser.add_argument(
        "--retain-media-dir",
        help="Opt-in media directory within the run artifact directory",
    )
    parser.add_argument("--max-retained-bytes", type=int, default=64 * 1024**2)
    add_arrivals(parser)


def cmd_profile_audio(args: argparse.Namespace) -> int:
    from .cli import (
        _arrival_trace,
        _request_count,
        _usage_errors,
        _validate_arrival_arguments,
    )

    with _usage_errors():
        _validate_arrival_arguments(args)
        config = AudioProfileConfig(
            endpoint=resolve_endpoint(
                endpoint=args.endpoint, base_url=args.base_url, task=args.task
            ),
            engine=args.engine,
            task=args.task,
            model=args.model,
            manifest_path=args.manifest,
            output_path=args.output,
            concurrency=tuple(
                parse_int_list(args.concurrency or "1", field_name="concurrency")
            ),
            request_count=_request_count(args),
            duration_seconds=args.duration,
            warmup_requests=args.warmup,
            stream=args.stream,
            response_format=args.response_format,
            timeout_seconds=args.timeout,
            max_response_bytes=args.max_response_bytes,
            max_fixture_bytes=args.max_fixture_bytes,
            max_manifest_bytes=args.max_manifest_bytes,
            max_total_fixture_bytes=args.max_total_fixture_bytes,
            max_audio_seconds=args.max_audio_seconds,
            seed=args.seed,
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
            run_id=args.run_id,
            server_version=args.server_version,
            arrival_mode=args.arrival,
            rates=(
                tuple(parse_float_list(args.rate, field_name="rate"))
                if args.rate
                else ()
            ),
            burst_size=args.burst_size,
            burst_interval_seconds=args.burst_interval,
            arrival_trace=_arrival_trace(args),
            max_in_flight=args.max_in_flight or 128,
            overflow=args.overflow or "wait",
            drain_timeout_seconds=args.drain_timeout,
            retain_media_dir=args.retain_media_dir,
            max_retained_bytes=args.max_retained_bytes,
        )
        profiler = AudioProfiler(config)
    report = profiler.run()
    print(format_analysis_text(report))
    print(f"Artifact saved to: {args.output}")
    return int(
        ExitCode.FINDINGS
        if report["summary"]["total_requests"]
        and not report["summary"]["successful_requests"]
        else ExitCode.OK
    )
