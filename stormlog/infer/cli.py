"""CLI for Stormlog inference profiling."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from .analysis import analyze_inference_events, format_analysis_text
from .config import ProfileConfig, parse_int_list, resolve_endpoint
from .profile import run_profile


def main(argv: Sequence[str] | None = None) -> int:
    """Run the inference CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.infer_command is None:
        parser.print_help()
        return 0
    try:
        if args.infer_command == "profile":
            return cmd_profile(args)
        if args.infer_command == "analyze":
            return cmd_analyze(args)
    except BrokenPipeError:
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"Unsupported infer command: {args.infer_command}")


def build_parser() -> argparse.ArgumentParser:
    """Build the `stormlog infer` parser."""
    parser = argparse.ArgumentParser(
        prog="stormlog infer",
        description="Profile OpenAI-compatible inference endpoints",
    )
    subparsers = parser.add_subparsers(
        dest="infer_command",
        help="Inference commands",
    )

    profile_parser = subparsers.add_parser(
        "profile",
        help="Run active inference profiling traffic",
    )
    endpoint_group = profile_parser.add_mutually_exclusive_group(required=True)
    endpoint_group.add_argument(
        "--endpoint",
        help="Full /v1/chat/completions endpoint URL",
    )
    endpoint_group.add_argument(
        "--base-url",
        help="OpenAI-compatible /v1 base URL",
    )
    profile_parser.add_argument("--model", required=True, help="Model name")
    profile_parser.add_argument(
        "--concurrency",
        default="1",
        help="Comma-separated concurrency levels (default: 1)",
    )
    profile_parser.add_argument(
        "--input-tokens",
        default="512",
        help="Comma-separated prompt token targets (default: 512)",
    )
    profile_parser.add_argument(
        "--output-tokens",
        default="128",
        help="Comma-separated output token caps (default: 128)",
    )
    profile_parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Measured duration per workload case in seconds",
    )
    profile_parser.add_argument(
        "--requests",
        type=int,
        default=1,
        help="Total measured request count per workload case (default: 1)",
    )
    profile_parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL artifact path",
    )
    profile_parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds (default: 60)",
    )
    profile_parser.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="Warmup requests per workload case, excluded from analysis",
    )
    profile_parser.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        default=True,
        help="Use streaming chat completions (default)",
    )
    profile_parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Use non-streaming chat completions",
    )
    profile_parser.add_argument(
        "--stream-usage",
        dest="stream_usage",
        action="store_true",
        default=True,
        help="Request streaming usage metadata with stream_options.include_usage",
    )
    profile_parser.add_argument(
        "--no-stream-usage",
        dest="stream_usage",
        action="store_false",
        help="Do not send stream_options.include_usage for streaming requests",
    )
    profile_parser.add_argument(
        "--api-key",
        default=None,
        help="Bearer token. Defaults to OPENAI_API_KEY when set.",
    )
    profile_parser.add_argument(
        "--max-tokens-field",
        choices=["max_tokens", "max_completion_tokens"],
        default="max_tokens",
        help="Output cap field to send (default: max_tokens)",
    )
    profile_parser.add_argument(
        "--tokenizer",
        choices=["auto", "none", "tiktoken", "transformers"],
        default="auto",
        help="Tokenizer for prompt generation and fallback counts",
    )
    profile_parser.add_argument(
        "--tokenizer-model",
        default=None,
        help="Tokenizer model/path override",
    )
    profile_parser.add_argument(
        "--tiktoken-encoding",
        default=None,
        help="Explicit tiktoken encoding, such as cl100k_base or o200k_base",
    )
    profile_parser.add_argument(
        "--strict-token-counts",
        action="store_true",
        help="Fail instead of falling back to estimated token counts",
    )
    profile_parser.add_argument(
        "--system-sampler",
        choices=["auto", "none", "psutil", "nvidia-smi"],
        default="auto",
        help="Best-effort telemetry sampler (default: auto)",
    )
    profile_parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="System sample interval in seconds (default: 1)",
    )
    profile_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic prompt seed (default: 0)",
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze an inference profiling JSONL artifact",
    )
    analyze_parser.add_argument("input_file", help="Input inference JSONL artifact")
    analyze_parser.add_argument(
        "--output",
        default=None,
        help="Optional report output path",
    )
    analyze_parser.add_argument(
        "--format",
        choices=["txt", "json"],
        default="txt",
        help="Report format (default: txt)",
    )
    return parser


def cmd_profile(args: argparse.Namespace) -> int:
    """Run active inference profiling."""
    _validate_profile_arguments(args)

    endpoint = resolve_endpoint(endpoint=args.endpoint, base_url=args.base_url)
    config = ProfileConfig(
        endpoint=endpoint,
        model=args.model,
        concurrency=tuple(parse_int_list(args.concurrency, field_name="concurrency")),
        input_tokens=tuple(
            parse_int_list(args.input_tokens, field_name="input-tokens")
        ),
        output_tokens=tuple(
            parse_int_list(args.output_tokens, field_name="output-tokens")
        ),
        duration_seconds=args.duration,
        request_count=None if args.duration is not None else args.requests,
        stream=bool(args.stream),
        stream_include_usage=bool(args.stream_usage),
        timeout_seconds=float(args.timeout),
        warmup_requests=int(args.warmup_requests),
        output_path=args.output,
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
        max_tokens_field=args.max_tokens_field,
        tokenizer=args.tokenizer,
        tokenizer_model=args.tokenizer_model,
        tiktoken_encoding=args.tiktoken_encoding,
        strict_token_counts=bool(args.strict_token_counts),
        system_sampler=args.system_sampler,
        sample_interval_seconds=float(args.sample_interval),
        seed=int(args.seed),
    )
    report = run_profile(config)
    print(format_analysis_text(report))
    print(f"Artifact saved to: {Path(args.output)}")
    summary = report.get("summary", {})
    if (
        int(summary.get("total_requests", 0)) > 0
        and int(summary.get("successful_requests", 0)) == 0
    ):
        print("Error: no measured inference requests succeeded", file=sys.stderr)
        return 1
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze an inference JSONL artifact."""
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        return 1
    report = analyze_inference_events(input_path)
    if args.format == "json":
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    else:
        payload = format_analysis_text(report) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        print(f"Analysis report saved to: {output_path}")
    else:
        print(payload, end="")
    return 0


def _validate_profile_arguments(args: argparse.Namespace) -> None:
    if args.duration is not None and args.duration <= 0:
        raise ValueError("--duration must be > 0")
    if args.requests is not None and args.requests <= 0:
        raise ValueError("--requests must be >= 1")
    if args.duration is not None and args.requests != 1:
        raise ValueError("Use either --duration or --requests, not both")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests must be >= 0")
    if args.sample_interval <= 0:
        raise ValueError("--sample-interval must be > 0")
