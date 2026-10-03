"""CLI for Stormlog inference profiling."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import urllib.parse
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..exit_codes import ExitCode
from .analysis import analyze_inference_events, format_analysis_text
from .arrivals import (
    ARRIVAL_MODES,
    BURST,
    CLOSED,
    RATE_MODES,
    REPLAY,
    ArrivalTrace,
    load_arrival_trace,
)
from .cache_state import CACHE_STATES, COLD, UNSPECIFIED
from .config import ProfileConfig, parse_float_list, parse_int_list, resolve_endpoint
from .errors import InferInputError, InferUsageError
from .profile import InferenceProfiler
from .prompts import MIN_CONTROLLED_TOKENS, PROMPT_MODES, REPEAT, SHARED_PREFIX
from .server_collector import (
    STOP_GPU_IDENTITY_CHANGED,
    STOP_SERVER_PROCESS_ENDED,
    CollectionResult,
    NvmlUnavailableError,
    collect_server_telemetry,
)
from .trace_capture import TRACE_MODES, TRACE_PHASES, TraceCaptureConfig, server_root
from .trace_import import import_traces_into_artifact, parse_device_uuids
from .vllm_execution_import import import_execution_into_artifact
from .vllm_scraper import AUTO_METRICS_URL, resolve_metrics_url
from .vllm_spans import DEFAULT_SPANS_LISTEN, parse_listen_address


def main(argv: Sequence[str] | None = None) -> int:
    """Run the inference CLI and return a code from ``stormlog.exit_codes``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.infer_command is None:
        parser.print_help()
        return int(ExitCode.OK)
    try:
        return _run_command(parser, args)
    except BrokenPipeError:
        return int(ExitCode.ERROR)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return int(ExitCode.INTERRUPTED)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return int(_exit_code_for(exc))


def _run_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.infer_command == "profile":
        return cmd_profile(args)
    if args.infer_command == "analyze":
        return cmd_analyze(args)
    if args.infer_command == "collect-server":
        return cmd_collect_server(args)
    if args.infer_command == "import-trace":
        return cmd_import_trace(args)
    if args.infer_command == "import-execution":
        return cmd_import_execution(args)
    parser.error(f"Unsupported infer command: {args.infer_command}")


def _exit_code_for(exc: Exception) -> ExitCode:
    """Usage and input errors have their own codes; anything else is a failure."""
    if isinstance(exc, InferUsageError):
        return ExitCode.USAGE
    if isinstance(exc, InferInputError):
        return ExitCode.INVALID_INPUT
    return ExitCode.ERROR


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
        "--run-id", default=None, help="Shared run ID for an on-host collector"
    )
    profile_parser.add_argument(
        "--concurrency",
        default=None,
        help="Comma-separated closed-loop concurrency levels (default: 1)",
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
        default=None,
        help=(
            "Total measured request count per workload case (default: 1; "
            "a replay sends the whole trace)"
        ),
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
        help="Seed for prompts and Poisson arrivals (default: 0)",
    )
    profile_parser.add_argument(
        "--vllm-metrics",
        nargs="?",
        const=AUTO_METRICS_URL,
        default=None,
        metavar="URL",
        help=(
            "Scrape vLLM's Prometheus metrics at the start and end of every "
            "phase and every --vllm-metrics-interval seconds inside it; without "
            "a URL, the endpoint's origin plus /metrics"
        ),
    )
    profile_parser.add_argument(
        "--vllm-metrics-interval",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Seconds between vLLM metrics scrapes inside a phase (default: 1)",
    )
    profile_parser.add_argument(
        "--vllm-execution-dir",
        default=None,
        metavar="DIR",
        help=(
            "The vLLM execution hook's STORMLOG_VLLM_HOOK_DIR as this host sees "
            "it; when the run ends, its final scheduler steps are imported and "
            "its worker hellos name the GPU of each traced process"
        ),
    )
    profile_parser.add_argument(
        "--vllm-spans-listen",
        nargs="?",
        const=DEFAULT_SPANS_LISTEN,
        default=None,
        metavar="HOST:PORT",
        help=(
            "Receive vLLM's OpenTelemetry request spans over OTLP/HTTP at "
            "HOST:PORT (default 127.0.0.1:4318) for the length of the run; "
            "start vLLM with --otlp-traces-endpoint pointing at it and "
            "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf, since vLLM "
            "exports over gRPC by default"
        ),
    )
    profile_parser.add_argument(
        "--vllm-spans-drain",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Keep the span receiver listening this long after the last phase, "
            "for the exporter's final batch (default: 6; vLLM flushes every 5 s)"
        ),
    )
    _add_arrival_arguments(profile_parser)
    _add_prompt_arguments(profile_parser)
    _add_cache_arguments(profile_parser)
    _add_trace_arguments(profile_parser)

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
    analyze_parser.add_argument(
        "--server-telemetry",
        action="append",
        default=[],
        metavar="JSONL",
        help="On-host collector artifact; may be supplied more than once",
    )
    analyze_parser.add_argument(
        "--direct-server",
        action="store_true",
        help="Assert requests went to the single server identity in telemetry",
    )
    analyze_parser.add_argument(
        "--vllm-spans",
        action="append",
        default=[],
        metavar="FILE",
        help=(
            "vLLM request spans collected elsewhere, as OTLP JSON or one span "
            "per line; may be supplied more than once"
        ),
    )
    analyze_parser.add_argument(
        "--clock-offset-ns",
        type=int,
        default=None,
        help=(
            "Server timestamp plus this offset equals client timestamp; "
            "replaces infer.clock_alignment records for the same clocks"
        ),
    )
    analyze_parser.add_argument(
        "--clock-uncertainty-ns",
        type=int,
        default=None,
        help=(
            "Absolute uncertainty of --clock-offset-ns; on one host and boot "
            "it may be given alone"
        ),
    )
    collector_parser = subparsers.add_parser(
        "collect-server",
        help="Collect scoped process and NVML memory on the inference host",
    )
    collector_parser.add_argument(
        "--run-id", required=True, help="Run ID also passed to `infer profile`"
    )
    collector_parser.add_argument(
        "--pid",
        required=True,
        type=int,
        help="Server process to watch; use the worker that owns the GPU work",
    )
    collector_parser.add_argument("--output", required=True, help="JSONL path")
    collector_parser.add_argument(
        "--interval", type=float, default=0.1, help="Seconds between polls"
    )
    collector_parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds (default: until Ctrl+C or SIGTERM)",
    )
    collector_parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="NVML index (PCI bus order, not the server's CUDA ordinal)",
    )
    collector_parser.add_argument(
        "--device-uuid",
        default=None,
        help="GPU or MIG UUID; preferred over --device-index",
    )
    collector_parser.add_argument(
        "--no-gpu", action="store_true", help="Collect process RSS only"
    )
    collector_parser.add_argument("--replica-id", default=None)
    collector_parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="This process's rank; required with --group-id",
    )
    collector_parser.add_argument(
        "--group-id",
        default=None,
        help="Shared by every collector of one server, e.g. its tensor-parallel workers",
    )
    collector_parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Number of group members; each rank 0..N-1 needs a collector",
    )
    _add_import_trace_parser(subparsers)
    _add_import_execution_parser(subparsers)
    return parser


def _add_import_trace_parser(subparsers: Any) -> None:
    import_parser = subparsers.add_parser(
        "import-trace",
        help="Add a profiler trace's GPU activity to an artifact",
    )
    import_parser.add_argument("artifact", help="Inference JSONL with a run identity")
    import_parser.add_argument(
        "traces",
        nargs="+",
        help=(
            "Kineto Chrome traces (.json, .json.gz) or Nsight Systems SQLite "
            "exports (.sqlite); an .nsys-rep report is registered only"
        ),
    )
    import_parser.add_argument(
        "--device-uuid",
        action="append",
        default=[],
        metavar="[TRACE_FILE:]INDEX=UUID",
        help=(
            "GPU UUID for a CUDA device ordinal in the traced process (after "
            "CUDA_VISIBLE_DEVICES); repeat per device. Prefix a trace's path as "
            "given here, or its file name when only one trace has it, to scope "
            "the entry to that trace; a prefix that names no trace or several is "
            "refused, as is an unscoped ordinal that traces from different "
            "processes use. Without it, GPU activity is kept but not measured"
        ),
    )
    import_parser.add_argument(
        "--detail",
        choices=("launch", "kernel"),
        default="launch",
        help=(
            "launch: one record per launch call (default, compact); kernel: one "
            "record per GPU event (exact, large)"
        ),
    )
    import_parser.add_argument(
        "--vllm-execution-dir",
        default=None,
        metavar="DIR",
        help=(
            "The vLLM execution hook's STORMLOG_VLLM_HOOK_DIR; its worker hellos "
            "name the GPU of each traced process (by host, pid and lifetime), so "
            "--device-uuid is only needed where that leaves a gap"
        ),
    )
    import_parser.add_argument(
        "--envelope", default=None, help="Run envelope (default: beside the artifact)"
    )


def _add_import_execution_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "import-execution",
        help="Add a vLLM execution hook's scheduler steps to an artifact",
    )
    parser.add_argument("artifact", help="Inference JSONL with a run identity")
    parser.add_argument(
        "directory",
        help=(
            "The hook's STORMLOG_VLLM_HOOK_DIR, copied or mounted from the "
            "server host; only steps that are final since the last import are added"
        ),
    )
    parser.add_argument(
        "--raw-foreign-ids",
        action="store_true",
        help=(
            "Record other clients' request IDs as vLLM saw them instead of keyed "
            "pseudonyms (the artifact then names requests that are not yours)"
        ),
    )
    parser.add_argument(
        "--server-stopped",
        action="store_true",
        help=(
            "The server that wrote this log is no longer running: an epoch "
            "without a goodbye record is gone and its pending steps are final. "
            "Without it, silence is judged only on the server's own host and "
            "boot; from anywhere else such an epoch's pending steps wait"
        ),
    )
    parser.add_argument(
        "--envelope", default=None, help="Run envelope (default: beside the artifact)"
    )


def _add_arrival_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--arrival",
        choices=ARRIVAL_MODES,
        default=CLOSED,
        help=(
            "How requests arrive. closed (default) sends the next request when "
            "a worker is free; the others send on a fixed schedule"
        ),
    )
    parser.add_argument(
        "--rate",
        default=None,
        help="Comma-separated requests/second for fixed-rate and poisson arrivals",
    )
    parser.add_argument(
        "--burst-size", type=int, default=None, help="Requests per burst"
    )
    parser.add_argument(
        "--burst-interval",
        type=float,
        default=None,
        help="Seconds between bursts",
    )
    parser.add_argument(
        "--arrival-trace",
        default=None,
        help=(
            "Arrivals to replay: JSON lines with offset_ms, or a Stormlog "
            "inference artifact"
        ),
    )
    parser.add_argument(
        "--arrival-trace-case",
        default=None,
        help="Measured case to replay from an inference artifact",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        help="Open-loop limit on outstanding requests (default: 128)",
    )
    parser.add_argument(
        "--drain-timeout",
        type=float,
        default=None,
        help=(
            "Seconds that requests still running when the measured window ends "
            "may take to finish before they are recorded as cancelled "
            "(default: --timeout)"
        ),
    )
    parser.add_argument(
        "--overflow",
        choices=["wait", "drop"],
        default=None,
        help=(
            "When every in-flight slot is busy, wait for one (default, and the "
            "wait is recorded) or drop the arrival"
        ),
    )


def _add_prompt_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prompt-mode",
        choices=PROMPT_MODES,
        default=REPEAT,
        help=(
            "repeat (default) sends one prompt per case; unique gives every "
            "request its own prefix; shared-prefix gives groups a common prefix"
        ),
    )
    parser.add_argument(
        "--shared-prefix-ratio",
        type=float,
        default=None,
        help="Share of each prompt's tokens in its group prefix, between 0 and 1",
    )
    parser.add_argument(
        "--prefix-groups",
        type=int,
        default=None,
        help="Number of distinct shared prefixes, assigned by seed (default: 1)",
    )


def _add_cache_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-state",
        choices=CACHE_STATES,
        default=UNSPECIFIED,
        help=(
            "Prefix-cache state each case should start from; cold is recorded "
            "but cannot be verified yet"
        ),
    )
    parser.add_argument(
        "--extra-body",
        default=None,
        help=(
            "JSON object of extra request fields, such as "
            '\'{"temperature": 0, "ignore_eos": true}\'; recorded with the workload'
        ),
    )
    parser.add_argument(
        "--cache-reset-url",
        default=None,
        help=(
            "URL to POST before each case to clear the prefix cache, such as "
            "vLLM /reset_prefix_cache or SGLang /flush_cache"
        ),
    )


def _add_trace_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "profiler trace",
        "Bounded vLLM torch-profiler windows. The server must be started with "
        "--profiler-config.profiler=torch and torch_profiler_dir; Stormlog only "
        "starts and stops the profiler and imports the worker traces.",
    )
    group.add_argument(
        "--trace", choices=TRACE_MODES, default=None, help="Capture a profiler trace"
    )
    group.add_argument(
        "--trace-dir",
        default=None,
        help=(
            "The server's torch_profiler_dir as this host sees it; without it the "
            "traces stay on the server for `stormlog infer import-trace`"
        ),
    )
    group.add_argument(
        "--trace-control-url",
        default=None,
        help="Server root for /start_profile and /stop_profile (default: endpoint host)",
    )
    group.add_argument(
        "--trace-phase",
        choices=TRACE_PHASES,
        default="measured",
        help="Phase of each case to profile (default: measured)",
    )
    group.add_argument(
        "--trace-max-seconds",
        type=float,
        default=None,
        help="Stop the profiler after this many seconds even if the phase continues",
    )
    group.add_argument(
        "--trace-max-bytes",
        type=int,
        default=None,
        help="Register but do not import a trace file larger than this",
    )
    group.add_argument(
        "--trace-device-uuid",
        action="append",
        default=[],
        metavar="INDEX=UUID",
        help="GPU UUID for a CUDA device ordinal in the server process; repeatable",
    )
    group.add_argument(
        "--trace-detail",
        choices=("launch", "kernel"),
        default="launch",
        help="Import one record per launch (default) or per GPU event",
    )


def _trace_config(args: argparse.Namespace, endpoint: str) -> TraceCaptureConfig | None:
    if args.trace is None:
        return None
    return TraceCaptureConfig(
        mode=args.trace,
        control_url=args.trace_control_url or server_root(endpoint),
        trace_dir=Path(args.trace_dir) if args.trace_dir else None,
        phase=args.trace_phase,
        max_seconds=args.trace_max_seconds,
        max_bytes=args.trace_max_bytes,
        device_uuids=parse_device_uuids(args.trace_device_uuid),
        detail=args.trace_detail,
    )


def _validate_trace_arguments(args: argparse.Namespace) -> None:
    if args.trace is not None:
        _validate_http_url(args.trace_control_url, "--trace-control-url")
        if args.trace_phase == "warmup" and args.warmup_requests < 1:
            raise ValueError(
                "--trace-phase warmup needs --warmup-requests >= 1; without "
                "warmup requests there is no warmup phase to profile"
            )
        if args.trace_dir is not None and not Path(args.trace_dir).is_dir():
            _print_warning(
                f"--trace-dir {args.trace_dir} does not exist yet; traces are found "
                "only if the server creates it and this host can read it"
            )
        return
    options: tuple[tuple[str, object, object], ...] = (
        ("--trace-dir", args.trace_dir, None),
        ("--trace-control-url", args.trace_control_url, None),
        ("--trace-phase", args.trace_phase, "measured"),
        ("--trace-max-seconds", args.trace_max_seconds, None),
        ("--trace-max-bytes", args.trace_max_bytes, None),
        ("--trace-device-uuid", args.trace_device_uuid, []),
        ("--trace-detail", args.trace_detail, "launch"),
    )
    given = [flag for flag, value, default in options if value != default]
    if given:
        raise ValueError(f"{', '.join(given)} needs --trace")


def cmd_profile(args: argparse.Namespace) -> int:
    """Run active inference profiling."""
    with _usage_errors():
        _validate_profile_arguments(args)
    _warn_about_short_prompts(args)
    _warn_about_repeated_prompts(args)
    if args.cache_state == COLD and args.cache_reset_url is None:
        _print_warning(
            "--cache-state cold without --cache-reset-url: nothing will reset "
            "the cache, and each case records its cache state as unverified"
        )
    with _usage_errors():
        profiler = InferenceProfiler(_profile_config(args), on_warning=_print_warning)
    report = profiler.run()
    print(format_analysis_text(report))
    print(f"Artifact saved to: {Path(args.output)}")
    summary = report.get("summary", {})
    if (
        int(summary.get("total_requests", 0)) > 0
        and int(summary.get("successful_requests", 0)) == 0
    ):
        # The run worked; what it measured is a server that failed every request.
        print(
            "Findings: no measured inference requests succeeded; "
            "the report above gives failures by status",
            file=sys.stderr,
        )
        return int(ExitCode.FINDINGS)
    return int(ExitCode.OK)


@contextmanager
def _usage_errors() -> Iterator[None]:
    """Report a setting the profile cannot use as a usage error.

    Flags are checked, and the profile built from them, before anything is
    sent, so a ``ValueError`` here is about the settings. A missing optional
    package, such as an explicitly requested tokenizer, is a usage error too.
    An input file the settings name keeps its ``InferInputError``.
    """
    try:
        yield
    except (InferUsageError, InferInputError):
        raise
    except ValueError as exc:
        raise InferUsageError(str(exc)) from exc
    except ImportError as exc:
        raise InferUsageError(
            f"{exc}; install it or choose another --tokenizer or --system-sampler"
        ) from exc


def _profile_config(args: argparse.Namespace) -> ProfileConfig:
    endpoint = resolve_endpoint(endpoint=args.endpoint, base_url=args.base_url)
    return ProfileConfig(
        endpoint=endpoint,
        model=args.model,
        concurrency=tuple(
            parse_int_list(
                "1" if args.concurrency is None else args.concurrency,
                field_name="concurrency",
            )
        ),
        input_tokens=tuple(
            parse_int_list(args.input_tokens, field_name="input-tokens")
        ),
        output_tokens=tuple(
            parse_int_list(args.output_tokens, field_name="output-tokens")
        ),
        duration_seconds=args.duration,
        request_count=_request_count(args),
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
        run_id=args.run_id,
        seed=int(args.seed),
        arrival_mode=args.arrival,
        rates=(
            tuple(parse_float_list(args.rate, field_name="rate")) if args.rate else ()
        ),
        burst_size=args.burst_size,
        burst_interval_seconds=args.burst_interval,
        arrival_trace=_arrival_trace(args),
        max_in_flight=128 if args.max_in_flight is None else int(args.max_in_flight),
        overflow=args.overflow or "wait",
        drain_timeout_seconds=args.drain_timeout,
        prompt_mode=args.prompt_mode,
        shared_prefix_ratio=args.shared_prefix_ratio,
        prefix_groups=args.prefix_groups,
        cache_state=args.cache_state,
        cache_reset_url=args.cache_reset_url,
        extra_body=_extra_body(args.extra_body),
        vllm_metrics_url=resolve_metrics_url(endpoint, args.vllm_metrics),
        vllm_metrics_interval_seconds=(
            1.0 if args.vllm_metrics_interval is None else args.vllm_metrics_interval
        ),
        vllm_spans_listen=args.vllm_spans_listen,
        vllm_spans_drain_seconds=(
            6.0 if args.vllm_spans_drain is None else args.vllm_spans_drain
        ),
        trace=_trace_config(args, endpoint),
        vllm_execution_dir=(
            Path(args.vllm_execution_dir) if args.vllm_execution_dir else None
        ),
    )


def _extra_body(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--extra-body is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("--extra-body must be a JSON object")
    return value


def _request_count(args: argparse.Namespace) -> int | None:
    """Measured requests per case; None means the duration or trace decides."""
    if args.duration is not None:
        return None
    if args.requests is not None:
        return int(args.requests)
    return None if args.arrival == REPLAY else 1


def _arrival_trace(args: argparse.Namespace) -> ArrivalTrace | None:
    if args.arrival != REPLAY:
        return None
    try:
        return load_arrival_trace(args.arrival_trace, case_id=args.arrival_trace_case)
    except OSError as exc:
        raise InferInputError(
            f"--arrival-trace {args.arrival_trace}: {exc.strerror or exc}"
        ) from exc
    except InferUsageError as exc:
        raise InferUsageError(f"--arrival-trace {args.arrival_trace}: {exc}") from exc
    except ValueError as exc:
        raise InferInputError(f"--arrival-trace {args.arrival_trace}: {exc}") from exc


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze an inference JSONL artifact."""
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        return int(ExitCode.INVALID_INPUT)
    report = analyze_inference_events(
        input_path,
        server_telemetry_paths=args.server_telemetry,
        direct_server=args.direct_server,
        clock_offset_ns=args.clock_offset_ns,
        clock_uncertainty_ns=args.clock_uncertainty_ns,
        vllm_span_paths=args.vllm_spans,
    )
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
    return int(ExitCode.OK)


def cmd_collect_server(args: argparse.Namespace) -> int:
    """Collect telemetry on the server while a profile uses the same run ID."""
    stop_event = threading.Event()
    previous_handlers = _stop_on_signals(stop_event)
    try:
        result = collect_server_telemetry(
            run_id=args.run_id,
            pid=args.pid,
            output_path=args.output,
            interval_seconds=args.interval,
            duration_seconds=args.duration,
            device_index=args.device_index,
            device_uuid=args.device_uuid,
            no_gpu=args.no_gpu,
            replica_id=args.replica_id,
            rank=args.rank,
            group_id=args.group_id,
            world_size=args.world_size,
            stop_event=stop_event,
            on_warning=_print_warning,
        )
    except NvmlUnavailableError as exc:
        raise InferUsageError(
            f"{exc}; pass --no-gpu to collect without GPU memory"
        ) from exc
    finally:
        _restore_signal_handlers(previous_handlers)
    print(
        f"Collected {result.polls} server polls to: {Path(args.output)} "
        f"(stopped: {result.stop_reason})"
    )
    return _collection_exit_code(result)


def _collection_exit_code(result: CollectionResult) -> int:
    if result.stop_reason == STOP_GPU_IDENTITY_CHANGED:
        # What was recorded is sound; the server's GPU changed under it.
        print(
            f"Findings: GPU identity changed ({result.detail}); later polls were "
            "not recorded and later case windows will not be joined",
            file=sys.stderr,
        )
        return int(ExitCode.FINDINGS)
    if result.stop_reason == STOP_SERVER_PROCESS_ENDED:
        _print_warning(
            f"{result.detail}; case windows that extend past the last "
            "confirmed poll will not be joined"
        )
    return int(ExitCode.OK)


def _print_warning(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def _stop_on_signals(stop_event: threading.Event) -> dict[int, Any]:
    """Turn Ctrl+C and SIGTERM into a clean stop instead of a traceback."""
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(
            signum, lambda _signum, _frame: stop_event.set()
        )
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _validate_profile_arguments(args: argparse.Namespace) -> None:
    if args.duration is not None and args.duration <= 0:
        raise ValueError("--duration must be > 0")
    if args.requests is not None and args.requests <= 0:
        raise ValueError("--requests must be >= 1")
    # --requests 1 was the default, so command lines that combined it with
    # --duration keep working.
    if args.duration is not None and args.requests not in (None, 1):
        raise ValueError("Use either --duration or --requests, not both")
    _validate_arrival_arguments(args)
    _validate_prompt_arguments(args)
    _validate_http_url(args.cache_reset_url, "--cache-reset-url")
    _validate_trace_arguments(args)
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests must be >= 0")
    if args.sample_interval <= 0:
        raise ValueError("--sample-interval must be > 0")
    _validate_vllm_metrics_arguments(args)
    _validate_vllm_span_arguments(args)
    _validate_execution_dir_argument(args)


def _validate_execution_dir_argument(args: argparse.Namespace) -> None:
    directory = args.vllm_execution_dir
    if directory is not None and not Path(directory).is_dir():
        _print_warning(
            f"--vllm-execution-dir {directory} does not exist yet; the execution "
            "log is imported only if the hook writes it there and this host can "
            "read it"
        )


def _validate_vllm_metrics_arguments(args: argparse.Namespace) -> None:
    if args.vllm_metrics not in (None, AUTO_METRICS_URL):
        _validate_http_url(args.vllm_metrics, "--vllm-metrics")
    interval = args.vllm_metrics_interval
    if interval is None:
        return
    if args.vllm_metrics is None:
        raise ValueError("--vllm-metrics-interval only applies with --vllm-metrics")
    if not math.isfinite(interval) or interval < 0.1:
        raise ValueError("--vllm-metrics-interval must be a number of seconds >= 0.1")


def _validate_vllm_span_arguments(args: argparse.Namespace) -> None:
    if args.vllm_spans_listen is not None:
        parse_listen_address(args.vllm_spans_listen)
    drain = args.vllm_spans_drain
    if drain is None:
        return
    if args.vllm_spans_listen is None:
        raise ValueError("--vllm-spans-drain only applies with --vllm-spans-listen")
    if not math.isfinite(drain) or drain < 0:
        raise ValueError("--vllm-spans-drain must be a number of seconds >= 0")


def _validate_arrival_arguments(args: argparse.Namespace) -> None:
    """Name the flag a setting belongs to before the arrival spec checks values."""
    mode = args.arrival
    _flag_for(mode, RATE_MODES, args.rate, "--rate")
    _flag_for(mode, {BURST}, args.burst_size, "--burst-size")
    _flag_for(mode, {BURST}, args.burst_interval, "--burst-interval")
    _flag_for(mode, {REPLAY}, args.arrival_trace, "--arrival-trace")
    if args.arrival_trace_case is not None and mode != REPLAY:
        raise ValueError("--arrival-trace-case only applies to --arrival replay")
    _validate_loop_flags(args)
    if args.max_in_flight is not None and args.max_in_flight < 1:
        raise ValueError("--max-in-flight must be >= 1")
    if args.drain_timeout is not None and args.drain_timeout <= 0:
        raise ValueError("--drain-timeout must be > 0")


def _flag_for(
    mode: str, modes: Any, value: object, flag: str, option: str = "--arrival"
) -> None:
    if mode in modes and value is None:
        raise ValueError(f"{option} {mode} needs {flag}")
    if mode not in modes and value is not None:
        raise ValueError(f"{flag} does not apply to {option} {mode}")


def _validate_prompt_arguments(args: argparse.Namespace) -> None:
    _flag_for(
        args.prompt_mode,
        {SHARED_PREFIX},
        args.shared_prefix_ratio,
        "--shared-prefix-ratio",
        "--prompt-mode",
    )
    if args.prefix_groups is not None and args.prompt_mode != SHARED_PREFIX:
        raise ValueError("--prefix-groups only applies to --prompt-mode shared-prefix")


def _validate_http_url(url: str | None, flag: str) -> None:
    if url is not None and urllib.parse.urlparse(url).scheme not in {"http", "https"}:
        raise ValueError(f"{flag} must use http:// or https://")


def _validate_loop_flags(args: argparse.Namespace) -> None:
    """Closed loops take workers; open loops take an in-flight limit."""
    if args.arrival == CLOSED:
        for flag, value in (
            ("--max-in-flight", args.max_in_flight),
            ("--overflow", args.overflow),
        ):
            if value is not None:
                raise ValueError(f"{flag} applies to open-loop --arrival modes")
    elif args.concurrency is not None:
        raise ValueError(
            "--concurrency applies to --arrival closed; open-loop arrivals "
            "use --max-in-flight"
        )


def _warn_about_short_prompts(args: argparse.Namespace) -> None:
    if args.prompt_mode == REPEAT:
        return
    lengths = parse_int_list(args.input_tokens, field_name="input-tokens")
    if min(lengths) < MIN_CONTROLLED_TOKENS:
        _print_warning(
            f"--input-tokens below {MIN_CONTROLLED_TOKENS} leaves little room "
            f"beside the nonce that --prompt-mode {args.prompt_mode} adds; "
            "prompts may exceed the target and the shared share may drift"
        )


def _warn_about_repeated_prompts(args: argparse.Namespace) -> None:
    if args.prompt_mode == REPEAT or args.cache_reset_url is not None:
        return
    _print_warning(
        f"prompts are the same for every run with --seed {args.seed}; a server "
        "that already served this workload starts with them cached. Pass "
        "--cache-reset-url or change --seed to start cold"
    )


def cmd_import_trace(args: argparse.Namespace) -> int:
    """Append profiler-trace GPU activity to an existing inference artifact."""
    capture = import_traces_into_artifact(
        args.artifact,
        args.traces,
        device_uuids=parse_device_uuids(args.device_uuid),
        detail=args.detail,
        envelope_path=args.envelope,
        execution_dir=args.vllm_execution_dir,
    )
    for summary in (capture.summary or {}).get("traces", []):
        _print_trace_summary(summary)
    for path in (capture.summary or {}).get("already_imported", []):
        print(f"Skipped {path}: already imported into this run")
    return int(ExitCode.OK)


def cmd_import_execution(args: argparse.Namespace) -> int:
    """Append a vLLM execution log's final steps to an existing artifact."""
    capture = import_execution_into_artifact(
        args.artifact,
        args.directory,
        raw_foreign_ids=args.raw_foreign_ids,
        envelope_path=args.envelope,
        server_stopped=args.server_stopped,
    )
    _print_execution_summary((capture.summary or {}).get("execution", {}))
    return int(ExitCode.OK)


def _print_execution_summary(summary: dict[str, Any]) -> None:
    counts = summary.get("records", {})
    print(
        f"Imported execution log {summary.get('directory')}: "
        f"{counts.get('iterations', 0)} iterations, "
        f"{counts.get('memberships', 0)} memberships, "
        f"{counts.get('requests', 0)} requests, "
        f"{counts.get('clock_alignment', 0)} clock alignments"
    )
    for name, epoch in sorted(summary.get("epochs", {}).items()):
        print(f"  {name}: {_epoch_line(epoch)}")
        for error in epoch.get("errors", []):
            print(f"    error: {error}")
    for note in summary.get("notes", []):
        print(f"  note: {note}")


def _epoch_line(epoch: dict[str, Any]) -> str:
    state = _state_text(epoch)
    if not epoch.get("reduced"):
        return f"{epoch.get('role')} epoch, {state}; not reduced"
    waiting = epoch.get("iterations_pending", 0)
    line = (
        f"{state}; kept {epoch.get('iterations_kept', 0)} steps "
        f"({epoch.get('iterations_incomplete', 0)} incomplete), {waiting} pending, "
        f"{epoch.get('iterations_already_imported', 0)} already imported, "
        f"{epoch.get('foreign_only_counted', 0)} foreign-only counted, "
        f"{epoch.get('empty_counted', 0)} empty; "
        f"high-water seq {epoch.get('high_water_seq')}"
    )
    dropped = sum(int(value) for value in (epoch.get("dropped") or {}).values())
    if dropped or epoch.get("gaps"):
        line += (
            f"; {dropped} records dropped by the hook, {epoch.get('gaps', 0)} missing"
        )
    withheld = epoch.get("withheld") or {}
    if any(withheld.values()):
        line += (
            f"; no epoch key: {withheld.get('memberships', 0)} memberships, "
            f"{withheld.get('executions', 0)} requests and "
            f"{withheld.get('foreign_only_steps', 0)} steps of other clients withheld"
        )
    return line


def _state_text(epoch: dict[str, Any]) -> str:
    """The epoch's liveness; an unjudged one says why and what to do."""
    state = str(epoch.get("state"))
    if state != "unknown":
        return state
    reason = epoch.get("state_reason") or "liveness not judged"
    return (
        f"unknown ({reason}: pending steps wait; pass --server-stopped if the "
        "server that wrote this log has stopped)"
    )


def _print_trace_summary(summary: dict[str, Any]) -> None:
    if summary.get("skipped") == "not_exported":
        print(
            f"Registered {summary['file']} without importing it; export it with "
            "`nsys export --type sqlite` and import the .sqlite file"
        )
        return
    if summary.get("skipped"):
        print(
            f"Registered trace {summary['file']} ({summary['bytes']} bytes) "
            f"without importing it: over the {summary['skipped']} bound"
        )
        return
    unresolved = sum(summary["unresolved_gpu_events"].values())
    print(
        f"Imported trace {summary['trace_id'] or '(no trace id)'}: "
        f"{summary['gpu_events']} GPU events as {summary['activity_records']} "
        f"records; {summary['linked_gpu_events']} linked to iterations, "
        f"{unresolved} unresolved"
    )
    for reason, count in summary["unresolved_gpu_events"].items():
        print(f"  unresolved ({reason}): {count}")
    _print_trace_devices(summary)
    for note in summary.get("notes", []):
        print(f"  note: {note}")


def _print_trace_devices(summary: dict[str, Any]) -> None:
    for device, values in summary["devices"].items():
        uuid = values["device_uuid"] or "unknown UUID, not measured"
        if values.get("device_uuid_source") == "execution_log":
            uuid += ", from the vLLM execution log"
        pid, _, ordinal = str(device).rpartition("/")
        label = f"process {pid} device {ordinal}" if pid else f"device {device}"
        print(
            f"  {label} ({uuid}): busy {values['busy_ns'] / 1e6:.3f} ms, "
            f"summed {values['summed_ns'] / 1e6:.3f} ms"
        )
    binding = summary.get("execution_log")
    if binding is not None and binding.get("status") != "bound":
        print(f"  execution log: {_binding_note(binding)}")


def _binding_note(binding: dict[str, Any]) -> str:
    parts = []
    if binding.get("unmatched"):
        pids = ", ".join(str(pid) for pid in binding["unmatched"])
        parts.append(f"no worker epoch covers process {pids}")
    for pid, epochs in binding.get("ambiguous", {}).items():
        parts.append(f"process {pid} matches {len(epochs)} worker epochs")
    if not parts:
        parts.append(f"status {binding.get('status')}")
    return "; ".join(parts) + "; give --device-uuid for it"
