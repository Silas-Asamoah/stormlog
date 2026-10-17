"""Dedicated MLX CLI; help and offline commands never discover MLX."""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import TYPE_CHECKING, Any, Sequence

from stormlog.exit_codes import ExitCode

if TYPE_CHECKING:
    from .artifacts import CaptureOptions
    from .models import MemorySnapshot


def _duration(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return result


def _positive(value: str) -> float:
    result = _duration(value)
    if result == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _count(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlxmemprof",
        description="Observe the MLX allocator in this process; read saved artifacts offline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("info", "monitor", "track", "run", "analyze", "diagnose"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--json", action="store_true", help="Print machine-readable JSON"
        )
        command.add_argument("--format", choices=("json", "txt"), default="txt")
        command.add_argument(
            "--output",
            help="New bundle directory; an existing directory receives a unique child",
        )
        if name in ("monitor", "track", "run", "diagnose"):
            _capture_arguments(command)
        if name in ("monitor", "track", "diagnose"):
            command.add_argument(
                "--duration",
                type=_duration,
                default=0 if name == "diagnose" else 10,
                help="Seconds to observe this command's own allocator",
            )
        if name == "run":
            command.add_argument(
                "script_args",
                nargs=argparse.REMAINDER,
                help="-- script.py [workload arguments]; executed in this interpreter",
            )
        elif name == "analyze":
            command.add_argument(
                "input", help="MLX telemetry, profile JSON or capture bundle"
            )
            command.add_argument(
                "--session-id",
                help="Select one session; default follows the shared session policy",
            )
    return parser


def _capture_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--interval", type=_positive, default=0.1)
    parser.add_argument("--max-history", type=_count, default=1000)
    parser.add_argument(
        "--alert-threshold-mb",
        type=_positive,
        help="Opt-in alert-only active allocation threshold in MiB",
    )
    parser.add_argument("--rollover-max-events", type=_count, default=10000)
    parser.add_argument("--retention-max-files", type=_count, default=8)
    parser.add_argument("--max-buffer-bytes", type=_count, default=4 * 1024**2)
    parser.add_argument("--oom-max-dumps", type=_count, default=5)
    parser.add_argument("--oom-max-total-mb", type=_count, default=256)
    parser.add_argument("--job-id")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--world-size", type=_count)


def _emit(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    if args.json or args.format == "json":
        print(json.dumps(payload, indent=2, allow_nan=False))
    else:
        print(payload.get("summary", "MLX allocator in this process"))
        if payload.get("artifact_dir"):
            print(f"Artifacts: {payload['artifact_dir']}")
        if payload.get("error"):
            print(payload["error"], file=sys.stderr)


def _info(args: argparse.Namespace) -> int:
    from stormlog.report import Artifact, build_report, write_report
    from stormlog.system_info import get_system_info

    from .artifacts import create_artifact_dir
    from .collector import MLXCollector
    from .runtime import MLXRuntime, MLXRuntimeError
    from .utils import get_device_info

    payload: dict[str, Any] = {
        "framework": "mlx",
        "memory_scope": "process_allocator",
        "system": get_system_info(),
    }
    code = ExitCode.OK
    try:
        runtime = MLXRuntime()
        payload.update(
            available=True,
            device=get_device_info(runtime),
            memory_capabilities=MLXCollector(runtime).capabilities,
            summary=f"MLX {runtime.version}: Metal allocator in this process",
        )
    except MLXRuntimeError as exc:
        code = ExitCode.USAGE
        payload.update(
            available=False,
            error=f"{type(exc).__name__}: {exc}",
            summary="MLX runtime unavailable",
        )
    if args.output:
        directory = create_artifact_dir(args.output, "info")
        (directory / "environment.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        write_report(
            directory / "report.json",
            build_report(
                report_kind="mlx_info",
                tool_name="mlxmemprof",
                command="info",
                exit_code=int(code),
                summary=payload["summary"],
                payload=payload,
                artifacts=[Artifact("environment", "environment.json")],
            ),
        )
        payload["artifact_dir"] = str(directory)
    _emit(args, payload)
    return int(code)


def _options(args: argparse.Namespace) -> CaptureOptions:
    from .artifacts import CaptureOptions

    return CaptureOptions(
        **{name: getattr(args, name) for name in CaptureOptions.__dataclass_fields__}
    )


def _capture(args: argparse.Namespace) -> int:
    from stormlog.mlx.diagnose import capture_process
    from stormlog.mlx.runner import run_script

    options = _options(args)
    if args.command == "run":
        script_args = args.script_args
        if script_args and script_args[0] == "--":
            script_args = script_args[1:]
        if not script_args:
            raise ValueError("run requires -- script.py [args...]")
        directory, outcome = run_script(
            script_args[0], script_args[1:], output=args.output, options=options
        )
    else:
        callback = (
            _monitor_sample
            if args.command == "monitor" and not (args.json or args.format == "json")
            else None
        )
        directory, outcome = capture_process(
            output=args.output,
            duration=args.duration,
            options=options,
            command=args.command,
            on_sample=callback,
        )
    _emit(
        args,
        {
            "artifact_dir": str(directory),
            "exit_code": outcome.exit_code,
            "summary": "MLX allocator capture (this process)",
            "error": outcome.error,
            "workload": outcome.workload,
            "artifact_errors": outcome.artifact_errors,
        },
    )
    return outcome.exit_code


def _monitor_sample(snapshot: MemorySnapshot) -> None:
    from stormlog.formatting import format_bytes

    def label(value: int | None) -> str:
        return "unavailable" if value is None else format_bytes(value)

    print(
        f"This process: active {label(snapshot.active_bytes)}, cache {label(snapshot.cache_bytes)}, lifetime peak {label(snapshot.runtime_peak_bytes)}",
        flush=True,
    )


def _analysis(args: argparse.Namespace) -> int:
    from stormlog.report import Artifact, Evidence, Finding, build_report, write_report

    from .analyzer import analyze
    from .artifacts import create_artifact_dir

    directory = create_artifact_dir(args.output, "analyze")
    code = ExitCode.OK
    findings = []
    payload: dict[str, Any] = {}
    artifacts = []
    try:
        payload = analyze(args.input, session_id=args.session_id)
        findings = [
            Finding(
                id=f"mlx.{f['kind']}",
                kind=f["kind"],
                severity="warning",
                title=f["title"],
                evidence=[
                    Evidence(
                        kind="mlx_analysis", path="analysis.json", pointer=f["pointer"]
                    )
                ],
            )
            for f in payload["findings"]
        ]
        code = ExitCode.FINDINGS if findings else ExitCode.OK
    except (OSError, ValueError, KeyError, TypeError) as exc:
        code = ExitCode.INVALID_INPUT
        payload = {"error": f"{type(exc).__name__}: {exc}"}
    if code != ExitCode.INVALID_INPUT:
        try:
            (directory / "analysis.json").write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            artifacts.append(
                Artifact("mlx_analysis", "analysis.json", "stormlog.mlx.analysis", 1)
            )
        except OSError as exc:
            code = ExitCode.ERROR
            findings = []
            payload["error"] = f"Cannot write analysis artifact: {exc}"
    report = build_report(
        report_kind="mlx_analysis",
        tool_name="mlxmemprof",
        command="analyze",
        exit_code=int(code),
        summary=payload.get("error", "Analyzed saved MLX allocator measurements"),
        findings=findings,
        metrics=payload.get("metrics"),
        artifacts=artifacts,
        session_id=payload.get("session", {}).get("session_id"),
        payload=payload,
    )
    write_report(directory / "report.json", report)
    _emit(
        args,
        {
            **payload,
            "summary": report["verdict"]["summary"],
            "artifact_dir": str(directory),
            "exit_code": int(code),
        },
    )
    return int(code)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else int(ExitCode.USAGE)
    try:
        if args.command == "info":
            return _info(args)
        if args.command == "analyze":
            return _analysis(args)
        return _capture(args)
    except KeyboardInterrupt:
        return int(ExitCode.INTERRUPTED)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return int(ExitCode.USAGE)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return int(ExitCode.ERROR)


if __name__ == "__main__":
    raise SystemExit(main())
