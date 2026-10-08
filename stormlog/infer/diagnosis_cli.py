"""``stormlog infer diagnose``: explain an artifact, or inspect a report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..exit_codes import ExitCode
from ..report import load_report, write_report
from .diagnosis import DiagnoseOptions, diagnose_artifact
from .diagnosis_text import InspectError, inspect, render_text
from .errors import InferInputError, InferUsageError


def add_diagnose_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "diagnose",
        help="Explain slow requests or windows of an inference artifact",
        description=(
            "Explain an artifact's incidents, or declared windows, requests or "
            "cases, from the evidence it holds. Exits 3 when a finding is a "
            "warning. With --inspect, print the records behind a finding of a "
            "saved report instead."
        ),
    )
    parser.add_argument("artifact", nargs="?", help="Inference JSONL to diagnose")
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        metavar="START,END",
        help="A declared window, in ns on the artifact's clock; repeatable",
    )
    parser.add_argument(
        "--request", action="append", default=[], help="A request ID; repeatable"
    )
    parser.add_argument("--case", action="append", default=[], help="A case ID")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Base window of automatic incident selection (default 1)",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        metavar="FILE",
        help="JSON object overriding threshold table entries by key",
    )
    parser.add_argument(
        "--metrics-from-engine",
        action="store_true",
        help=(
            "Assert that the scraped metrics exporter is the engine whose hook "
            "log was imported, so its metrics may witness hook findings"
        ),
    )
    parser.add_argument(
        "--server-telemetry",
        action="append",
        default=[],
        metavar="JSONL",
        help="On-host collector artifact for the memory ledger; repeatable",
    )
    parser.add_argument("--format", choices=["txt", "json"], default="txt")
    parser.add_argument(
        "--output", default=None, help="Write the validated JSON report here"
    )
    parser.add_argument(
        "--inspect",
        nargs=2,
        metavar=("REPORT", "FINDING_ID"),
        default=None,
        help="Print the records behind a finding of a saved report",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="With --inspect, every supporting record, not the display subset",
    )


def cmd_diagnose(args: argparse.Namespace, argv: list[str] | None = None) -> int:
    if args.inspect is not None:
        return _inspect(args)
    if args.artifact is None:
        raise InferUsageError("diagnose needs an artifact, or --inspect")
    output = Path(args.output) if args.output else None
    options = DiagnoseOptions(
        request_ids=tuple(args.request),
        case_ids=tuple(args.case),
        thresholds=_thresholds(args),
        metrics_from_engine=args.metrics_from_engine,
        server_telemetry=tuple(args.server_telemetry),
        report_dir=output.parent if output is not None else None,
        argv=tuple(argv) if argv is not None else None,
    )
    try:
        report = diagnose_artifact(
            args.artifact, windows=_windows(args.window), options=options
        )
    except OSError as exc:
        raise InferInputError(f"cannot read {args.artifact}: {exc}") from exc
    except ValueError as exc:
        raise InferUsageError(str(exc)) from exc
    if output is not None:
        write_report(output, report)
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report), end="")
    return int(report["verdict"]["exit_code"])


def _inspect(args: argparse.Namespace) -> int:
    report_path, finding_id = Path(args.inspect[0]), args.inspect[1]
    try:
        report = load_report(report_path)
    except ValueError as exc:
        raise InferInputError(str(exc)) from exc
    try:
        for line in inspect(report, finding_id, report_path, everything=args.all):
            print(line)
    except InspectError as exc:
        raise InferInputError(str(exc)) from exc
    return int(ExitCode.OK)


def _windows(values: list[str]) -> list[tuple[int, int]]:
    windows = []
    for value in values:
        start, _, end = value.partition(",")
        try:
            windows.append((int(start), int(end)))
        except ValueError as exc:
            raise InferUsageError(f"--window {value!r} is not START,END") from exc
    return windows


def _thresholds(args: argparse.Namespace) -> dict[str, float]:
    overrides: dict[str, float] = {}
    if args.thresholds:
        try:
            data = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise InferInputError(f"cannot read {args.thresholds}: {exc}") from exc
        if not isinstance(data, dict) or not all(
            isinstance(v, (int, float)) for v in data.values()
        ):
            raise InferUsageError("--thresholds must hold a JSON object of numbers")
        overrides.update({str(k): float(v) for k, v in data.items()})
    if args.window_seconds is not None:
        overrides["selection.window_seconds"] = args.window_seconds
    return overrides


__all__ = ["add_diagnose_parser", "cmd_diagnose"]
