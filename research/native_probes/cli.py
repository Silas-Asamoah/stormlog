"""Command-line entry points for native probe research artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .analysis import analyze_trials
from .preflight import collect_environment, write_manifest
from .validation import build_unvalidated_matrix


def main(argv: Sequence[str] | None = None) -> int:
    """Run a research command and return a process exit code."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "preflight":
        return _preflight(arguments)
    if arguments.command == "initialize-matrix":
        return _initialize_matrix(arguments)
    return _analyze(arguments)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="record host capabilities")
    preflight.add_argument("--host-id", required=True)
    preflight.add_argument("--repository", type=Path, default=Path.cwd())
    preflight.add_argument("--output", required=True, type=Path)

    analyze = subparsers.add_parser("analyze", help="aggregate trial JSONL")
    analyze.add_argument("--input", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    analyze.add_argument("--bootstrap-samples", type=int, default=10_000)

    matrix = subparsers.add_parser(
        "initialize-matrix",
        help="reset source-backed claims to unvalidated experiment cells",
    )
    matrix.add_argument("--source", required=True, type=Path)
    matrix.add_argument("--environment-artifact", required=True)
    matrix.add_argument("--output", required=True, type=Path)
    return parser


def _preflight(arguments: argparse.Namespace) -> int:
    manifest = collect_environment(arguments.host_id, arguments.repository)
    checksum = write_manifest(arguments.output, manifest)
    print(json.dumps({"output": str(arguments.output), "sha256": checksum}))
    return 0


def _analyze(arguments: argparse.Namespace) -> int:
    trials = _read_jsonl(arguments.input)
    analysis = analyze_trials(trials, bootstrap_samples=arguments.bootstrap_samples)
    checksum = write_manifest(arguments.output, analysis)
    print(json.dumps({"output": str(arguments.output), "sha256": checksum}))
    return 0


def _initialize_matrix(arguments: argparse.Namespace) -> int:
    with arguments.source.open(encoding="utf-8") as source:
        source_matrix = json.load(source)
    matrix = build_unvalidated_matrix(source_matrix, arguments.environment_artifact)
    checksum = write_manifest(arguments.output, matrix)
    print(json.dumps({"output": str(arguments.output), "sha256": checksum}))
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: trial must be a JSON object")
            trials.append(value)
    return trials


if __name__ == "__main__":
    raise SystemExit(main())
