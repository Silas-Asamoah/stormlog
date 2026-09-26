"""Command-line entry points for native probe research artifacts."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Sequence

from .analysis import analyze_trials
from .models import (
    ArtifactExpectation,
    CommandSpec,
    ExperimentMode,
    ProcessRole,
    ProcessRoleSpec,
    TrialSpec,
    WorkloadId,
)
from .normalization import normalize_trial
from .planning import build_plan
from .preflight import collect_environment, write_manifest
from .runner import run_trial
from .validation import build_unvalidated_matrix, validate_matrix_promotions

_SCHEMAS = Path(__file__).with_name("schemas")
jsonschema = importlib.import_module("jsonschema")


def main(argv: Sequence[str] | None = None) -> int:
    """Run a research command and return a process exit code."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "preflight":
        return _preflight(arguments)
    handlers = {
        "preflight": _preflight,
        "plan": _plan,
        "run": _run,
        "normalize": _normalize,
        "analyze": _analyze,
        "initialize-matrix": _initialize_matrix,
        "validate-matrix": _validate_matrix,
    }
    return handlers[arguments.command](arguments)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="record host capabilities")
    preflight.add_argument("--host-id", required=True)
    preflight.add_argument("--repository", type=Path, default=Path.cwd())
    preflight.add_argument("--output", required=True, type=Path)

    plan = subparsers.add_parser("plan", help="create an immutable trial plan")
    plan.add_argument("--configuration-id", required=True)
    plan.add_argument("--vendor", required=True, choices=("nvidia", "amd"))
    plan.add_argument(
        "--workload",
        action="append",
        required=True,
        choices=[row.value for row in WorkloadId],
    )
    plan.add_argument(
        "--mode",
        action="append",
        required=True,
        choices=[row.value for row in ExperimentMode],
    )
    plan.add_argument("--repetitions", type=int, default=5)
    plan.add_argument("--seed", type=int, default=118)
    plan.add_argument("--environment-artifact", required=True)
    plan.add_argument("--artifact-root", required=True, type=Path)
    plan.add_argument("--cupti-library", type=Path)
    plan.add_argument("--output", required=True, type=Path)

    run = subparsers.add_parser("run", help="execute every trial in a plan")
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)

    normalize = subparsers.add_parser("normalize", help="normalize one trial")
    normalize.add_argument("--trial", required=True, type=Path)
    normalize.add_argument("--output", required=True, type=Path)

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

    validate = subparsers.add_parser(
        "validate-matrix", help="apply only fully evidenced matrix promotions"
    )
    validate.add_argument("--matrix", required=True, type=Path)
    validate.add_argument("--promotions", required=True, type=Path)
    validate.add_argument("--repository", type=Path, default=Path.cwd())
    validate.add_argument("--output", required=True, type=Path)
    return parser


def _preflight(arguments: argparse.Namespace) -> int:
    manifest = collect_environment(arguments.host_id, arguments.repository)
    _validate(manifest, "environment.schema.json")
    checksum = write_manifest(arguments.output, manifest)
    print(json.dumps({"output": str(arguments.output), "sha256": checksum}))
    return 0


def _analyze(arguments: argparse.Namespace) -> int:
    trials = _read_jsonl(arguments.input)
    _validate_analysis_inputs(trials)
    analysis = analyze_trials(trials, bootstrap_samples=arguments.bootstrap_samples)
    checksum = write_manifest(arguments.output, analysis)
    print(json.dumps({"output": str(arguments.output), "sha256": checksum}))
    return 0


def _plan(arguments: argparse.Namespace) -> int:
    plan = build_plan(
        configuration_id=arguments.configuration_id,
        vendor=arguments.vendor,
        workloads=[WorkloadId(row) for row in arguments.workload],
        modes=[ExperimentMode(row) for row in arguments.mode],
        repetitions=arguments.repetitions,
        seed=arguments.seed,
        environment_artifact=arguments.environment_artifact,
        artifact_root=arguments.artifact_root,
        cupti_library=arguments.cupti_library,
    )
    environment_path = Path(arguments.environment_artifact)
    if environment_path.is_file():
        environment = _read_object(environment_path)
        source = environment.get("source")
        if isinstance(source, dict) and isinstance(source.get("revision"), str):
            plan["revision"] = source["revision"]
    _validate(plan, "plan.schema.json")
    checksum = write_manifest(arguments.output, plan)
    print(json.dumps({"output": str(arguments.output), "sha256": checksum}))
    return 0


def _run(arguments: argparse.Namespace) -> int:
    plan = _read_object(arguments.plan)
    _validate(plan, "plan.schema.json")
    root = Path(plan["artifact_root"])
    results = []
    failed = False
    for row in plan["trials"]:
        spec = TrialSpec(
            trial_id=row["trial_id"],
            configuration_id=row["configuration_id"],
            workload_id=WorkloadId(row["workload_id"]),
            mode=ExperimentMode(row["mode"]),
            repetition=row["repetition"],
            command=CommandSpec.from_mapping(row["command"]),
            expected_artifacts=tuple(
                ArtifactExpectation(**item) for item in row["expected_artifacts"]
            ),
            process_roles=tuple(
                ProcessRoleSpec(
                    role=ProcessRole(item["role"]),
                    discovery=item["discovery"],
                    argv_contains=item.get("argv_contains"),
                )
                for item in row["process_roles"]
            ),
            measurement_range_id=row["measurement_range_id"],
            pressure_controls=row.get("pressure_controls", {}),
        )
        manifest = run_trial(spec, root, revision=plan.get("revision"))
        _validate(manifest, "trial.schema.json")
        results.append(
            {
                "trial_id": spec.trial_id,
                "path": str(
                    root
                    / spec.configuration_id
                    / "trials"
                    / spec.trial_id
                    / "manifest.json"
                ),
                "status": manifest["status"],
            }
        )
        failed = failed or manifest["status"] != "pass"
    checksum = write_manifest(
        arguments.output,
        {
            "schema_version": 1,
            "artifact_kind": "native_probe_run_index",
            "plan": str(arguments.plan),
            "trial_manifests": results,
            "status": "fail" if failed else "pass",
        },
    )
    print(json.dumps({"output": str(arguments.output), "sha256": checksum}))
    return 1 if failed else 0


def _normalize(arguments: argparse.Namespace) -> int:
    trial = _read_object(arguments.trial)
    if trial.get("artifact_kind") != "native_probe_trial":
        raise ValueError("normalize input must be a native_probe_trial document")
    _validate(trial, "trial.schema.json")
    result = normalize_trial(trial)
    _validate(result, "normalized.schema.json")
    checksum = write_manifest(arguments.output, result)
    print(json.dumps({"output": str(arguments.output), "sha256": checksum}))
    return 0


def _initialize_matrix(arguments: argparse.Namespace) -> int:
    with arguments.source.open(encoding="utf-8") as source:
        source_matrix = json.load(source)
    matrix = build_unvalidated_matrix(source_matrix, arguments.environment_artifact)
    checksum = write_manifest(arguments.output, matrix)
    print(json.dumps({"output": str(arguments.output), "sha256": checksum}))
    return 0


def _validate_matrix(arguments: argparse.Namespace) -> int:
    matrix = _read_object(arguments.matrix)
    _validate(matrix, "capability_matrix.schema.json")
    promotions = _read_object(arguments.promotions).get("promotions", [])
    result = validate_matrix_promotions(matrix, promotions, arguments.repository)
    _validate(result, "capability_matrix.schema.json")
    checksum = write_manifest(arguments.output, result)
    print(json.dumps({"output": str(arguments.output), "sha256": checksum}))
    return 0


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _validate(value: object, schema_name: str) -> None:
    schema = _read_object(_SCHEMAS / schema_name)
    jsonschema.Draft202012Validator(schema).validate(value)


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


def _validate_analysis_inputs(trials: list[dict[str, Any]]) -> None:
    if not trials:
        raise ValueError("analysis requires at least one trial")
    seen: set[str] = set()
    for trial in trials:
        kind = trial.get("artifact_kind")
        if kind == "native_probe_trial":
            _validate(trial, "trial.schema.json")
        elif kind == "native_probe_normalized_trial":
            _validate(trial, "normalized.schema.json")
        else:
            raise ValueError("analysis input has an unsupported document kind")
        trial_id = trial.get("trial_id")
        if not isinstance(trial_id, str) or trial_id in seen:
            raise ValueError("analysis input has a missing or duplicate trial_id")
        seen.add(trial_id)


if __name__ == "__main__":
    raise SystemExit(main())
