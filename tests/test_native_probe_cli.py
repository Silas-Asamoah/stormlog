"""Subprocess integration tests for the native-probe CLI pipeline."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def test_cli_runs_cpu_fixture_through_run_normalize_and_analyze(
    tmp_path: Path,
) -> None:
    plan_path, artifact_root = _fixture_plan(tmp_path)
    run_index = tmp_path / "index.json"
    _run_cli("run", "--plan", plan_path, "--output", run_index)
    index = json.loads(run_index.read_text(encoding="utf-8"))
    assert index["status"] == "pass"
    assert index["schema_version"] == 2
    assert len(index["trial_manifests"]) == 10
    assert isinstance(index["trial_manifests"][0], dict)

    trial_path = Path(index["trial_manifests"][0]["path"])
    normalized_path = tmp_path / "normalized.json"
    _run_cli("normalize", "--trial", trial_path, "--output", normalized_path)
    normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
    assert normalized["status"] == "pass"

    jsonl = tmp_path / "trials.jsonl"
    jsonl.write_text(json.dumps(normalized) + "\n", encoding="utf-8")
    analysis_path = tmp_path / "analysis.json"
    _run_cli("analyze", "--input", jsonl, "--output", analysis_path)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert analysis["trial_count"] == 1
    assert analysis["schema_version"] == 2
    assert artifact_root.exists()


def test_cli_retains_missing_executable_failures_and_index(tmp_path: Path) -> None:
    plan_path, artifact_root = _fixture_plan(
        tmp_path, executable="missing-native-probe"
    )
    run_index = tmp_path / "failed-index.json"

    result = _run_cli("run", "--plan", plan_path, "--output", run_index, check=False)

    assert result.returncode != 0
    index = json.loads(run_index.read_text(encoding="utf-8"))
    assert index["status"] == "fail"
    assert len(index["trial_manifests"]) == 10
    assert all(row["status"] == "fail" for row in index["trial_manifests"])
    assert all(Path(row["path"]).is_file() for row in index["trial_manifests"])
    assert artifact_root.exists()


def test_cli_timeout_retains_trials_and_returns_nonzero(tmp_path: Path) -> None:
    plan_path, _ = _fixture_plan(tmp_path, timeout=True)
    run_index = tmp_path / "timeout-index.json"

    result = _run_cli("run", "--plan", plan_path, "--output", run_index, check=False)

    assert result.returncode != 0
    index = json.loads(run_index.read_text(encoding="utf-8"))
    assert index["status"] == "fail"
    assert all(row["status"] == "timeout" for row in index["trial_manifests"])
    assert all(Path(row["path"]).is_file() for row in index["trial_manifests"])


def test_cli_rejects_malformed_normalize_and_analysis_inputs(tmp_path: Path) -> None:
    malformed = tmp_path / "bad.json"
    malformed.write_text("{}", encoding="utf-8")
    normalized = tmp_path / "normalized.json"
    normalize = _run_cli(
        "normalize", "--trial", malformed, "--output", normalized, check=False
    )
    assert normalize.returncode != 0
    assert not normalized.exists()

    jsonl = tmp_path / "bad.jsonl"
    jsonl.write_text("{}\n", encoding="utf-8")
    analysis = tmp_path / "analysis.json"
    analyze = _run_cli("analyze", "--input", jsonl, "--output", analysis, check=False)
    assert analyze.returncode != 0
    assert not analysis.exists()


def test_cli_rejects_forged_but_checksum_shaped_promotion(tmp_path: Path) -> None:
    matrix = REPOSITORY / "research/native_probes/matrices/stormlog_validated.json"
    matrix_copy = tmp_path / "matrix.json"
    matrix_copy.write_text(matrix.read_text(encoding="utf-8"), encoding="utf-8")
    value = json.loads(matrix_copy.read_text(encoding="utf-8"))
    candidate = value["candidates"][0]
    claim_id = next(iter(candidate["claims"]))
    evidence = []
    for role in ("environment", "command", "raw_artifact", "trial", "analysis"):
        path = tmp_path / f"{role}.json"
        path.write_text("{}", encoding="utf-8")
        evidence.append(
            {
                "role": role,
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    promotion_path = tmp_path / "promotions.json"
    promotion_path.write_text(
        json.dumps(
            {
                "promotions": [
                    {
                        "candidate_id": candidate["id"],
                        "claim_id": claim_id,
                        "evidence": evidence,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "promoted.json"

    result = _run_cli(
        "validate-matrix",
        "--matrix",
        matrix_copy,
        "--promotions",
        promotion_path,
        "--repository",
        tmp_path,
        "--output",
        output,
        check=False,
    )

    assert result.returncode != 0
    assert not output.exists()


def _fixture_plan(
    tmp_path: Path, executable: str | None = None, timeout: bool = False
) -> tuple[Path, Path]:
    artifact_root = tmp_path / "artifacts"
    plan_path = tmp_path / "plan.json"
    _run_cli(
        "plan",
        "--configuration-id",
        "cpu-fixture",
        "--vendor",
        "nvidia",
        "--workload",
        "w1-eager",
        "--mode",
        "off",
        "--mode",
        "public-pytorch",
        "--repetitions",
        "5",
        "--seed",
        "118",
        "--environment-artifact",
        "environment.json",
        "--artifact-root",
        artifact_root,
        "--output",
        plan_path,
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    result = {
        "artifact_kind": "workload_result",
        "metrics": {"latency_ms": 2.5},
        "measurement_window": {
            "range_id": "cpu-fixture",
            "marker": "cpu",
            "warmup_iterations": 1,
            "measured_iterations": 2,
            "host_started_ns": 1,
            "host_finished_ns": 2,
            "clock": "CLOCK_REALTIME",
            "flush_completed": True,
        },
        "ground_truth": {},
    }
    code = (
        "import time; time.sleep(10)"
        if timeout
        else f"import json; print({json.dumps(result)!r})"
    )
    command = [sys.executable, "-c", code]
    for index, row in enumerate(plan["trials"]):
        row["mode"] = "off"
        row["command"] = {
            "argv": [executable] if executable else command,
            "environment": {},
            "timeout_seconds": 0.05 if timeout else 5,
        }
        row["expected_artifacts"] = []
        row["process_roles"] = [{"role": "target", "discovery": "root"}]
        row["trial_id"] = f"cpu-fixture-{index:02d}"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path, artifact_root


def _run_cli(
    *arguments: str | Path, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "research.native_probes.cli", *map(str, arguments)],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=check,
    )
