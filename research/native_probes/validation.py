"""Build an experiment-only matrix without promoting literature claims."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, cast

from .models import CommandSpec
from .normalization import validate_measurement_window
from .runner import _artifact_digest
from .schema_validation import validate_document


def build_unvalidated_matrix(
    source_matrix: Mapping[str, Any], environment_artifact: str
) -> dict[str, Any]:
    """Return a matrix whose cells explicitly await compatible-hardware trials.

    Source-backed claims are useful for designing experiments, but they are not
    Stormlog measurements.  This transformation deliberately resets every
    capability cell to ``UNKNOWN`` and points it at the control-host preflight.
    """
    candidates = deepcopy(source_matrix.get("candidates"))
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("source matrix must contain candidates")
    if not environment_artifact or environment_artifact.startswith("/"):
        raise ValueError("environment artifact must be a repository-relative path")

    for candidate in candidates:
        claims = candidate.get("claims")
        if not isinstance(claims, dict) or not claims:
            raise ValueError("every candidate must contain claims")
        for claim in claims.values():
            source_status = claim["status"]
            claim["status"] = "UNKNOWN"
            claim["detail"] = (
                "UNTESTED - COMPATIBLE HARDWARE UNAVAILABLE. "
                f"Source-backed status was {source_status}; no Stormlog trial ran."
            )
            claim["evidence"] = [environment_artifact]
        assumptions = candidate.get("unverified_assumptions")
        if not isinstance(assumptions, list):
            raise ValueError("every candidate must list unverified assumptions")
        assumptions.append(
            "No compatible accelerator trial was executed on the control host."
        )

    return {
        "schema_version": 1,
        "artifact_kind": "native_probe_capability_matrix",
        "matrix_kind": "stormlog_validated",
        "reviewed_on": source_matrix["reviewed_on"],
        "candidates": candidates,
    }


_EVIDENCE_ROLES = ("environment", "command", "raw_artifact", "trial", "analysis")
_CANDIDATE_MODES = {
    "pytorch-kineto": {"public-pytorch"},
    "vllm-proton": {"proton"},
    "direct-cupti": {"direct-cupti"},
    "ebpf-semantic": {"ebpf-semantic"},
    "cupti-ebpf-hybrid": {"hybrid-cupti-ebpf"},
    "neutrino": {"programmable"},
    "rocprofiler": {"amd-rocprofiler"},
    "no-native-collector": {"off"},
}


def validate_matrix_promotions(
    matrix: Mapping[str, Any], promotions: list[Mapping[str, Any]], repository: Path
) -> dict[str, Any]:
    """Promote claims only when all immutable evidence roles are verifiable."""
    if matrix.get("matrix_kind") != "stormlog_validated":
        raise ValueError(
            "promotions may be applied only to stormlog_validated matrices"
        )
    result: dict[str, Any] = deepcopy(dict(matrix))
    candidates = {row["id"]: row for row in result.get("candidates", [])}
    for promotion in promotions:
        candidate_id, claim_id, claims = _promotion_target(promotion, candidates)
        if claims[claim_id].get("status") != "UNKNOWN":
            raise ValueError("only UNKNOWN matrix cells may be promoted")
        evidence = _promotion_evidence(promotion, repository)
        claims[claim_id]["status"] = "STORMLOG_VALIDATED"
        claims[claim_id]["detail"] = str(
            promotion.get("detail", "validated by hardware experiment")
        )
        claims[claim_id]["evidence"] = [
            str(row.get("path") or row["durable_location"]) for row in evidence
        ]
        claims[claim_id]["evidence_roles"] = [
            {"role": row["role"], "path": row["path"], "sha256": row["sha256"]}
            for row in evidence
        ]
    return result


def _promotion_target(
    promotion: Mapping[str, Any], candidates: Mapping[str, Any]
) -> tuple[str, str, dict[str, Any]]:
    candidate_id = promotion.get("candidate_id")
    claim_id = promotion.get("claim_id")
    if not isinstance(candidate_id, str) or candidate_id not in candidates:
        raise ValueError(f"unknown candidate: {candidate_id}")
    claims = candidates[candidate_id].get("claims", {})
    if not isinstance(claim_id, str) or claim_id not in claims:
        raise ValueError(f"unknown claim: {candidate_id}.{claim_id}")
    return candidate_id, claim_id, claims


def _promotion_evidence(
    promotion: Mapping[str, Any], repository: Path
) -> list[Mapping[str, Any]]:
    rows, by_role = _evidence_rows_by_role(promotion)
    paths = [
        _validate_evidence_row(role, by_role[role], repository)
        for role in _EVIDENCE_ROLES
    ]
    environment, plan, trial, analysis = (
        _read_document(paths[index]) for index in (0, 1, 3, 4)
    )
    _validate_document_kinds(environment, plan, trial, analysis)
    for document, schema_name in (
        (environment, "environment.schema.json"),
        (plan, "plan.schema.json"),
        (trial, "trial.schema.json"),
        (analysis, "analysis.schema.json"),
    ):
        validate_document(document, schema_name)
    _validate_evidence_links(
        promotion,
        environment,
        str(by_role["environment"]["path"]),
        plan,
        repository,
        str(by_role["raw_artifact"]["path"]),
        paths[2],
        trial,
        analysis,
    )
    return rows


def _evidence_rows_by_role(
    promotion: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    evidence = promotion.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != len(_EVIDENCE_ROLES):
        raise ValueError("promotion requires exactly five unique evidence roles")
    rows = [row for row in evidence if isinstance(row, Mapping)]
    if len(rows) != len(evidence) or any(
        not isinstance(row.get("role"), str) for row in rows
    ):
        raise ValueError("promotion requires exactly five unique evidence roles")
    by_role = cast(dict[str, Mapping[str, Any]], {row["role"]: row for row in rows})
    if tuple(sorted(by_role)) != tuple(sorted(_EVIDENCE_ROLES)):
        raise ValueError("promotion requires exactly five unique evidence roles")
    return rows, by_role


def _validate_evidence_row(role: str, row: Mapping[str, Any], repository: Path) -> Path:
    checksum = row.get("sha256")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError(f"{role} evidence requires a SHA-256 checksum")
    path = row.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"{role} evidence must be a local file")
    candidate = repository / path
    if any(
        part.is_symlink()
        for part in (candidate, *candidate.parents)
        if part != repository.parent
    ):
        raise ValueError(f"{role} evidence may not be a symlink")
    resolved = candidate.resolve()
    root = repository.resolve()
    if root not in resolved.parents:
        raise ValueError(f"{role} evidence path is unavailable or unsafe")
    actual = _evidence_digest(role, resolved)
    if actual != checksum:
        raise ValueError(f"{role} evidence checksum mismatch")
    return resolved


def _evidence_digest(role: str, resolved: Path) -> str:
    if role == "raw_artifact" and resolved.is_dir():
        members = sorted(resolved.rglob("*"))
        for member in members:
            if member.is_symlink():
                raise ValueError("raw_artifact evidence may not contain symlinks")
            member_resolved = member.resolve()
            if resolved not in member_resolved.parents:
                raise ValueError("raw_artifact evidence contains a path escape")
        actual, _ = _artifact_digest(resolved)
    elif resolved.is_file():
        actual, _ = _artifact_digest(resolved)
    else:
        raise ValueError(f"{role} evidence path is unavailable or unsafe")
    return actual


def _read_document(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"evidence document is not valid JSON: {path.name}")
    if not isinstance(value, dict):
        raise ValueError(f"evidence document must be a JSON object: {path.name}")
    return value


def _validate_document_kinds(
    environment: Mapping[str, Any],
    plan: Mapping[str, Any],
    trial: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> None:
    expected = (
        (environment, "native_probe_environment", "environment"),
        (plan, "native_probe_plan", "command/plan"),
        (trial, "native_probe_trial", "trial"),
        (analysis, "native_probe_analysis", "analysis"),
    )
    for document, kind, role in expected:
        if document.get("artifact_kind") != kind:
            raise ValueError(f"{role} evidence has the wrong document kind")


def _validate_evidence_links(
    promotion: Mapping[str, Any],
    environment: Mapping[str, Any],
    environment_evidence_path: str,
    plan: Mapping[str, Any],
    repository: Path,
    raw_evidence_path: str,
    raw_artifact: Path,
    trial: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> None:
    _validate_revision_links(environment, plan, trial)
    _validate_environment_artifact_link(plan, repository, environment_evidence_path)
    trial_id = _validate_promotion_identity(promotion, trial)
    _validate_planned_trial(plan, trial, trial_id)
    _validate_complete_trial(trial)
    artifact = _linked_raw_artifact(trial, repository, raw_evidence_path, raw_artifact)
    _validate_analysis_group(analysis, trial, trial_id)
    _validate_claim_proof(promotion, analysis, trial_id, artifact)


def _validate_revision_links(
    environment: Mapping[str, Any],
    plan: Mapping[str, Any],
    trial: Mapping[str, Any],
) -> None:
    environment_source = environment.get("source")
    revision = (
        environment_source.get("revision")
        if isinstance(environment_source, Mapping)
        else None
    )
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or plan.get("revision") != revision
        or trial.get("revision") != revision
    ):
        raise ValueError("environment, plan, and trial revision links do not match")


def _validate_environment_artifact_link(
    plan: Mapping[str, Any], repository: Path, environment_evidence_path: str
) -> None:
    planned_environment = plan.get("environment_artifact")
    if not isinstance(planned_environment, str):
        raise ValueError("plan does not reference its environment artifact")
    planned_environment_path = Path(planned_environment)
    if not planned_environment_path.is_absolute():
        planned_environment_path = repository / planned_environment_path
    if (
        planned_environment_path.resolve()
        != (repository / environment_evidence_path).resolve()
    ):
        raise ValueError("plan/environment artifact link does not match")


def _validate_promotion_identity(
    promotion: Mapping[str, Any], trial: Mapping[str, Any]
) -> str:
    trial_id = trial.get("trial_id")
    if not isinstance(trial_id, str) or not trial_id:
        raise ValueError("trial evidence has no trial_id")
    identity = {
        "trial_id": trial_id,
        "configuration_id": trial.get("configuration_id"),
        "workload_id": trial.get("workload_id"),
        "mode": trial.get("mode"),
        "repetition": trial.get("repetition"),
        "revision": trial.get("revision"),
    }
    if any(promotion.get(key) != value for key, value in identity.items()):
        raise ValueError("promotion selector does not match its evidence identity")
    candidate_id = promotion.get("candidate_id")
    candidate_modes = (
        _CANDIDATE_MODES.get(candidate_id, set())
        if isinstance(candidate_id, str)
        else set()
    )
    if trial.get("mode") not in candidate_modes:
        raise ValueError("trial mode is unrelated to the promoted candidate")
    return trial_id


def _validate_planned_trial(
    plan: Mapping[str, Any], trial: Mapping[str, Any], trial_id: str
) -> Mapping[str, Any]:
    planned = [
        row
        for row in plan.get("trials", [])
        if isinstance(row, Mapping) and row.get("trial_id") == trial_id
    ]
    if len(planned) != 1:
        raise ValueError("plan must contain exactly one linked trial")
    planned_trial = planned[0]
    _require_executed_command(planned_trial.get("command"), "plan trial")
    _require_executed_command(trial.get("command"), "trial")
    if plan.get("configuration_id") != trial.get("configuration_id"):
        raise ValueError("plan/trial mismatch for configuration_id")
    links = ("configuration_id", "workload_id", "mode", "repetition")
    for field in links:
        if planned_trial.get(field) != trial.get(field):
            raise ValueError(f"plan/trial mismatch for {field}")
    if planned_trial.get("command") != trial.get("command"):
        raise ValueError("plan/trial mismatch for command")
    return planned_trial


def _require_executed_command(command: Any, role: str) -> None:
    if not isinstance(command, Mapping) or not {
        "argv",
        "environment",
        "timeout_seconds",
    }.issubset(command):
        raise ValueError(f"{role} must contain a structured executed command")
    try:
        CommandSpec.from_mapping(command)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{role} must contain a structured executed command"
        ) from error


def _validate_complete_trial(trial: Mapping[str, Any]) -> None:
    if trial.get("status") != "pass" or trial.get("return_code") != 0:
        raise ValueError("only a passing complete trial can be promoted")
    if validate_measurement_window(trial.get("measurement_window")):
        raise ValueError("passing trial has no complete measurement window")
    required_missing = any(
        isinstance(row, Mapping)
        and row.get("required") is True
        and row.get("status") != "present"
        for row in trial.get("artifacts", [])
    )
    if required_missing:
        raise ValueError("passing trial is missing a required artifact")


def _linked_raw_artifact(
    trial: Mapping[str, Any],
    repository: Path,
    raw_evidence_path: str,
    raw_artifact: Path,
) -> Mapping[str, Any]:
    artifact = _find_raw_artifact(trial, repository, raw_evidence_path, raw_artifact)
    _validate_linked_raw_artifact(artifact)
    return artifact


def _find_raw_artifact(
    trial: Mapping[str, Any],
    repository: Path,
    raw_evidence_path: str,
    raw_artifact: Path,
) -> Mapping[str, Any]:
    raw_digest, _ = _artifact_digest(raw_artifact)
    evidence_path = (repository / raw_evidence_path).resolve()
    artifact = next(
        (
            row
            for row in trial.get("artifacts", [])
            if isinstance(row, Mapping)
            and _trial_artifact_path(row, repository) == evidence_path
            and row.get("sha256") == raw_digest
        ),
        None,
    )
    if artifact is None:
        raise ValueError("raw artifact is not linked from the trial")
    return artifact


def _validate_linked_raw_artifact(artifact: Mapping[str, Any]) -> None:
    if (
        artifact.get("status") != "present"
        or not isinstance(artifact.get("kind"), str)
        or not artifact.get("kind")
        or not isinstance(artifact.get("producer"), str)
        or not artifact.get("producer")
    ):
        raise ValueError("raw artifact is not linked from the trial")


def _trial_artifact_path(row: Mapping[str, Any], repository: Path) -> Path | None:
    value = row.get("path")
    if not isinstance(value, str):
        return None
    path = Path(value)
    return (path if path.is_absolute() else repository / path).resolve()


def _validate_analysis_group(
    analysis: Mapping[str, Any], trial: Mapping[str, Any], trial_id: str
) -> None:
    group_key = ":".join(
        str(trial[field]) for field in ("configuration_id", "workload_id", "mode")
    )
    groups = analysis.get("groups")
    group = groups.get(group_key) if isinstance(groups, Mapping) else None
    group_trial_ids = group.get("trial_ids") if isinstance(group, Mapping) else None
    if not isinstance(group_trial_ids, list) or trial_id not in group_trial_ids:
        raise ValueError(
            "analysis does not include the promoted trial in its exact group"
        )


def _validate_claim_proof(
    promotion: Mapping[str, Any],
    analysis: Mapping[str, Any],
    trial_id: str,
    artifact: Mapping[str, Any],
) -> None:
    claim_id = promotion.get("claim_id")
    claims = analysis.get("claim_evidence")
    proof = claims.get(claim_id) if isinstance(claims, Mapping) else None
    proof = _validate_explicit_claim_proof(proof)
    _validate_claim_trial_reference(proof, trial_id)
    _validate_claim_artifact_reference(proof, artifact)


def _validate_explicit_claim_proof(proof: Any) -> Mapping[str, Any]:
    if (
        not isinstance(proof, Mapping)
        or proof.get("status") != "pass"
        or not isinstance(proof.get("basis"), str)
        or not proof.get("basis", "").strip()
        or not isinstance(proof.get("criterion"), str)
        or not proof.get("criterion", "").strip()
        or not isinstance(proof.get("observed"), str)
        or not proof.get("observed", "").strip()
    ):
        raise ValueError("analysis has no explicit passing evidence for this claim")
    return proof


def _validate_claim_trial_reference(proof: Mapping[str, Any], trial_id: str) -> None:
    if (
        not isinstance(proof.get("trial_ids"), list)
        or trial_id not in proof["trial_ids"]
    ):
        raise ValueError("claim evidence does not reference the promoted trial")


def _validate_claim_artifact_reference(
    proof: Mapping[str, Any], artifact: Mapping[str, Any]
) -> None:
    artifact_ids = proof.get("artifact_ids")
    if (
        not isinstance(artifact_ids, list)
        or not isinstance(artifact.get("artifact_id"), str)
        or artifact.get("artifact_id") not in artifact_ids
    ):
        raise ValueError("claim evidence does not reference the promoted raw artifact")
