"""Build an experiment-only matrix without promoting literature claims."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


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


_EVIDENCE_ROLES = {"environment", "command", "raw_artifact", "trial", "analysis"}


def validate_matrix_promotions(
    matrix: Mapping[str, Any], promotions: list[Mapping[str, Any]], repository: Path
) -> dict[str, Any]:
    """Promote claims only when all immutable evidence roles are verifiable."""
    result: dict[str, Any] = deepcopy(dict(matrix))
    candidates = {row["id"]: row for row in result.get("candidates", [])}
    for promotion in promotions:
        candidate_id, claim_id, claims = _promotion_target(promotion, candidates)
        evidence = _promotion_evidence(promotion, repository)
        claims[claim_id]["status"] = "STORMLOG_VALIDATED"
        claims[claim_id]["detail"] = str(
            promotion.get("detail", "validated by hardware experiment")
        )
        claims[claim_id]["evidence"] = [
            str(row.get("path") or row["durable_location"]) for row in evidence
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
    evidence = promotion.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("promotion evidence must be an array")
    rows = [row for row in evidence if isinstance(row, Mapping)]
    by_role = {row.get("role"): row for row in rows}
    if set(by_role) != _EVIDENCE_ROLES:
        missing = sorted(_EVIDENCE_ROLES - set(by_role))
        raise ValueError(f"promotion missing evidence roles: {missing}")
    for role, row in by_role.items():
        _validate_evidence_row(str(role), row, repository)
    return rows


def _validate_evidence_row(role: str, row: Mapping[str, Any], repository: Path) -> None:
    checksum = row.get("sha256")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError(f"{role} evidence requires a SHA-256 checksum")
    path = row.get("path")
    if isinstance(path, str):
        resolved = (repository / path).resolve()
        if repository.resolve() not in resolved.parents or not resolved.is_file():
            raise ValueError(f"{role} evidence path is unavailable or unsafe")
        return
    durable = row.get("durable_location")
    if not isinstance(durable, str) or not durable.startswith("https://"):
        raise ValueError(f"{role} evidence requires a file or durable HTTPS location")
