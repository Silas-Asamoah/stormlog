"""Build an experiment-only matrix without promoting literature claims."""

from __future__ import annotations

from copy import deepcopy
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
