"""The ``stormlog.report`` v1 verdict envelope.

A report is the machine-readable summary a command leaves behind for CI jobs
and agents: the verdict paired with the process exit code, findings with
evidence pointers, flat metrics, and pointers to the versioned artifacts the
command wrote. Tool-specific detail goes in ``payload`` and is versioned by
the producing command, so this envelope stays small and strict.

The published schema is ``docs/schemas/stormlog_report_v1.schema.json``.
``validate_report`` applies the same rules without a JSON Schema dependency.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .exit_codes import VERDICT_STATUSES, verdict_status

REPORT_FORMAT = "stormlog.report"
REPORT_SCHEMA_VERSION = 1
REPORT_FILENAME = "report.json"

# Mirrors the patterns in docs/schemas/stormlog_report_v1.schema.json. They
# are applied with fullmatch, i.e. with the ECMA-262 anchoring JSON Schema
# specifies: no trailing newline is tolerated.
_REPORT_KIND_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_FINDING_ID_PATTERN = re.compile(r"[a-z][a-z0-9_.]*")
_POINTER_PATTERN = re.compile(r"|/[^\n]*")

_TOOL_KEYS = frozenset({"name", "command", "version", "argv"})
_VERDICT_KEYS = frozenset({"status", "exit_code", "summary"})
_FINDING_KEYS = frozenset(
    {"id", "kind", "severity", "title", "message", "metrics", "evidence"}
)
_ARTIFACT_KEYS = frozenset({"kind", "path", "format", "schema_version"})

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"
SEVERITIES = frozenset({SEVERITY_INFO, SEVERITY_WARNING, SEVERITY_CRITICAL})

_REQUIRED_KEYS = (
    "schema_version",
    "format",
    "report_kind",
    "generated_at_utc",
    "tool",
    "verdict",
    "findings",
)
_KNOWN_KEYS = frozenset(
    {
        *_REQUIRED_KEYS,
        "metrics",
        "artifacts",
        "recommendations",
        "session_id",
        "run_id",
        "payload",
    }
)
_EVIDENCE_KEYS = frozenset(
    {
        "kind",
        "path",
        "pointer",
        "session_id",
        "record_id",
        "start_ns",
        "end_ns",
        "description",
    }
)


def _compact(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True)
class Evidence:
    """Pointer to the data behind a finding.

    ``path`` resolves against the directory holding the report; ``pointer`` is
    a JSON pointer inside that file.
    """

    kind: str
    path: str | None = None
    pointer: str | None = None
    session_id: str | None = None
    record_id: str | None = None
    start_ns: int | None = None
    end_ns: int | None = None
    description: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return _compact(self.__dict__)


@dataclass(frozen=True)
class Finding:
    """One detected condition with its severity and evidence."""

    id: str
    kind: str
    severity: str
    title: str
    message: str | None = None
    metrics: Mapping[str, float | int | None] | None = None
    evidence: Sequence[Evidence] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "severity": self.severity,
            "title": self.title,
            "evidence": [item.as_dict() for item in self.evidence],
        }
        if self.message is not None:
            payload["message"] = self.message
        if self.metrics is not None:
            payload["metrics"] = dict(self.metrics)
        return payload


@dataclass(frozen=True)
class Artifact:
    """A file the command wrote or relied on."""

    kind: str
    path: str
    format: str | None = None
    schema_version: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return _compact(self.__dict__)


def build_report(
    *,
    report_kind: str,
    tool_name: str,
    command: str,
    exit_code: int,
    summary: str,
    findings: Sequence[Finding] = (),
    metrics: Mapping[str, float | int | None] | None = None,
    artifacts: Sequence[Artifact] = (),
    recommendations: Sequence[str] = (),
    session_id: str | None = None,
    run_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
    argv: Sequence[str] | None = None,
    tool_version: str | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Assemble a v1 report dict whose verdict matches ``exit_code``.

    Raises:
        ValueError: if ``exit_code`` is not in the contract.
    """
    tool: dict[str, Any] = {
        "name": tool_name,
        "command": command,
        "version": _tool_version(tool_version),
    }
    if argv is not None:
        tool["argv"] = list(argv)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "format": REPORT_FORMAT,
        "report_kind": report_kind,
        "generated_at_utc": generated_at_utc or _now_utc(),
        "tool": tool,
        "verdict": {
            "status": verdict_status(exit_code),
            "exit_code": int(exit_code),
            "summary": summary,
        },
        "findings": [finding.as_dict() for finding in findings],
        "metrics": dict(metrics or {}),
        "artifacts": [artifact.as_dict() for artifact in artifacts],
        "recommendations": list(recommendations),
        "session_id": session_id,
        "run_id": run_id,
    }
    if payload is not None:
        report["payload"] = dict(payload)
    return report


def _tool_version(explicit: str | None) -> str | None:
    if explicit is not None:
        return explicit
    from . import __version__

    return str(__version__)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    """Validate ``report`` and write it as indented JSON."""
    validate_report(report)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def load_report(path: Path) -> dict[str, Any]:
    """Read and validate a report file.

    Raises:
        ValueError: if the file is not a valid v1 report.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read report {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Report {path} is not a JSON object")
    validate_report(payload)
    return dict(payload)


def validate_report(report: Mapping[str, Any]) -> None:
    """Check ``report`` against the v1 contract.

    Raises:
        ValueError: naming the first rule the report breaks.
    """
    _validate_header(report)
    _validate_tool(report["tool"])
    _validate_verdict(report["verdict"])
    _validate_list(report, "findings", _validate_finding)
    _validate_metrics(report.get("metrics", {}), "metrics")
    _validate_list(report, "artifacts", _validate_artifact)
    _validate_recommendations(report.get("recommendations", []))
    for key in ("session_id", "run_id"):
        _require_nullable_nonempty_string(report, key)
    if "payload" in report and not isinstance(report["payload"], Mapping):
        raise ValueError("payload must be an object")


def _validate_header(report: Mapping[str, Any]) -> None:
    if not isinstance(report, Mapping):
        raise ValueError("report must be an object")
    missing = [key for key in _REQUIRED_KEYS if key not in report]
    if missing:
        raise ValueError(f"report is missing required fields: {', '.join(missing)}")
    _reject_unknown(report, _KNOWN_KEYS, "report")
    version = report["schema_version"]
    if not _is_integer_value(version) or int(version) != REPORT_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {REPORT_SCHEMA_VERSION}")
    if report["format"] != REPORT_FORMAT:
        raise ValueError(f"format must be {REPORT_FORMAT!r}")
    _require_pattern(report, "report_kind", _REPORT_KIND_PATTERN)
    _require_nonempty_string(report, "generated_at_utc")


def _validate_tool(tool: Any) -> None:
    if not isinstance(tool, Mapping):
        raise ValueError("tool must be an object")
    _reject_unknown(tool, _TOOL_KEYS, "tool")
    _require_nonempty_string(tool, "name", prefix="tool")
    _require_nonempty_string(tool, "command", prefix="tool")
    _require_nullable_nonempty_string(tool, "version", prefix="tool")
    if "argv" in tool:
        argv = tool["argv"]
        if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
            raise ValueError("tool.argv must be a list of strings")


def _validate_verdict(verdict: Any) -> None:
    if not isinstance(verdict, Mapping):
        raise ValueError("verdict must be an object")
    _reject_unknown(verdict, _VERDICT_KEYS, "verdict")
    status = verdict.get("status")
    if not isinstance(status, str) or status not in VERDICT_STATUSES:
        raise ValueError(f"verdict.status {status!r} is not a verdict status")
    exit_code = verdict.get("exit_code")
    if not isinstance(exit_code, (int, float)) or not _is_integer_value(exit_code):
        raise ValueError("verdict.exit_code must be an integer")
    if verdict_status(int(exit_code)) != status:
        raise ValueError(
            f"verdict.status {status!r} does not pair with exit_code {exit_code}"
        )
    _require_nonempty_string(verdict, "summary", prefix="verdict")


def _validate_finding(finding: Any, label: str) -> None:
    if not isinstance(finding, Mapping):
        raise ValueError(f"{label} must be an object")
    _reject_unknown(finding, _FINDING_KEYS, label)
    _require_pattern(finding, "id", _FINDING_ID_PATTERN, prefix=label)
    for key in ("kind", "title"):
        _require_nonempty_string(finding, key, prefix=label)
    severity = finding.get("severity")
    if not isinstance(severity, str) or severity not in SEVERITIES:
        raise ValueError(f"{label}.severity must be one of {sorted(SEVERITIES)}")
    if "message" in finding and not isinstance(finding["message"], (str, type(None))):
        raise ValueError(f"{label}.message must be a string or null")
    if "metrics" in finding:
        _validate_metrics(finding["metrics"], f"{label}.metrics")
    evidence = finding.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError(f"{label}.evidence must be a list")
    for index, item in enumerate(evidence):
        _validate_evidence(item, f"{label}.evidence[{index}]")


def _validate_evidence(evidence: Any, label: str) -> None:
    if not isinstance(evidence, Mapping):
        raise ValueError(f"{label} must be an object")
    _reject_unknown(evidence, _EVIDENCE_KEYS, label)
    _require_nonempty_string(evidence, "kind", prefix=label)
    for key in ("path", "session_id", "record_id"):
        _require_nullable_nonempty_string(evidence, key, prefix=label)
    _validate_pointer(evidence.get("pointer"), f"{label}.pointer")
    for key in ("start_ns", "end_ns"):
        _validate_nullable_int(evidence.get(key), f"{label}.{key}", minimum=0)
    description = evidence.get("description")
    if description is not None and not isinstance(description, str):
        raise ValueError(f"{label}.description must be a string or null")


def _validate_pointer(pointer: Any, label: str) -> None:
    if pointer is None:
        return
    if not isinstance(pointer, str) or _POINTER_PATTERN.fullmatch(pointer) is None:
        raise ValueError(f"{label} must be a single-line JSON pointer")


def _validate_nullable_int(value: Any, label: str, *, minimum: int) -> None:
    if value is None:
        return
    if not _is_integer_value(value) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def _is_integer_value(value: Any) -> bool:
    """True for JSON integers: ints (not bools) and integral floats like 3.0."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and value.is_integer()


def _reject_unknown(
    payload: Mapping[str, Any], allowed: frozenset[str], label: str
) -> None:
    unknown = sorted(str(key) for key in set(payload) - allowed)
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")


def _validate_artifact(artifact: Any, label: str) -> None:
    if not isinstance(artifact, Mapping):
        raise ValueError(f"{label} must be an object")
    _reject_unknown(artifact, _ARTIFACT_KEYS, label)
    _require_nonempty_string(artifact, "kind", prefix=label)
    _require_nonempty_string(artifact, "path", prefix=label)
    _require_nullable_nonempty_string(artifact, "format", prefix=label)
    _validate_nullable_int(
        artifact.get("schema_version"), f"{label}.schema_version", minimum=1
    )


def _validate_metrics(metrics: Any, label: str) -> None:
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{label} must be an object")
    for key, value in metrics.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} keys must be strings")
        if value is not None and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            raise ValueError(f"{label}.{key} must be a number or null")


def _validate_recommendations(recommendations: Any) -> None:
    if not isinstance(recommendations, list) or not all(
        isinstance(item, str) and item for item in recommendations
    ):
        raise ValueError("recommendations must be a list of non-empty strings")


def _validate_list(report: Mapping[str, Any], key: str, check: Any) -> None:
    items = report.get(key, [])
    if not isinstance(items, list):
        raise ValueError(f"{key} must be a list")
    for index, item in enumerate(items):
        check(item, f"{key}[{index}]")


def _require_nonempty_string(
    payload: Mapping[str, Any], key: str, *, prefix: str | None = None
) -> None:
    label = f"{prefix}.{key}" if prefix else key
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")


def _require_nullable_nonempty_string(
    payload: Mapping[str, Any], key: str, *, prefix: str | None = None
) -> None:
    if payload.get(key) is not None:
        _require_nonempty_string(payload, key, prefix=prefix)


def _require_pattern(
    payload: Mapping[str, Any],
    key: str,
    pattern: re.Pattern[str],
    *,
    prefix: str | None = None,
) -> None:
    _require_nonempty_string(payload, key, prefix=prefix)
    label = f"{prefix}.{key}" if prefix else key
    if pattern.fullmatch(payload[key]) is None:
        raise ValueError(f"{label} must match ^{pattern.pattern}$")


__all__ = [
    "Artifact",
    "Evidence",
    "Finding",
    "REPORT_FILENAME",
    "REPORT_FORMAT",
    "REPORT_SCHEMA_VERSION",
    "SEVERITIES",
    "SEVERITY_CRITICAL",
    "SEVERITY_INFO",
    "SEVERITY_WARNING",
    "build_report",
    "load_report",
    "validate_report",
    "write_report",
]
