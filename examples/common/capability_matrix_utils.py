"""Helpers for the launch capability matrix runner."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence


@dataclass
class CheckResult:
    """Result of one capability check."""

    name: str
    status: str
    duration_s: float
    details: Dict[str, object] = field(default_factory=dict)


def run_command(
    cmd: Sequence[str], timeout_s: Optional[float] = None
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command and capture stdout/stderr."""
    return subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_s,
    )


def write_report(path: Path, report: Dict[str, object]) -> None:
    """Write a JSON report to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def result_to_dict(result: CheckResult) -> Dict[str, object]:
    """Serialize CheckResult to a plain dictionary."""
    return asdict(result)


def timed_result(
    name: str,
    fn: Callable[[], object],
) -> CheckResult:
    """Execute callable and normalize PASS/SKIP/FAIL output."""
    started = time.perf_counter()
    try:
        payload = fn()
        duration = time.perf_counter() - started
        if not isinstance(payload, dict):
            return CheckResult(
                name=name,
                status="FAIL",
                duration_s=duration,
                details={
                    "error": f"Expected dict payload, got {type(payload).__name__}"
                },
            )
        status = str(payload.get("status", "PASS")).upper()
        if status not in {"PASS", "SKIP", "FAIL"}:
            status = "FAIL"
        return CheckResult(
            name=name,
            status=status,
            duration_s=duration,
            details=payload,
        )
    except Exception as exc:  # noqa: BLE001
        duration = time.perf_counter() - started
        return CheckResult(
            name=name,
            status="FAIL",
            duration_s=duration,
            details={"error": str(exc), "exception_type": type(exc).__name__},
        )


def summarize_results(results: List[CheckResult]) -> Dict[str, int]:
    """Summarize pass/skip/fail counts."""
    summary = {"PASS": 0, "SKIP": 0, "FAIL": 0}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1
    return summary
