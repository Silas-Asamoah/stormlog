"""Process exit-code contract shared by every Stormlog command line.

The table is fixed so that automated consumers (CI jobs, agents, the TUI
command runner) can branch on the code without parsing output. Values the
platform already owns keep their meaning: ``argparse`` exits 2 on a usage
error, Python exits 1 on an uncaught exception, and the shell reports a
SIGINT-terminated process as 128 + 2.

Codes are never renumbered or reused. A new outcome takes the next free value
below 128; values from 128 upward stay reserved for signals.
"""

from __future__ import annotations

from enum import IntEnum
from types import MappingProxyType
from typing import Mapping


class ExitCode(IntEnum):
    """Deterministic process exit codes for Stormlog commands."""

    OK = 0
    """The command completed, no findings reached failure severity, gates passed."""

    ERROR = 1
    """An unexpected failure; any output written may be incomplete."""

    USAGE = 2
    """Invalid command line, or the installation cannot serve the request."""

    FINDINGS = 3
    """The command completed but detected memory risk, OOM, or leak findings."""

    GATE_FAILED = 4
    """The command completed but a configured budget or tolerance was exceeded."""

    INVALID_INPUT = 5
    """An input artifact or asset is missing, unreadable, or unsupported."""

    INTERRUPTED = 130
    """Stopped by SIGINT before completion."""


VERDICT_PASS = "pass"
VERDICT_ERROR = "error"
VERDICT_USAGE = "usage"
VERDICT_FINDINGS = "findings"
VERDICT_GATE_FAILED = "gate_failed"
VERDICT_INVALID_INPUT = "invalid_input"
VERDICT_INTERRUPTED = "interrupted"

VERDICT_STATUS_BY_EXIT_CODE: Mapping[ExitCode, str] = MappingProxyType(
    {
        ExitCode.OK: VERDICT_PASS,
        ExitCode.ERROR: VERDICT_ERROR,
        ExitCode.USAGE: VERDICT_USAGE,
        ExitCode.FINDINGS: VERDICT_FINDINGS,
        ExitCode.GATE_FAILED: VERDICT_GATE_FAILED,
        ExitCode.INVALID_INPUT: VERDICT_INVALID_INPUT,
        ExitCode.INTERRUPTED: VERDICT_INTERRUPTED,
    }
)

VERDICT_STATUSES = frozenset(VERDICT_STATUS_BY_EXIT_CODE.values())


def verdict_status(exit_code: int) -> str:
    """Return the verdict status string that the contract pairs with ``exit_code``.

    Raises:
        ValueError: if ``exit_code`` is not part of the contract.
    """
    try:
        return VERDICT_STATUS_BY_EXIT_CODE[ExitCode(exit_code)]
    except ValueError as exc:
        raise ValueError(f"Exit code {exit_code!r} is not in the contract") from exc


def exit_code_for_status(status: str) -> ExitCode:
    """Return the exit code the contract pairs with a verdict ``status``.

    Raises:
        ValueError: if ``status`` is not a verdict status.
    """
    for code, candidate in VERDICT_STATUS_BY_EXIT_CODE.items():
        if candidate == status:
            return code
    raise ValueError(f"Verdict status {status!r} is not in the contract")


def completed_with_findings(risk_detected: bool) -> ExitCode:
    """Map a completed diagnosis to ``FINDINGS`` or ``OK``."""
    return ExitCode.FINDINGS if risk_detected else ExitCode.OK


__all__ = [
    "ExitCode",
    "VERDICT_ERROR",
    "VERDICT_FINDINGS",
    "VERDICT_GATE_FAILED",
    "VERDICT_INTERRUPTED",
    "VERDICT_INVALID_INPUT",
    "VERDICT_PASS",
    "VERDICT_STATUSES",
    "VERDICT_STATUS_BY_EXIT_CODE",
    "VERDICT_USAGE",
    "completed_with_findings",
    "exit_code_for_status",
    "verdict_status",
]
