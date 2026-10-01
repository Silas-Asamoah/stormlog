"""Tests for the shared exit-code contract."""

from __future__ import annotations

import pytest

from stormlog.exit_codes import (
    VERDICT_STATUS_BY_EXIT_CODE,
    VERDICT_STATUSES,
    ExitCode,
    completed_with_findings,
    exit_code_for_status,
    verdict_status,
)

EXPECTED_TABLE: dict[str, tuple[int, str]] = {
    "OK": (0, "pass"),
    "ERROR": (1, "error"),
    "USAGE": (2, "usage"),
    "FINDINGS": (3, "findings"),
    "GATE_FAILED": (4, "gate_failed"),
    "INVALID_INPUT": (5, "invalid_input"),
    "INTERRUPTED": (130, "interrupted"),
}


def test_exit_code_values_are_fixed() -> None:
    assert {member.name: int(member) for member in ExitCode} == {
        name: value for name, (value, _status) in EXPECTED_TABLE.items()
    }


def test_every_exit_code_has_exactly_one_verdict_status() -> None:
    assert set(VERDICT_STATUS_BY_EXIT_CODE) == set(ExitCode)
    assert len(VERDICT_STATUSES) == len(ExitCode)
    for name, (value, status) in EXPECTED_TABLE.items():
        assert verdict_status(value) == status
        assert exit_code_for_status(status) is ExitCode[name]


def test_platform_owned_values_keep_their_meaning() -> None:
    # argparse exits 2 on a usage error; Python exits 1 on an uncaught exception;
    # the shell reports SIGINT as 128 + 2.
    assert int(ExitCode.USAGE) == 2
    assert int(ExitCode.ERROR) == 1
    assert int(ExitCode.INTERRUPTED) == 128 + 2


def test_outcome_codes_do_not_collide_with_platform_values() -> None:
    outcome_codes = {ExitCode.FINDINGS, ExitCode.GATE_FAILED, ExitCode.INVALID_INPUT}
    assert all(2 < int(code) < 128 for code in outcome_codes)


def test_exit_codes_are_plain_ints_for_sys_exit() -> None:
    assert isinstance(ExitCode.FINDINGS, int)
    assert int(ExitCode.FINDINGS) == 3
    assert ExitCode(3) is ExitCode.FINDINGS


def test_verdict_status_rejects_unknown_code() -> None:
    with pytest.raises(ValueError, match="not in the contract"):
        verdict_status(42)


def test_exit_code_for_status_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="not in the contract"):
        exit_code_for_status("maybe")


def test_completed_with_findings_maps_risk_flag() -> None:
    assert completed_with_findings(True) is ExitCode.FINDINGS
    assert completed_with_findings(False) is ExitCode.OK
