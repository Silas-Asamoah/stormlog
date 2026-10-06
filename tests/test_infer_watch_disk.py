"""Byte budgets for the incident store: reservations, allowances, capped writes."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from stormlog.infer.watch.disk import (
    Allowance,
    BudgetExceeded,
    CappedWriter,
    DiskBudget,
    StoreLimits,
    bytes_on_disk,
)

KIB = 1024


def _limits(max_total_bytes: int, max_incident_bytes: int) -> StoreLimits:
    return StoreLimits(
        max_total_bytes=max_total_bytes, max_incident_bytes=max_incident_bytes
    )


def test_limits_are_validated() -> None:
    with pytest.raises(ValueError, match="max_incident_bytes"):
        StoreLimits(max_total_bytes=10, max_incident_bytes=11)
    with pytest.raises(ValueError):
        StoreLimits(max_incidents=0)


def test_a_reservation_over_either_limit_is_refused() -> None:
    budget = DiskBudget(_limits(10 * KIB, 8 * KIB))
    assert budget.reserve(9 * KIB) is None  # over one incident's limit
    first = budget.reserve(6 * KIB)
    assert first is not None
    assert budget.reserve(6 * KIB) is None  # 12 KiB would pass the total
    first.release(keep=1 * KIB)
    assert budget.used_bytes == 1 * KIB
    assert budget.reserved_bytes == 0
    assert budget.reserve(6 * KIB) is not None


def test_a_capped_writer_charges_before_it_writes(tmp_path: Path) -> None:
    allowance = Allowance(10)
    with CappedWriter(tmp_path / "f", allowance) as out:
        out.write(b"0123456789")
        with pytest.raises(BudgetExceeded):
            out.write(b"x")
    # The refused byte never reached the disk.
    assert (tmp_path / "f").read_bytes() == b"0123456789"
    assert allowance.used == 10


def test_hard_links_are_counted_once(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"x" * 100)
    os.link(tmp_path / "a", tmp_path / "b")
    (tmp_path / "c").write_bytes(b"y" * 10)
    assert bytes_on_disk([tmp_path], seen=set()) == 110


def test_a_released_allowance_takes_nothing_more() -> None:
    budget = DiskBudget(_limits(10 * KIB, 8 * KIB))
    allowance = budget.reserve(KIB)
    assert allowance is not None
    allowance.charge(100)
    allowance.release()
    allowance.release()  # idempotent
    with pytest.raises(BudgetExceeded):
        allowance.charge(1)
    assert budget.used_bytes == 100 and budget.reserved_bytes == 0
