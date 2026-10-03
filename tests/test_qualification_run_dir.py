"""A run directory's label, layout and atomic publication, and the victim's
SLO outcomes for the impact layer."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from examples.qualification.outcomes import Slo, count_outcomes, outcome
from examples.qualification.run_dir import RunDirectory, new_label, verify


def test_labels_are_opaque() -> None:
    first, second = new_label(), new_label()
    assert re.fullmatch(r"q221-[0-9a-f]{16}", first)
    assert first != second


def test_a_run_appears_whole_or_not_at_all(tmp_path: Path) -> None:
    directory = RunDirectory(tmp_path, "q221-0000000000000001").create()
    (directory.run / "victim.jsonl").write_text("{}\n")
    (directory.truth / "injections.jsonl").write_text("{}\n")
    assert not directory.final.exists()
    published = directory.publish()
    assert published == tmp_path / "q221-0000000000000001"
    assert not directory.partial.exists()
    sums = (published / "SHA256SUMS").read_text().splitlines()
    assert [line.split("  ")[1] for line in sums] == [
        "run/victim.jsonl",
        "truth/injections.jsonl",
    ]
    assert verify(published) == []
    (published / "run" / "victim.jsonl").write_text("{}\n{}\n")
    assert verify(published) == ["changed run/victim.jsonl"]
    with pytest.raises(FileExistsError):
        RunDirectory(tmp_path, "q221-0000000000000001").create()


def _request(
    status: str = "ok", ttft: float = 100, e2e: float = 500, at: int = 10
) -> dict[str, Any]:
    return {
        "event_type": "infer.request",
        "status": status,
        "ttft_ms": ttft,
        "e2e_latency_ms": e2e,
        "started_at_ns": at,
    }


def test_outcomes_follow_213s_rule() -> None:
    slo = Slo(ttft_ms=200, e2e_ms=1000)
    assert outcome(_request(), slo) == "met"
    assert outcome(_request(ttft=300), slo) == "violation"
    assert outcome(_request(e2e=None), slo) == "violation"  # type: ignore[arg-type]
    for failed in ("timeout", "rejected", "error", "dropped"):
        assert outcome(_request(failed), slo) == "violation"
    assert outcome(_request("cancelled"), slo) == "unknown"


def test_outcomes_are_counted_by_arrival() -> None:
    slo = Slo(ttft_ms=200)
    records = [
        _request(at=5),
        _request(ttft=900, at=10),
        _request("cancelled", at=15),
        {**_request(at=99), "intended_at_ns": 20},  # arrived at 20, sent at 99
        _request(at=30),
        {"event_type": "infer.phase_window"},
    ]
    counts = count_outcomes(records, 10, 20, slo)
    assert (counts.violations, counts.met, counts.unknown) == (1, 1, 1)
