"""A run directory's label, layout and atomic publication, and the victim's
SLO outcomes for the impact layer."""

from __future__ import annotations

import hashlib
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
    assert [line.split("  ")[1] for line in sums[:-1]] == [
        "run/victim.jsonl",
        "truth/injections.jsonl",
    ]
    assert sums[-1].startswith("# files 2 sha256 ")
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
    # #213's evaluate_request: anything but ok is missed, cancelled
    # included; a successful request misses on any failed criterion, else is
    # unknown on any value no latency can have.
    slo = Slo(ttft_ms=200, e2e_ms=1000)
    assert outcome(_request(), slo) == "met"
    assert outcome(_request(ttft=200), slo) == "met"
    assert outcome(_request(ttft=300), slo) == "violation"
    for failed in ("timeout", "rejected", "error", "dropped", "cancelled"):
        assert outcome(_request(failed), slo) == "violation"
    for unjudged in (None, float("nan"), float("inf"), -5.0):
        assert outcome(_request(e2e=unjudged), slo) == "unknown"  # type: ignore[arg-type]
    assert outcome(_request(ttft=300, e2e=None), slo) == "violation"  # type: ignore[arg-type]


def test_outcomes_are_counted_by_arrival() -> None:
    slo = Slo(ttft_ms=200)
    records = [
        _request(at=5),
        _request(ttft=900, at=10),
        _request(ttft=-1, at=15),
        {**_request(at=99), "intended_at_ns": 20},  # arrived at 20, sent at 99
        _request(at=30),
        {"event_type": "infer.phase_window"},
    ]
    counts = count_outcomes(records, 10, 20, slo)
    assert (counts.violations, counts.met, counts.unknown) == (1, 1, 1)


def _run(tmp_path: Path) -> Path:
    directory = RunDirectory(tmp_path, "q221-00000000000000aa").create()
    (directory.run / "victim.jsonl").write_text('{"event_type": "infer.request"}\n')
    (directory.truth / "injections.jsonl").write_text('{"status": "not_realized"}\n')
    return directory.publish()


def _drop(sums: Path, fragment: str) -> None:
    lines = sums.read_text().splitlines(keepends=True)
    sums.write_text("".join(line for line in lines if fragment not in line))


@pytest.mark.parametrize(
    "tamper",
    [
        "truncate the sums",
        "edit the truth and drop its line",
        "add a file",
        "truncate the victim and drop its line",
        "rewrite the sums whole",
        "remove a listed file",
    ],
)
def test_verify_catches_tampering_and_truncation(tmp_path: Path, tamper: str) -> None:
    run = _run(tmp_path)
    sums = run / "SHA256SUMS"
    if tamper == "truncate the sums":
        sums.write_text("")
    elif tamper == "edit the truth and drop its line":
        (run / "truth" / "injections.jsonl").write_text('{"status": "valid"}\n')
        _drop(sums, "injections")
    elif tamper == "add a file":
        (run / "truth" / "injections-extra.jsonl").write_text('{"status": "valid"}\n')
    elif tamper == "truncate the victim and drop its line":
        (run / "run" / "victim.jsonl").write_text("")
        _drop(sums, "victim")
    elif tamper == "rewrite the sums whole":
        # Consistent sums for a changed file: only the digest kept outside
        # the directory shows it.
        (run / "truth" / "injections.jsonl").write_text('{"status": "valid"}\n')
        other = RunDirectory(tmp_path / "other", run.name).create()
        for path in run.rglob("*"):
            if path.is_file() and path.name != "SHA256SUMS":
                target = other.partial / path.relative_to(run)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
        sums.write_bytes((other.publish() / "SHA256SUMS").read_bytes())
    else:
        (run / "run" / "victim.jsonl").unlink()
    assert verify(run), tamper


def test_only_the_run_that_claimed_a_label_publishes_it(tmp_path: Path) -> None:
    # close-221-final: a label is claimed by one rename of a directory laid
    # out with the run's nonce; a second run given it can neither claim it
    # nor publish what is there, and the owner file isn't published.
    first = RunDirectory(tmp_path, "q221-0000000000000003").create()
    second = RunDirectory(tmp_path, first.label)
    with pytest.raises(FileExistsError):
        second.create()
    with pytest.raises(FileExistsError):
        second.publish()
    assert first.owned() and not second.owned()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        f".{first.label}.partial"
    ]
    published = first.publish()
    assert verify(published) == [] and not (published / ".owner").exists()
    with pytest.raises(FileExistsError):
        first.publish()


def test_a_label_published_while_it_is_claimed_is_given_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # close-221-final: a run that published between this one's check and
    # its claim freed the partial directory's name, but the label is its.
    from examples.qualification import run_dir

    first = RunDirectory(tmp_path, "q221-0000000000000004").create()
    (first.run / "victim.jsonl").write_text('{"event_type": "infer.request"}\n')
    real = run_dir._claim

    def published_meanwhile(staging: Path, partial: Path) -> None:
        first.publish()
        real(staging, partial)

    monkeypatch.setattr(run_dir, "_claim", published_meanwhile)
    with pytest.raises(FileExistsError):
        RunDirectory(tmp_path, first.label).create()
    assert verify(first.final) == []
    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == [first.label, f"{first.label}.sha256"]


def test_a_published_run_keeps_its_sums_digest_beside_it(tmp_path: Path) -> None:
    run = _run(tmp_path)
    assert verify(run) == []
    beside = tmp_path / f"{run.name}.sha256"
    assert (
        beside.read_text().split()[0]
        == hashlib.sha256((run / "SHA256SUMS").read_bytes()).hexdigest()
    )
    beside.unlink()
    assert verify(run) == [f"no digest of SHA256SUMS beside the run ({beside.name})"]


def test_the_sums_are_complete_before_the_run_is_renamed_into_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples.qualification import run_dir

    seen: list[tuple[bool, bool]] = []
    real = run_dir.os.replace

    def replace(source: Any, target: Any) -> None:
        sums = Path(source) / "SHA256SUMS"
        complete = sums.is_file() and "# files " in sums.read_text()
        seen.append((complete, (tmp_path / f"{Path(target).name}.sha256").is_file()))
        real(source, target)

    monkeypatch.setattr(run_dir.os, "replace", replace)
    _run(tmp_path)
    assert seen == [(True, True)]


def test_the_count_line_catches_a_dropped_file_without_the_sidecar(
    tmp_path: Path,
) -> None:
    # Fable's A2 delta N2: the count line was never the only detector, yet
    # it is exactly when the run travels without its <label>.sha256
    # sidecar. A truth file deleted along with its line in SHA256SUMS then
    # leaves every listed file present and nothing unlisted.
    run = _run(tmp_path)
    (tmp_path / "q221-00000000000000aa.sha256").unlink()
    (run / "truth" / "injections.jsonl").unlink()
    _drop(run / "SHA256SUMS", "truth/injections.jsonl")
    problems = verify(run)
    assert any("count line doesn't match" in problem for problem in problems)
    assert problems == [
        "no digest of SHA256SUMS beside the run (q221-00000000000000aa.sha256)",
        "SHA256SUMS is truncated or edited: its count line doesn't match",
    ]
