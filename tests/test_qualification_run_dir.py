"""A run directory's label, layout and atomic publication."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pytest

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
