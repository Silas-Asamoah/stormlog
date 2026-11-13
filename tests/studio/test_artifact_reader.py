"""ArtifactReader: bounded, resumable reads of one artifact (design §4.3, §9.1).

The reader's lines must be exactly the lines ``diagnosis_inputs`` numbers
and hashes, its records exactly what the correlation codec decodes, and its
change detection must tell an append, an import's replace and a rerun's
truncate-and-rewrite apart on the same inode.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, BinaryIO

import pytest

from stormlog.infer import artifact_lines
from stormlog.infer.artifact_reader import ArtifactReader, ReadItem
from stormlog.infer.correlation_codec import expand_inference_records
from stormlog.infer.diagnosis_inputs import read_input
from stormlog.studio.contracts import LineProblem
from tests.studio.artifacts import (
    MIXED_CASES,
    STANDARD_ARTIFACTS,
    UNSUPPORTED,
    CaseSpec,
    GrowingCopy,
    rewrite_in_place,
    with_execution_import,
)
from tests.studio.conftest import StandardArtifacts
from tests.studio.test_index import same


class Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def opener(path: Path) -> Callable[[], BinaryIO]:
    def open_it() -> BinaryIO:
        return path.open("rb")

    return open_it


def reader_for(path: Path, **options: Any) -> ArtifactReader:
    return ArtifactReader(opener(path), **options)


def read_all(reader: ArtifactReader, *, max_bytes: int = 8 * 2**20) -> list[ReadItem]:
    items: list[ReadItem] = []
    while True:
        before = reader.offset
        batch = reader.read_batch(max_bytes=max_bytes)
        items.extend(batch)
        if reader.offset == before:
            return items


def tail_of(reader: ArtifactReader) -> LineProblem | None:
    """The reader's tail problem, read through a call so mypy does not
    narrow the property across the reads that change it."""
    return reader.tail_problem


def lines_of(items: list[ReadItem]) -> list[ReadItem]:
    """The items that are lines, not a reported pending tail."""
    return [
        item
        for item in items
        if item.problem is None or item.problem.code != "truncated_final_line"
    ]


def write_lines(path: Path, lines: list[bytes]) -> Path:
    path.write_bytes(b"".join(line + b"\n" for line in lines))
    return path


def record_line(record: dict[str, Any]) -> bytes:
    return json.dumps(record).encode("utf-8")


# --- Every standard artifact ------------------------------------------------


@pytest.mark.parametrize("name", sorted(STANDARD_ARTIFACTS))
def test_lines_and_hashes_match_diagnosis_inputs(
    studio_artifacts: StandardArtifacts, name: str
) -> None:
    path = studio_artifacts[name]
    reader = reader_for(path)
    assert reader.identity_change() == "grown"
    items = lines_of(read_all(reader, max_bytes=4096))
    expected = read_input(path)
    data = path.read_bytes()
    complete = data[: data.rfind(b"\n") + 1]
    assert [item.ref.line for item in items] == [
        line.number for line in expected.lines
    ][: len(items)]
    assert [item.ref.sha256 for item in items] == [
        line.sha256 for line in expected.lines
    ][: len(items)]
    for item in items:
        assert data[item.ref.offset : item.ref.offset + item.ref.length + 1] == (
            data[item.ref.offset : item.ref.offset + item.ref.length] + b"\n"
        )
    assert reader.offset == len(complete)
    assert reader.pending_tail_bytes == len(data) - len(complete)
    assert reader.prefix_sha256 == hashlib.sha256(complete).hexdigest()
    if not reader.pending_tail_bytes:
        assert reader.prefix_sha256 == expected.sha256


@pytest.mark.parametrize(
    "name", sorted(set(STANDARD_ARTIFACTS) - {"multi_session", "truncated_mid_line"})
)
def test_records_equal_the_codecs_decoding(
    studio_artifacts: StandardArtifacts, name: str
) -> None:
    path = studio_artifacts[name]
    items = read_all(reader_for(path), max_bytes=1024)
    records = [item.record for item in items if item.record is not None]
    lines = [line for line in path.read_bytes().split(b"\n") if line]
    try:
        expected = expand_inference_records(json.loads(line) for line in lines)
    except ValueError:  # C5's not_an_inference_artifact: the codec refuses it
        assert name in UNSUPPORTED
        problems = [item.problem for item in items if item.problem is not None]
        assert problems and {p.code for p in problems} == {"undecodable"}
        return
    assert same(records, expected)
    assert all(item.problem is None for item in items)
    contexts = sum(1 for line in lines if b'"infer.context"' in line)
    assert sum(1 for item in items if item.kind == "context") == contexts


def test_a_compact_line_finds_its_context_defined_batches_earlier(
    studio_artifacts: StandardArtifacts,
) -> None:
    """Contexts defined far back: one decoder lives across batches."""
    path = studio_artifacts["execution_import"]
    reader = reader_for(path)
    while not any(item.kind == "context" for item in reader.read_batch(max_bytes=600)):
        assert reader.offset < 4096
    rest = read_all(reader, max_bytes=600)
    server = [
        item.record
        for item in rest
        if item.record is not None
        and item.record.get("event_type") == "infer.request"
        and "context" in item.record
    ]
    assert len(server) == 6
    assert all(record["schema_version"] == 2 for record in server)


def test_the_multi_session_files_second_context_is_a_problem_not_a_crash(
    studio_artifacts: StandardArtifacts,
) -> None:
    items = read_all(reader_for(studio_artifacts["multi_session"]))
    problems = [item.problem for item in items if item.problem is not None]
    assert all(problem.code == "undecodable" for problem in problems)
    assert sum(1 for item in items if item.record is not None) > 0


# --- Lines: splitting, numbering, problems ----------------------------------


def test_blank_lines_count_and_a_carriage_return_does_not_split(
    tmp_path: Path,
) -> None:
    record = record_line({"event_type": "infer.x", "text": "a\rb"})
    raw = record.replace(b"a\\rb", b"a\rb")  # a raw CR inside the line
    path = write_lines(tmp_path / "a.jsonl", [b"", raw, b"   ", record])
    items = read_all(reader_for(path))
    assert [item.ref.line for item in items] == [0, 1, 2, 3]
    assert [item.kind for item in items] == ["blank", "problem", "blank", "record"]
    assert items[1].problem is not None and items[1].problem.code == "not_json"
    assert items[3].record == {"event_type": "infer.x", "text": "a\rb"}
    assert items[1].ref.offset == 1 and items[1].ref.length == len(raw)


def test_each_bad_line_is_one_problem_and_reading_goes_on(tmp_path: Path) -> None:
    lines = [
        b"{not json",
        b"[1, 2]",
        b"\xff\xfe{}",
        record_line({"schema_version": 4, "event_type": "infer.iteration"}),
        record_line(
            {"schema_version": 4, "event_type": "infer.context", "context_id": "x"}
        ),
        record_line({"event_type": "infer.session", "session_id": "s"}),
    ]
    items = read_all(reader_for(write_lines(tmp_path / "bad.jsonl", lines)))
    codes = [None if item.problem is None else item.problem.code for item in items]
    assert codes == [
        "not_json",
        "not_object",
        "not_json",
        "undecodable",
        "undecodable",
        None,
    ]
    for item in items[:5]:
        assert item.problem is not None
        assert item.problem.ref == item.ref
        assert len(item.problem.detail) <= 200
    assert items[5].record == {"event_type": "infer.session", "session_id": "s"}


def test_unknown_types_and_newer_versions_never_crash(tmp_path: Path) -> None:
    context = {
        "schema_version": 4,
        "event_type": "infer.context",
        "context_id": "ctx-1",
        "context": {"bogus": 1},
    }
    lines = [
        record_line({"schema_version": 9, "event_type": "infer.future", "x": [1]}),
        record_line({"schema_version": 5, "event_type": "infer.request"}),
        record_line(
            {"schema_version": 4, "event_type": "infer.future", "context_id": "nope"}
        ),
        record_line(context),
        record_line({"event_type": "not.inference", "nested": {"a": None}}),
        b'{"event_type": "infer.x", "deep": ' + b"[" * 100_000 + b"]" * 100_000 + b"}",
    ]
    items = read_all(reader_for(write_lines(tmp_path / "new.jsonl", lines)))
    assert [item.kind for item in items] == [
        "record",
        "record",
        "problem",
        "problem",
        "record",
        "problem",
    ]
    assert items[0].record == json.loads(lines[0])
    assert items[1].record == {"schema_version": 5, "event_type": "infer.request"}


def nested_line(depth: int) -> bytes:
    """An infer.x line nesting ``depth`` objects and arrays, its own included."""
    inner = b"[" * (depth - 1) + b"]" * (depth - 1)
    return b'{"event_type": "infer.x", "deep": ' + inner + b"}"


def test_nesting_past_the_limit_is_not_json_on_any_python_or_thread(
    tmp_path: Path,
) -> None:
    """json.loads's own limit moves with the version and, from 3.14, the
    thread's stack; the reader's does not, and what it keeps can be walked."""
    from concurrent.futures import ThreadPoolExecutor

    from stormlog.infer.diagnosis_inputs import MAX_NESTING
    from stormlog.studio.contracts import encode_json, finite_copy

    brackets = b"[" * 3 * MAX_NESTING
    lines = [
        nested_line(MAX_NESTING),
        nested_line(MAX_NESTING + 1),
        nested_line(50_000),
        b'{"event_type": "infer.x", "text": "' + brackets + b'"}',
        b'{"event_type": "infer.x", "text": "\\"' + brackets + b'", "n": [[1]]}',
    ]
    path = write_lines(tmp_path / "deep.jsonl", lines)
    items = read_all(reader_for(path))
    codes = [item.kind if item.problem is None else item.problem.code for item in items]
    assert codes == ["record", "not_json", "not_json", "record", "record"]
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(read_all, reader_for(path)).result() == items
    assert items[1].problem is not None
    assert items[1].problem.detail == f"nested deeper than {MAX_NESTING}"
    record = items[0].record
    assert record is not None
    encode_json(finite_copy(record)[0])


def test_nan_in_a_record_is_read_as_written(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "nan.jsonl", [b'{"event_type": "x", "v": NaN}'])
    (item,) = read_all(reader_for(path))
    assert item.record is not None and math.isnan(item.record["v"])


def test_an_oversized_line_is_hashed_and_skipped(tmp_path: Path) -> None:
    big = record_line({"event_type": "infer.big", "pad": "x" * 5000})
    small = record_line({"event_type": "infer.small"})
    path = write_lines(tmp_path / "big.jsonl", [small, big, small])
    reader = reader_for(path, max_line_bytes=1000)
    items = read_all(reader, max_bytes=256)
    assert [item.kind for item in items] == ["record", "problem", "record"]
    problem = items[1].problem
    assert problem is not None and problem.code == "oversized"
    assert problem.ref.length == len(big)
    assert problem.ref.sha256 == hashlib.sha256(big).hexdigest()
    assert items[2].ref.offset == len(small) + 1 + len(big) + 1
    assert reader.prefix_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_line_longer_than_a_batch_is_still_read_whole(tmp_path: Path) -> None:
    long = {"event_type": "infer.long", "pad": "y" * 50_000}
    path = write_lines(tmp_path / "long.jsonl", [record_line(long), b"{}"])
    reader = reader_for(path)
    first = reader.read_batch(max_bytes=1024)
    assert [item.record for item in first] == [long]
    assert [item.record for item in reader.read_batch(max_bytes=1024)] == [{}]


# --- The pending tail -------------------------------------------------------


def test_a_pending_tail_is_reported_after_five_still_seconds(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    source = studio_artifacts["truncated_mid_line"].read_bytes()
    path = tmp_path / "t.jsonl"
    path.write_bytes(source)
    clock = Clock()
    reader = reader_for(path, clock=clock)
    items = read_all(reader)
    tail = len(source) - (source.rfind(b"\n") + 1)
    assert reader.pending_tail_bytes == tail > 0
    assert tail_of(reader) is None
    assert all(item.problem is None for item in items)
    clock.now += 4.9
    assert reader.read_batch() == []
    clock.now += 0.2
    (item,) = reader.read_batch()
    assert item.problem is not None and item.problem.code == "truncated_final_line"
    assert item.ref.line == reader.next_line and item.ref.length == tail
    assert item.ref.sha256 == hashlib.sha256(source[-tail:]).hexdigest()
    assert tail_of(reader) == item.problem
    assert reader.read_batch() == []  # reported once
    with path.open("ab") as handle:
        handle.write(b"\n")
    assert reader.identity_change() == "grown"
    (last,) = reader.read_batch()
    assert last.kind == "problem" and last.ref.line == item.ref.line
    assert last.problem is not None and last.problem.code == "not_json"
    assert tail_of(reader) is None and reader.pending_tail_bytes == 0


def test_a_cut_after_a_newline_leaves_no_tail(
    studio_artifacts: StandardArtifacts,
) -> None:
    reader = reader_for(studio_artifacts["truncated_after_newline"], tail_grace_s=0)
    read_all(reader)
    assert reader.pending_tail_bytes == 0 and reader.tail_problem is None


# --- Change detection -------------------------------------------------------


def test_growth_across_calls_reaches_the_whole_file(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    source = studio_artifacts["mixed"]
    growing = GrowingCopy(source, tmp_path / "g.jsonl", lines=5)
    reader = reader_for(growing.target)
    assert reader.identity_change() == "grown"
    assert len(read_all(reader)) == 5
    assert reader.identity_change() == "same"
    growing.append(3)
    growing.append_partial()
    assert reader.identity_change() == "grown"
    assert len(read_all(reader)) == 3 and reader.pending_tail_bytes > 0
    growing.finish()
    assert reader.identity_change() == "grown"
    read_all(reader)
    assert reader.next_line == len(read_input(source).lines)
    assert reader.prefix_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()


def test_a_rerun_to_the_same_inode_is_changed(
    fresh_artifact: Callable[[str], Path],
) -> None:
    path = fresh_artifact("tokens_exact")
    inode = os.stat(path).st_ino
    reader = reader_for(path)
    read_all(reader)
    rewrite_in_place(path, cases=MIXED_CASES[:1])  # longer than before
    assert os.stat(path).st_ino == inode
    assert os.stat(path).st_size > reader.offset
    assert reader.identity_change() == "changed"
    reader.reset()
    assert reader.identity is None and reader.offset == 0
    assert reader.identity_change() == "grown"
    read_all(reader)
    assert reader.prefix_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_rewrite_that_keeps_line_0_but_changes_the_last_line_is_changed(
    tmp_path: Path,
) -> None:
    path = write_lines(tmp_path / "a.jsonl", [b'{"a": 1}', b'{"b": 2}'])
    reader = reader_for(path)
    read_all(reader)
    with path.open("r+b") as handle:  # same inode, longer, last anchor differs
        handle.write(b'{"a": 1}\n{"b": 3}\n{"c": 4}\n')
    assert reader.identity_change() == "changed"


def test_a_shorter_file_on_the_same_inode_is_changed(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "a.jsonl", [b'{"a": 1}', b'{"b": 2}'])
    reader = reader_for(path)
    read_all(reader)
    with path.open("r+b") as handle:
        handle.truncate(4)
    assert reader.identity_change() == "changed"


def test_an_import_replace_keeps_the_prefix(
    fresh_artifact: Callable[[str], Path],
) -> None:
    path = fresh_artifact("tokens_exact")
    reader = reader_for(path)
    read_all(reader)
    consumed = reader.offset
    with_execution_import(path)
    assert reader.identity_change() == "replaced_same_prefix"
    items = read_all(reader)
    assert items and items[0].ref.offset == consumed
    assert reader.prefix_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_new_inode_with_other_bytes_is_changed(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    path = tmp_path / "a.jsonl"
    path.write_bytes(studio_artifacts["tokens_exact"].read_bytes())
    reader = reader_for(path)
    read_all(reader)
    other = tmp_path / "b.jsonl"
    other.write_bytes(studio_artifacts["mixed"].read_bytes())
    os.replace(other, path)
    assert reader.identity_change() == "changed"


def test_a_touch_without_new_bytes_is_grown_by_nothing(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "a.jsonl", [b"{}"])
    reader = reader_for(path)
    read_all(reader)
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    assert reader.identity_change() == "grown"
    assert reader.read_batch() == []


def test_an_empty_file_is_same_and_reads_nothing(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")
    reader = reader_for(path)
    assert reader.identity_change() == "same"
    assert reader.read_batch() == []
    assert reader.prefix_sha256 == hashlib.sha256(b"").hexdigest()


def test_read_batch_reads_nothing_from_another_inode(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "a.jsonl", [b"{}"])
    reader = reader_for(path)
    reader.identity_change()
    other = write_lines(tmp_path / "b.jsonl", [b"[]"])
    os.replace(other, path)
    assert reader.read_batch() == []


def test_an_unreadable_file_raises_what_the_opener_raises(tmp_path: Path) -> None:
    reader = reader_for(tmp_path / "missing.jsonl")
    with pytest.raises(FileNotFoundError):
        reader.identity_change()
    assert reader.identity is None


# --- Rewind and reset -------------------------------------------------------


def test_rewind_unconsumes_a_line_of_the_last_batch(tmp_path: Path) -> None:
    lines = [record_line({"n": n}) for n in range(6)]
    path = write_lines(tmp_path / "a.jsonl", lines)
    reader = reader_for(path)
    first = reader.read_batch(max_bytes=20)
    second = reader.read_batch(max_bytes=40)
    reader.rewind(second[1].ref)
    assert reader.offset == second[1].ref.offset
    assert reader.next_line == second[1].ref.line
    prefix = path.read_bytes()[: reader.offset]
    assert reader.prefix_sha256 == hashlib.sha256(prefix).hexdigest()
    with pytest.raises(ValueError):
        reader.rewind(first[0].ref)
    rest = read_all(reader)
    assert [item.record for item in rest] == [{"n": n} for n in range(3, 6)]
    assert reader.identity_change() == "same"


def test_rewind_to_line_0_and_reset(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "a.jsonl", [b"{}", b"[]"])
    reader = reader_for(path)
    items = reader.read_batch()
    reader.rewind(items[0].ref)
    assert (reader.offset, reader.next_line) == (0, 0)
    assert reader.prefix_sha256 == hashlib.sha256(b"").hexdigest()
    read_all(reader)
    reader.reset()
    assert (reader.offset, reader.next_line, reader.identity) == (0, 0, None)


def test_bad_arguments_are_refused(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "a.jsonl", [b"{}"])
    with pytest.raises(ValueError):
        reader_for(path, max_line_bytes=0)
    with pytest.raises(ValueError):
        reader_for(path).read_batch(max_bytes=0)


def test_the_reader_closes_every_handle_it_opens(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "a.jsonl", [b"{}", b"[]", b"x"])
    handles: list[BinaryIO] = []

    def tracking() -> BinaryIO:
        handle = path.open("rb")
        handles.append(handle)
        return handle

    reader = ArtifactReader(tracking)
    reader.identity_change()
    items = reader.read_batch()
    reader.rewind(items[-1].ref)
    read_all(reader)
    assert handles and all(handle.closed for handle in handles)


def context_lines(source: Path, count: int) -> list[bytes]:
    """``count`` distinct valid context definitions, each with one compact
    record that refers to it, made from ``source``'s first definition."""
    template = next(
        json.loads(line)
        for line in source.read_bytes().split(b"\n")
        if b'"infer.context"' in line
    )
    lines = []
    for i in range(count):
        identity = f"ctx-{i:012x}"
        context = dict(template["context"], pid=i + 1)
        definition = {**template, "context_id": identity, "context": context}
        reference = {
            "schema_version": 4,
            "event_type": "infer.artifact",
            "event_id": f"artifact-{i}",
            "artifact_kind": "inference_jsonl",
            "created_at_ns": 1,
            "metadata": {},
            "context_id": identity,
        }
        lines += [json.dumps(definition).encode(), json.dumps(reference).encode()]
    return lines


def test_past_the_context_budget_definitions_and_their_uses_are_problems(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    lines = context_lines(studio_artifacts["execution_import"], 40)
    path = tmp_path / "contexts.jsonl"
    path.write_bytes(b"\n".join(lines) + b"\n")
    budget = 10 * (len(lines[0]) + 400)
    reader = reader_for(path, max_context_bytes=budget)
    items = read_all(reader, max_bytes=4096)
    kinds = [item.kind for item in items]
    accepted = kinds[0::2].count("context")
    assert 0 < accepted < 40 and reader.contexts_limited
    assert kinds[1::2] == ["record"] * accepted + ["problem"] * (40 - accepted)
    for item in items[2 * accepted :]:
        assert item.problem is not None and item.problem.code == "undecodable"
        assert item.problem.detail.startswith("context limit")
    assert 0 < reader.context_bytes <= budget
    again = reader_for(path, max_context_bytes=budget)
    first = again.read_batch(max_bytes=2**20)
    before = again.context_bytes
    again.rewind(first[0].ref)  # reading definitions again counts nothing
    assert [item.kind for item in read_all(again)] == kinds
    assert again.context_bytes == before == reader.context_bytes
    unbounded = reader_for(path)
    read_all(unbounded)
    assert not unbounded.contexts_limited and unbounded.context_bytes > budget
    unbounded.reset()
    assert unbounded.context_bytes == 0


def test_the_core_reader_imports_no_studio_module() -> None:
    probe = (
        "import sys, stormlog.infer.artifact_reader; "
        "print(sorted(m for m in sys.modules if m.startswith('stormlog.studio')))"
    )
    output = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        timeout=120,
    ).stdout
    assert output.strip() == "[]"
    assert LineProblem is artifact_lines.LineProblem


def iter_lines(path: Path) -> Iterator[bytes]:
    yield from path.read_bytes().split(b"\n")


def test_a_profile_with_two_cases_reads_like_its_codec(
    fresh_artifact: Callable[[str], Path],
) -> None:
    path = fresh_artifact("warmup_measured")
    rewrite_in_place(path, cases=[CaseSpec(2, 8, 4, 4), CaseSpec(2, 8, 16, 4)])
    items = read_all(reader_for(path), max_bytes=333)
    records = [item.record for item in items if item.record is not None]
    expected = expand_inference_records(
        json.loads(line) for line in iter_lines(path) if line
    )
    assert records == expected
