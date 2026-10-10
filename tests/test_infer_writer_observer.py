"""The artifact writer's observer: told about each record, never in the way."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from stormlog.infer.events import JsonlEventWriter


def test_an_observer_sees_each_record_after_it_is_written(tmp_path: Path) -> None:
    seen: list[tuple[dict[str, Any], Mapping[str, Any] | None, int]] = []
    path = tmp_path / "infer.jsonl"

    def observe(record: dict[str, Any], extras: Mapping[str, Any] | None) -> None:
        written = len(path.read_text().splitlines())
        seen.append((record, extras, written))

    with JsonlEventWriter(path, observer=observe) as writer:
        writer.append({"event_type": "a"})
        writer.append({"event_type": "b"}, {"chunk_summary": (1, 2)})
    assert [entry[0]["event_type"] for entry in seen] == ["a", "b"]
    assert seen[0][1] is None and seen[1][1] == {"chunk_summary": (1, 2)}
    assert [entry[2] for entry in seen] == [1, 2]  # on disk before observed


def test_a_failing_observer_never_reaches_the_artifact(tmp_path: Path) -> None:
    def broken(record: dict[str, Any], extras: Mapping[str, Any] | None) -> None:
        raise RuntimeError("exporter bug")

    path = tmp_path / "infer.jsonl"
    with JsonlEventWriter(path, observer=broken) as writer:
        writer.append({"event_type": "a"})
        writer.append({"event_type": "b"})
    assert writer.observer_errors == 2
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["event_type"] for line in lines] == ["a", "b"]
