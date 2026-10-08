"""Detection is causal: what a diagnosis found at first_detectable_ns, a
genuine prefix of the run, imported afresh, finds again."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stormlog.infer.diagnosis import diagnose_artifact
from stormlog.infer.vllm_execution_import import import_execution_into_artifact
from tests.diagnosis_scenarios import MS, Engine, build_run, poisson_free
from tests.vllm_execution_helpers import SECOND, WALL_OFFSET, importer

# When each client record was appended: at its send, its first content, its end.
_APPENDED = {
    "infer.dispatch": "started_at_ns",
    "infer.first_content": "first_content_at_ns",
    "infer.request": "ended_at_ns",
}


def _appended_at(record: dict[str, Any]) -> int:
    field = _APPENDED.get(record.get("event_type", ""))
    return int(record[field]) if field else 0


def _stamp(record: dict[str, Any]) -> int:
    for key in ("mono_ns", "start_mono_ns"):
        if isinstance(record.get(key), int):
            return int(record[key])
    clock = record.get("clock") or {}
    return int(clock.get("mono_ns", 0))


def _prefix(full: Path, into: Path, at_wall: int) -> Path:
    """The run as it stood at ``at_wall``: client records appended by then,
    hook records written by then, imported into a fresh artifact."""
    into.mkdir()
    client = [json.loads(line) for line in full.read_text().splitlines()]
    kept = [
        r
        for r in client
        if r.get("schema_version", 1) == 1 or r["event_type"] == "infer.artifact"
    ]
    kept = sorted((r for r in kept if _appended_at(r) <= at_wall), key=_appended_at)
    artifact = into / "infer.jsonl"
    artifact.write_text("".join(json.dumps(r) + "\n" for r in kept))
    for source in sorted((full.parent / "hook").rglob("*.jsonl")):
        records = [json.loads(line) for line in source.read_text().splitlines()]
        target = into / "hook" / source.relative_to(full.parent / "hook")
        target.parent.mkdir(parents=True, exist_ok=True)
        written = [r for r in records if _stamp(r) <= at_wall - WALL_OFFSET]
        target.write_text("".join(json.dumps(r) + "\n" for r in written))
    import_execution_into_artifact(
        artifact, into / "hook", importer=importer(at_wall - WALL_OFFSET)
    )
    return artifact


def test_a_finding_reappears_in_a_genuine_prefix_import(tmp_path: Path) -> None:
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    burst = poisson_free(300, 90 * SECOND, 5 * MS, prefix="b")
    full = build_run(tmp_path / "full", calm + burst, Engine(max_num_seqs=4))
    report = diagnose_artifact(full)
    (subject,) = report["payload"]["selection"]["subjects"]
    detected = subject["first_detectable_ns"]

    replayed = diagnose_artifact(_prefix(full, tmp_path / "prefix", detected))

    kinds = {f["kind"]: f for f in replayed["findings"]}
    assert "queue_saturation" in kinds
    (again,) = replayed["payload"]["selection"]["subjects"]
    assert again["first_detectable_ns"] is not None
    assert again["first_detectable_ns"] <= detected
    assert kinds["queue_saturation"]["severity"] == "warning"
