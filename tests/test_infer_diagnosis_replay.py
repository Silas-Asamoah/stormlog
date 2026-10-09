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
    """The run as it stood at first_detectable_ns selects the same incident,
    detected at the same instant, and assesses the queue on it; a moment
    earlier there is no incident. The class's own evidence can come later.
    Two seconds on, the queue is the fault it is on the whole run, a pause
    ruled out by admissions alone; at the hook's last heartbeat before its
    goodbye, a read of a live epoch, the heartbeats vouch for the whole of
    the waits, and coverage rules the pause out."""
    calm = poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")
    burst = poisson_free(300, 90 * SECOND, 5 * MS, prefix="b")
    full = build_run(tmp_path / "full", calm + burst, Engine(max_num_seqs=4))
    report = diagnose_artifact(full)
    (subject,) = report["payload"]["selection"]["subjects"]
    detected = subject["first_detectable_ns"]

    replayed = diagnose_artifact(_prefix(full, tmp_path / "at", detected))
    earlier = diagnose_artifact(_prefix(full, tmp_path / "before", detected - 1))
    later = diagnose_artifact(_prefix(full, tmp_path / "later", detected + 2 * SECOND))
    covered = diagnose_artifact(
        _prefix(full, tmp_path / "covered", _last_heartbeat(full))
    )

    (again,) = replayed["payload"]["selection"]["subjects"]
    assert (again["key"], again["first_detectable_ns"]) == (subject["key"], detected)
    assert "queue_saturation" in {f["kind"] for f in replayed["findings"]}
    assert earlier["payload"]["selection"]["subjects"] == []
    kinds = {f["kind"]: f for f in later["findings"]}
    assert kinds["queue_saturation"]["severity"] == "warning"
    assert [engine["state"] for engine in covered["payload"]["join"]["engines"]] == [
        "alive"
    ]
    for prefix, by in ((later, "admissions stopped"), (covered, "nothing lost")):
        assert by in _paused(prefix)["reason"]
        assert _paused(prefix)["status"] == "ruled_out"


def _last_heartbeat(full: Path) -> int:
    """When the hook wrote its last heartbeat before its goodbye, on the
    client's clock."""
    records = [
        json.loads(line)
        for source in (full.parent / "hook").rglob("*.jsonl")
        for line in source.read_text().splitlines()
    ]
    (goodbye,) = [_stamp(r) for r in records if r["kind"] == "goodbye"]
    beats = [_stamp(r) for r in records if r["kind"] == "heartbeat"]
    return max(at for at in beats if at < goodbye) + WALL_OFFSET


def _paused(report: dict[str, Any]) -> dict[str, Any]:
    """The queue's paused-scheduler competitor."""
    details = report["payload"]["findings_detail"].values()
    (queue,) = [d for d in details if d["kind"] == "queue_saturation"]
    (paused,) = [a for a in queue["alternatives"] if a["kind"] == "scheduler_paused"]
    return dict(paused)
