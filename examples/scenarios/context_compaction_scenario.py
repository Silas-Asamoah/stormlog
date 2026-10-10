"""Compare paired embedded/compact encodings of a saved trace without a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from stormlog.infer.correlation_accounting import (
    account_gpu_time,
    resolve_inference_events,
)
from stormlog.infer.correlation_events import (
    ArtifactIdentityEvent,
    CorrelationContext,
    load_inference_artifact,
)
from stormlog.infer.events import JsonlEventWriter
from stormlog.infer.trace_import import import_traces_into_artifact, parse_device_uuids


def _seed(path: Path) -> None:
    identity = ArtifactIdentityEvent(
        context=CorrelationContext(
            run_id="context-compaction",
            session_id="context-compaction",
            producer_id="scenario",
            source="context_compaction_scenario",
            clock_domain="scenario/wall",
            clock_kind="wall",
            collection_mode="imported",
            provenance="observed",
        ),
        event_id="artifact",
        artifact_kind="inference_jsonl",
        created_at_ns=0,
    )
    with JsonlEventWriter(path) as writer:
        writer.append(identity.to_record())


def _file_facts(path: Path) -> dict[str, int]:
    with path.open() as handle:
        lines = sum(1 for line in handle if line.strip())
    return {"bytes": path.stat().st_size, "physical_lines": lines}


def _trace_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"trace": str(path), "trace_sha256": digest.hexdigest(), "trace_bytes": size}


def compare(args: argparse.Namespace) -> dict[str, Any]:
    trace = args.trace.resolve(strict=True)
    identity = _trace_identity(trace)
    output = args.output_dir.resolve()
    # A fresh directory also protects the importer's envelope from accidental reuse.
    output.mkdir(parents=True, exist_ok=False)
    compact, embedded = output / "compact.jsonl", output / "embedded.jsonl"
    _seed(compact)
    capture = import_traces_into_artifact(
        compact,
        [trace],
        detail=args.detail,
        device_uuids=parse_device_uuids(args.device_uuid),
    )
    records = load_inference_artifact(compact)
    assert records[1 : 1 + len(capture.events)] == list(capture.events)
    with embedded.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_record(), sort_keys=True) + "\n")
    baseline = load_inference_artifact(embedded)
    assert baseline == records
    graph = resolve_inference_events(records)
    baseline_graph = resolve_inference_events(baseline)
    assert graph == baseline_graph
    accounting = account_gpu_time(graph)
    assert accounting == account_gpu_time(baseline_graph)
    compact_facts, embedded_facts = _file_facts(compact), _file_facts(embedded)
    saved = embedded_facts["bytes"] - compact_facts["bytes"]
    assert saved > 0, "repeated-context workload must produce a smaller artifact"
    assert _trace_identity(trace) == identity, "input trace changed during comparison"
    return {
        "comparison": "paired encodings of the same production import",
        **identity,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "python": platform.python_version(),
        "detail": args.detail,
        "device_uuid_overrides": args.device_uuid,
        "command": shlex.join(
            [
                sys.executable,
                "-m",
                "examples.scenarios.context_compaction_scenario",
                *sys.argv[1:],
            ]
        ),
        "compact": compact_facts,
        "embedded": embedded_facts,
        "saved_bytes": saved,
        "saved_percent": 100 * saved / embedded_facts["bytes"],
        "semantic_events": len(records),
        "event_types": dict(
            Counter(record.to_record()["event_type"] for record in records)
        ),
        "context_definitions": compact_facts["physical_lines"] - len(records),
        "activity_count": len(graph.activities),
        "linked": sum(
            a.attribution_status == "linked" for a in graph.activities.values()
        ),
        "unresolved": len(accounting.unattributed_activity_refs),
        "unmeasured": len(accounting.unmeasured_gpu_activity_refs),
        "device_clock_totals": [
            {**asdict(dimension), **asdict(timing)}
            for dimension, timing in accounting.device_totals.items()
        ],
        "capture_summary": capture.summary,
        "invariants": {
            "events_equal": True,
            "graphs_equal": True,
            "accounting_equal": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--detail", choices=("launch", "kernel"), default="launch")
    parser.add_argument("--device-uuid", action="append", default=[])
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Must not already exist"
    )
    args = parser.parse_args()
    results = compare(args)
    payload = json.dumps(results, indent=2, sort_keys=True) + "\n"
    (args.output_dir / "results.json").write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
