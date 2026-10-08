"""The memory ledger: what each source says, never a sum."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.diagnosis import DiagnoseOptions, diagnose_artifact
from stormlog.infer.telemetry import ServerIdentity, TelemetrySample
from tests.diagnosis_scenarios import MS, Engine, build_run, poisson_free
from tests.vllm_execution_helpers import SECOND
from tests.vllm_scrape_helpers import exposition, scrape


def _ledger(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = report["payload"]["memory"]["entries"]
    return {entry["category"]: entry for entry in entries}


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    engine = Engine(config={"num_gpu_blocks": 1000, "enable_cumem_allocator": False})
    path = build_run(
        tmp_path_factory.mktemp("m"), poisson_free(20, 10 * SECOND, 100 * MS), engine
    )
    usage = [0.25, 0.75, 0.5]
    scrapes = [
        scrape(exposition(gauges={"vllm:kv_cache_usage_perc": value}), 10 + i)
        for i, value in enumerate(usage)
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.writelines(json.dumps(s.to_record()) + "\n" for s in scrapes)
    return path


def test_without_telemetry_every_category_says_why(artifact: Path) -> None:
    ledger = _ledger(diagnose_artifact(artifact))

    assert ledger["physical_device"]["status"] == "not_collected"
    assert ledger["allocator_reserved"]["status"] == "not_collected"
    assert ledger["runtime"]["status"] == "unsupported"
    assert ledger["cuda_graph_pools"]["status"] == "unsupported"


def test_kv_blocks_are_bound_to_the_engine_only_by_assertion(artifact: Path) -> None:
    scoped = _ledger(diagnose_artifact(artifact))["kv_blocks_allocated"]
    asserted = _ledger(
        diagnose_artifact(artifact, options=DiagnoseOptions(metrics_from_engine=True))
    )["kv_blocks_allocated"]

    assert (scoped["status"], scoped["binding"]) == (
        "exporter_scoped",
        "exporter_scoped",
    )
    assert (asserted["status"], asserted["binding"]) == ("observed", "asserted")
    assert asserted["peak"] == {"scope": "sampled_max", "value": 750}
    assert asserted["unit"] == "blocks"


def test_collector_samples_are_observed_with_their_peak(
    artifact: Path, tmp_path: Path
) -> None:
    identity = ServerIdentity(
        host="node-7", pid=2600, process_start_ns=1, device_uuid="GPU-a"
    )
    samples = [
        TelemetrySample(
            run_id="run-1",
            identity=identity,
            observed_at_ns=1_790_000_010_000_000_000 + i * SECOND,
            metric="device_memory_used_bytes",
            value_bytes=value,
            state="valid",
            source="nvml-v2",
            interval_ms=1000,
        )
        for i, value in enumerate((100, 300, 200))
    ]
    telemetry = tmp_path / "server.jsonl"
    telemetry.write_text("".join(json.dumps(s.to_record()) + "\n" for s in samples))

    report = diagnose_artifact(
        artifact, options=DiagnoseOptions(server_telemetry=(str(telemetry),))
    )

    device = _ledger(report)["physical_device"]
    assert device["status"] == "observed" and device["samples"] == 3
    assert device["peak"]["value"] == 300 and device["cadence_ms"] == 1000
    assert report["payload"]["memory"]["nesting"]["holds"] is True
