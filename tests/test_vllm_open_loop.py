"""CPU checks for the fixed-arrival vLLM trial boundary."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from scripts.native_probes.mode_commands import vllm_command, vllm_expected_artifacts
from scripts.native_probes.models import ExperimentMode, WorkloadId
from scripts.native_probes.planning import build_plan
from scripts.native_probes.workloads.vllm_open_loop import (
    MEASURED_REQUESTS,
    REVISION,
    _memory_metrics,
    _server_argv,
    offer_requests,
    request_body,
    run,
    schedule,
)


def test_overloaded_arrivals_are_retained_without_shifting_schedule() -> None:
    def slow_sender(request_id: str) -> dict[str, object]:
        started = time.time_ns()
        time.sleep(0.04)
        return {
            "request_id": request_id,
            "status": "ok",
            "started_at_ns": started,
            "ended_at_ns": time.time_ns(),
            "first_chunk_at_ns": None,
        }

    rows = asyncio.run(
        offer_requests(
            5, "measured", slow_sender, interval_seconds=0.005, max_in_flight=1
        )
    )

    assert len(rows) == 5
    assert [row["request_id"] for row in rows] == [
        f"stormlog-118-measured-{index:04d}" for index in range(5)
    ]
    assert [row["scheduled_offset_ms"] for row in rows] == [0, 5, 10, 15, 20]
    assert [row["status"] for row in rows].count("client_rejected") == 4
    assert rows[0]["e2e_latency_ms"] >= rows[0]["dispatch_delay_ms"]


def test_sender_failure_is_retained_as_an_offered_request() -> None:
    def failing_sender(request_id: str) -> dict[str, object]:
        raise RuntimeError(f"failed {request_id}")

    rows = asyncio.run(
        offer_requests(2, "measured", failing_sender, interval_seconds=0.001)
    )

    assert len(rows) == 2
    assert all(row["status"] == "error" for row in rows)
    assert all("RuntimeError" in row["error"] for row in rows)


def test_pinned_request_and_server_configuration(tmp_path: Path) -> None:
    request = request_body()
    command = _server_argv("public-engine", tmp_path, 8000)

    assert request["max_tokens"] == 64
    assert request["temperature"] == 0
    assert request["stream_options"] == {"include_usage": True}
    assert command.count(REVISION) == 2
    assert "--enforce-eager" in command
    assert "--no-enable-prefix-caching" in command
    assert schedule(MEASURED_REQUESTS)[-1] == pytest.approx(219.9)


def test_version_mismatch_stops_before_server_and_retains_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "scripts.native_probes.workloads.vllm_open_loop.importlib.metadata.version",
        lambda _: "0.29.0",
    )
    output = tmp_path / "trial"

    with pytest.raises(RuntimeError, match="pinned vLLM"):
        run("off", output, 8000)

    assert '"vllm": "0.29.0"' in (output / "versions.json").read_text()
    assert not (output / "server-command.json").exists()


def test_unknown_resource_domain_is_not_reported_as_zero() -> None:
    metrics = _memory_metrics(
        [
            {
                "processes": [
                    {"role": "helper_agent", "rss_bytes": 10},
                    {"role": "target", "rss_bytes": None},
                ]
            }
        ]
    )

    assert metrics["host_rss_valid_samples"] == 0
    assert metrics["host_rss_measured_peak_bytes"] is None
    assert metrics["gpu_memory_bytes"] is None


def test_vllm_plan_separates_modes_and_rejects_unimplemented_modes(
    tmp_path: Path,
) -> None:
    plan = build_plan(
        configuration_id="local-vllm",
        vendor="nvidia",
        workloads=[WorkloadId.VLLM],
        modes=[ExperimentMode.OFF, ExperimentMode.PUBLIC_ENGINE, ExperimentMode.PROTON],
        repetitions=5,
        seed=118,
        environment_artifact="environment.json",
        artifact_root=tmp_path,
    )

    assert len(plan["trials"]) == 15
    assert all(
        "scripts.native_probes.workloads.vllm_open_loop" in row["command"]["argv"]
        for row in plan["trials"]
    )
    assert any(
        "vllm/public-engine" in artifact["relative_path"]
        for row in plan["trials"]
        if row["mode"] == "public-engine"
        for artifact in row["expected_artifacts"]
    )
    assert any(
        row["role"] == "helper_agent"
        for row in plan["trials"]
        for row in row["process_roles"]
    )
    with pytest.raises(ValueError, match="no vLLM session adapter"):
        vllm_command(ExperimentMode.PUBLIC_PYTORCH, tmp_path)
    with pytest.raises(ValueError, match="pinned injection library"):
        vllm_command(ExperimentMode.DIRECT_CUPTI, tmp_path)
    assert any(
        row.relative_path == "vllm/nsys"
        for row in vllm_expected_artifacts(ExperimentMode.TRUSTED)
    )
