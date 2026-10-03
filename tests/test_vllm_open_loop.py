"""CPU checks for the fixed-arrival vLLM trial boundary."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.native_probes.mode_commands import vllm_command, vllm_expected_artifacts
from scripts.native_probes.models import ExperimentMode, WorkloadId
from scripts.native_probes.planning import build_plan
from scripts.native_probes.workloads.vllm_open_loop import (
    MEASURED_REQUESTS,
    REVISION,
    _cupti_capture_status,
    _memory_metrics,
    _request,
    _result,
    _server_argv,
    _slo_goodput,
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


def test_incomplete_resource_sampling_cannot_produce_memory_claim() -> None:
    metrics = _memory_metrics(
        [
            {
                "processes": [
                    {"pid": 1, "role": "helper_agent", "rss_bytes": 10},
                    {"pid": 2, "role": "target", "rss_bytes": 20},
                ]
            },
            {
                "processes": [
                    {"pid": 1, "role": "helper_agent", "rss_bytes": 11},
                    {"pid": 2, "role": "target", "rss_bytes": None},
                ]
            },
        ]
    )
    assert metrics["host_rss_valid_samples"] == 1
    assert metrics["host_rss_measured_peak_bytes"] is None
    assert metrics["host_rss_measured_median_bytes"] is None


def test_resource_process_population_change_is_inconclusive() -> None:
    metrics = _memory_metrics(
        [
            {
                "processes": [
                    {"pid": 1, "role": "helper_agent", "rss_bytes": 10},
                    {"pid": 2, "role": "target", "rss_bytes": 20},
                ]
            },
            {
                "processes": [
                    {"pid": 1, "role": "helper_agent", "rss_bytes": 11},
                    {"pid": 2, "role": "target", "rss_bytes": 21},
                    {"pid": 3, "role": "target", "rss_bytes": 22},
                ]
            },
        ]
    )
    assert metrics["host_rss_valid_samples"] == 2
    assert metrics["host_rss_measured_peak_bytes"] is None


def test_incomplete_stream_is_a_failed_offered_request() -> None:
    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter([b'data: {"choices": [{"delta": {"content": "x"}}]}\n'])

    with patch(
        "scripts.native_probes.workloads.vllm_open_loop.urllib.request.urlopen",
        return_value=Response(),
    ):
        row = _request("http://127.0.0.1:8000", "id", 1)

    assert row["status"] == "error"
    assert row["error"] == "stream ended before [DONE]"


def test_cupti_capture_requires_every_target_and_clean_flush(tmp_path: Path) -> None:
    cupti = tmp_path / "cupti"
    cupti.mkdir()
    (cupti / "activity.ndjson").write_text('{"kind":"kernel"}\n')
    status = {
        "pid": 12,
        "finalized": True,
        "initialization_error": None,
        "delivered_records": 1,
        "cupti_dropped_records": 0,
        "local_dropped_records": 0,
        "bytes_dropped": 0,
    }
    (cupti / "cupti_status.json").write_text(json.dumps(status))
    samples = [{"processes": [{"pid": 12, "role": "target"}]}]

    capture, loss = _cupti_capture_status(tmp_path, samples)
    assert capture["complete"] is True
    assert loss["vendor_activity"]["lost_records"] == 0

    samples[0]["processes"].append({"pid": 13, "role": "target"})
    capture, _ = _cupti_capture_status(tmp_path, samples)
    assert capture["complete"] is False
    assert "13" in capture["errors"][0]

    samples[0]["processes"].pop()
    status["local_dropped_records"] = 2
    (cupti / "cupti_status.json").write_text(json.dumps(status))
    capture, loss = _cupti_capture_status(tmp_path, samples)
    assert capture["complete"] is False
    assert loss["vendor_activity"]["lost_records"] == 2


def test_cupti_capture_rejects_invalid_or_unobserved_process_status(
    tmp_path: Path,
) -> None:
    cupti = tmp_path / "cupti"
    cupti.mkdir()
    (cupti / "activity.ndjson").write_text('{"kind":"kernel"}\n')
    status = {
        "pid": {"bad": "identity"},
        "finalized": True,
        "initialization_error": None,
        "delivered_records": 1,
        "cupti_dropped_records": 0,
        "local_dropped_records": 0,
        "bytes_dropped": 0,
    }
    (cupti / "cupti_status.json").write_text(json.dumps(status))
    samples = [{"processes": [{"pid": 12, "role": "target"}]}]
    capture, _ = _cupti_capture_status(tmp_path, samples)
    assert capture["complete"] is False
    assert any("invalid pid" in error for error in capture["errors"])
    status["pid"] = 13
    (cupti / "cupti_status.json").write_text(json.dumps(status))
    capture, _ = _cupti_capture_status(tmp_path, samples)
    assert capture["complete"] is False
    assert any("unobserved" in error for error in capture["errors"])


def test_cupti_capture_rejects_partial_or_malformed_trace(tmp_path: Path) -> None:
    cupti = tmp_path / "cupti"
    cupti.mkdir()
    trace = cupti / "activity.ndjson"
    trace.write_text('{"kind":"kernel"}\n')
    status = {
        "pid": 12,
        "finalized": True,
        "initialization_error": None,
        "delivered_records": 2,
        "cupti_dropped_records": 0,
        "local_dropped_records": 0,
        "bytes_dropped": 0,
    }
    (cupti / "cupti_status.json").write_text(json.dumps(status))
    samples = [{"processes": [{"pid": 12, "role": "target"}]}]
    capture, _ = _cupti_capture_status(tmp_path, samples)
    assert capture["complete"] is False
    assert any("record count" in error for error in capture["errors"])
    trace.write_text('{"kind":"kernel"}\ninvalid\n')
    capture, _ = _cupti_capture_status(tmp_path, samples)
    assert capture["complete"] is False
    assert any("malformed activity trace" in error for error in capture["errors"])


def test_goodput_requires_declared_slo_and_keeps_failures_in_population() -> None:
    rows = [
        {"status": "ok", "e2e_latency_ms": 40},
        {"status": "ok", "e2e_latency_ms": 120},
        {"status": "timeout", "e2e_latency_ms": 20},
    ]
    assert _slo_goodput(rows, 2.0, None) is None
    assert _slo_goodput(rows, 2.0, 100) == 0.5
    with pytest.raises(ValueError, match="positive"):
        _slo_goodput(rows, 2.0, 0)


def test_result_metrics_are_normalizable() -> None:
    from scripts.native_probes.normalization import _validated_trial_metrics

    row = {"status": "ok", "e2e_latency_ms": 10, "ttft_ms": 2}
    result = _result([row], 1, 1_000_000_001)
    result["metrics"]["server_exited_early_count"] = 0
    assert _validated_trial_metrics(result) == result["metrics"]


def test_approved_protocol_matches_frozen_proposal_and_adapter() -> None:
    root = Path(__file__).resolve().parents[1]
    proposal_bytes = (
        root / "benchmarks/native_probes/final_run_proposal.json"
    ).read_bytes()
    proposal = json.loads(proposal_bytes)
    approval = json.loads(
        (root / "benchmarks/native_probes/final_run_approval.json").read_text()
    )
    experiment = json.loads(
        (root / "benchmarks/native_probes/experiment.json").read_text()
    )

    assert proposal["status"] == "proposed_not_approved"
    assert approval["status"] == "policy_and_workload_approved_pending_preflight"
    assert approval["paid_hardware"]["status"] == "not_authorized"
    assert approval["proposal_sha256"] == hashlib.sha256(proposal_bytes).hexdigest()
    thresholds = experiment["acceptance_thresholds"]
    assert thresholds["proposal_sha256"] == approval["proposal_sha256"]
    assert thresholds["latency_perturbation_max"] == 0.05
    assert thresholds["memory_reduction_min"] == 0.25
    assert thresholds["correlation_coverage_min"] == 0.99
    assert thresholds["event_loss_max"] == 0
    assert thresholds["sample_sufficiency_min"] == 5
    assert thresholds["paid_hardware_authorization_required"] is True
    assert proposal["vllm"]["model_revision"] == REVISION
    assert experiment["workloads"]["vllm"]["model_revision"] == REVISION
    assert experiment["workloads"]["vllm"]["tokenizer_revision"] == REVISION
    assert (
        proposal["vllm"]["request"]["user_message"]
        == request_body()["messages"][0]["content"]
    )
    assert (
        proposal["vllm"]["arrivals"]["measured_offered_requests"] == MEASURED_REQUESTS
    )


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
