"""CPU checks for the fixed-arrival vLLM trial boundary."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import struct
import tempfile
import threading
import time
import zlib
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.native_probes.mode_commands import (
    Workload,
    microbenchmark_command,
    vllm_command,
    vllm_expected_artifacts,
)
from scripts.native_probes.models import ExperimentMode, WorkloadId
from scripts.native_probes.planning import _pressure_controls, build_plan
from scripts.native_probes.workloads.vllm_open_loop import (
    MEASURED_REQUESTS,
    REVISION,
    SLO_DEADLINE_MS,
    SLO_OFFER_INTERVAL_SECONDS,
    _cupti_capture_status,
    _inspect_cupti_trace,
    _memory_metrics,
    _request,
    _result,
    _server_argv,
    _slo_goodput,
    _stop_cupti_helpers,
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
        status = 200

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


def test_cupti_stop_requires_private_socket_ack_and_actual_controls() -> None:
    with tempfile.TemporaryDirectory(prefix="cupti-stop-", dir="/tmp") as temporary:
        root = Path(temporary)
        process = root / "pid-123-test"
        process.mkdir(mode=0o700)
        path = process / "stop.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(path))
            os.chmod(path, 0o600)
            listener.listen(1)

            def serve() -> None:
                connection, _ = listener.accept()
                with connection:
                    assert connection.recv(4) == b"STOP"
                    (process / "cupti_status.json").write_text(
                        json.dumps(
                            {
                                "pid": 123,
                                "finalized": True,
                                "activity_buffer_bytes": 8388608,
                            }
                        )
                    )
                    connection.sendall(b"OK\n")

            worker = threading.Thread(target=serve, daemon=True)
            worker.start()
            result = _stop_cupti_helpers(
                root, timeout=2, expected_controls={"activity_buffer_bytes": 8388608}
            )
            worker.join(timeout=2)
            assert result["complete"] is True
            assert result["reports"][0]["status"]["pid"] == 123
        path.unlink()
        assert _stop_cupti_helpers(root)["complete"] is False


def test_cupti_stop_retains_missing_ack() -> None:
    with tempfile.TemporaryDirectory(prefix="cupti-stop-", dir="/tmp") as temporary:
        root = Path(temporary)
        process = root / "pid-123-test"
        process.mkdir(mode=0o700)
        path = process / "stop.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(path))
            os.chmod(path, 0o600)
            listener.listen(1)

            def serve() -> None:
                connection, _ = listener.accept()
                with connection:
                    assert connection.recv(4) == b"STOP"

            worker = threading.Thread(target=serve, daemon=True)
            worker.start()
            result = _stop_cupti_helpers(root, timeout=2)
            worker.join(timeout=2)
            assert result["complete"] is False
            assert result["reports"][0]["ack"] == ""
            assert result["errors"]


def test_cupti_stop_times_out_and_keeps_partial_trace() -> None:
    with tempfile.TemporaryDirectory(prefix="cupti-stop-", dir="/tmp") as temporary:
        root = Path(temporary)
        process = root / "pid-123-test"
        process.mkdir(mode=0o700)
        partial = process / "activity.partial"
        partial.write_bytes(b"incomplete trace")
        path = process / "stop.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(path))
            os.chmod(path, 0o600)
            listener.listen(1)

            def serve() -> None:
                connection, _ = listener.accept()
                with connection:
                    assert connection.recv(4) == b"STOP"
                    time.sleep(0.2)

            worker = threading.Thread(target=serve, daemon=True)
            worker.start()
            result = _stop_cupti_helpers(root, timeout=0.05)
            worker.join(timeout=1)
            assert result["complete"] is False
            assert any("TimeoutError" in error for error in result["errors"])
            assert partial.read_bytes() == b"incomplete trace"


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
    capture, _ = _cupti_capture_status(tmp_path, samples, required_gpu_pids={12})
    assert capture["complete"] is True
    assert capture["required_gpu_pids"] == [12]
    capture, _ = _cupti_capture_status(tmp_path, samples, required_gpu_pids=set())
    assert capture["complete"] is False

    samples[0]["processes"].pop()
    status["local_dropped_records"] = 2
    (cupti / "cupti_status.json").write_text(json.dumps(status))
    capture, loss = _cupti_capture_status(tmp_path, samples)
    assert capture["complete"] is False
    assert loss["vendor_activity"]["lost_records"] == 2


def test_cupti_capture_keeps_two_process_traces_distinct(tmp_path: Path) -> None:
    for pid in (12, 13):
        process = tmp_path / "cupti" / f"pid-{pid}-distinct"
        process.mkdir(parents=True)
        (process / "activity.ndjson").write_text(
            json.dumps({"pid": pid}) + "\n", encoding="utf-8"
        )
        (process / "cupti_status.json").write_text(
            json.dumps(
                {
                    "pid": pid,
                    "finalized": True,
                    "initialization_error": None,
                    "delivered_records": 1,
                    "cupti_dropped_records": 0,
                    "local_dropped_records": 0,
                    "bytes_dropped": 0,
                }
            ),
            encoding="utf-8",
        )
    samples = [{"processes": [{"pid": pid, "role": "target"} for pid in (12, 13)]}]
    capture, loss = _cupti_capture_status(tmp_path, samples, {12, 13})
    assert capture["complete"] is True
    assert capture["reported_pids"] == [12, 13]
    assert loss["vendor_activity"]["lost_records"] == 0


def test_compressed_cupti_trace_verifies_framing_hash_and_bound(tmp_path: Path) -> None:
    process = tmp_path / "cupti" / "pid-12-distinct"
    process.mkdir(parents=True)
    raw = b'{"pid":12}\n'
    encoded = zlib.compress(raw, level=3)
    frame = struct.pack("<III", len(encoded), len(raw), 1) + encoded
    trace = process / "activity.sclz"
    trace.write_bytes(b"SLCPTZ1\n" + frame + bytes(12))
    inspected = _inspect_cupti_trace(trace)
    assert inspected["decoded_sha256"] == hashlib.sha256(raw).hexdigest()
    assert inspected["records"] == 1
    status = {
        "pid": 12,
        "finalized": True,
        "initialization_error": None,
        "trace_encoding": "stormlog-zlib-frames-v1",
        "delivered_records": 1,
        "cupti_dropped_records": 0,
        "local_dropped_records": 0,
        "bytes_dropped": 0,
        "bytes_written": trace.stat().st_size,
        "uncompressed_bytes_delivered": len(raw),
        "maximum_output_bytes": trace.stat().st_size,
    }
    (process / "cupti_status.json").write_text(json.dumps(status))
    samples = [{"processes": [{"pid": 12, "role": "target"}]}]
    capture, _ = _cupti_capture_status(tmp_path, samples, {12})
    assert capture["complete"] is True
    assert (
        capture["trace_validation"][0]["decoded_sha256"] == inspected["decoded_sha256"]
    )
    status["maximum_output_bytes"] = trace.stat().st_size - 1
    (process / "cupti_status.json").write_text(json.dumps(status))
    assert _cupti_capture_status(tmp_path, samples, {12})[0]["complete"] is False
    trace.write_bytes(trace.read_bytes()[:-1])
    with pytest.raises(ValueError, match="end frame"):
        _inspect_cupti_trace(trace)
    trace.write_bytes(b"SLCPTZ1\n" + frame[:-1] + bytes(12))
    with pytest.raises(ValueError):
        _inspect_cupti_trace(trace)


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


def test_approved_goodput_uses_fixed_offer_interval_and_complete_population() -> None:
    rows = [
        {
            "request_id": f"stormlog-118-measured-{index:04d}",
            "scheduled_offset_ms": index * 100,
            "offered_at_ns": index * 100_000_000,
            "status": "client_rejected",
        }
        for index in range(MEASURED_REQUESTS)
    ]
    rows[0].update(
        status="ok",
        http_status=200,
        stream_done=True,
        ended_at_ns=1_800_000_000,
        e2e_latency_ms=1800,
        ttft_ms=100,
    )
    rows[1].update(
        status="ok",
        http_status=200,
        stream_done=True,
        ended_at_ns=2_200_000_000,
        e2e_latency_ms=2100,
        ttft_ms=100,
    )
    rows[-1].update(
        status="ok",
        http_status=200,
        stream_done=True,
        ended_at_ns=221_500_000_000,
        e2e_latency_ms=1600,
        ttft_ms=100,
    )
    result = _result(rows, 0, 300_000_000_000)
    assert result["metrics"]["slo_deadline_ms"] == 2000
    assert result["metrics"]["slo_offer_interval_seconds"] == 220
    assert result["metrics"]["slo_goodput_requests_per_second"] == 2 / 220
    assert (
        _result(rows[:-1], 0, 300_000_000_000)["metrics"][
            "slo_goodput_requests_per_second"
        ]
        is None
    )
    rows[0]["stream_done"] = False
    assert (
        _result(rows, 0, 300_000_000_000)["metrics"]["slo_goodput_requests_per_second"]
        is None
    )


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


def test_slo_and_w4_values_match_approved_amendment() -> None:
    root = Path(__file__).resolve().parents[1] / "benchmarks/native_probes"
    amendment_bytes = (
        root / "protocol_amendment_proposal_2026-10-02.json"
    ).read_bytes()
    amendment = json.loads(amendment_bytes)
    approval = json.loads(
        (root / "protocol_amendment_approval_2026-10-02.json").read_text()
    )
    assert approval["proposal_sha256"] == hashlib.sha256(amendment_bytes).hexdigest()
    assert approval["approved_scope"] == [
        "slo_goodput_reporting",
        "w4_direct_cupti_pressure.variants",
    ]
    assert approval["paid_hardware_authorized"] is False
    assert approval["final_trials_authorized"] is False
    assert amendment["slo_goodput_reporting"]["proposed_end_to_end_deadline_ms"] == (
        SLO_DEADLINE_MS
    )
    assert amendment["slo_goodput_reporting"]["proposed_interval_seconds"] == (
        SLO_OFFER_INTERVAL_SECONDS
    )
    variants = amendment["w4_direct_cupti_pressure"]["variants"]
    planned = _pressure_controls(
        Workload(WorkloadId.W4_STRESS), ExperimentMode.DIRECT_CUPTI
    )
    assert planned["approved_variants"] == variants
    assert planned["variant_status"] == "unsupported"
    assert "producer_buffer_bytes" in planned["supported"]
    assert "consumer_delay_ms" in planned["supported"]
    assert variants["nominal"]["producer_activity_buffer_bytes"] == 8 * 1024**2
    assert variants["small_buffer"]["producer_activity_buffer_bytes"] == 64 * 1024
    assert variants["slow_consumer"]["consumer_delay_ms_per_completed_buffer"] == 25
    assert variants["output_limit"]["output_byte_bound"] == 64 * 1024
    assert (
        variants["target_timeout"]["target_timeout_after_measurement_start_ms"] == 500
    )


def test_approved_stop_rule_and_cupti_pressure_environment(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1] / "benchmarks/native_probes"
    proposal_bytes = (root / "protocol_stop_proposal_2026-10-02.json").read_bytes()
    approval = json.loads((root / "protocol_stop_approval_2026-10-02.json").read_text())
    assert approval["proposal_sha256"] == hashlib.sha256(proposal_bytes).hexdigest()
    assert approval["approved_scope"] == "direct_cupti_stop_rule"
    assert approval["final_trials_authorized"] is False

    library = tmp_path / "libstormlog_cupti_injection.so"
    library.write_bytes(b"placeholder")
    command = microbenchmark_command(
        ExperimentMode.DIRECT_CUPTI,
        Workload(
            WorkloadId.W4_STRESS,
            producer_buffer_bytes=65536,
            output_byte_bound=65536,
            consumer_delay_ms=25,
        ),
        tmp_path,
        cupti_library=library,
    )
    assert command.environment["STORMLOG_CUPTI_BUFFER_BYTES"] == "65536"
    assert command.environment["STORMLOG_CUPTI_CONSUMER_DELAY_MS"] == "25"
    assert command.environment["STORMLOG_CUPTI_MAX_BYTES"] == "65536"
    with pytest.raises(ValueError, match="pressure control"):
        microbenchmark_command(
            ExperimentMode.DIRECT_CUPTI,
            Workload(WorkloadId.W4_STRESS, consumer_delay_ms=0.5),
            tmp_path,
            cupti_library=library,
        )


def test_w4_plan_expands_approved_single_intervention_variants(tmp_path: Path) -> None:
    approved = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "benchmarks/native_probes/protocol_amendment_proposal_2026-10-02.json"
        ).read_text(encoding="utf-8")
    )["w4_direct_cupti_pressure"]["variants"]
    library = tmp_path / "libstormlog_cupti_injection.so"
    library.write_bytes(b"placeholder")
    plan = build_plan(
        configuration_id="w4-local",
        vendor="nvidia",
        workloads=[WorkloadId.W4_STRESS],
        modes=[ExperimentMode.OFF, ExperimentMode.DIRECT_CUPTI],
        repetitions=5,
        seed=118,
        environment_artifact="environment.json",
        artifact_root=tmp_path,
        cupti_library=library,
    )
    assert len(plan["trials"]) == 30
    direct = [row for row in plan["trials"] if row["mode"] == "direct-cupti"]
    assert len({row["trial_id"] for row in plan["trials"]}) == 30
    assert {row["pressure_controls"]["variant_name"] for row in direct} == {
        "nominal",
        "small_buffer",
        "slow_consumer",
        "output_limit",
        "target_timeout",
    }
    assert all(
        row["pressure_controls"]["variant_status"] == "unsupported" for row in direct
    )
    for row in direct:
        controls = row["pressure_controls"]
        environment = row["command"]["environment"]
        expected = approved[controls["variant_name"]]
        assert (
            int(environment["STORMLOG_CUPTI_BUFFER_BYTES"])
            == expected["producer_activity_buffer_bytes"]
        )
        assert (
            int(environment["STORMLOG_CUPTI_MAX_BYTES"])
            == expected["output_byte_bound"]
        )
        assert (
            int(environment["STORMLOG_CUPTI_CONSUMER_DELAY_MS"])
            == expected["consumer_delay_ms_per_completed_buffer"]
        )
        if controls["variant_name"] == "target_timeout":
            assert controls["target_timeout_after_measurement_start_ms"] == 500
            assert environment["STORMLOG_MEASUREMENT_START_FILE"].endswith(
                "/measurement-start.ns"
            )
        else:
            assert "STORMLOG_MEASUREMENT_START_FILE" not in environment


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
    library = tmp_path / "libstormlog_cupti_injection.so"
    library.touch()
    direct = vllm_command(ExperimentMode.DIRECT_CUPTI, tmp_path, cupti_library=library)
    assert direct.environment["STORMLOG_CUPTI_MAX_BYTES"] == "4294967296"
    assert any(
        row.relative_path == "vllm/nsys"
        for row in vllm_expected_artifacts(ExperimentMode.TRUSTED)
    )
