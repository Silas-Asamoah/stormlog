"""Do not turn unrelated server measurements into request-owned memory."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.analysis import analyze_inference_events, format_analysis_text
from stormlog.infer.correlation_events import ClockAlignmentEvent, CorrelationContext
from stormlog.infer.telemetry import ServerIdentity, TelemetrySample

SECOND = 1_000_000_000
MS = 1_000_000


def _profile(
    tmp_path: Path,
    *,
    host: str = "client",
    run_id: str = "run-1",
    windows: dict[str, tuple[int, int]] | None = None,
    boot: bool = True,
    extra_records: list[dict[str, Any]] | None = None,
) -> Path:
    path = tmp_path / "profile.jsonl"
    records: list[dict[str, Any]] = [
        {
            "schema_version": 2,
            "event_type": "infer.artifact",
            "context": {"run_id": run_id, "host": host},
            "metadata": {"boot_id": f"boot-{host}"} if boot else {},
        },
        *(extra_records or []),
        {
            "schema_version": 1,
            "event_type": "infer.system_sample",
            "timestamp_ns": 150,
            "device_used_bytes": 999,
            "observation_scope": "client_local",
        },
    ]
    for case_id, (started, ended) in (windows or {"case-a": (100, 300)}).items():
        records.append(
            {
                "schema_version": 1,
                "event_type": "infer.request",
                "phase": "measured",
                "status": "ok",
                "case_id": case_id,
                "started_at_ns": started,
                "ended_at_ns": ended,
                "output_tokens": 1,
                "total_tokens": 2,
            }
        )
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def _server_sample(
    *,
    host: str = "server-a",
    pid: int = 42,
    start_ns: int = 10,
    uuid: str = "GPU-A",
    instance: str | None = None,
    metric: str = "device_memory_used_bytes",
    state: str = "valid",
    value: int | None = 256,
    run_id: str = "run-1",
    boot_id: str | None = None,
    observed_at_ns: int = 150,
    interval_ms: int = 100,
    source: str = "nvml-v2",
    provenance: str = "observed",
    detail: str | None = None,
) -> TelemetrySample:
    return TelemetrySample(
        run_id=run_id,
        identity=ServerIdentity(
            host=host,
            pid=pid,
            process_start_ns=start_ns,
            device_uuid=uuid,
            gpu_instance_id=instance,
            replica_id=f"replica-{host}",
            boot_id=boot_id or f"boot-{host}",
        ),
        observed_at_ns=observed_at_ns,
        metric=metric,
        value_bytes=value,
        state=state,
        source=source,
        interval_ms=interval_ms,
        detail=detail,
        provenance=provenance,
    )


def _polls(
    start_ns: int, end_ns: int, *, step_ns: int = 100 * MS, **changes: Any
) -> list[TelemetrySample]:
    """One valid device sample per poll, like a collector running in that span."""
    return [
        _server_sample(observed_at_ns=observed, value=observed // MS, **changes)
        for observed in range(start_ns, end_ns + 1, step_ns)
    ]


def _ended(observed_at_ns: int) -> TelemetrySample:
    return _server_sample(
        observed_at_ns=observed_at_ns,
        state="invalid",
        value=None,
        detail="server process ended or its PID was reused",
    )


SERVER_CLOCK = "server-a/boot-server-a/unix_epoch_ns"
CLIENT_CLOCK = "client/boot-client/unix_epoch_ns"


def _alignment_record(
    event_id: str,
    *,
    offset_ns: int = 0,
    uncertainty_ns: int = 0,
    valid_from_ns: int | None = None,
    valid_to_ns: int | None = None,
    from_clock_domain: str = SERVER_CLOCK,
    to_clock_domain: str = CLIENT_CLOCK,
) -> dict[str, Any]:
    """An ``infer.clock_alignment`` record, as a clock probe would append it."""
    return ClockAlignmentEvent(
        context=CorrelationContext(
            run_id="run-1",
            session_id="session-probe",
            producer_id="clock-probe",
            source="clock-probe",
            clock_domain=to_clock_domain,
            clock_kind="wall",
            collection_mode="passive",
            provenance="observed",
        ),
        event_id=event_id,
        from_clock_domain=from_clock_domain,
        to_clock_domain=to_clock_domain,
        offset_ns=offset_ns,
        uncertainty_ns=uncertainty_ns,
        valid_from_ns=valid_from_ns,
        valid_to_ns=valid_to_ns,
    ).to_record()


def _aligned_report(
    tmp_path: Path,
    alignments: list[dict[str, Any]],
    *samples: TelemetrySample,
    windows: dict[str, tuple[int, int]] | None = None,
    **options: Any,
) -> dict[str, Any]:
    profile = _profile(tmp_path, windows=windows, extra_records=alignments)
    telemetry = _telemetry(tmp_path, "server.jsonl", *samples)
    return analyze_inference_events(
        profile, server_telemetry_paths=[telemetry], direct_server=True, **options
    )


def _telemetry(tmp_path: Path, name: str, *samples: TelemetrySample) -> Path:
    path = tmp_path / name
    path.write_text("".join(json.dumps(s.to_record()) + "\n" for s in samples))
    return path


def _same_host_report(
    tmp_path: Path,
    windows: dict[str, tuple[int, int]] | None,
    *samples: TelemetrySample,
    **options: Any,
) -> dict[str, Any]:
    profile = _profile(tmp_path, host="server-a", windows=windows)
    telemetry = _telemetry(tmp_path, "server.jsonl", *samples)
    return analyze_inference_events(
        profile, server_telemetry_paths=[telemetry], direct_server=True, **options
    )


def test_endpoint_only_memory_is_explicitly_client_local(tmp_path: Path) -> None:
    report = analyze_inference_events(_profile(tmp_path))
    memory = report["cases"]["case-a"]["memory"]
    assert memory["observation_scope"] == "client_local"
    assert memory["peak_device_used_bytes"] == 999
    assert memory["server_observations"] == {}
    assert memory["server_coverage"] == {"status": "not_joined"}
    assert report["telemetry"]["server_join"]["status"] == "not_configured"


def test_direct_remote_join_requires_clock_alignment(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    telemetry = _telemetry(tmp_path, "server.jsonl", _server_sample())
    unaligned = analyze_inference_events(
        profile, server_telemetry_paths=[telemetry], direct_server=True
    )
    assert unaligned["telemetry"]["server_join"]["reason"] == "clock_alignment_required"
    assert unaligned["cases"]["case-a"]["memory"]["server_observations"] == {}
    report = analyze_inference_events(
        profile,
        server_telemetry_paths=[telemetry],
        direct_server=True,
        clock_offset_ns=0,
        clock_uncertainty_ns=5,
    )
    join = report["telemetry"]["server_join"]
    assert join["status"] == "joined"
    assert join["route_evidence"] == "operator_declared_direct"
    assert join["invalidation"] is None
    assert join["clock_alignment_evidence"] == "operator_supplied"
    assert join["clock_alignments"] == [
        {
            "source": "operator",
            "event_id": "cli:clock-offset",
            "from_clock_domain": SERVER_CLOCK,
            "to_clock_domain": CLIENT_CLOCK,
            "offset_ns": 0,
            "uncertainty_ns": 5,
            "valid_from_ns": None,
            "valid_to_ns": None,
            "aligned_samples": 1,
        }
    ]
    assert join["unaligned_samples"] == {"uncovered": 0, "ambiguous": 0}
    memory = report["cases"]["case-a"]["memory"]
    assert memory["peak_device_used_bytes"] == 999
    assert memory["server_coverage"]["status"] == "observed"
    assert memory["server_observations"]["device_memory_used_bytes"] == {
        "observation_scope": "gpu_device",
        "counter_owner": "gpu_device",
        "sources": ["nvml-v2"],
        "provenance": ["observed"],
        "maximum_recorded_bytes": 256,
        "valid_samples": 1,
        "missing_samples": 0,
        "stale_samples": 0,
        "invalid_samples": 0,
        "intervals_ms": [100],
    }


def test_equal_hostname_without_equal_boot_id_needs_alignment(tmp_path: Path) -> None:
    profile = _profile(tmp_path, host="server-a")
    telemetry = _telemetry(
        tmp_path,
        "same-name.jsonl",
        _server_sample(boot_id="another-boot"),
    )
    report = analyze_inference_events(
        profile, server_telemetry_paths=[telemetry], direct_server=True
    )
    assert report["telemetry"]["server_join"]["reason"] == "clock_alignment_required"


def test_two_servers_both_index_zero_are_not_merged(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    a = _telemetry(tmp_path, "a.jsonl", _server_sample(uuid="GPU-A"))
    b = _telemetry(
        tmp_path,
        "b.jsonl",
        _server_sample(host="server-b", pid=43, uuid="GPU-B"),
    )
    report = analyze_inference_events(
        profile,
        server_telemetry_paths=[a, b],
        direct_server=True,
        clock_offset_ns=0,
        clock_uncertainty_ns=0,
    )
    assert report["telemetry"]["server_join"]["reason"] == "multiple_server_identities"
    assert {
        t["identity"]["device_uuid"] for t in report["telemetry"]["server_targets"]
    } == {
        "GPU-A",
        "GPU-B",
    }


@pytest.mark.parametrize("change", [{"pid": 43}, {"start_ns": 11}, {"uuid": "GPU-B"}])
def test_process_restart_or_gpu_change_cannot_silently_join(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    profile = _profile(tmp_path)
    telemetry = _telemetry(
        tmp_path, "changed.jsonl", _server_sample(), _server_sample(**change)
    )
    report = analyze_inference_events(
        profile, server_telemetry_paths=[telemetry], direct_server=True
    )
    assert report["telemetry"]["server_join"]["reason"] == "multiple_server_identities"


def test_missing_counter_is_null_and_instance_scope_remains_distinct(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path, host="server-a")
    telemetry = _telemetry(
        tmp_path,
        "mig.jsonl",
        _server_sample(
            uuid="GPU-A",
            instance="MIG-A",
            metric="instance_memory_used_bytes",
        ),
        _server_sample(
            uuid="GPU-A",
            instance="MIG-A",
            metric="instance_memory_reserved_bytes",
            state="missing",
            value=None,
        ),
    )
    report = analyze_inference_events(
        profile, server_telemetry_paths=[telemetry], direct_server=True
    )
    memory = report["cases"]["case-a"]["memory"]["server_observations"]
    assert memory["instance_memory_used_bytes"]["observation_scope"] == "gpu_instance"
    assert memory["instance_memory_reserved_bytes"]["maximum_recorded_bytes"] is None
    assert memory["instance_memory_reserved_bytes"]["missing_samples"] == 1


def test_server_stopping_after_the_run_keeps_every_case(tmp_path: Path) -> None:
    # The collector writes an invalid sample when the server goes away; after
    # the last case that must not discard readings taken during the run.
    report = _same_host_report(
        tmp_path,
        {"case-a": (1 * SECOND, 2 * SECOND)},
        *_polls(900 * MS, 2_500 * MS),
        _ended(2_600 * MS),
    )
    join = report["telemetry"]["server_join"]
    assert join["status"] == "joined"
    assert join["invalidation"] == {
        "observed_at_ns": 2_600 * MS,
        "detail": "server process ended or its PID was reused",
        "last_confirmed_at_ns": 2_500 * MS,
    }
    memory = report["cases"]["case-a"]["memory"]
    assert memory["server_coverage"]["status"] == "observed"
    observation = memory["server_observations"]["device_memory_used_bytes"]
    assert observation["maximum_recorded_bytes"] == 2_000
    assert observation["valid_samples"] == 11
    assert observation["invalid_samples"] == 0


def test_invalidation_only_affects_cases_after_the_last_confirmed_poll(
    tmp_path: Path,
) -> None:
    report = _same_host_report(
        tmp_path,
        {"case-a": (1 * SECOND, 2 * SECOND), "case-b": (3 * SECOND, 4 * SECOND)},
        *_polls(900 * MS, 2_500 * MS),
        _ended(2_600 * MS),
    )
    cases = report["cases"]
    assert cases["case-a"]["memory"]["server_coverage"]["status"] == "observed"
    assert cases["case-b"]["memory"]["server_observations"] == {}
    assert cases["case-b"]["memory"]["server_coverage"] == {
        "status": "empty",
        "reason": "identity_invalidated",
        "counted_window_ns": [3 * SECOND, 4 * SECOND],
    }
    assert report["telemetry"]["server_join"]["case_coverage"] == {
        "observed": 1,
        "partial": 0,
        "empty": 1,
    }


def test_invalidation_within_clock_uncertainty_of_a_case_excludes_it(
    tmp_path: Path,
) -> None:
    # Confirmed through 2.0 s, but a 200 ms clock uncertainty means that poll
    # could have happened before the 1.9 s end of the case.
    report = _same_host_report(
        tmp_path,
        {"case-a": (1 * SECOND, 1_900 * MS)},
        *_polls(900 * MS, 2_000 * MS),
        _ended(2_100 * MS),
        clock_offset_ns=0,
        clock_uncertainty_ns=200 * MS,
    )
    coverage = report["cases"]["case-a"]["memory"]["server_coverage"]
    assert coverage["reason"] == "identity_invalidated"


def test_invalid_first_poll_leaves_no_trusted_window(tmp_path: Path) -> None:
    report = _same_host_report(
        tmp_path,
        None,
        _server_sample(),
        _server_sample(state="invalid", value=None),
    )
    assert report["telemetry"]["server_join"]["status"] == "joined"
    memory = report["cases"]["case-a"]["memory"]
    assert memory["server_observations"] == {}
    assert memory["server_coverage"]["reason"] == "identity_invalidated"


def test_mismatched_run_id_leaves_client_report_intact(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    telemetry = _telemetry(
        tmp_path, "wrong-run.jsonl", _server_sample(run_id="other-run")
    )
    report = analyze_inference_events(
        profile, server_telemetry_paths=[telemetry], direct_server=True
    )
    assert report["telemetry"]["server_join"] == {
        "status": "unjoined",
        "reason": "run_id_mismatch",
        "artifact_run_id": "run-1",
        "telemetry_run_ids": ["other-run"],
    }
    memory = report["cases"]["case-a"]["memory"]
    assert memory["peak_device_used_bytes"] == 999
    assert memory["server_observations"] == {}
    assert report["summary"]["successful_requests"] == 1
    assert "unjoined (run_id_mismatch)" in format_analysis_text(report)


def test_window_shorter_than_twice_the_uncertainty_is_reported_empty(
    tmp_path: Path,
) -> None:
    report = _same_host_report(
        tmp_path,
        {"case-a": (1 * SECOND, 1_600 * MS)},
        *_polls(900 * MS, 1_700 * MS),
        clock_offset_ns=0,
        clock_uncertainty_ns=350 * MS,
    )
    memory = report["cases"]["case-a"]["memory"]
    assert memory["server_observations"] == {}
    assert memory["server_coverage"] == {
        "status": "empty",
        "reason": "window_shorter_than_uncertainty",
        "counted_window_ns": None,
    }
    text = format_analysis_text(report)
    assert "server telemetry: none (window_shorter_than_uncertainty)" in text
    assert "Server coverage: 0 observed, 0 partial, 1 empty" in text


def test_collector_coverage_is_reported_per_case(tmp_path: Path) -> None:
    report = _same_host_report(
        tmp_path,
        {
            "case-a": (1 * SECOND, 1_900 * MS),
            "case-b": (2 * SECOND, 2_900 * MS),
            "case-c": (3 * SECOND, 3_900 * MS),
        },
        *_polls(900 * MS, 2_500 * MS),
    )
    coverage = {
        case_id: (
            case["memory"]["server_coverage"]["status"],
            case["memory"]["server_coverage"]["reason"],
        )
        for case_id, case in report["cases"].items()
    }
    assert coverage == {
        "case-a": ("observed", None),
        "case-b": ("partial", "collector_stopped_before_window_end"),
        "case-c": ("empty", "no_collector_coverage"),
    }
    text = format_analysis_text(report)
    assert "server telemetry: partial (collector_stopped_before_window_end)" in text
    assert "server telemetry: none (no_collector_coverage)" in text


def test_collector_started_late_is_partial(tmp_path: Path) -> None:
    report = _same_host_report(
        tmp_path,
        {"case-a": (1 * SECOND, 1_900 * MS)},
        *_polls(1_500 * MS, 2_500 * MS),
    )
    coverage = report["cases"]["case-a"]["memory"]["server_coverage"]
    assert (coverage["status"], coverage["reason"]) == (
        "partial",
        "collector_started_after_window_start",
    )


def test_same_artifact_twice_is_not_double_counted(tmp_path: Path) -> None:
    profile = _profile(tmp_path, host="server-a")
    telemetry = _telemetry(tmp_path, "server.jsonl", _server_sample())
    report = analyze_inference_events(
        profile, server_telemetry_paths=[telemetry, telemetry], direct_server=True
    )
    observation = report["cases"]["case-a"]["memory"]["server_observations"][
        "device_memory_used_bytes"
    ]
    assert observation["valid_samples"] == 1
    target = report["telemetry"]["server_targets"][0]
    assert target["sample_states"]["valid"] == 1


def test_same_host_join_honors_supplied_uncertainty(tmp_path: Path) -> None:
    report = _same_host_report(
        tmp_path, None, _server_sample(), clock_uncertainty_ns=40
    )
    join = report["telemetry"]["server_join"]
    assert join["clock_alignment_evidence"] == "same_host"
    assert join["clock_offset_ns"] == 0
    assert join["clock_uncertainty_ns"] == 40
    assert join["clock_alignments"][0]["source"] == "same_host"
    # 150 ns is inside [100 + 40, 300 - 40].
    memory = report["cases"]["case-a"]["memory"]["server_observations"]
    assert memory["device_memory_used_bytes"]["valid_samples"] == 1


def test_metric_metadata_describes_only_samples_in_the_window(tmp_path: Path) -> None:
    report = _same_host_report(
        tmp_path,
        None,
        _server_sample(observed_at_ns=150),
        _server_sample(
            observed_at_ns=500,
            source="dcgm",
            provenance="reported",
            interval_ms=1000,
        ),
    )
    observation = report["cases"]["case-a"]["memory"]["server_observations"][
        "device_memory_used_bytes"
    ]
    assert observation["sources"] == ["nvml-v2"]
    assert observation["provenance"] == ["observed"]
    assert observation["intervals_ms"] == [100]


def test_non_object_profile_line_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("[1, 2]\n")
    with pytest.raises(ValueError, match="not a JSON object"):
        analyze_inference_events(path)


def test_artifact_alignment_record_joins_without_flags(tmp_path: Path) -> None:
    report = _aligned_report(
        tmp_path, [_alignment_record("probe-1", uncertainty_ns=5)], _server_sample()
    )
    join = report["telemetry"]["server_join"]
    assert join["status"] == "joined"
    assert join["clock_alignment_evidence"] == "artifact_record"
    assert [item["event_id"] for item in join["clock_alignments"]] == ["probe-1"]
    assert join["clock_alignments"][0]["source"] == "artifact"
    memory = report["cases"]["case-a"]["memory"]["server_observations"]
    assert memory["device_memory_used_bytes"]["valid_samples"] == 1


def test_validity_windows_place_each_sample_with_its_own_offset(
    tmp_path: Path,
) -> None:
    report = _aligned_report(
        tmp_path,
        [
            _alignment_record("early", valid_to_ns=1_000),
            _alignment_record("late", offset_ns=-700, valid_from_ns=1_000),
        ],
        _server_sample(observed_at_ns=150, value=10),
        _server_sample(observed_at_ns=1_150, value=20),
        windows={"case-a": (100, 300), "case-b": (400, 600)},
    )
    join = report["telemetry"]["server_join"]
    assert join["clock_offset_ns"] is None
    assert {
        item["event_id"]: item["aligned_samples"] for item in join["clock_alignments"]
    } == {"early": 1, "late": 1}
    maxima = {
        case_id: case["memory"]["server_observations"]["device_memory_used_bytes"][
            "maximum_recorded_bytes"
        ]
        for case_id, case in report["cases"].items()
    }
    assert maxima == {"case-a": 10, "case-b": 20}


def test_samples_outside_every_window_are_counted_not_joined(tmp_path: Path) -> None:
    covered = _alignment_record("covered", valid_to_ns=1_000)
    report = _aligned_report(
        tmp_path,
        [covered],
        _server_sample(observed_at_ns=150),
        _server_sample(observed_at_ns=1_150),
    )
    join = report["telemetry"]["server_join"]
    assert join["status"] == "joined"
    assert join["unaligned_samples"] == {"uncovered": 1, "ambiguous": 0}
    report = _aligned_report(
        tmp_path,
        [_alignment_record("later", valid_from_ns=2_000)],
        _server_sample(observed_at_ns=150),
    )
    assert report["telemetry"]["server_join"] == {
        "status": "unjoined",
        "reason": "clock_alignment_uncovered",
    }


def test_overlapping_alignment_records_are_ambiguous(tmp_path: Path) -> None:
    report = _aligned_report(
        tmp_path,
        [_alignment_record("a"), _alignment_record("b", offset_ns=3)],
        _server_sample(),
    )
    assert report["telemetry"]["server_join"]["reason"] == "clock_alignment_ambiguous"


def test_operator_flags_replace_artifact_records(tmp_path: Path) -> None:
    report = _aligned_report(
        tmp_path,
        [_alignment_record("probe-1", offset_ns=1_000)],
        _server_sample(),
        clock_offset_ns=0,
        clock_uncertainty_ns=5,
    )
    join = report["telemetry"]["server_join"]
    assert join["clock_alignment_evidence"] == "operator_supplied"
    assert join["overridden_clock_alignments"] == ["probe-1"]
    memory = report["cases"]["case-a"]["memory"]["server_observations"]
    assert memory["device_memory_used_bytes"]["valid_samples"] == 1


def test_same_hostname_on_another_boot_can_be_aligned(tmp_path: Path) -> None:
    profile = _profile(
        tmp_path,
        host="server-a",
        extra_records=[
            _alignment_record(
                "probe-1",
                from_clock_domain="server-a/other-boot/unix_epoch_ns",
                to_clock_domain=SERVER_CLOCK,
            )
        ],
    )
    telemetry = _telemetry(
        tmp_path, "server.jsonl", _server_sample(boot_id="other-boot")
    )
    report = analyze_inference_events(
        profile, server_telemetry_paths=[telemetry], direct_server=True
    )
    join = report["telemetry"]["server_join"]
    assert (join["status"], join["clock_alignment_evidence"]) == (
        "joined",
        "artifact_record",
    )


def test_same_hostname_without_boot_ids_is_never_one_clock(tmp_path: Path) -> None:
    sample = _server_sample()
    bootless = replace(sample, identity=replace(sample.identity, boot_id=None))
    profile = _profile(tmp_path, host="server-a", boot=False)
    telemetry = _telemetry(tmp_path, "server.jsonl", bootless)
    report = analyze_inference_events(
        profile, server_telemetry_paths=[telemetry], direct_server=True
    )
    assert report["telemetry"]["server_join"]["reason"] == "clock_domain_unverified"


def test_malformed_alignment_record_is_rejected(tmp_path: Path) -> None:
    record = _alignment_record("probe-1")
    record["uncertainty_ns"] = -1
    with pytest.raises(ValueError, match="uncertainty_ns"):
        _aligned_report(tmp_path, [record], _server_sample())


def test_each_sample_keeps_its_own_alignment_uncertainty(tmp_path: Path) -> None:
    # A precise early alignment must not inherit a later, looser one's margin.
    report = _aligned_report(
        tmp_path,
        [
            _alignment_record("early", uncertainty_ns=1, valid_to_ns=1_000),
            _alignment_record("late", uncertainty_ns=1_000, valid_from_ns=1_000),
        ],
        _server_sample(observed_at_ns=150),
        _server_sample(observed_at_ns=5_000),
        _server_sample(observed_at_ns=6_500),
        windows={"case-a": (100, 300), "case-b": (4_000, 7_000)},
    )
    assert report["telemetry"]["server_join"]["clock_uncertainty_ns"] == 1_000
    cases = report["cases"]
    assert cases["case-a"]["memory"]["server_coverage"] == {
        "status": "observed",
        "reason": None,
        "counted_window_ns": [101, 299],
    }
    counts = {
        case_id: case["memory"]["server_observations"]["device_memory_used_bytes"][
            "valid_samples"
        ]
        for case_id, case in cases.items()
    }
    # 6,500 lies within 1,000 of case-b's end, so its own uncertainty excludes it.
    assert counts == {"case-a": 1, "case-b": 1}
