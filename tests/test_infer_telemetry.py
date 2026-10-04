"""Scoped inference telemetry contract and identity safeguards."""

from __future__ import annotations

import contextlib
import ctypes
import io
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest import mock

import psutil
import pytest
from jsonschema import Draft202012Validator

from stormlog.exit_codes import ExitCode
from stormlog.infer.cli import main as infer_main
from stormlog.infer.host_clock import is_boot_qualified, wall_clock_domain
from stormlog.infer.server_collector import (
    STOP_DURATION_ELAPSED,
    STOP_GPU_IDENTITY_CHANGED,
    STOP_REQUESTED,
    STOP_SERVER_PROCESS_ENDED,
    CollectionResult,
    GpuMemoryReading,
    NvmlMemorySource,
    collect_server_telemetry,
    describe_gpu_process_match,
    next_poll_time,
    read_process_rss,
    running_compute_pids,
)
from stormlog.infer.telemetry import (
    ServerIdentity,
    TelemetrySample,
    load_telemetry,
    validate_group_membership,
)

SCHEMA = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "docs/schemas/inference_telemetry_v1.schema.json"
    ).read_text()
)


def _identity(**changes: Any) -> ServerIdentity:
    values: dict[str, Any] = {
        "host": "server-a",
        "pid": 321,
        "process_start_ns": 1_000_000_000,
        "device_uuid": "GPU-a",
        "replica_id": "replica-a",
        **changes,
    }
    return ServerIdentity(**values)


def _sample(**changes: Any) -> TelemetrySample:
    values: dict[str, Any] = {
        "run_id": "run-a",
        "identity": _identity(),
        "observed_at_ns": 2_000_000_000,
        "metric": "device_memory_used_bytes",
        "value_bytes": 100,
        "state": "valid",
        "source": "nvml-v2",
        "interval_ms": 100,
        **changes,
    }
    return TelemetrySample(**values)


class _FakeGpu:
    """A GPU source that reports fixed values and can act on each read."""

    device_uuid = "GPU-live"
    gpu_instance_id: str | None = None

    def __init__(
        self,
        reading: GpuMemoryReading | None = None,
        on_read: Callable[[int], None] | None = None,
    ) -> None:
        self.reading = reading or GpuMemoryReading(256, 64, "valid")
        self.on_read = on_read
        self.reads = 0

    def read(self) -> GpuMemoryReading:
        self.reads += 1
        if self.on_read is not None:
            self.on_read(self.reads)
        return self.reading

    def close(self) -> None:
        pass


def _assert_schema_valid(samples: list[TelemetrySample]) -> None:
    validator = Draft202012Validator(SCHEMA)
    for sample in samples:
        validator.validate(sample.to_record())


def test_telemetry_has_explicit_scope_owner_and_provenance(tmp_path: Path) -> None:
    device = _sample()
    process = _sample(metric="process_rss_bytes", source="psutil")
    instance = _sample(
        identity=_identity(device_uuid="MIG-a", gpu_instance_id="MIG-a"),
        metric="instance_memory_used_bytes",
    )
    path = tmp_path / "telemetry.jsonl"
    path.write_text(
        "".join(
            json.dumps(sample.to_record()) + "\n"
            for sample in (device, process, instance)
        )
    )
    assert [sample.scope for sample in load_telemetry(path)] == [
        "gpu_device",
        "server_process",
        "gpu_instance",
    ]
    assert device.to_record()["counter_owner"] == "gpu_device"
    assert device.to_record()["provenance"] == "observed"
    Draft202012Validator.check_schema(SCHEMA)
    _assert_schema_valid([device, process, instance])


def test_unavailable_counter_is_null_and_scope_cannot_be_forged() -> None:
    missing = _sample(value_bytes=None, state="missing")
    assert missing.to_record()["value_bytes"] is None
    with pytest.raises(ValueError, match="null value_bytes"):
        _sample(value_bytes=0, state="missing")
    forged = _sample().to_record()
    forged["scope"] = "server_process"
    with pytest.raises(ValueError, match="scope"):
        TelemetrySample.from_record(forged)
    with pytest.raises(ValueError, match="non-negative value_bytes"):
        _sample(value_bytes=True)
    with pytest.raises(ValueError, match="positive pid"):
        _identity(pid=True)
    with pytest.raises(ValueError, match="non-empty strings"):
        _identity(boot_id="")
    forged = _sample().to_record()
    forged["schema_version"] = True
    with pytest.raises(ValueError, match="unsupported telemetry"):
        TelemetrySample.from_record(forged)


def _set_identity(**changes: object) -> Callable[[dict[str, Any]], None]:
    return lambda record: record["identity"].update(changes)


def _set(**changes: object) -> Callable[[dict[str, Any]], None]:
    return lambda record: record.update(changes)


def _drop(key: str) -> Callable[[dict[str, Any]], None]:
    return lambda record: record.pop(key)


def _set_with_domain(
    clock_domain: str, **identity: object
) -> Callable[[dict[str, Any]], None]:
    def mutate(record: dict[str, Any]) -> None:
        record["identity"].update(identity)
        record["clock_domain"] = clock_domain

    return mutate


def _group_without(key: str) -> Callable[[dict[str, Any]], None]:
    def mutate(record: dict[str, Any]) -> None:
        record["identity"].update(group_id="tp", rank=0, world_size=2)
        del record["identity"][key]

    return mutate


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_set_identity(boot_id=""), id="empty-boot-id"),
        pytest.param(_set_identity(replica_id=""), id="empty-replica-id"),
        pytest.param(_set_identity(device_uuid=None), id="gpu-metric-without-device"),
        pytest.param(
            _set_identity(device_uuid=None, gpu_instance_id="MIG-a"),
            id="instance-without-device",
        ),
        pytest.param(
            _set_identity(group_id="tp", rank=0, world_size=None),
            id="group-with-null-world-size",
        ),
        pytest.param(_group_without("world_size"), id="group-without-world-size"),
        pytest.param(_group_without("rank"), id="group-without-rank"),
        pytest.param(_set_identity(world_size=2), id="world-size-without-group"),
        pytest.param(
            _set(
                metric="instance_memory_used_bytes",
                scope="gpu_instance",
                counter_owner="gpu_instance",
            ),
            id="instance-metric-without-instance",
        ),
        pytest.param(_set(scope="server_process"), id="scope-not-matching-metric"),
        pytest.param(_set(counter_owner="allocator"), id="owner-not-matching-metric"),
        pytest.param(_set(metric="process_rss_bytes"), id="metric-not-matching-scope"),
        pytest.param(
            _set(clock_domain="server-a/monotonic_ns"), id="domain-not-wall-clock"
        ),
        # Each identity part is one segment of the clock domain.
        pytest.param(
            _set_with_domain("rack/a/unix_epoch_ns", host="rack/a"),
            id="slash-in-host",
        ),
        pytest.param(
            _set_with_domain("server-a/boot/1/unix_epoch_ns", boot_id="boot/1"),
            id="slash-in-boot-id",
        ),
        # The domain names the boot exactly when the identity has a boot ID.
        pytest.param(
            _set_with_domain("server-a/unix_epoch_ns", boot_id="boot-1"),
            id="boot-id-missing-from-domain",
        ),
        pytest.param(
            _set(clock_domain="server-a/boot-1/unix_epoch_ns"),
            id="boot-in-domain-without-boot-id",
        ),
        pytest.param(_drop("detail"), id="missing-detail"),
        pytest.param(_drop("provenance"), id="missing-provenance"),
    ],
)
def test_schema_rejects_what_the_loader_rejects(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    record = _sample().to_record()
    mutate(record)
    with pytest.raises((KeyError, TypeError, ValueError)):
        TelemetrySample.from_record(record)
    assert list(Draft202012Validator(SCHEMA).iter_errors(record))


@pytest.mark.parametrize("field", sorted(TelemetrySample.__dataclass_fields__))
def test_schema_lists_every_record_field(field: str) -> None:
    # Every dataclass field is written to the record, so the schema must list it.
    assert field in SCHEMA["properties"]


def test_clock_domain_names_the_host_boot() -> None:
    booted = _sample(identity=_identity(boot_id="boot-1"))
    record = booted.to_record()
    assert record["clock_domain"] == "server-a/boot-1/unix_epoch_ns"
    assert TelemetrySample.from_record(record) == booted
    assert _sample().to_record()["clock_domain"] == "server-a/unix_epoch_ns"
    # A hostname-only domain no longer matches an identity that has a boot ID.
    record["clock_domain"] = "server-a/unix_epoch_ns"
    with pytest.raises(ValueError, match="clock domain"):
        TelemetrySample.from_record(record)
    _assert_schema_valid([booted])


@pytest.mark.parametrize("changes", [{"host": "rack/a"}, {"boot_id": "boot/1"}])
def test_identity_parts_cannot_split_the_clock_domain(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="cannot contain '/'"):
        _identity(**changes)


def test_only_boot_qualified_domains_can_prove_one_clock() -> None:
    assert wall_clock_domain("host", "boot") == "host/boot/unix_epoch_ns"
    assert wall_clock_domain("host", None) == "host/unix_epoch_ns"
    assert is_boot_qualified("host/boot/unix_epoch_ns")
    assert not is_boot_qualified("host/unix_epoch_ns")
    assert not is_boot_qualified("worker-a/monotonic")


def test_instance_metric_requires_instance_identity() -> None:
    with pytest.raises(ValueError, match="gpu_instance_id"):
        _sample(metric="instance_memory_used_bytes")


def test_allocator_and_engine_cache_are_separate_reported_counters() -> None:
    allocator = _sample(
        metric="allocator_allocated_bytes",
        source="pytorch",
        provenance="reported",
    )
    cache = _sample(
        metric="engine_cache_occupied_bytes",
        source="vllm",
        provenance="reported",
    )
    assert allocator.scope == cache.scope == "server_process"
    assert allocator.to_record()["counter_owner"] == "allocator"
    assert cache.to_record()["counter_owner"] == "engine_cache"
    assert TelemetrySample.from_record(cache.to_record()) == cache
    _assert_schema_valid([allocator, cache])


def test_on_host_collector_writes_process_and_gpu_with_same_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "server.jsonl"
    result = collect_server_telemetry(
        run_id="run-live",
        pid=os.getpid(),
        output_path=path,
        interval_seconds=0.01,
        duration_seconds=0.025,
        gpu_source=_FakeGpu(),
    )
    assert result.stop_reason == STOP_DURATION_ELAPSED
    samples = load_telemetry(path)
    assert len(samples) == result.polls * 3
    assert {sample.metric for sample in samples} == {
        "process_rss_bytes",
        "device_memory_used_bytes",
        "device_memory_reserved_bytes",
    }
    assert {sample.identity.process_start_ns for sample in samples} == {
        samples[0].identity.process_start_ns
    }
    assert {sample.identity.device_uuid for sample in samples} == {"GPU-live"}
    _assert_schema_valid(samples)


def test_on_host_collector_emits_missing_nvml_counter(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"
    collect_server_telemetry(
        run_id="run-live",
        pid=os.getpid(),
        output_path=path,
        interval_seconds=0.01,
        duration_seconds=0.015,
        gpu_source=_FakeGpu(GpuMemoryReading(None, None, "missing", "unavailable")),
    )
    gpu_samples = [s for s in load_telemetry(path) if s.scope == "gpu_device"]
    assert gpu_samples
    assert {s.state for s in gpu_samples} == {"missing"}
    assert all(s.value_bytes is None for s in gpu_samples)


def test_on_host_collector_stops_after_gpu_identity_failure(tmp_path: Path) -> None:
    path = tmp_path / "changed.jsonl"
    result = collect_server_telemetry(
        run_id="run-live",
        pid=os.getpid(),
        output_path=path,
        interval_seconds=0.01,
        duration_seconds=0.1,
        gpu_source=_FakeGpu(
            GpuMemoryReading(None, None, "invalid", "device UUID changed")
        ),
    )
    assert result == CollectionResult(
        polls=1, stop_reason=STOP_GPU_IDENTITY_CHANGED, detail="device UUID changed"
    )
    samples = load_telemetry(path)
    assert [s.state for s in samples] == ["valid", "invalid", "invalid"]
    _assert_schema_valid(samples)


def test_access_denied_is_missing_and_collection_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def deny(process: psutil.Process) -> None:
        raise psutil.AccessDenied(process.pid)

    monkeypatch.setattr(psutil.Process, "memory_info", deny)
    path = tmp_path / "denied.jsonl"
    result = collect_server_telemetry(
        run_id="run-live",
        pid=os.getpid(),
        output_path=path,
        interval_seconds=0.01,
        duration_seconds=0.06,
        gpu_source=_FakeGpu(),
    )
    assert result.stop_reason == STOP_DURATION_ELAPSED
    assert result.polls >= 2
    samples = load_telemetry(path)
    rss = [s for s in samples if s.metric == "process_rss_bytes"]
    assert {s.state for s in rss} == {"missing"}
    assert {s.detail for s in rss} == {"access denied reading server process memory"}
    assert {s.state for s in samples if s.scope == "gpu_device"} == {"valid"}
    _assert_schema_valid(samples)


def test_collector_stops_cleanly_when_the_server_process_ends(tmp_path: Path) -> None:
    server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])

    def stop_server() -> None:
        server.terminate()
        server.wait()

    timer = threading.Timer(0.3, stop_server)
    path = tmp_path / "ended.jsonl"
    try:
        timer.start()
        result = collect_server_telemetry(
            run_id="run-live",
            pid=server.pid,
            output_path=path,
            interval_seconds=0.02,
            duration_seconds=20,
            no_gpu=True,
        )
    finally:
        timer.cancel()
        server.kill()
        server.wait()
    assert result.stop_reason == STOP_SERVER_PROCESS_ENDED
    samples = load_telemetry(path)
    assert len(samples) == result.polls
    assert samples[-1].state == "invalid"
    assert samples[-1].detail == "server process ended or its PID was reused"
    assert {s.state for s in samples[:-1]} == {"valid"}


def test_stop_event_ends_collection_as_a_requested_stop(tmp_path: Path) -> None:
    stop_event = threading.Event()

    def stop_after_second_read(reads: int) -> None:
        if reads == 2:
            stop_event.set()

    path = tmp_path / "stopped.jsonl"
    result = collect_server_telemetry(
        run_id="run-live",
        pid=os.getpid(),
        output_path=path,
        interval_seconds=0.01,
        gpu_source=_FakeGpu(on_read=stop_after_second_read),
        stop_event=stop_event,
    )
    assert (result.polls, result.stop_reason) == (2, STOP_REQUESTED)
    assert len(load_telemetry(path)) == 6


def test_interrupt_keeps_every_completed_poll_readable(tmp_path: Path) -> None:
    def interrupt_on_third_read(reads: int) -> None:
        if reads == 3:
            raise KeyboardInterrupt

    path = tmp_path / "interrupted.jsonl"
    result = collect_server_telemetry(
        run_id="run-live",
        pid=os.getpid(),
        output_path=path,
        interval_seconds=0.01,
        gpu_source=_FakeGpu(on_read=interrupt_on_third_read),
    )
    assert result == CollectionResult(2, STOP_REQUESTED, "interrupted")
    assert len(load_telemetry(path)) == 6


@pytest.mark.parametrize(
    ("interval", "duration"),
    [
        (math.nan, None),
        (math.inf, None),
        (0.005, None),
        (0.1, math.nan),
        (0.1, math.inf),
        (0.1, 0.0),
    ],
)
def test_non_finite_or_invalid_timing_is_rejected_before_writing(
    tmp_path: Path, interval: float, duration: float | None
) -> None:
    path = tmp_path / "never.jsonl"
    with pytest.raises(ValueError):
        collect_server_telemetry(
            run_id="run-live",
            pid=os.getpid(),
            output_path=path,
            interval_seconds=interval,
            duration_seconds=duration,
            gpu_source=_FakeGpu(),
        )
    assert not path.exists()


def test_poll_schedule_skips_missed_polls_instead_of_bursting() -> None:
    assert next_poll_time(10.0, 0.1, now=10.05) == pytest.approx(10.1)
    # A 3 s stall schedules one poll an interval later, not 30 catch-up polls.
    assert next_poll_time(10.0, 0.1, now=13.0) == pytest.approx(13.1)


class _Process:
    pid = 7

    def __init__(self, *, running: bool = True, error: Exception | None = None):
        self.running = running
        self.error = error

    def is_running(self) -> bool:
        return self.running

    def memory_info(self) -> SimpleNamespace:
        if self.error is not None:
            raise self.error
        return SimpleNamespace(rss=4096)


@pytest.mark.parametrize(
    ("process", "expected_state"),
    [
        (_Process(), "valid"),
        (_Process(running=False), "invalid"),
        (_Process(error=psutil.NoSuchProcess(7)), "invalid"),
        (_Process(error=psutil.ZombieProcess(7)), "invalid"),
        (_Process(error=psutil.AccessDenied(7)), "missing"),
        (_Process(error=psutil.Error()), "missing"),
    ],
)
def test_only_a_gone_process_invalidates_rss(
    process: _Process, expected_state: str
) -> None:
    state, rss, _detail = read_process_rss(process)
    assert state == expected_state
    assert (rss is not None) == (expected_state == "valid")


def test_gpu_process_match_explains_mismatches() -> None:
    assert describe_gpu_process_match(100, "GPU-x", {100, 5}, set()) == []
    (child,) = describe_gpu_process_match(100, "GPU-x", {523}, {523, 600})
    assert "child process(es) 523" in child
    assert "--pid 523" in child
    (other,) = describe_gpu_process_match(100, "GPU-x", {7}, set())
    assert "--device-uuid" in other
    assert "PIDs: 7" in other
    (empty,) = describe_gpu_process_match(100, "GPU-x", set(), set())
    assert "PIDs: none" in empty
    (unknown,) = describe_gpu_process_match(100, "GPU-x", None, set())
    assert "could not list compute processes" in unknown


def test_collector_reports_gpu_process_warnings(tmp_path: Path) -> None:
    class GpuWithoutServer(_FakeGpu):
        def compute_pids(self) -> set[int]:
            return {999_999}

    warnings: list[str] = []
    result = collect_server_telemetry(
        run_id="run-live",
        pid=os.getpid(),
        output_path=tmp_path / "warned.jsonl",
        interval_seconds=0.01,
        duration_seconds=0.015,
        gpu_source=GpuWithoutServer(),
        on_warning=warnings.append,
    )
    assert len(warnings) == 1
    assert result.warnings == tuple(warnings)
    assert f"PID {os.getpid()} has no compute context on GPU-live" in warnings[0]


class _FakeNvmlLibrary:
    def __init__(self, uuids: dict[str, str], *, uuid_code: int = 0) -> None:
        self.uuids = uuids
        self.uuid_code = uuid_code
        self.memory_code = 0

    def nvmlDeviceGetUUID(  # noqa: N802
        self, handle: str, buffer: Any, _length: int
    ) -> int:
        if self.uuid_code:
            return self.uuid_code
        buffer.value = self.uuids[handle].encode()
        return 0

    def nvmlDeviceGetMemoryInfo_v2(  # noqa: N802
        self, _handle: str, info_ref: Any
    ) -> int:
        if self.memory_code:
            return self.memory_code
        info_ref._obj.used, info_ref._obj.reserved = 300, 40
        return 0


def _nvml_source(
    library: _FakeNvmlLibrary, *, parent_uuid: str | None = None
) -> NvmlMemorySource:
    source = object.__new__(NvmlMemorySource)
    # Fill in what __init__ would take from a real NVML library.
    state: Any = source
    state._lib = library
    state._closed = True
    state._handle = "gpu"
    state._handle_uuid = "MIG-a" if parent_uuid else "GPU-a"
    state._parent_handle = "parent" if parent_uuid else None
    state.device_uuid = parent_uuid or "GPU-a"
    state.gpu_instance_id = "MIG-a" if parent_uuid else None
    return source


def test_nvml_read_reports_values_for_the_same_gpu() -> None:
    reading = _nvml_source(_FakeNvmlLibrary({"gpu": "GPU-a"})).read()
    assert reading == GpuMemoryReading(300, 40, "valid")


def test_unreadable_gpu_uuid_is_missing_not_invalid() -> None:
    library = _FakeNvmlLibrary({"gpu": "GPU-a"}, uuid_code=999)
    reading = _nvml_source(library).read()
    assert reading.state == "missing"
    assert "device identity unreadable" in (reading.detail or "")


def test_a_different_gpu_uuid_invalidates() -> None:
    reading = _nvml_source(_FakeNvmlLibrary({"gpu": "GPU-b"})).read()
    assert (reading.state, reading.detail) == ("invalid", "device UUID changed")
    library = _FakeNvmlLibrary({"gpu": "MIG-a", "parent": "GPU-other"})
    reading = _nvml_source(library, parent_uuid="GPU-a").read()
    assert (reading.state, reading.detail) == ("invalid", "parent device UUID changed")


def test_failed_nvml_memory_read_is_missing() -> None:
    library = _FakeNvmlLibrary({"gpu": "GPU-a"})
    library.memory_code = 3
    reading = _nvml_source(library).read()
    assert reading.state == "missing"
    assert reading.used_bytes is None


def test_compute_process_query_grows_the_buffer() -> None:
    calls: list[bool] = []

    def query(_handle: object, count_ref: Any, infos: Any) -> int:
        calls.append(infos is not None)
        if infos is None:
            count_ref._obj.value = 2
            return 7  # NVML_ERROR_INSUFFICIENT_SIZE
        infos[0].pid, infos[1].pid = 11, 12
        count_ref._obj.value = 2
        return 0

    assert running_compute_pids(query, ctypes.c_void_p()) == {11, 12}
    assert calls == [False, True]
    assert running_compute_pids(lambda *_: 0, ctypes.c_void_p()) == set()
    assert running_compute_pids(lambda *_: 999, ctypes.c_void_p()) is None


def _run_collect_cli(
    result_factory: Callable[..., CollectionResult], *extra: str
) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with mock.patch(
        "stormlog.infer.cli.collect_server_telemetry", side_effect=result_factory
    ):
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = infer_main(
                [
                    "collect-server",
                    "--run-id",
                    "r",
                    "--pid",
                    "1",
                    "--output",
                    "o",
                    *extra,
                ]
            )
    return code, stdout.getvalue(), stderr.getvalue()


def test_collect_cli_reports_why_collection_stopped() -> None:
    code, stdout, stderr = _run_collect_cli(
        lambda **_: CollectionResult(4, STOP_DURATION_ELAPSED)
    )
    assert code == 0
    assert "Collected 4 server polls to: o (stopped: duration_elapsed)" in stdout
    assert stderr == ""


def test_collect_cli_reports_a_gpu_identity_change_as_a_finding() -> None:
    code, stdout, stderr = _run_collect_cli(
        lambda **_: CollectionResult(3, STOP_GPU_IDENTITY_CHANGED, "device changed")
    )
    assert code == ExitCode.FINDINGS
    assert "(stopped: gpu_identity_changed)" in stdout
    assert "Findings: GPU identity changed (device changed)" in stderr


def test_collect_cli_warns_when_the_server_process_ended() -> None:
    code, _stdout, stderr = _run_collect_cli(
        lambda **_: CollectionResult(9, STOP_SERVER_PROCESS_ENDED, "server ended")
    )
    assert code == 0
    assert "Warning: server ended; case windows that extend past" in stderr


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")
def test_collect_cli_turns_ctrl_c_into_a_clean_stop() -> None:
    before = signal.getsignal(signal.SIGINT)

    def collect(*, stop_event: threading.Event, **_options: object) -> CollectionResult:
        os.kill(os.getpid(), signal.SIGINT)
        deadline = time.monotonic() + 5
        while not stop_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert stop_event.is_set()
        return CollectionResult(2, STOP_REQUESTED)

    code, stdout, _stderr = _run_collect_cli(collect)
    assert code == 0
    assert "(stopped: stop_requested)" in stdout
    assert signal.getsignal(signal.SIGINT) is before


@pytest.mark.parametrize(
    ("group_id", "rank", "world_size", "message"),
    [
        (None, None, 2, "world_size requires group_id"),
        ("", 0, 2, "group_id must be a non-empty string"),
        ("tp", 0, 0, "positive world_size"),
        ("tp", None, 2, "rank below world_size"),
        ("tp", 2, 2, "rank below world_size"),
    ],
)
def test_group_members_need_a_rank_within_the_world_size(
    tmp_path: Path,
    group_id: str | None,
    rank: int | None,
    world_size: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_group_membership(group_id, rank, world_size)
    path = tmp_path / "never.jsonl"
    with pytest.raises(ValueError, match=message):
        collect_server_telemetry(
            run_id="run-live",
            pid=os.getpid(),
            output_path=path,
            rank=rank,
            group_id=group_id,
            world_size=world_size,
            gpu_source=_FakeGpu(),
        )
    assert not path.exists()


def test_collector_records_group_membership(tmp_path: Path) -> None:
    path = tmp_path / "member.jsonl"
    collect_server_telemetry(
        run_id="run-live",
        pid=os.getpid(),
        output_path=path,
        interval_seconds=0.01,
        duration_seconds=0.015,
        rank=1,
        group_id="tp",
        world_size=2,
        gpu_source=_FakeGpu(),
    )
    samples = load_telemetry(path)
    assert {
        (s.identity.group_id, s.identity.rank, s.identity.world_size) for s in samples
    } == {("tp", 1, 2)}
    _assert_schema_valid(samples)


def test_collect_cli_passes_group_flags() -> None:
    seen: dict[str, Any] = {}

    def collect(**options: Any) -> CollectionResult:
        seen.update(options)
        return CollectionResult(1, STOP_DURATION_ELAPSED)

    code, _stdout, _stderr = _run_collect_cli(
        collect, "--group-id", "tp", "--rank", "1", "--world-size", "2"
    )
    assert code == 0
    assert (seen["group_id"], seen["rank"], seen["world_size"]) == ("tp", 1, 2)


def test_records_written_before_groups_stay_valid() -> None:
    # Group fields are optional in v1: older artifacts omit them entirely.
    sample = _sample()
    record = sample.to_record()
    del record["identity"]["group_id"], record["identity"]["world_size"]
    Draft202012Validator(SCHEMA).validate(record)
    assert TelemetrySample.from_record(record) == sample
