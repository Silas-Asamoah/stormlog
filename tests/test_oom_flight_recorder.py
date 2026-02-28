"""Tests for OOM flight recorder classification and dump behavior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpumemprof.device_collectors import DeviceMemorySample
from gpumemprof.oom_flight_recorder import (
    OOMFlightRecorder,
    OOMFlightRecorderConfig,
    classify_oom_exception,
)
from gpumemprof.tracker import MemoryTracker


class _ResourceExhaustedError(RuntimeError):
    pass


class _TrackerHarness:
    """Minimal harness that exercises MemoryTracker OOM methods without a GPU."""

    def __init__(
        self, dump_dir: Path, *, enabled: bool = True, buffer_size: int = 8
    ) -> None:
        self.backend = "cuda"
        self.collector_capabilities = {"telemetry_collector": "gpumemprof.cuda_tracker"}
        self.total_memory = 1024 * 1024 * 1024
        self.sampling_interval = 0.1
        self.last_oom_dump_path = None
        self.distributed_identity = {
            "job_id": "test-job",
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
        }
        self._oom_flight_recorder = OOMFlightRecorder(
            OOMFlightRecorderConfig(
                enabled=enabled,
                dump_dir=str(dump_dir),
                buffer_size=buffer_size,
                max_dumps=5,
                max_total_mb=32,
            )
        )

    def get_statistics(self) -> dict[str, int]:
        return {"total_events": len(self._oom_flight_recorder.snapshot_events())}

    def _safe_sample(self) -> DeviceMemorySample:
        return DeviceMemorySample(
            allocated_bytes=256,
            reserved_bytes=512,
            used_bytes=512,
            free_bytes=512,
            total_bytes=1024,
            active_bytes=256,
            inactive_bytes=256,
            device_id=0,
        )

    def _add_event(
        self,
        event_type: str,
        memory_change: int,
        context: str,
        metadata: dict[str, object] | None = None,
        sample: object = None,
    ) -> None:
        _ = sample
        payload = {
            "event_type": event_type,
            "memory_change": memory_change,
            "context": context,
            "metadata": dict(metadata or {}),
            "backend": self.backend,
        }
        self._oom_flight_recorder.record_event(payload)

    def handle_exception(
        self,
        exc: BaseException,
        context: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str | None:
        return MemoryTracker.handle_exception(
            self, exc, context=context, metadata=metadata  # type: ignore[arg-type, unused-ignore]
        )

    def capture_oom(
        self, context: str = "runtime", metadata: dict[str, object] | None = None
    ) -> Any:
        return MemoryTracker.capture_oom(self, context=context, metadata=metadata)  # type: ignore[arg-type, unused-ignore]


def test_classify_oom_runtime_error_pattern() -> None:
    classified = classify_oom_exception(
        RuntimeError("CUDA out of memory. Tried to allocate")
    )
    assert classified.is_oom is True
    assert classified.reason == "message_pattern:out of memory"


def test_classify_tensorflow_style_resource_exhausted_error() -> None:
    classified = classify_oom_exception(
        _ResourceExhaustedError("OOM when allocating tensor")
    )
    assert classified.is_oom is True
    assert classified.reason == "tensorflow.ResourceExhaustedError"


def test_classify_non_oom_exception() -> None:
    classified = classify_oom_exception(ValueError("bad input"))
    assert classified.is_oom is False
    assert classified.reason is None


def test_ring_buffer_keeps_only_latest_events(tmp_path: Path) -> None:
    recorder = OOMFlightRecorder(
        OOMFlightRecorderConfig(
            enabled=True,
            dump_dir=str(tmp_path / "oom_dumps"),
            buffer_size=3,
            max_dumps=5,
            max_total_mb=32,
        )
    )
    for idx in range(5):
        recorder.record_event({"i": idx})

    events = recorder.snapshot_events()
    assert [event["i"] for event in events] == [2, 3, 4]


def test_dump_bundle_contains_expected_artifacts(tmp_path: Path) -> None:
    recorder = OOMFlightRecorder(
        OOMFlightRecorderConfig(
            enabled=True,
            dump_dir=str(tmp_path / "oom_dumps"),
            buffer_size=4,
            max_dumps=5,
            max_total_mb=32,
        )
    )
    recorder.record_event({"event": "allocation", "size": 123})

    dump_path = recorder.dump(
        reason="message_pattern:out of memory",
        exception=RuntimeError("CUDA out of memory"),
        context="unit-test",
        backend="cuda",
        metadata={"test_case": "artifact_integrity"},
    )

    assert dump_path is not None
    bundle = Path(dump_path)
    assert (bundle / "manifest.json").exists()
    assert (bundle / "events.json").exists()
    assert (bundle / "metadata.json").exists()
    assert (bundle / "environment.json").exists()

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    events = json.loads((bundle / "events.json").read_text(encoding="utf-8"))

    assert manifest["reason"] == "message_pattern:out of memory"
    assert metadata["context"] == "unit-test"
    assert metadata["custom_metadata"]["test_case"] == "artifact_integrity"
    assert len(events) == 1


def test_retention_enforces_max_dump_count(tmp_path: Path) -> None:
    recorder = OOMFlightRecorder(
        OOMFlightRecorderConfig(
            enabled=True,
            dump_dir=str(tmp_path / "oom_dumps"),
            buffer_size=4,
            max_dumps=2,
            max_total_mb=64,
        )
    )

    for idx in range(4):
        recorder.record_event({"event": "allocation", "idx": idx})
        recorder.dump(
            reason="message_pattern:out of memory",
            exception=RuntimeError(f"oom-{idx}"),
            context="retention-count",
            backend="cuda",
            metadata={"idx": idx},
        )

    bundles = sorted((tmp_path / "oom_dumps").glob("oom_dump_*"))
    assert len(bundles) == 2


def test_retention_enforces_total_size_limit(tmp_path: Path) -> None:
    recorder = OOMFlightRecorder(
        OOMFlightRecorderConfig(
            enabled=True,
            dump_dir=str(tmp_path / "oom_dumps"),
            buffer_size=4,
            max_dumps=10,
            max_total_mb=1,
        )
    )

    blob = "x" * 900_000
    for idx in range(3):
        recorder.record_event({"event": "allocation", "idx": idx, "blob": blob})
        recorder.dump(
            reason="message_pattern:out of memory",
            exception=RuntimeError(f"oom-{idx}"),
            context="retention-size",
            backend="cuda",
            metadata={"blob": blob},
        )

    bundles = list((tmp_path / "oom_dumps").glob("oom_dump_*"))
    total_size = 0
    for bundle in bundles:
        for file_path in bundle.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size

    assert total_size <= 1 * 1024 * 1024
    assert len(bundles) < 3


def test_handle_exception_writes_dump_for_simulated_oom(tmp_path: Path) -> None:
    harness = _TrackerHarness(tmp_path / "oom_dumps")

    dump_path = harness.handle_exception(
        RuntimeError("CUDA out of memory during allocation"),
        context="simulated-oom",
        metadata={"source": "test"},
    )

    assert dump_path is not None
    assert Path(dump_path).exists()
    assert harness.last_oom_dump_path == dump_path


def test_handle_exception_ignores_non_oom(tmp_path: Path) -> None:
    harness = _TrackerHarness(tmp_path / "oom_dumps")

    dump_path = harness.handle_exception(
        ValueError("not oom"),
        context="non-oom",
    )

    assert dump_path is None
    assert harness.last_oom_dump_path is None


def test_capture_oom_context_triggers_dump_then_reraises(tmp_path: Path) -> None:
    harness = _TrackerHarness(tmp_path / "oom_dumps")

    with pytest.raises(RuntimeError, match="out of memory"):
        with harness.capture_oom(context="capture-oom", metadata={"source": "ctx"}):
            raise RuntimeError("CUDA out of memory in context")

    assert harness.last_oom_dump_path is not None  # type: ignore[unreachable, unused-ignore]
    assert Path(harness.last_oom_dump_path).exists()  # type: ignore[unreachable, unused-ignore]
