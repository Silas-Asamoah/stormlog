from types import SimpleNamespace

import pytest

import stormlog.device_collectors as collectors


def test_detect_torch_runtime_backend_reports_rocm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(collectors.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(collectors.torch, "version", SimpleNamespace(hip="6.3.0"))

    backend = collectors.detect_torch_runtime_backend()

    assert backend == "rocm"


def test_detect_torch_runtime_backend_reports_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(collectors.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(collectors.torch, "version", SimpleNamespace(hip=None))

    backend = collectors.detect_torch_runtime_backend()

    assert backend == "cuda"


def test_detect_torch_runtime_backend_reports_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(collectors.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(collectors, "_is_mps_available", lambda: True)

    backend = collectors.detect_torch_runtime_backend()

    assert backend == "mps"


def test_detect_torch_runtime_backend_reports_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(collectors.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(collectors, "_is_mps_available", lambda: False)

    backend = collectors.detect_torch_runtime_backend()

    assert backend == "cpu"


def test_build_device_memory_collector_rejects_cpu_device() -> None:
    with pytest.raises(ValueError, match="Only CUDA/ROCm and MPS"):
        collectors.build_device_memory_collector("cpu")


def test_resolve_device_rejects_int_when_cuda_backends_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(collectors, "detect_torch_runtime_backend", lambda: "mps")

    with pytest.raises(
        ValueError, match="Integer device IDs are only supported for CUDA/ROCm"
    ):
        collectors._resolve_device(0)


def test_build_device_memory_collector_allows_mps_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collectors,
        "_resolve_device",
        lambda _device: SimpleNamespace(type="mps"),
    )

    collector = collectors.build_device_memory_collector("mps")

    assert collector.name() == "mps"


def test_cuda_collector_reports_full_sample_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collectors,
        "_resolve_device",
        lambda _device: SimpleNamespace(type="cuda", index=2),
    )
    monkeypatch.setattr(collectors.torch.cuda, "memory_allocated", lambda _device: 1024)
    monkeypatch.setattr(collectors.torch.cuda, "memory_reserved", lambda _device: 2048)
    monkeypatch.setattr(
        collectors.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=4096),
    )
    monkeypatch.setattr(
        collectors.torch.cuda,
        "memory_stats",
        lambda _device: {
            "active_bytes.all.current": 512,
            "inactive_split_bytes.all.current": 256,
        },
    )

    collector = collectors.CudaDeviceCollector("cuda:2")
    result = collector.sample_with_diagnostics()

    assert result.core_error is None
    assert result.partial_fields == ()
    assert result.errors == {}
    assert result.sample == collectors.DeviceMemorySample(
        allocated_bytes=1024,
        reserved_bytes=2048,
        used_bytes=2048,
        free_bytes=2048,
        total_bytes=4096,
        active_bytes=512,
        inactive_bytes=256,
        device_id=2,
    )
    capabilities = collector.capabilities()
    assert isinstance(capabilities, collectors.DeviceMemoryCapabilities)
    assert capabilities.supports_allocator_allocated is True
    assert capabilities.supports_device_used is True
    assert capabilities.supports_native_allocator_history is True


def test_validate_device_only_sample_accepts_absent_allocator_metrics() -> None:
    capabilities = collectors.DeviceMemoryCapabilities(
        backend="future",
        telemetry_collector="stormlog.future_tracker",
        sampling_source="future.device_memory",
        supports_device_used=True,
        supports_device_free=True,
        supports_device_total=True,
    )
    sample = collectors.DeviceMemorySample(
        allocated_bytes=None,
        reserved_bytes=None,
        used_bytes=3072,
        free_bytes=1024,
        total_bytes=4096,
        active_bytes=None,
        inactive_bytes=None,
        device_id=0,
    )

    collectors.validate_device_memory_sample(sample, capabilities)


def test_validate_sample_rejects_counter_declared_unsupported() -> None:
    capabilities = collectors.DeviceMemoryCapabilities(
        backend="future",
        telemetry_collector="stormlog.future_tracker",
        sampling_source="future.device_memory",
        supports_device_used=True,
    )
    sample = collectors.DeviceMemorySample(
        allocated_bytes=1,
        reserved_bytes=None,
        used_bytes=2,
        free_bytes=None,
        total_bytes=None,
        active_bytes=None,
        inactive_bytes=None,
        device_id=0,
    )

    with pytest.raises(ValueError, match="declares it unsupported"):
        collectors.validate_device_memory_sample(sample, capabilities)


def test_validate_sample_requires_partial_diagnostic_for_missing_counter() -> None:
    capabilities = collectors.DeviceMemoryCapabilities(
        backend="future",
        telemetry_collector="stormlog.future_tracker",
        sampling_source="future.device_memory",
        supports_device_used=True,
        supports_device_total=True,
    )
    sample = collectors.DeviceMemorySample(
        allocated_bytes=None,
        reserved_bytes=None,
        used_bytes=2,
        free_bytes=None,
        total_bytes=None,
        active_bytes=None,
        inactive_bytes=None,
        device_id=0,
    )

    with pytest.raises(ValueError, match="missing without a partial-field"):
        collectors.validate_device_memory_sample(sample, capabilities)
    collectors.validate_device_memory_sample(
        sample,
        capabilities,
        partial_fields=("device_total_bytes",),
    )


def test_capabilities_reject_allocator_feature_without_core_counters() -> None:
    with pytest.raises(ValueError, match="allocator features require"):
        collectors.DeviceMemoryCapabilities(
            backend="future",
            telemetry_collector="stormlog.future_tracker",
            sampling_source="future.device_memory",
            supports_device_used=True,
            supports_fragmentation_analysis=True,
        )


def test_validate_sample_rejects_populated_partial_field() -> None:
    capabilities = collectors.DeviceMemoryCapabilities(
        backend="future",
        telemetry_collector="stormlog.future_tracker",
        sampling_source="future.device_memory",
        supports_device_used=True,
        supports_device_total=True,
    )
    sample = collectors.DeviceMemorySample(
        allocated_bytes=None,
        reserved_bytes=None,
        used_bytes=2,
        free_bytes=None,
        total_bytes=4,
        active_bytes=None,
        inactive_bytes=None,
        device_id=0,
    )

    with pytest.raises(ValueError, match="populated but marked partial"):
        collectors.validate_device_memory_sample(
            sample,
            capabilities,
            partial_fields=("device_total_bytes",),
        )


def test_cuda_collector_reports_partial_probe_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collectors,
        "_resolve_device",
        lambda _device: SimpleNamespace(type="cuda", index=0),
    )
    monkeypatch.setattr(collectors.torch.cuda, "memory_allocated", lambda _device: 100)
    monkeypatch.setattr(collectors.torch.cuda, "memory_reserved", lambda _device: 200)

    def _fail_total(_device: object) -> object:
        raise RuntimeError("total unavailable")

    def _fail_stats(_device: object) -> dict[str, int]:
        raise RuntimeError("stats unavailable")

    monkeypatch.setattr(collectors.torch.cuda, "get_device_properties", _fail_total)
    monkeypatch.setattr(collectors.torch.cuda, "memory_stats", _fail_stats)

    collector = collectors.CudaDeviceCollector("cuda:0")
    result = collector.sample_with_diagnostics()

    assert result.sample is not None
    assert result.is_partial is True
    assert result.sample.total_bytes is None
    assert result.sample.free_bytes is None
    assert result.sample.active_bytes is None
    assert result.sample.inactive_bytes is None
    assert result.partial_fields == (
        "device_total_bytes",
        "device_free_bytes",
        "allocator_active_bytes",
        "allocator_inactive_bytes",
    )
    assert result.errors["device_total_bytes"] == "total unavailable"
    assert result.errors["allocator_active_bytes"] == "stats unavailable"


def test_cuda_collector_reports_core_failures_without_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collectors,
        "_resolve_device",
        lambda _device: SimpleNamespace(type="cuda", index=0),
    )

    def _fail_allocated(_device: object) -> int:
        raise RuntimeError("allocated unavailable")

    monkeypatch.setattr(collectors.torch.cuda, "memory_allocated", _fail_allocated)
    monkeypatch.setattr(collectors.torch.cuda, "memory_reserved", lambda _device: 200)

    collector = collectors.CudaDeviceCollector("cuda:0")
    result = collector.sample_with_diagnostics()

    assert result.sample is None
    assert result.is_core_failure is True
    assert result.core_error == "allocated unavailable"
    assert result.errors == {"core_metrics": "allocated unavailable"}

    with pytest.raises(RuntimeError, match="allocated unavailable"):
        collector.sample()
