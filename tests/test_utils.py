import os
import platform
from types import SimpleNamespace

import pytest

import gpumemprof.utils as gpumemprof_utils
import tfmemprof.utils as tfmemprof_utils


def test_get_system_info_contains_expected_keys() -> None:
    system_info = gpumemprof_utils.get_system_info()

    assert "platform" in system_info
    assert "architecture" in system_info
    assert "python_version" in system_info
    assert "mps_available" in system_info
    assert "mps_built" in system_info
    assert "detected_backend" in system_info
    assert system_info["platform"]
    assert system_info["architecture"]


def test_get_system_info_falls_back_to_platform_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy_uname = SimpleNamespace(system="TestOS", machine="TestArch")
    monkeypatch.delattr(os, "uname", raising=False)
    monkeypatch.setattr(platform, "uname", lambda: dummy_uname)

    system_info = gpumemprof_utils.get_system_info()

    assert system_info["platform"] == "TestOS"
    assert system_info["architecture"] == "TestArch"


def _patch_torch_backend(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    mps_backend: object,
    hip_version: str | None = None,
) -> None:
    cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        device_count=lambda: 1,
        current_device=lambda: 0,
    )
    backends = SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 0))
    if mps_backend is not None:
        setattr(backends, "mps", mps_backend)

    dummy_torch = SimpleNamespace(
        cuda=cuda,
        backends=backends,
        version=SimpleNamespace(cuda="12.1", hip=hip_version),
    )
    monkeypatch.setattr(gpumemprof_utils, "torch", dummy_torch)


def test_get_system_info_detects_mps_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    mps_backend = SimpleNamespace(is_built=lambda: True, is_available=lambda: True)
    _patch_torch_backend(monkeypatch, cuda_available=False, mps_backend=mps_backend)

    system_info = gpumemprof_utils.get_system_info()

    assert system_info["mps_built"] is True
    assert system_info["mps_available"] is True
    assert system_info["detected_backend"] == "mps"


def test_get_system_info_prefers_cuda_over_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    mps_backend = SimpleNamespace(is_built=lambda: True, is_available=lambda: True)
    _patch_torch_backend(monkeypatch, cuda_available=True, mps_backend=mps_backend)

    system_info = gpumemprof_utils.get_system_info()

    assert system_info["detected_backend"] == "cuda"


def test_get_system_info_reports_cpu_when_no_cuda_or_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_torch_backend(monkeypatch, cuda_available=False, mps_backend=None)

    system_info = gpumemprof_utils.get_system_info()

    assert system_info["mps_built"] is False
    assert system_info["mps_available"] is False
    assert system_info["detected_backend"] == "cpu"


def test_get_system_info_detects_rocm_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_torch_backend(
        monkeypatch,
        cuda_available=True,
        mps_backend=None,
        hip_version="6.3.0",
    )

    system_info = gpumemprof_utils.get_system_info()

    assert system_info["rocm_available"] is True
    assert system_info["rocm_version"] == "6.3.0"
    assert system_info["detected_backend"] == "rocm"


def _build_dummy_tf(runtime_gpu_count: int, build_info: dict[str, object]) -> object:
    config = SimpleNamespace(
        list_physical_devices=lambda device_type: (
            [object()] * runtime_gpu_count if device_type == "GPU" else []
        )
    )
    sysconfig = SimpleNamespace(get_build_info=lambda: build_info)
    return SimpleNamespace(__version__="2.13.0", config=config, sysconfig=sysconfig)


def _patch_tf_backend(
    monkeypatch: pytest.MonkeyPatch,
    runtime_gpu_count: int,
    build_info: dict[str, object],
    apple_silicon: bool,
    metal_installed: bool,
) -> None:
    dummy_tf = _build_dummy_tf(runtime_gpu_count, build_info)
    monkeypatch.setattr(tfmemprof_utils, "TF_AVAILABLE", True)
    monkeypatch.setattr(tfmemprof_utils, "tf", dummy_tf)
    monkeypatch.setattr(
        tfmemprof_utils,
        "get_gpu_info",
        lambda: {
            "available": runtime_gpu_count > 0,
            "count": runtime_gpu_count,
            "devices": [],
            "total_memory": 0,
        },
    )
    monkeypatch.setattr(
        tfmemprof_utils, "_is_package_installed", lambda _: metal_installed
    )
    if apple_silicon:
        monkeypatch.setattr(tfmemprof_utils.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(tfmemprof_utils.platform, "machine", lambda: "arm64")
    else:
        monkeypatch.setattr(tfmemprof_utils.platform, "system", lambda: "Linux")
        monkeypatch.setattr(tfmemprof_utils.platform, "machine", lambda: "x86_64")


def test_tf_get_system_info_reports_apple_metal_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tf_backend(
        monkeypatch,
        runtime_gpu_count=0,
        build_info={
            "is_cuda_build": False,
            "is_rocm_build": False,
            "is_tensorrt_build": False,
        },
        apple_silicon=True,
        metal_installed=True,
    )

    system_info = tfmemprof_utils.get_system_info()
    backend = system_info["backend"]

    assert backend["is_apple_silicon"] is True
    assert backend["hardware_gpu_detected"] is True
    assert backend["runtime_gpu_count"] == 0
    assert backend["tensorflow_metal_installed"] is True
    assert backend["runtime_backend"] == "metal"


def test_tf_get_system_info_reports_cuda_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tf_backend(
        monkeypatch,
        runtime_gpu_count=1,
        build_info={
            "is_cuda_build": True,
            "is_rocm_build": False,
            "is_tensorrt_build": True,
        },
        apple_silicon=False,
        metal_installed=False,
    )

    system_info = tfmemprof_utils.get_system_info()
    backend = system_info["backend"]

    assert backend["is_apple_silicon"] is False
    assert backend["hardware_gpu_detected"] is True
    assert backend["runtime_gpu_count"] == 1
    assert backend["is_cuda_build"] is True
    assert backend["is_tensorrt_build"] is True
    assert backend["runtime_backend"] == "cuda"
