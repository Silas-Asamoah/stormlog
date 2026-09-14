import json
import os
import platform
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

import stormlog.tensorflow.utils as tfmemprof_utils
import stormlog.utils as gpumemprof_utils


def test_format_bytes_preserves_negative_sign_and_scales_magnitude() -> None:
    assert gpumemprof_utils.format_bytes(-3 * 1024**3) == "-3.00 GB"


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


def test_check_memory_fragmentation_adds_formatted_keys_without_mutation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy_properties = SimpleNamespace(total_memory=8 * 1024**3)
    dummy_device_type = type("DummyDevice", (), {})
    dummy_torch = SimpleNamespace(
        device=dummy_device_type,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            memory_stats=lambda device_id: {
                "allocated_bytes.all.current": 2 * 1024**3,
                "reserved_bytes.all.current": 3 * 1024**3,
                "active_bytes.all.current": 2 * 1024**3,
                "inactive_split_bytes.all.current": 512 * 1024**2,
            },
            get_device_properties=lambda device_id: dummy_properties,
        ),
    )
    monkeypatch.setattr(gpumemprof_utils, "torch", dummy_torch)

    fragmentation = gpumemprof_utils.check_memory_fragmentation()

    assert fragmentation["total_memory"] == 8 * 1024**3
    assert fragmentation["allocated_memory"] == 2 * 1024**3
    assert fragmentation["reserved_memory"] == 3 * 1024**3
    assert fragmentation["allocated_memory_formatted"] == "2.00 GB"
    assert fragmentation["reserved_memory_formatted"] == "3.00 GB"


class _CudaDevice:
    def __init__(self, index: int | None) -> None:
        self.index = index


@pytest.mark.parametrize(
    "device,expected",
    [
        (None, 3),
        (2, 2),
        ("cuda:4", 4),
        ("cuda", 0),
        (_CudaDevice(None), 0),
        (_CudaDevice(5), 5),
        (SimpleNamespace(index=7), 7),
        (SimpleNamespace(), 0),
    ],
)
def test_gpu_queries_preserve_device_index_resolution(
    monkeypatch: pytest.MonkeyPatch, device: Any, expected: int
) -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 3,
        get_device_name=lambda index: "GPU",
        get_device_capability=lambda index: (8, 0),
        get_device_properties=lambda index: SimpleNamespace(
            total_memory=1024, multi_processor_count=8
        ),
        memory_allocated=lambda index: 0,
        memory_reserved=lambda index: 0,
        max_memory_allocated=lambda index: 0,
        max_memory_reserved=lambda index: 0,
        memory_stats=lambda index: {},
    )
    fake_torch = SimpleNamespace(
        device=_CudaDevice,
        cuda=cuda,
        version=SimpleNamespace(cuda="12"),
        __version__="2",
    )
    monkeypatch.setattr(gpumemprof_utils, "torch", fake_torch)
    monkeypatch.setattr(gpumemprof_utils, "_get_nvidia_smi_info", lambda index: {})
    assert gpumemprof_utils.get_gpu_info(device)["device_id"] == expected
    assert gpumemprof_utils.check_memory_fragmentation(device)["device_id"] == expected


def test_detect_gpu_hardware_windows_prefers_powershell_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpumemprof_utils.platform, "system", lambda: "Windows")

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["shell"] is False
        assert kwargs["timeout"] == 2
        assert cmd[0] == "powershell"
        return SimpleNamespace(
            returncode=0,
            stdout="\nAMD Radeon RX 7900 XTX\nMicrosoft Basic Render Driver\n",
            stderr="",
        )

    monkeypatch.setattr(gpumemprof_utils.subprocess, "run", _fake_run)

    hardware_info = gpumemprof_utils._detect_gpu_hardware()

    assert hardware_info["hardware_gpu_detected"] is True
    assert hardware_info["devices"] == [
        {
            "name": "AMD Radeon RX 7900 XTX",
            "source": "powershell",
            "vendor": "amd",
        }
    ]


def test_detect_gpu_hardware_preserves_identical_device_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpumemprof_utils.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        gpumemprof_utils.subprocess,
        "run",
        lambda cmd, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="\nNVIDIA GeForce RTX 4090\nNVIDIA GeForce RTX 4090\n",
            stderr="",
        ),
    )

    hardware_info = gpumemprof_utils._detect_gpu_hardware()

    assert hardware_info["hardware_gpu_detected"] is True
    assert hardware_info["devices"] == [
        {
            "name": "NVIDIA GeForce RTX 4090",
            "source": "powershell",
            "vendor": "nvidia",
        },
        {
            "name": "NVIDIA GeForce RTX 4090",
            "source": "powershell",
            "vendor": "nvidia",
        },
    ]


def test_detect_gpu_hardware_windows_falls_back_to_wmic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpumemprof_utils.platform, "system", lambda: "Windows")

    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(list(cmd))
        if cmd[0] == "powershell":
            return SimpleNamespace(returncode=1, stdout="", stderr="boom")
        return SimpleNamespace(
            returncode=0, stdout="Name\nNVIDIA GeForce RTX 4090\n", stderr=""
        )

    monkeypatch.setattr(gpumemprof_utils.subprocess, "run", _fake_run)

    hardware_info = gpumemprof_utils._detect_gpu_hardware()

    assert [cmd[0] for cmd in calls] == ["powershell", "wmic"]
    assert hardware_info["devices"][0]["vendor"] == "nvidia"
    assert hardware_info["devices"][0]["source"] == "wmic"


def test_detect_gpu_hardware_linux_parses_lspci(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpumemprof_utils.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        gpumemprof_utils.subprocess,
        "run",
        lambda cmd, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 23 [Radeon RX 6600]\n"
                "04:00.0 Ethernet controller: Intel Corporation Ethernet Controller\n"
            ),
            stderr="",
        ),
    )

    hardware_info = gpumemprof_utils._detect_gpu_hardware()

    assert hardware_info["hardware_gpu_detected"] is True
    assert hardware_info["devices"] == [
        {
            "name": "Advanced Micro Devices, Inc. [AMD/ATI] Navi 23 [Radeon RX 6600]",
            "source": "lspci",
            "vendor": "amd",
        }
    ]


def test_detect_gpu_hardware_macos_parses_system_profiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpumemprof_utils.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        gpumemprof_utils.subprocess,
        "run",
        lambda cmd, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "SPDisplaysDataType": [
                        {
                            "_name": "Apple M2",
                            "spdisplays_vendor": "Apple",
                        }
                    ]
                }
            ),
            stderr="",
        ),
    )

    hardware_info = gpumemprof_utils._detect_gpu_hardware()

    assert hardware_info["hardware_gpu_detected"] is True
    assert hardware_info["devices"] == [
        {
            "name": "Apple M2",
            "source": "system_profiler",
            "vendor": "apple",
        }
    ]


def test_detect_gpu_hardware_returns_empty_on_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpumemprof_utils.platform, "system", lambda: "Linux")

    def _boom(cmd, **kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs["timeout"])

    monkeypatch.setattr(gpumemprof_utils.subprocess, "run", _boom)

    hardware_info = gpumemprof_utils._detect_gpu_hardware()

    assert hardware_info == {"hardware_gpu_detected": False, "devices": []}


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
