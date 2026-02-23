import contextlib
from types import SimpleNamespace

import pytest

import gpumemprof.cli as gpumemprof_cli
import tfmemprof.cli as tfmemprof_cli


def _patch_cpu_process(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyProcess:
        def oneshot(self) -> object:
            return contextlib.nullcontext()

        def memory_info(self) -> object:
            return SimpleNamespace(rss=1024, vms=2048)

    monkeypatch.setattr(gpumemprof_cli.psutil, "Process", lambda: DummyProcess())
    monkeypatch.setattr(
        gpumemprof_cli.psutil,
        "cpu_count",
        lambda logical=None: 8 if logical else 4,
    )


def test_gpumemprof_info_reports_mps_without_cpu_only_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        gpumemprof_cli,
        "get_system_info",
        lambda: {
            "platform": "Darwin",
            "python_version": "3.10",
            "cuda_available": False,
            "mps_built": True,
            "mps_available": True,
            "detected_backend": "mps",
        },
    )
    _patch_cpu_process(monkeypatch)

    gpumemprof_cli.cmd_info(SimpleNamespace(device=None, detailed=False))  # type: ignore[arg-type, unused-ignore]
    output = capsys.readouterr().out

    assert "Detected Backend: mps" in output
    assert "MPS Built: True" in output
    assert "MPS Available: True" in output
    assert "MPS backend is available" in output
    assert "Falling back to CPU-only profiling." not in output


def test_gpumemprof_info_reports_cpu_fallback_when_mps_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        gpumemprof_cli,
        "get_system_info",
        lambda: {
            "platform": "Darwin",
            "python_version": "3.10",
            "cuda_available": False,
            "mps_built": False,
            "mps_available": False,
            "detected_backend": "cpu",
        },
    )
    _patch_cpu_process(monkeypatch)

    gpumemprof_cli.cmd_info(SimpleNamespace(device=None, detailed=False))  # type: ignore[arg-type, unused-ignore]
    output = capsys.readouterr().out

    assert "Detected Backend: cpu" in output
    assert "MPS Available: False" in output
    assert "CUDA is not available. Falling back to CPU-only profiling." in output


def test_gpumemprof_info_keeps_cuda_output_when_available(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        gpumemprof_cli,
        "get_system_info",
        lambda: {
            "platform": "Linux",
            "python_version": "3.10",
            "cuda_available": True,
            "cuda_version": "12.1",
            "cuda_device_count": 1,
            "current_device": 0,
            "detected_backend": "cuda",
        },
    )
    monkeypatch.setattr(
        gpumemprof_cli,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 0)),
    )
    monkeypatch.setattr(
        gpumemprof_cli,
        "get_gpu_info",
        lambda device_id: {
            "device_name": "GPU0",
            "total_memory": 1024**3,
            "allocated_memory": 0,
            "reserved_memory": 0,
            "multiprocessor_count": 1,
        },
    )

    gpumemprof_cli.cmd_info(SimpleNamespace(device=None, detailed=False))  # type: ignore[arg-type, unused-ignore]
    output = capsys.readouterr().out

    assert "Detected Backend: cuda" in output
    assert "CUDA Version: 12.1" in output
    assert "GPU 0 Information:" in output
    assert "Falling back to CPU-only profiling." not in output


def test_gpumemprof_info_reports_rocm_backend_details(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        gpumemprof_cli,
        "get_system_info",
        lambda: {
            "platform": "Linux",
            "python_version": "3.10",
            "cuda_available": True,
            "cuda_version": "12.1",
            "rocm_version": "6.3.0",
            "cuda_device_count": 1,
            "current_device": 0,
            "detected_backend": "rocm",
        },
    )
    monkeypatch.setattr(
        gpumemprof_cli,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 0)),
    )
    monkeypatch.setattr(
        gpumemprof_cli,
        "get_gpu_info",
        lambda device_id: {
            "device_name": "AMD GPU0",
            "total_memory": 1024**3,
            "allocated_memory": 0,
            "reserved_memory": 0,
            "multiprocessor_count": 1,
        },
    )

    gpumemprof_cli.cmd_info(SimpleNamespace(device=None, detailed=False))  # type: ignore[arg-type, unused-ignore]
    output = capsys.readouterr().out

    assert "Detected Backend: rocm" in output
    assert "ROCm Version: 6.3.0" in output


def test_tfmemprof_info_reports_backend_diagnostics_for_apple(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        tfmemprof_cli,
        "get_system_info",
        lambda: {
            "platform": "macOS-15.3-arm64-arm-64bit",
            "python_version": "3.10.19",
            "tensorflow_version": "2.13.0",
            "cpu_count": 8,
            "total_memory_gb": 16.0,
            "available_memory_gb": 8.0,
            "gpu": {"available": False, "error": "No GPU devices found"},
            "backend": {
                "is_apple_silicon": True,
                "hardware_gpu_detected": True,
                "runtime_gpu_count": 0,
                "runtime_backend": "metal",
                "is_cuda_build": False,
                "is_rocm_build": False,
                "is_tensorrt_build": False,
                "tensorflow_metal_installed": True,
            },
        },
    )
    monkeypatch.setattr(tfmemprof_cli, "TF_AVAILABLE", True)
    monkeypatch.setattr(
        tfmemprof_cli,
        "tf",
        SimpleNamespace(
            sysconfig=SimpleNamespace(
                get_build_info=lambda: {
                    "is_cuda_build": False,
                    "cuda_version": "Unknown",
                    "cudnn_version": "Unknown",
                }
            )
        ),
    )

    tfmemprof_cli.cmd_info(SimpleNamespace())  # type: ignore[arg-type, unused-ignore]
    output = capsys.readouterr().out

    assert "GPU Hardware Detected: Yes" in output
    assert "GPU Available to TensorFlow Runtime: No" in output
    assert "Runtime Backend: metal" in output
    assert "Hardware GPU Detected: True" in output
    assert "Apple Silicon: True" in output
    assert "tensorflow-metal Installed: True" in output
    assert "ROCm Build: False" in output
    assert "TensorRT Build: False" in output
    assert "CUDA Build: False" in output


def test_tfmemprof_info_keeps_cuda_build_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        tfmemprof_cli,
        "get_system_info",
        lambda: {
            "platform": "Linux",
            "python_version": "3.10.19",
            "tensorflow_version": "2.15.0",
            "cpu_count": 8,
            "total_memory_gb": 16.0,
            "available_memory_gb": 8.0,
            "gpu": {
                "available": True,
                "count": 1,
                "total_memory": 4096,
                "devices": [
                    {"name": "GPU 0", "current_memory_mb": 10.0, "peak_memory_mb": 20.0}
                ],
            },
            "backend": {
                "is_apple_silicon": False,
                "hardware_gpu_detected": True,
                "runtime_gpu_count": 1,
                "runtime_backend": "cuda",
                "is_cuda_build": True,
                "is_rocm_build": False,
                "is_tensorrt_build": True,
                "tensorflow_metal_installed": False,
            },
        },
    )
    monkeypatch.setattr(tfmemprof_cli, "TF_AVAILABLE", True)
    monkeypatch.setattr(
        tfmemprof_cli,
        "tf",
        SimpleNamespace(
            sysconfig=SimpleNamespace(
                get_build_info=lambda: {
                    "is_cuda_build": True,
                    "cuda_version": "12.1",
                    "cudnn_version": "8.9",
                }
            )
        ),
    )

    tfmemprof_cli.cmd_info(SimpleNamespace())  # type: ignore[arg-type, unused-ignore]
    output = capsys.readouterr().out

    assert "Runtime Backend: cuda" in output
    assert "Hardware GPU Detected: True" in output
    assert "CUDA Build: True" in output
    assert "CUDA Version: 12.1" in output
    assert "cuDNN Version: 8.9" in output
