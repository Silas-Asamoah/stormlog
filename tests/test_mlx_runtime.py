from types import SimpleNamespace

import pytest

from stormlog.mlx import runtime as module
from stormlog.mlx.runtime import (
    MLXInitializationError,
    MLXMetalUnavailableError,
    MLXNotInstalledError,
    MLXPlatformError,
    MLXRuntime,
    MLXRuntimeError,
)
from tests.mlx_fakes import FakeCore, make_runtime


@pytest.mark.parametrize("system,machine", [("Linux", "arm64"), ("Darwin", "x86_64")])
def test_platform_gate(system, machine):
    with pytest.raises(MLXPlatformError):
        MLXRuntime(platform_system=system, platform_machine=machine)


@pytest.mark.parametrize("version", ["0.32.2", "0.32.3rc1", "unknown"])
def test_version_gate(version):
    with pytest.raises(MLXRuntimeError):
        MLXRuntime(
            core=FakeCore(),
            runtime_version=version,
            platform_system="Darwin",
            platform_machine="arm64",
        )


def test_availability_and_partial_info():
    core = FakeCore()
    core.metal = SimpleNamespace(is_available=lambda: False)
    with pytest.raises(MLXMetalUnavailableError):
        make_runtime(core)
    core.metal.is_available = lambda: True
    core.device_info = lambda _: (_ for _ in ()).throw(RuntimeError("info unavailable"))
    adapter = make_runtime(core)
    assert adapter.device_info == {}
    assert "info unavailable" in adapter.device_info_error
    assert adapter.read_bytes("get_active_memory") == 100


@pytest.mark.parametrize(
    "cause,error",
    [
        (ModuleNotFoundError("missing", name="mlx"), MLXNotInstalledError),
        (ModuleNotFoundError("native", name="native_helper"), MLXInitializationError),
        (OSError("loader failure"), MLXInitializationError),
    ],
)
def test_discovery_errors_and_retry(monkeypatch, cause, error):
    monkeypatch.setattr(module, "_CORE", None)
    monkeypatch.setattr(module, "_CORE_VERSION", None)
    monkeypatch.setattr(
        module.importlib, "import_module", lambda _: (_ for _ in ()).throw(cause)
    )
    with pytest.raises(error) as found:
        MLXRuntime(platform_system="Darwin", platform_machine="arm64")
    assert found.value.__cause__ is cause
    assert module._CORE is None
    core = FakeCore()
    monkeypatch.setattr(module.importlib, "import_module", lambda _: core)
    monkeypatch.setattr(module, "version", lambda _: "0.32.3")
    adapter = MLXRuntime(platform_system="Darwin", platform_machine="arm64")
    assert adapter.core is core


@pytest.mark.parametrize("device", [1, -1, True, "gpu"])
def test_no_fictitious_device(device):
    with pytest.raises(ValueError):
        MLXRuntime(device_id=device)


def test_native_unavailable_metal_loader_is_distinct(monkeypatch):
    monkeypatch.setattr(module, "_CORE", None)
    monkeypatch.setattr(module, "_CORE_VERSION", None)
    original = ImportError("[metal::load_device] No Metal device available. sandbox")
    monkeypatch.setattr(
        module.importlib, "import_module", lambda _: (_ for _ in ()).throw(original)
    )
    with pytest.raises(MLXMetalUnavailableError) as found:
        MLXRuntime(platform_system="Darwin", platform_machine="arm64")
    assert found.value.__cause__ is original
