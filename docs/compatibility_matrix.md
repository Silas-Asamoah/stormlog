[← Back to main docs](index.md)

# Compatibility Matrix

This matrix reflects the current behavior of the repository and is versioned for the v0.2 documentation refresh.

- Matrix version: `v0.2`
- Last verified: `2026-09-05`
- Source of truth:
  - `pyproject.toml` (`requires-python >=3.10`, framework dependency floors)
  - CLI entry points in `pyproject.toml`
  - Runtime backend detection in `stormlog/device_collectors.py` and `stormlog/tensorflow/utils.py`
  - CI test/lint/docs/build lanes in `.github/workflows/ci.yml`

## Runtime + Version Support

| Surface | Supported | Notes |
| --- | --- | --- |
| Python | 3.10, 3.11, 3.12 | 3.8/3.9 are no longer supported for v0.2 |
| PyTorch package floor | `torch>=1.8.0` | Runtime backend can be CUDA, ROCm, MPS, or CPU |
| TensorFlow package floor | `tensorflow>=2.4.0` | Runtime backend can be CUDA, ROCm, Metal, or CPU |
| OS | Linux, macOS, Windows | Backend availability depends on installed framework/runtime |

## CLI Feature Availability by Environment

| Environment | `gpumemprof info` | `gpumemprof monitor` | `gpumemprof track` | `gpumemprof analyze` | `gpumemprof diagnose` | `tfmemprof info` | `tfmemprof monitor` | `tfmemprof track` | `tfmemprof analyze` | `tfmemprof diagnose` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CUDA (NVIDIA) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ROCm (AMD Linux) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Apple Metal / MPS | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| CPU-only host | ✅ | ✅ (CPU fallback) | ✅ (CPU fallback) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

## Backend Capability Details

### PyTorch APIs (`stormlog`)

| Runtime backend | Telemetry collector | Allocator counters | Device used/free/total | Allocator-native analysis |
| --- | --- | --- | --- | --- |
| `cuda` | `stormlog.cuda_tracker` | ✅ | ✅ | history, fragmentation, attribution, bounded profiling |
| `rocm` | `stormlog.rocm_tracker` | ✅ | ✅ | history, fragmentation, attribution, bounded profiling |
| `mps` | `stormlog.mps_tracker` | allocated/reserved | used; free/total when `recommended_max_memory()` is available | fragmentation |
| injected device-only | collector-defined | N/A | capability-defined | explicitly unavailable |
| `cpu` | `stormlog.cpu_tracker` | N/A | process/system memory contract | dedicated CPU profiler/tracker |

### TensorFlow APIs (`stormlog.tensorflow`)

| Runtime backend | Typical platform | Telemetry collector | Notes |
| --- | --- | --- | --- |
| `cuda` | NVIDIA + CUDA | `stormlog.tensorflow.memory_tracker` | Build/runtime diagnostics shown in `tfmemprof info` |
| `rocm` | AMD + ROCm (Linux) | `stormlog.tensorflow.memory_tracker` | Build/runtime diagnostics shown in `tfmemprof info` |
| `metal` | Apple Silicon | `stormlog.tensorflow.memory_tracker` | Counters can be runtime-dependent on Metal stack |
| `cpu` | Any host | `stormlog.tensorflow.memory_tracker` | Full CLI surface remains available |

## Validation Notes

- The compatibility matrix is linked from the documentation home page.
- The documented CLI-only pip workflow is exercised from the built wheel in a fresh environment by the `artifact-cli-smoke` job in `.github/workflows/ci.yml`.
- Source-only example-module smoke remains covered by the `examples-smoke` job in `.github/workflows/ci.yml`.
- CI currently validates framework test matrices, wheel smoke, source example smoke, TUI gates, lint, docs, and package build lanes.
- V4 tracker exports include the complete typed capability object under
  `metadata["memory_capabilities"]`. The legacy flat backend, device total/free,
  and sampling-source keys remain for compatibility.
- Device-only CLI and TUI diagnostics plot device-used memory, show allocator
  values as `N/A`, and explain that allocator-native analyses are unavailable.
