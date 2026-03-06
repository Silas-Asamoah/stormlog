[← Back to main docs](index.md)

# Compatibility Matrix

This matrix reflects the current behavior of the repository and is versioned for the v0.2 documentation refresh.

- Matrix version: `v0.2`
- Last verified: `2026-03-05`
- Source of truth:
  - `pyproject.toml` (`requires-python >=3.10`, framework dependency floors)
  - CLI entry points in `pyproject.toml`
  - Runtime backend detection in `gpumemprof/device_collectors.py` and `tfmemprof/utils.py`
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

### PyTorch (`gpumemprof`)

| Runtime backend | Typical platform | Telemetry collector | `device_total/free` support | Notes |
| --- | --- | --- | --- | --- |
| `cuda` | NVIDIA + CUDA | `gpumemprof.cuda_tracker` | ✅ | Uses `torch.cuda.memory_*` |
| `rocm` | AMD + ROCm (Linux) | `gpumemprof.rocm_tracker` | ✅ | Uses HIP-backed `torch.cuda.memory_*` |
| `mps` | Apple Silicon (macOS) | `gpumemprof.mps_tracker` | Partial | Depends on `torch.mps.recommended_max_memory()` availability |
| `cpu` | Any host | `gpumemprof.cpu_tracker` | N/A | `CPUMemoryProfiler` / `CPUMemoryTracker` fallback |

### TensorFlow (`tfmemprof`)

| Runtime backend | Typical platform | Telemetry collector | Notes |
| --- | --- | --- | --- |
| `cuda` | NVIDIA + CUDA | `tfmemprof.memory_tracker` | Build/runtime diagnostics shown in `tfmemprof info` |
| `rocm` | AMD + ROCm (Linux) | `tfmemprof.memory_tracker` | Build/runtime diagnostics shown in `tfmemprof info` |
| `metal` | Apple Silicon | `tfmemprof.memory_tracker` | Counters can be runtime-dependent on Metal stack |
| `cpu` | Any host | `tfmemprof.memory_tracker` | Full CLI surface remains available |

## Validation Notes

- The compatibility matrix is linked from `docs/index.md`.
- The example smoke commands documented elsewhere in the repo are maintained validation paths and are exercised by the dedicated `cli-test` GitHub Actions job in `.github/workflows/ci.yml`.
- CI currently validates framework test matrices, TUI gates, lint, docs, and package build lanes.
- Backend capability metadata emitted in tracker exports includes:
  - `backend`
  - `supports_device_total`
  - `supports_device_free`
  - `sampling_source`
