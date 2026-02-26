[← Back to main docs](index.md)

# GPU Stack Installation & CUDA Enablement

Use this guide to install the GPU-capable versions of PyTorch and TensorFlow,
verify CUDA availability, and switch between CPU and GPU smoke tests.

## 1. Prerequisites

- **NVIDIA driver** that supports the CUDA version you plan to use. Update
  through NVIDIA Experience (Windows) or your distro’s driver manager (Linux).
- **CUDA toolkit** (optional). Most pip wheels bundle the needed runtime, but
  native builds may require the toolkit/`nvcc`.
- **Python 3.10–3.12** in a virtual environment.

## 2. PyTorch (CUDA) Installation

Pick the CUDA build that matches your driver/toolkit. Example for CUDA 12.1:

```bash
# Windows / macOS / Linux (same pip command)
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

Need CUDA 11.8 instead? Swap the wheel URL:

```bash
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```

### Verify PyTorch CUDA

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PY
```

## 3. TensorFlow (GPU) Installation

TensorFlow ≥2.11 ships a unified wheel that includes GPU support as long as
CUDA/cuDNN are present. For this repo we pin 2.15:

```bash
pip install tensorflow==2.15.0
```

### Verify TensorFlow CUDA

```bash
python - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("Memory growth enabled for", gpus[0].name)
PY
```

## 4. Switching Between CPU and GPU Runs

- **CPU-only smoke test:** clear `CUDA_VISIBLE_DEVICES` and run
  `python -m examples.cli.quickstart` plus `pytest tests/test_utils.py` (see
  `docs/testing.md`).
- **GPU path:** unset `CUDA_VISIBLE_DEVICES` (or set it to a GPU index) and run
  the PyTorch/TensorFlow demos:

  ```bash
  python -m examples.basic.pytorch_demo
  python -m examples.basic.tensorflow_demo
  pytest tests/test_profiler.py
  ```

## 5. Common Issues

| Symptom | Fix |
| --- | --- |
| `torch.cuda.is_available()` is False | Confirm NVIDIA driver is installed, retry with the correct CUDA wheel, or reboot after driver install. |
| TensorFlow cannot find cuDNN | Install the CUDA/cuDNN versions listed in the TF release notes or use `tensorflow==2.15.0` (which bundles cuDNN on Windows/Linux). |
| `RuntimeError: CUDA driver not found` | Check that `nvidia-smi` works on the command line; reinstall the driver if necessary. |
| CI path installing wrong framework | Follow `.github/workflows/ci.yml` logic: install the base deps, then exactly one framework (PyTorch or TensorFlow). |

## 6. Next Steps

Once the GPU stack is working, run the Textual TUI for an interactive check:

```bash
pip install "stormlog[tui]"
stormlog
```

Use the overview tab to confirm the profiler sees your GPUs before running the
full benchmarking or release checklist.
