"""Device and seeding helpers shared across examples."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment, unused-ignore]

try:
    import tensorflow as tf
except ImportError:
    tf = None


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, PyTorch, and TensorFlow for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if tf is not None:
        tf.random.set_seed(seed)


def _require_module(module: Any, name: str) -> None:
    if module is None:
        raise RuntimeError(
            f"{name} is required for this example but is not installed. "
            f"Install by running 'pip install {name}' or one of the optional extras."
        )


def get_torch_device(prefer_cuda: bool = True) -> "torch.device":
    """Return a torch.device, gracefully falling back to CPU."""
    _require_module(torch, "torch")
    assert torch is not None  # satisfies type-checkers

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_torch_environment(
    device: Optional["torch.device"] = None,
) -> Dict[str, Any]:
    """Collect lightweight metadata about the PyTorch runtime."""
    _require_module(torch, "torch")
    assert torch is not None

    if device is None:
        device = get_torch_device()

    info: Dict[str, Any] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        cuda_device_name = torch.cuda.get_device_name(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        info.update(
            {
                "cuda_device": cuda_device_name,
                "cuda_memory_total": total_memory,
            }
        )

    return info


def get_tf_device() -> str:
    """Return a TensorFlow device string, defaulting to CPU."""
    _require_module(tf, "tensorflow")
    assert tf is not None

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return str(gpus[0].name)
    return "/device:CPU:0"


def describe_tf_environment() -> Dict[str, Any]:
    """Return metadata about the TensorFlow runtime."""
    _require_module(tf, "tensorflow")
    assert tf is not None

    gpus = tf.config.list_physical_devices("GPU")
    return {
        "tensorflow_version": tf.__version__,
        "gpu_count": len(gpus),
        "gpus": [gpu.name for gpu in gpus],
    }
