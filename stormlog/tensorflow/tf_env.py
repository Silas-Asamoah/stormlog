"""TensorFlow runtime environment configuration."""

import os


def configure_tensorflow_logging() -> None:
    """Suppress noisy TensorFlow C++ info logs unless user overrides."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
