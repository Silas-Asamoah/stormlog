"""Reusable TensorFlow helpers for examples."""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

from .device import seed_everything


def build_simple_tf_model(
    input_size: int = 1024,
    hidden_size: int = 512,
    num_layers: int = 3,
    num_classes: int = 10,
) -> "tf.keras.Model":
    if tf is None:
        raise RuntimeError("TensorFlow is required for this example.")

    inputs = tf.keras.Input(shape=(input_size,))
    x = inputs
    for _ in range(num_layers - 1):
        x = tf.keras.layers.Dense(hidden_size, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return model


def generate_tf_batch(
    batch_size: int = 256,
    input_size: int = 1024,
    num_classes: int = 10,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """Create a synthetic classification batch compatible with tf tensors."""
    if tf is None:
        raise RuntimeError("TensorFlow is required for this example.")

    seed_everything()
    inputs = np.random.randn(batch_size, input_size).astype("float32")
    targets = np.random.randint(0, num_classes, size=(batch_size,), dtype="int32")
    return inputs, targets


def run_tf_train_step(model: "tf.keras.Model") -> float:
    """Run a single training step on synthetic data."""
    if tf is None:
        raise RuntimeError("TensorFlow is required for this example.")

    inputs, targets = generate_tf_batch()
    history = model.train_on_batch(inputs, targets, return_dict=True)
    return float(history.get("loss", 0.0))
