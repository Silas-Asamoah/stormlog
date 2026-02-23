"""TensorFlow Context Profiling"""

import functools
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar, Union, cast

from .tf_env import configure_tensorflow_logging

configure_tensorflow_logging()

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

from .profiler import TFMemoryProfiler

# Global profiler instance
_global_profiler: Optional[TFMemoryProfiler] = None
_profiler_lock = threading.Lock()
F = TypeVar("F", bound=Callable[..., Any])


def get_global_profiler() -> TFMemoryProfiler:
    """Get or create global profiler instance."""
    global _global_profiler

    with _profiler_lock:
        if _global_profiler is None:
            _global_profiler = TFMemoryProfiler()
        return _global_profiler


def set_global_profiler(profiler: TFMemoryProfiler) -> None:
    """Set global profiler instance."""
    global _global_profiler

    with _profiler_lock:
        _global_profiler = profiler


def profile_function(
    func: Optional[F] = None,
    *,
    profiler: Optional[TFMemoryProfiler] = None,
    name: Optional[str] = None,
) -> Union[Callable[[F], F], F]:
    """
    Decorator to profile function memory usage.

    Args:
        func: Function to profile
        profiler: Profiler instance (uses global if None)
        name: Custom name for profiling
    """

    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            prof = profiler or get_global_profiler()
            _ = name or f.__name__

            # Use the profiler's function profiling
            return prof.profile_function(f)(*args, **kwargs)

        return cast(F, wrapper)

    if func is None:
        return decorator
    else:
        return decorator(func)


@contextmanager
def profile_context(
    name: str = "context", profiler: Optional[TFMemoryProfiler] = None
) -> Iterator[None]:
    """
    Context manager for profiling code blocks.

    Args:
        name: Name for the profiling context
        profiler: Profiler instance (uses global if None)
    """
    prof = profiler or get_global_profiler()

    with prof.profile_context(name):
        yield


class ProfiledLayer:
    """Wrapper for TensorFlow layers with automatic profiling."""

    def __init__(
        self,
        layer: Any,
        profiler: Optional[TFMemoryProfiler] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize profiled layer.

        Args:
            layer: TensorFlow layer to profile
            profiler: Profiler instance
            name: Custom name for profiling
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")

        self.layer = layer
        self.profiler = profiler or get_global_profiler()
        self.name = name or getattr(layer, "name", layer.__class__.__name__)

        # Wrap the call method
        self._original_call = layer.call
        layer.call = self._profiled_call

    def _profiled_call(self, *args: Any, **kwargs: Any) -> Any:
        """Profiled version of layer call."""
        with self.profiler.profile_context(f"layer_{self.name}"):
            return self._original_call(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped layer."""
        return getattr(self.layer, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Make the wrapper callable."""
        return self.layer(*args, **kwargs)


def profile_model(model: Any, profiler: Optional[TFMemoryProfiler] = None) -> Any:
    """
    Profile all layers in a TensorFlow model.

    Args:
        model: TensorFlow model
        profiler: Profiler instance

    Returns:
        Model with profiled layers
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available")

    prof = profiler or get_global_profiler()

    # Profile each layer
    for i, layer in enumerate(model.layers):
        ProfiledLayer(layer, prof, f"{layer.name}_{i}")

    return model


class TensorFlowProfiler:
    """High-level TensorFlow profiling interface."""

    def __init__(self, device: Optional[str] = None) -> None:
        """Initialize TensorFlow profiler."""
        self.profiler = TFMemoryProfiler(device=device)
        set_global_profiler(self.profiler)

    def profile_training(
        self,
        model: Any,
        dataset: Any,
        epochs: int = 1,
        steps_per_epoch: Optional[int] = None,
    ) -> None:
        """
        Profile model training.

        Args:
            model: TensorFlow model
            dataset: Training dataset
            epochs: Number of epochs
            steps_per_epoch: Steps per epoch
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")

        # Profile the entire training process
        with self.profiler.profile_context("training"):
            for epoch in range(epochs):
                with self.profiler.profile_context(f"epoch_{epoch}"):
                    step_count = 0

                    for batch in dataset:
                        if steps_per_epoch and step_count >= steps_per_epoch:
                            break

                        with self.profiler.profile_context(f"step_{step_count}"):
                            # Assume the model has a train_step method or similar
                            if hasattr(model, "train_step"):
                                model.train_step(batch)
                            else:
                                # Generic training step
                                with tf.GradientTape() as tape:
                                    if isinstance(batch, tuple):
                                        x, y = batch
                                        predictions = model(x, training=True)
                                        loss = model.compiled_loss(y, predictions)
                                    else:
                                        predictions = model(batch, training=True)
                                        loss = model.compiled_loss(batch, predictions)

                                gradients = tape.gradient(
                                    loss, model.trainable_variables
                                )
                                model.optimizer.apply_gradients(
                                    zip(gradients, model.trainable_variables)
                                )

                        step_count += 1

    def profile_inference(self, model: Any, data: Any, batch_size: int = 32) -> None:
        """
        Profile model inference.

        Args:
            model: TensorFlow model
            data: Input data
            batch_size: Batch size for inference
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")

        with self.profiler.profile_context("inference"):
            # Batch the data if needed
            if hasattr(data, "batch"):
                _batched_data = data.batch(batch_size)
            else:
                # Assume data is a tensor or numpy array
                import numpy as np

                if isinstance(data, np.ndarray):
                    data = tf.constant(data)

                # Create batches manually
                num_samples = tf.shape(data)[0]
                num_batches = (num_samples + batch_size - 1) // batch_size

                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_samples)
                    batch = data[start_idx:end_idx]

                    with self.profiler.profile_context(f"inference_batch_{i}"):
                        model(batch, training=False)

    def get_results(self) -> Any:
        """Get profiling results."""
        return self.profiler.get_results()

    def reset(self) -> None:
        """Reset profiler state."""
        self.profiler.reset()


# Convenience functions for common use cases
def profile_keras_training(
    model: Any,
    x_train: Any,
    y_train: Any,
    epochs: int = 1,
    batch_size: int = 32,
    validation_data: Optional[Any] = None,
    profiler: Optional[TFMemoryProfiler] = None,
) -> None:
    """
    Profile Keras model training.

    Args:
        model: Keras model
        x_train: Training data
        y_train: Training labels
        epochs: Number of epochs
        batch_size: Batch size
        validation_data: Validation data tuple (x_val, y_val)
        profiler: Profiler instance
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available")

    prof = profiler or get_global_profiler()

    with prof.profile_context("keras_training"):
        # Create dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.batch(batch_size)

        # Profile training
        for epoch in range(epochs):
            with prof.profile_context(f"epoch_{epoch}"):
                # Training
                with prof.profile_context("training_batches"):
                    for batch_x, batch_y in train_dataset:
                        with prof.profile_context("train_step"):
                            model.train_on_batch(batch_x, batch_y)

                # Validation
                if validation_data:
                    x_val, y_val = validation_data
                    with prof.profile_context("validation"):
                        model.evaluate(x_val, y_val, verbose=0)


def clear_global_profiler() -> None:
    """Clear global profiler state."""
    global _global_profiler

    with _profiler_lock:
        if _global_profiler:
            _global_profiler.reset()
            _global_profiler = None


def clear_profiles() -> None:
    """Reset profiling data without discarding the global profiler."""

    with _profiler_lock:
        if _global_profiler:
            _global_profiler.reset()


def get_profile_summaries(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return aggregated profiling summaries for recent functions/contexts."""

    with _profiler_lock:
        profiler = _global_profiler
        if not profiler or not profiler.function_profiles:
            return []

        entries: List[Dict[str, Any]] = []
        for name, stats in profiler.function_profiles.items():
            snapshots = stats.get("snapshots") or []
            last_snapshot = snapshots[-1] if snapshots else None
            last_timestamp = getattr(last_snapshot, "timestamp", None)

            entries.append(
                {
                    "name": name,
                    "calls": stats.get("calls", 0),
                    "total_duration": stats.get("total_duration", 0.0),
                    "total_memory_used": stats.get("total_memory_used", 0.0),
                    "peak_memory": stats.get("peak_memory", 0.0),
                    "last_timestamp": last_timestamp,
                }
            )

    entries.sort(key=lambda entry: entry.get("last_timestamp") or 0.0, reverse=True)

    if limit:
        return entries[:limit]
    return entries
