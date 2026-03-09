"""Context profiler for easy function and code block profiling."""

import functools
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar, Union, cast

import torch

from .profiler import GPUMemoryProfiler, ProfileResult

# Global profiler instance for convenience
_global_profiler: Optional[GPUMemoryProfiler] = None
F = TypeVar("F", bound=Callable[..., Any])


def get_global_profiler(
    device: Optional[Union[str, int, torch.device]] = None,
) -> GPUMemoryProfiler:
    """Get or create the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = GPUMemoryProfiler(device=device)
    return _global_profiler


def set_global_profiler(profiler: GPUMemoryProfiler) -> None:
    """Set the global profiler instance."""
    global _global_profiler
    _global_profiler = profiler


def profile_function(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    device: Optional[Union[str, int, torch.device]] = None,
    profiler: Optional[GPUMemoryProfiler] = None,
) -> Union[Callable[[F], F], F]:
    """
    Decorator to profile a function's GPU memory usage.

    Can be used as @profile_function or @profile_function(name="custom_name")

    Args:
        func: Function to profile (when used as @profile_function)
        name: Custom name for the profiled function
        device: GPU device to use for profiling
        profiler: Custom profiler instance to use

    Returns:
        Decorated function or ProfileResult if called directly
    """

    def decorator(f: F) -> F:
        resolved_name = (
            name if name is not None else getattr(f, "__name__", "unknown_function")
        )
        function_name = str(resolved_name)

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get profiler instance
            prof = profiler or get_global_profiler(device)
            result_marker = object()
            result_holder: Dict[str, Any] = {"value": result_marker}

            # Profile the function
            def profiled_func() -> Any:
                result_holder["value"] = f(*args, **kwargs)
                return result_holder["value"]

            profiled_func.__name__ = function_name
            prof.profile_function(profiled_func)

            # Return original function result, not profile result
            if result_holder["value"] is result_marker:
                return f(*args, **kwargs)
            return result_holder["value"]

        # Add profiling metadata to the wrapper
        setattr(wrapper, "_is_profiled", True)
        setattr(wrapper, "_profile_name", function_name)

        return cast(F, wrapper)

    # Handle different calling patterns
    if func is None:
        # Called as @profile_function(args)
        return decorator
    else:
        # Called as @profile_function
        return decorator(func)


@contextmanager
def profile_context(
    name: str = "context",
    device: Optional[Union[str, int, torch.device]] = None,
    profiler: Optional[GPUMemoryProfiler] = None,
) -> Iterator[GPUMemoryProfiler]:
    """
    Context manager for profiling a block of code.

    Args:
        name: Name for the profiled context
        device: GPU device to use for profiling
        profiler: Custom profiler instance to use

    Yields:
        ProfileResult after the context exits

    Example:
        with profile_context("model_forward") as prof:
            output = model(input)
    """
    # Get profiler instance
    prof = profiler or get_global_profiler(device)

    # Use the profiler's context manager
    with prof.profile_context(name):
        yield prof


class ProfiledModule(torch.nn.Module):
    """
    Wrapper for PyTorch modules that automatically profiles forward passes.

    Example:
        model = ProfiledModule(original_model, name="my_model")
        output = model(input)  # Automatically profiled
    """

    def __init__(
        self,
        module: torch.nn.Module,
        name: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        profiler: Optional[GPUMemoryProfiler] = None,
    ) -> None:
        super().__init__()
        self.module = module
        self.profile_name = name or module.__class__.__name__
        self.profiler = profiler or get_global_profiler(device)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass with automatic profiling."""
        with self.profiler.profile_context(f"{self.profile_name}_forward"):
            return self.module(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class MemoryProfiler:
    """
    High-level memory profiler with convenient methods.

    This class provides a simplified interface for common profiling tasks.
    """

    def __init__(self, device: Optional[Union[str, int, torch.device]] = None) -> None:
        self.profiler = GPUMemoryProfiler(device=device)
        self._monitoring = False

    def start_monitoring(self, interval: float = 0.1) -> None:
        """Start continuous memory monitoring."""
        self.profiler.start_monitoring(interval)
        self._monitoring = True

    def stop_monitoring(self) -> None:
        """Stop continuous memory monitoring."""
        self.profiler.stop_monitoring()
        self._monitoring = False

    def profile(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> ProfileResult:
        """Profile a function call."""
        return self.profiler.profile_function(func, *args, **kwargs)

    @contextmanager
    def context(self, name: str = "context") -> Iterator[None]:
        """Context manager for profiling code blocks."""
        with self.profiler.profile_context(name):
            yield

    def wrap_module(
        self, module: torch.nn.Module, name: Optional[str] = None
    ) -> ProfiledModule:
        """Wrap a PyTorch module for automatic profiling."""
        return ProfiledModule(module, name=name, profiler=self.profiler)

    def get_summary(self) -> Any:
        """Get profiling summary."""
        return self.profiler.get_summary()

    def clear(self) -> None:
        """Clear profiling results."""
        self.profiler.clear_results()

    def save_results(self, filename: str) -> None:
        """Save profiling results to file."""
        import json

        summary = self.get_summary()

        # Convert results to JSON-serializable format
        json_data = {
            "summary": summary,
            "results": [result.to_dict() for result in self.profiler.results],
            "snapshots": [snapshot.to_dict() for snapshot in self.profiler.snapshots],
        }

        with open(filename, "w") as f:
            json.dump(json_data, f, indent=2, default=str)

    def load_results(self, filename: str) -> Any:
        """Load profiling results from file."""
        import json

        with open(filename, "r") as f:
            data = json.load(f)
        return data

    def __enter__(self) -> "MemoryProfiler":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        if self._monitoring:
            self.stop_monitoring()


# Convenience functions for global profiler
def start_monitoring(
    interval: float = 0.1, device: Optional[Union[str, int, torch.device]] = None
) -> None:
    """Start global memory monitoring."""
    profiler = get_global_profiler(device)
    profiler.start_monitoring(interval)


def stop_monitoring() -> None:
    """Stop global memory monitoring."""
    if _global_profiler:
        _global_profiler.stop_monitoring()


def get_summary() -> Any:
    """Get global profiler summary."""
    if _global_profiler:
        return _global_profiler.get_summary()
    return {"message": "No global profiler instance"}


def clear_results() -> None:
    """Clear global profiler results."""
    if _global_profiler:
        _global_profiler.clear_results()


def get_profile_results(limit: Optional[int] = None) -> List[ProfileResult]:
    """Return recent profile results captured by the global profiler."""
    if not _global_profiler:
        return []

    results = list(_global_profiler.results)
    if limit:
        return results[-limit:]
    return results


def profile_model_training(
    model: torch.nn.Module,
    train_loader: Any,
    epochs: int = 1,
    device: Optional[Union[str, int, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Profile an entire training loop.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        epochs: Number of epochs to profile
        device: GPU device to use

    Returns:
        Dictionary with profiling results
    """
    profiler = get_global_profiler(device)

    # Profile training setup
    with profiler.profile_context("training_setup"):
        model.train()
        if hasattr(train_loader, "__len__"):
            _total_batches = len(train_loader) * epochs
        else:
            _total_batches = epochs * 100  # Estimate

    results: Dict[str, Any] = {
        "total_epochs": epochs,
        "batch_results": [],
        "epoch_summaries": [],
    }

    for epoch in range(epochs):
        with profiler.profile_context(f"epoch_{epoch}"):
            epoch_results: List[Dict[str, Any]] = []

            for batch_idx, batch_data in enumerate(train_loader):
                with profiler.profile_context(f"batch_{epoch}_{batch_idx}"):
                    # This is a simplified example - in practice, you'd need
                    # to provide the actual training step function
                    pass

                # Store batch result
                if profiler.results:
                    epoch_results.append(profiler.results[-1].to_dict())

                # Limit profiling to prevent memory issues
                if batch_idx >= 10:  # Profile first 10 batches per epoch
                    break

            results["batch_results"].extend(epoch_results)

            # Epoch summary
            epoch_summary = {
                "epoch": epoch,
                "batches_profiled": len(epoch_results),
                "total_memory_allocated": sum(
                    r.get("memory_allocated", 0) for r in epoch_results
                ),
                "average_batch_time": sum(
                    r.get("execution_time", 0) for r in epoch_results
                )
                / max(len(epoch_results), 1),
            }
            results["epoch_summaries"].append(epoch_summary)

    # Overall summary
    results["overall_summary"] = profiler.get_summary()

    return results
