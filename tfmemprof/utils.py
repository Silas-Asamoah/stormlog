"""
Utility functions for TensorFlow memory profiling.

This module provides helper functions for memory formatting, system information,
and TensorFlow-specific optimizations.
"""

import logging
import os
import platform
from importlib import metadata
from typing import Any, Dict, List, Optional, TypedDict, Union, cast

from .tf_env import configure_tensorflow_logging

configure_tensorflow_logging()

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None


def _is_package_installed(package_name: str) -> bool:
    """Return True when a package distribution is installed."""
    try:
        metadata.version(package_name)
        return True
    except metadata.PackageNotFoundError:
        return False
    except Exception as exc:
        logging.debug("Package check failed for %r: %s", package_name, exc)
        return False


def _is_apple_silicon() -> bool:
    """Return True when running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}


def _detect_runtime_backend(
    runtime_gpu_count: int,
    is_cuda_build: bool,
    is_rocm_build: bool,
    is_apple_silicon: bool,
    tensorflow_metal_installed: bool,
) -> str:
    """Classify the currently usable TensorFlow runtime backend."""
    if runtime_gpu_count > 0:
        if is_cuda_build:
            return "cuda"
        if is_rocm_build:
            return "rocm"
        if is_apple_silicon:
            return "metal"
        return "gpu"

    if is_apple_silicon and tensorflow_metal_installed:
        return "metal"

    return "cpu"


class BackendInfo(TypedDict):
    is_apple_silicon: bool
    hardware_gpu_detected: bool
    runtime_gpu_count: int
    runtime_backend: str
    is_cuda_build: bool
    is_rocm_build: bool
    is_tensorrt_build: bool
    tensorflow_metal_installed: bool


def get_backend_info() -> BackendInfo:
    """Return backend diagnostics used by CLI and system reporting."""
    backend_info: BackendInfo = {
        "is_apple_silicon": _is_apple_silicon(),
        "hardware_gpu_detected": False,
        "runtime_gpu_count": 0,
        "runtime_backend": "cpu",
        "is_cuda_build": False,
        "is_rocm_build": False,
        "is_tensorrt_build": False,
        "tensorflow_metal_installed": _is_package_installed("tensorflow-metal"),
    }

    if TF_AVAILABLE:
        try:
            runtime_gpu_count = len(tf.config.list_physical_devices("GPU"))
        except Exception as exc:
            logging.debug("Could not get TF GPU count: %s", exc)
            runtime_gpu_count = 0
        backend_info["runtime_gpu_count"] = runtime_gpu_count

        try:
            build_info = cast(Dict[str, Any], tf.sysconfig.get_build_info())
        except Exception as exc:
            logging.debug("Could not get TF build info: %s", exc)
            build_info = {}

        backend_info["is_cuda_build"] = bool(build_info.get("is_cuda_build", False))
        backend_info["is_rocm_build"] = bool(build_info.get("is_rocm_build", False))
        backend_info["is_tensorrt_build"] = bool(
            build_info.get("is_tensorrt_build", False)
        )

    backend_info["hardware_gpu_detected"] = bool(
        backend_info["is_apple_silicon"] or backend_info["runtime_gpu_count"] > 0
    )

    backend_info["runtime_backend"] = _detect_runtime_backend(
        runtime_gpu_count=backend_info["runtime_gpu_count"],
        is_cuda_build=backend_info["is_cuda_build"],
        is_rocm_build=backend_info["is_rocm_build"],
        is_apple_silicon=backend_info["is_apple_silicon"],
        tensorflow_metal_installed=backend_info["tensorflow_metal_installed"],
    )

    return backend_info


def format_memory(bytes_value: Optional[Union[int, float]]) -> str:
    """Format memory size in human-readable format."""
    if bytes_value is None:
        return "N/A"

    bytes_value = float(bytes_value)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def get_gpu_info() -> Dict[str, Any]:
    """Get detailed GPU information for TensorFlow."""
    gpu_info: Dict[str, Any] = {
        "available": False,
        "count": 0,
        "devices": [],
        "driver_version": "Unknown",
        "cuda_version": "Unknown",
        "total_memory": 0,
    }

    if not TF_AVAILABLE:
        gpu_info["error"] = "TensorFlow not available"
        return gpu_info

    try:
        # Get physical GPU devices
        physical_gpus = tf.config.list_physical_devices("GPU")

        if physical_gpus:
            gpu_info["available"] = True
            gpu_info["count"] = len(physical_gpus)

            # Get device details
            for i, gpu in enumerate(physical_gpus):
                try:
                    memory_info = tf.config.experimental.get_memory_info(f"/GPU:{i}")
                    device_info = {
                        "id": i,
                        "name": f"GPU {i}",
                        "current_memory_mb": memory_info.get("current", 0)
                        / (1024 * 1024),
                        "peak_memory_mb": memory_info.get("peak", 0) / (1024 * 1024),
                    }
                    gpu_info["devices"].append(device_info)
                    gpu_info["total_memory"] += device_info["peak_memory_mb"]
                except Exception as e:
                    logging.warning(f"Could not get memory info for GPU {i}: {e}")

            # Try to get CUDA version
            try:
                if hasattr(tf, "sysconfig"):
                    gpu_info["cuda_version"] = tf.sysconfig.get_build_info().get(
                        "cuda_version", "Unknown"
                    )
            except Exception as exc:
                logging.debug("Could not get CUDA version from sysconfig: %s", exc)

        else:
            gpu_info["error"] = "No GPU devices found"

    except Exception as e:
        gpu_info["error"] = f"Error getting GPU info: {str(e)}"
        logging.warning(f"Could not get GPU info: {e}")

    return gpu_info


def get_system_info() -> Dict[str, Any]:
    """Get system and TensorFlow environment information."""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "tensorflow_version": "Not installed",
        "cpu_count": os.cpu_count(),
        "total_memory_gb": 0,
        "available_memory_gb": 0,
    }

    # TensorFlow version
    if TF_AVAILABLE:
        info["tensorflow_version"] = tf.__version__

        # TensorFlow build info
        try:
            build_info = tf.sysconfig.get_build_info()
            info["tf_build_cuda"] = build_info.get("is_cuda_build", False)
            info["tf_cuda_version"] = build_info.get("cuda_version", "N/A")
            info["tf_cudnn_version"] = build_info.get("cudnn_version", "N/A")
        except Exception as exc:
            logging.debug("Could not read TF build info: %s", exc)

    # Memory information
    if PSUTIL_AVAILABLE and psutil is not None:
        memory = psutil.virtual_memory()
        info["total_memory_gb"] = memory.total / (1024**3)
        info["available_memory_gb"] = memory.available / (1024**3)
        info["memory_percent_used"] = memory.percent

    # GPU information
    gpu_info = get_gpu_info()
    info["gpu"] = gpu_info
    info["backend"] = get_backend_info()

    return info


def analyze_fragmentation(snapshots: List) -> Dict[str, float]:
    """Analyze memory fragmentation from snapshots."""
    if len(snapshots) < 2:
        return {"fragmentation_score": 0.0, "fragmentation_trend": 0.0}

    # Calculate memory usage patterns
    gpu_memories = [s.gpu_memory_mb for s in snapshots]
    reserved_memories = [s.gpu_memory_reserved_mb for s in snapshots]

    # Fragmentation score based on difference between reserved and used
    fragmentation_ratios = []
    for i in range(len(gpu_memories)):
        if reserved_memories[i] > 0:
            ratio = (reserved_memories[i] - gpu_memories[i]) / reserved_memories[i]
            fragmentation_ratios.append(ratio)

    if not fragmentation_ratios:
        return {"fragmentation_score": 0.0, "fragmentation_trend": 0.0}

    # Average fragmentation
    avg_fragmentation = sum(fragmentation_ratios) / len(fragmentation_ratios)

    # Fragmentation trend (increasing/decreasing)
    if len(fragmentation_ratios) >= 2:
        recent_frag = sum(fragmentation_ratios[-5:]) / min(5, len(fragmentation_ratios))
        early_frag = sum(fragmentation_ratios[:5]) / min(5, len(fragmentation_ratios))
        fragmentation_trend = recent_frag - early_frag
    else:
        fragmentation_trend = 0.0

    return {
        "fragmentation_score": avg_fragmentation,
        "fragmentation_trend": fragmentation_trend,
        "max_fragmentation": max(fragmentation_ratios),
        "min_fragmentation": min(fragmentation_ratios),
    }


def suggest_optimizations(profile_result: Any) -> List[str]:
    """Generate TensorFlow-specific optimization suggestions."""
    suggestions: List[str] = []

    if not hasattr(profile_result, "peak_memory_mb"):
        return suggestions

    # Memory usage analysis
    peak_memory = profile_result.peak_memory_mb
    _avg_memory = profile_result.average_memory_mb

    # High memory usage suggestions
    if peak_memory > 8000:  # > 8GB
        suggestions.extend(
            [
                "Consider using tf.keras.utils.Sequence for data loading to reduce memory usage",
                "Enable mixed precision training with tf.keras.mixed_precision.Policy('mixed_float16')",
                "Use gradient checkpointing with tf.recompute_grad for large models",
            ]
        )

    if peak_memory > 4000:  # > 4GB
        suggestions.extend(
            [
                "Consider reducing batch size or using gradient accumulation",
                "Use tf.data.Dataset.prefetch() and tf.data.Dataset.cache() for efficient data loading",
                "Enable memory growth: tf.config.experimental.set_memory_growth(gpu, True)",
            ]
        )

    # Memory growth analysis
    if (
        hasattr(profile_result, "memory_growth_rate")
        and profile_result.memory_growth_rate > 100
    ):  # >100MB/s
        suggestions.extend(
            [
                "High memory growth detected - check for memory leaks in custom ops",
                "Use tf.function decorator to optimize computation graphs",
                "Consider using tf.data.Dataset instead of feeding numpy arrays",
            ]
        )

    # Fragmentation analysis
    if hasattr(profile_result, "snapshots") and len(profile_result.snapshots) > 1:
        frag_info = analyze_fragmentation(profile_result.snapshots)
        if frag_info["fragmentation_score"] > 0.3:
            suggestions.append(
                "High memory fragmentation detected - consider using smaller batch sizes"
            )
            suggestions.append(
                "Use tf.config.experimental.set_virtual_device_configuration() to limit GPU memory"
            )

    # Function-specific suggestions
    if hasattr(profile_result, "function_profiles"):
        for func_name, profile in profile_result.function_profiles.items():
            if profile.get("peak_memory", 0) > 2000:  # > 2GB per function
                suggestions.append(
                    f"Function '{func_name}' uses high memory - consider optimization"
                )

            if (
                profile.get("calls", 0) > 100
                and profile.get("total_memory_used", 0) > 0
            ):
                suggestions.append(
                    f"Function '{func_name}' called frequently - consider @tf.function decoration"
                )

    # TensorFlow-specific optimizations
    suggestions.extend(
        [
            "Use tf.data.Dataset.map() with num_parallel_calls=tf.data.AUTOTUNE for preprocessing",
            "Enable XLA compilation with tf.config.optimizer.set_jit(True)",
            "Consider using tf.distribute.Strategy for multi-GPU training",
            "Use tf.keras.callbacks.ReduceLROnPlateau to prevent overfitting and reduce memory over time",
        ]
    )

    # Remove duplicates while preserving order
    seen = set()
    unique_suggestions = []
    for suggestion in suggestions:
        if suggestion not in seen:
            seen.add(suggestion)
            unique_suggestions.append(suggestion)

    return unique_suggestions[:10]  # Return top 10 suggestions


def generate_summary_report(profile_result: Any) -> str:
    """Generate a comprehensive summary report."""
    if not hasattr(profile_result, "peak_memory_mb"):
        return "No profiling data available"

    report = []
    report.append("TensorFlow GPU Memory Analysis Report")
    report.append("=" * 40)
    report.append("")

    # Basic memory statistics
    report.append(
        f"Peak Memory Usage: {format_memory(profile_result.peak_memory_mb * 1024 * 1024)}"
    )
    report.append(
        f"Average Memory Usage: {format_memory(profile_result.average_memory_mb * 1024 * 1024)}"
    )
    report.append(
        f"Minimum Memory Usage: {format_memory(profile_result.min_memory_mb * 1024 * 1024)}"
    )

    if hasattr(profile_result, "duration"):
        report.append(f"Profiling Duration: {profile_result.duration:.2f} seconds")

    if hasattr(profile_result, "memory_growth_rate"):
        report.append(
            f"Memory Growth Rate: {profile_result.memory_growth_rate:.2f} MB/s"
        )

    report.append("")

    # Memory allocations
    report.append(f"Total Memory Allocations: {profile_result.total_allocations}")
    report.append(f"Total Memory Deallocations: {profile_result.total_deallocations}")
    report.append("")

    # Function profiling
    if (
        hasattr(profile_result, "function_profiles")
        and profile_result.function_profiles
    ):
        report.append("Function Profiling:")
        report.append("-" * 20)
        for func_name, profile in profile_result.function_profiles.items():
            report.append(f"  {func_name}:")
            report.append(f"    Calls: {profile.get('calls', 0)}")
            report.append(
                f"    Peak Memory: {format_memory(profile.get('peak_memory', 0) * 1024 * 1024)}"
            )
            report.append(
                f"    Total Duration: {profile.get('total_duration', 0):.3f}s"
            )
            report.append("")

    # Fragmentation analysis
    if hasattr(profile_result, "snapshots") and len(profile_result.snapshots) > 1:
        frag_info = analyze_fragmentation(profile_result.snapshots)
        report.append("Memory Fragmentation Analysis:")
        report.append("-" * 30)
        report.append(f"  Fragmentation Score: {frag_info['fragmentation_score']:.3f}")
        report.append(f"  Fragmentation Trend: {frag_info['fragmentation_trend']:.3f}")
        report.append("")

    # Tensor lifecycle
    if hasattr(profile_result, "tensor_lifecycle") and profile_result.tensor_lifecycle:
        active_tensors = profile_result.tensor_lifecycle.get("active", {})
        if active_tensors:
            report.append("Tensor Information:")
            report.append("-" * 18)
            report.append(f"  Active Tensors: {active_tensors.get('count', 0)}")
            report.append(
                f"  Total Tensor Memory: {format_memory(active_tensors.get('total_size_mb', 0) * 1024 * 1024)}"
            )
            report.append("")

    # Optimization suggestions
    suggestions = suggest_optimizations(profile_result)
    if suggestions:
        report.append("Optimization Suggestions:")
        report.append("-" * 25)
        for i, suggestion in enumerate(suggestions, 1):
            report.append(f"  {i}. {suggestion}")
        report.append("")

    # System information
    system_info = get_system_info()
    report.append("System Information:")
    report.append("-" * 19)
    report.append(f"  TensorFlow Version: {system_info['tensorflow_version']}")
    report.append(f"  Python Version: {system_info['python_version']}")
    report.append(f"  Platform: {system_info['platform']}")
    if system_info["gpu"]["available"]:
        report.append(f"  GPU Count: {system_info['gpu']['count']}")
        report.append(
            f"  Total GPU Memory: {format_memory(system_info['gpu']['total_memory'] * 1024 * 1024)}"
        )
    else:
        report.append("  GPU: Not available")

    return "\n".join(report)


def optimize_tensorflow_memory() -> List[str]:
    """Apply TensorFlow memory optimizations."""
    if not TF_AVAILABLE:
        logging.warning("TensorFlow not available for memory optimization")
        return []

    optimizations_applied = []

    try:
        # Enable memory growth for all GPUs
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                optimizations_applied.append(f"Enabled memory growth for {gpu}")

        # Set mixed precision policy
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            optimizations_applied.append("Enabled mixed precision training")
        except Exception as exc:
            logging.debug("Mixed precision not applied: %s", exc)

        # Enable XLA compilation
        try:
            tf.config.optimizer.set_jit(True)
            optimizations_applied.append("Enabled XLA compilation")
        except Exception as exc:
            logging.debug("XLA compilation not enabled: %s", exc)

        logging.info(f"Applied TensorFlow optimizations: {optimizations_applied}")

    except Exception as e:
        logging.warning(f"Could not apply all TensorFlow optimizations: {e}")

    return optimizations_applied


def get_tensorflow_memory_usage() -> Dict[str, float]:
    """Get current TensorFlow memory usage."""
    memory_info = {"gpu_current_mb": 0.0, "gpu_peak_mb": 0.0, "cpu_mb": 0.0}

    if not TF_AVAILABLE:
        return memory_info

    try:
        # GPU memory
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            gpu_memory = tf.config.experimental.get_memory_info("/GPU:0")
            memory_info["gpu_current_mb"] = gpu_memory.get("current", 0) / (1024 * 1024)
            memory_info["gpu_peak_mb"] = gpu_memory.get("peak", 0) / (1024 * 1024)

        # CPU memory
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            memory_info["cpu_mb"] = process.memory_info().rss / (1024 * 1024)

    except Exception as e:
        logging.warning(f"Could not get TensorFlow memory usage: {e}")

    return memory_info


def clear_tensorflow_session() -> None:
    """Clear TensorFlow session and free memory."""
    if not TF_AVAILABLE:
        return

    try:
        tf.keras.backend.clear_session()
        logging.info("Cleared TensorFlow session")
    except Exception as e:
        logging.warning(f"Could not clear TensorFlow session: {e}")


def validate_tensorflow_environment() -> Dict[str, Any]:
    """Validate TensorFlow environment for memory profiling."""
    issues: List[str] = []
    validation: Dict[str, Any] = {
        "tensorflow_available": TF_AVAILABLE,
        "gpu_available": False,
        "memory_growth_enabled": False,
        "version_compatible": False,
        "issues": issues,
    }

    if not TF_AVAILABLE:
        issues.append("TensorFlow not installed")
        return validation

    # Check TensorFlow version
    try:
        version = tf.__version__
        major, minor = map(int, version.split(".")[:2])
        if major >= 2 and minor >= 4:
            validation["version_compatible"] = True
        else:
            issues.append(
                f"TensorFlow {version} may not be fully compatible (recommend 2.4+)"
            )
    except Exception as exc:
        logging.debug("TF version check failed: %s", exc)
        issues.append("Could not determine TensorFlow version")

    # Check GPU availability
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            validation["gpu_available"] = True

            # Check memory growth
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                validation["memory_growth_enabled"] = True
            except Exception as exc:
                logging.debug("GPU memory growth config failed: %s", exc)
                issues.append("Could not enable GPU memory growth")
        else:
            issues.append("No GPU devices found")
    except Exception as e:
        issues.append(f"Error checking GPU availability: {e}")

    return validation
