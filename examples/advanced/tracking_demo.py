"""Advanced GPU memory tracking demo with MemoryTracker + Watchdog."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch

    from gpumemprof.tracker import MemoryTracker, MemoryWatchdog

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment, unused-ignore]
    F = None  # type: ignore[assignment, unused-ignore]

from examples.common import (
    describe_torch_environment,
    print_header,
    print_kv,
    print_section,
    seed_everything,
)
from gpumemprof import get_gpu_info

LEAK_BUCKET: List[object] = []


def alert_handler(event: object) -> None:
    timestamp = time.strftime(
        "%H:%M:%S", time.localtime(getattr(event, "timestamp", 0))
    )
    event_type = str(getattr(event, "event_type", "")).upper()
    context = getattr(event, "context", "")
    print(f"\n⚠️  [{timestamp}] {event_type}: {context}")
    for key, value in (getattr(event, "metadata", None) or {}).items():
        print(f"    {key}: {value}")


def setup_tracker() -> tuple[MemoryTracker, MemoryWatchdog]:
    from gpumemprof.tracker import MemoryTracker, MemoryWatchdog

    tracker = MemoryTracker(
        sampling_interval=0.2,
        max_events=10_000,
        enable_alerts=True,
    )
    tracker.set_threshold("memory_warning_percent", 65.0)
    tracker.set_threshold("memory_critical_percent", 80.0)
    tracker.set_threshold("memory_leak_threshold", 25 * 1024 * 1024)
    tracker.add_alert_callback(alert_handler)

    watchdog = MemoryWatchdog(
        tracker=tracker,
        auto_cleanup=True,
        cleanup_threshold=0.75,
        aggressive_cleanup_threshold=0.9,
    )
    return tracker, watchdog


def _allocate_leaky_tensor(step: int, device: torch.device) -> torch.Tensor:
    size_mb = 16 + (step % 3) * 8
    elements = int(size_mb * 1024 * 1024 / 4)
    tensor = torch.randn(elements, device=device)
    LEAK_BUCKET.append(tensor)
    if len(LEAK_BUCKET) > 5:
        LEAK_BUCKET.pop(0)
    return tensor


def run_monitored_workload(duration: float = 20.0) -> None:
    tracker, watchdog = setup_tracker()
    tracker.start_tracking()

    device = torch.device("cuda")
    start_time = time.time()
    step = 0
    print_section("Simulating workload")
    print("Running memory churn for ~20 seconds...")

    try:
        while time.time() - start_time < duration:
            _allocate_leaky_tensor(step, device)

            if step % 3 == 0:
                tmp = torch.randn(2_000_000, device=device) * 0.01
                F.relu_(tmp)

            if step % 5 == 0:
                watchdog.force_cleanup()

            time.sleep(0.2)
            step += 1
    finally:
        tracker.stop_tracking()
        summarize_results(tracker, watchdog)
        export_results(tracker)


def summarize_results(tracker: MemoryTracker, watchdog: MemoryWatchdog) -> None:
    stats = tracker.get_statistics()
    cleanup_stats = watchdog.get_cleanup_stats()

    print_section("Tracker summary")
    duration = stats.get("tracking_duration_seconds", 0.0)
    print_kv("Tracking duration (s)", f"{duration:.1f}")
    print_kv("Total events", stats.get("total_events", 0))
    print_kv("Peak memory (GB)", f"{stats.get('peak_memory', 0) / (1024**3):.2f}")
    print_kv("Alerts emitted", stats.get("alert_count", 0))
    print_kv("Watchdog cleanups", cleanup_stats.get("cleanup_count", 0))


def export_results(tracker: MemoryTracker) -> None:
    output_dir = Path("artifacts/advanced_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tracking_events.csv"
    json_path = output_dir / "tracking_events.json"

    tracker.export_events(str(csv_path), format="csv")
    tracker.export_events(str(json_path), format="json")

    timeline = tracker.get_memory_timeline(interval=0.5)
    if timeline["timestamps"]:
        try:
            import matplotlib.pyplot as plt
        except (ImportError, ModuleNotFoundError) as e:
            raise ImportError(
                "matplotlib is required for plotting in this example. "
                "Install it with: pip install matplotlib"
            ) from e

        times = [t - timeline["timestamps"][0] for t in timeline["timestamps"]]
        allocated = [value / (1024**3) for value in timeline["allocated"]]
        plt.figure(figsize=(10, 4))
        plt.plot(times, allocated, label="Allocated GB", linewidth=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Allocated memory (GB)")
        plt.title("GPU memory usage over time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = output_dir / "memory_timeline.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print_kv("Timeline plot", plot_path)

    print_kv("Events CSV", csv_path)
    print_kv("Events JSON", json_path)


def main() -> None:
    seed_everything()
    print_header("Stormlog - Advanced Tracking Demo")

    if torch is None or F is None:
        print("PyTorch is not installed. Skipping advanced tracking demo.")  # type: ignore[unreachable, unused-ignore]
        return

    if not torch.cuda.is_available():
        print("CUDA is not available. This demo requires a GPU.")
        return

    env = describe_torch_environment()
    print_section("Environment")
    for key, value in env.items():
        print_kv(key, value)

    initial_state = get_gpu_info()
    print_section("Initial GPU state")
    print_kv("Allocated (GB)", f"{initial_state['allocated_memory'] / (1024**3):.2f}")
    print_kv("Reserved (GB)", f"{initial_state['reserved_memory'] / (1024**3):.2f}")

    run_monitored_workload()

    final_state = get_gpu_info()
    print_section("Final GPU state")
    print_kv("Allocated (GB)", f"{final_state['allocated_memory'] / (1024**3):.2f}")
    print_kv("Reserved (GB)", f"{final_state['reserved_memory'] / (1024**3):.2f}")
    delta = (final_state["allocated_memory"] - initial_state["allocated_memory"]) / (
        1024**3
    )
    print_kv("Delta from start (GB)", f"{delta:+.2f}")


if __name__ == "__main__":
    main()
