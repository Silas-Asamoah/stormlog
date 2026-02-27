"""Minimal PyTorch demo showing the key profiler workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment, unused-ignore]
    nn = None  # type: ignore[assignment, unused-ignore]

from examples.common import (
    build_simple_torch_model,
    describe_torch_environment,
    generate_torch_batch,
    get_torch_device,
    print_header,
    print_kv,
    print_profiler_summary,
    print_section,
    run_torch_train_step,
    seed_everything,
)
from gpumemprof import GPUMemoryProfiler


def _allocate_tensor_mb(size_mb: int, device: torch.device) -> torch.Tensor:
    elements = int(size_mb * 1024 * 1024 / 4)
    rows = max(1, elements // 1024)
    return torch.randn(rows, 1024, device=device)


def profile_tensor_allocation(profiler: GPUMemoryProfiler, repeats: int = 3) -> None:
    device = get_torch_device()

    for idx in range(repeats):
        size_mb = 32 * (idx + 1)

        def allocate(
            sz: int = size_mb, dev: "torch.device" = device
        ) -> float:  # capture via default args
            tensor = _allocate_tensor_mb(sz, dev)
            return float(tensor.mean().item())

        allocate.__name__ = f"tensor_alloc_{size_mb}mb"
        profiler.profile_function(allocate)


def profile_training_epoch(
    profiler: GPUMemoryProfiler, model: nn.Module, epochs: int = 2
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        with profiler.profile_context(f"epoch_{epoch+1}"):
            loss_value = run_torch_train_step(model, optimizer, criterion)
            print_kv(f"Epoch {epoch+1} loss", f"{loss_value:.4f}")


def profile_inference_context(profiler: GPUMemoryProfiler, model: nn.Module) -> None:
    model.eval()
    with torch.no_grad():
        inputs, _ = generate_torch_batch(batch_size=64)
        with profiler.profile_context("inference_pass"):
            logits = model(inputs)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * probs.log()).sum(dim=-1).mean()
            print_kv("Mean entropy", f"{entropy.item():.4f}")


def display_environment() -> None:
    info = describe_torch_environment()
    print_section("Environment")
    for key, value in info.items():
        if key.endswith("memory_total"):
            human_value = f"{value / (1024**3):.2f} GB"
            print_kv(key, human_value)
        else:
            print_kv(key, value)


def main() -> None:
    seed_everything()
    print_header("Stormlog - PyTorch Demo")

    if torch is None or nn is None:
        print("PyTorch is not installed. Skipping PyTorch demo.")  # type: ignore[unreachable, unused-ignore]
        return

    display_environment()

    if not torch.cuda.is_available():
        print("CUDA is not available on this machine. Skipping GPU demo.")
        return

    profiler = GPUMemoryProfiler(track_tensors=True)

    print_section("Tensor Allocation Profiling")
    profile_tensor_allocation(profiler)

    print_section("Training Loop Profiling")
    model = build_simple_torch_model()
    profile_training_epoch(profiler, model)

    print_section("Inference Profiling")
    profile_inference_context(profiler, model)

    summary = profiler.get_summary()
    print_profiler_summary(summary)


if __name__ == "__main__":
    main()
