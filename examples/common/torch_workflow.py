"""Reusable PyTorch helpers for examples."""

from __future__ import annotations

from typing import Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment, unused-ignore]
    nn = None  # type: ignore[assignment, unused-ignore]
    F = None  # type: ignore[assignment, unused-ignore]

from .device import get_torch_device, seed_everything


def build_simple_torch_model(
    input_size: int = 1024,
    hidden_size: int = 512,
    num_layers: int = 3,
    num_classes: int = 10,
) -> "nn.Module":
    """Construct a small fully connected network used by demos."""
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for this example.")

    layers = [nn.Linear(input_size, hidden_size)]
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_size, hidden_size))
    layers.append(nn.Linear(hidden_size, num_classes))

    class SimpleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(layers)
            self.dropout = nn.Dropout(0.1)

        def forward(self, x: object) -> object:
            for layer in self.layers[:-1]:
                x = self.dropout(F.relu(layer(x)))
            return self.layers[-1](x)

    model = SimpleModel()
    model.to(get_torch_device())
    return model


def generate_torch_batch(
    batch_size: int = 256,
    input_size: int = 1024,
    num_classes: int = 10,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Create a synthetic classification batch on the resolved device."""
    if torch is None:
        raise RuntimeError("PyTorch is required for this example.")

    device = get_torch_device()
    inputs = torch.randn(batch_size, input_size, device=device)
    targets = torch.randint(0, num_classes, (batch_size,), device=device)
    return inputs, targets


def run_torch_train_step(
    model: "nn.Module",
    optimizer: "torch.optim.Optimizer",
    criterion: "nn.Module",
) -> float:
    """Run a single training step and return the scalar loss."""
    if torch is None:
        raise RuntimeError("PyTorch is required for this example.")

    seed_everything()  # keep batches deterministic for demos
    inputs, targets = generate_torch_batch()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())
