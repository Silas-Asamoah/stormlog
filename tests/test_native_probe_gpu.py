"""Opt-in real NVIDIA initialization and Chrome trace smoke test."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from research.native_probes.runner import _is_device_activity_event


@pytest.mark.skipif(
    os.environ.get("STORMLOG_RUN_NVIDIA_SMOKE") != "1",
    reason="set STORMLOG_RUN_NVIDIA_SMOKE=1 to require a usable NVIDIA runtime",
)
def test_nvidia_initialization_and_chrome_trace(tmp_path: Path) -> None:
    try:
        import torch
    except ImportError:
        pytest.fail("NVIDIA smoke selected but PyTorch is not installed")
    if not torch.cuda.is_available():
        pytest.fail("NVIDIA smoke selected but CUDA is unavailable")

    device = torch.cuda.current_device()
    torch.cuda.init()
    value = torch.ones((32, 32), device="cuda")
    (value @ value).sum().item()
    torch.cuda.synchronize(device)

    trace_path = tmp_path / "nvidia-eager-trace.json"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        (value @ value).sum().item()
        torch.cuda.synchronize(device)
    profile.export_chrome_trace(str(trace_path))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    events = trace.get("traceEvents")
    usable_device_events = [
        event for event in events or [] if _is_device_activity_event(event)
    ]
    if not usable_device_events:
        pytest.fail("NVIDIA smoke produced no timed GPU device kernel events")
    print(
        "Real NVIDIA initialization and Chrome trace passed; full issue #118 "
        "adoption evidence remains pending."
    )
