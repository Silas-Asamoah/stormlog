"""Bounded in-process PyTorch profiler capture."""

from __future__ import annotations

from pathlib import Path

import pytest

from stormlog.infer.correlation_events import EntityRef
from stormlog.infer.trace_kineto import load_kineto_trace
from stormlog.infer.trace_ranges import iteration_range
from stormlog.infer.trace_torch import ProfilerBusyError, capture_torch_trace

torch = pytest.importorskip("torch")


def test_capture_writes_a_trace_with_iteration_ranges(tmp_path: Path) -> None:
    with capture_torch_trace(
        tmp_path / "out" / "cpu.pt.trace.json", cuda=False
    ) as path:
        for step in range(3):
            with iteration_range("loop", str(step)):
                torch.ones(4) * 2

    trace = load_kineto_trace(path)
    refs = [span.iteration_ref for spans in trace.spans.values() for span in spans]
    assert refs == [EntityRef("loop", str(step)) for step in range(3)]


def test_capture_writes_the_trace_when_the_block_raises(tmp_path: Path) -> None:
    path = tmp_path / "failed.pt.trace.json"
    with pytest.raises(RuntimeError, match="boom"):
        with capture_torch_trace(path, cuda=False):
            torch.ones(2)
            raise RuntimeError("boom")

    assert load_kineto_trace(path).spans == {}


def test_capture_refuses_to_take_over_a_running_profiler(tmp_path: Path) -> None:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU]):
        with pytest.raises(ProfilerBusyError):
            with capture_torch_trace(tmp_path / "x.json", cuda=False):
                pass


def test_a_failed_export_does_not_hide_the_blocks_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from torch.profiler import profile

    def broken_export(_self: object, _path: str) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(profile, "export_chrome_trace", broken_export)
    with pytest.raises(RuntimeError, match="boom"):
        with capture_torch_trace(tmp_path / "x.json", cuda=False):
            raise RuntimeError("boom")
