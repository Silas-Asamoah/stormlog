"""Bounded in-process PyTorch profiler capture."""

from __future__ import annotations

import subprocess
import sys
import textwrap
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


def _run_isolated(script: str) -> list[str]:
    """Two Kineto profilers in one process crash it when the first one stops,
    so these checks run in a subprocess: a regression fails the test instead of
    crashing the test run."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.split()


_HOLD_A_PROFILER = """
    import threading
    import torch
    from torch.profiler import ProfilerActivity, profile
    from stormlog.infer.trace_torch import ProfilerBusyError, capture_torch_trace

    started, done = threading.Event(), threading.Event()

    def hold(first):
        with first():
            torch.ones(2)
            started.set()
            done.wait(10)

    def attempt(path):
        try:
            with capture_torch_trace(path, cuda=False):
                print("entered")
        except ProfilerBusyError:
            print("refused")
        finally:
            done.set()
"""


def test_capture_refuses_a_profiler_running_on_another_thread(tmp_path: Path) -> None:
    out = _run_isolated(
        _HOLD_A_PROFILER
        + f"""
    thread = threading.Thread(
        target=hold, args=(lambda: profile(activities=[ProfilerActivity.CPU]),)
    )
    thread.start()
    started.wait(10)
    attempt({str(tmp_path / "x.json")!r})
    thread.join(10)
    """
    )

    assert out == ["refused"]


def test_one_capture_at_a_time_per_process(tmp_path: Path) -> None:
    out = _run_isolated(
        _HOLD_A_PROFILER
        + f"""
    first = {str(tmp_path / "first.json")!r}
    thread = threading.Thread(
        target=hold, args=(lambda: capture_torch_trace(first, cuda=False),)
    )
    thread.start()
    started.wait(10)
    attempt({str(tmp_path / "second.json")!r})
    thread.join(10)
    with capture_torch_trace({str(tmp_path / "third.json")!r}, cuda=False):
        print("third")
    """
    )

    assert out == ["refused", "third"]


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
