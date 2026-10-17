import asyncio
import subprocess
import sys
from unittest.mock import Mock

import pytest

pytest.importorskip("textual")

from stormlog.mlx.analyzer import analyze
from stormlog.mlx.context_profiler import (
    clear_global_profiler,
    get_profile_summaries,
    set_global_profiler,
)
from stormlog.mlx.profile_artifact import write_profiles
from stormlog.mlx.profiler import MLXMemoryProfiler
from stormlog.tui.app import GPUMemoryProfilerTUI
from stormlog.tui.builders import build_mlx_stats_rows
from stormlog.tui.monitor import TrackerSession, TrackerUnavailableError
from stormlog.tui.profiles import clear_mlx_profiles, fetch_mlx_profiles
from stormlog.tui.widgets.timeline import TimelineCanvas
from tests.mlx_fakes import make_runtime


def test_explicit_mlx_session_passive_counters_nulls_and_export(tmp_path):
    runtime = make_runtime()
    session = TrackerSession(framework="mlx", sampling_interval=10)
    session.start(runtime=runtime)
    try:
        assert session.backend == "mlx"
        assert session.get_session_summary().pid
        assert session.get_device_label() == "MLX Metal allocator (this process)"
        assert session.get_statistics()["cache_bytes"] == 20
        events = session.pull_events()
        assert events[0].reserved == "N/A"
        assert session.get_telemetry_events()[0].device_total_bytes is None
        assert session.get_memory_timeline()["reserved"] == [None]
        rows = build_mlx_stats_rows(session.get_statistics())
        assert any(label == "Reusable cache" for label, _ in rows)
        assert not any(
            "Utilization" in label or "Fragmentation" in label for label, _ in rows
        )
        assert not session.force_cleanup()
        assert session.export_events(str(tmp_path / "events.json"), format="json")
        exported = analyze(tmp_path / "events.json")
        assert (
            exported["session"]["session_id"]
            == session.get_session_summary().session_id
        )
        assert exported["metrics"]["sampled_peak_bytes"] == 100
        assert session.export_events(str(tmp_path / "events.csv"), format="csv")
        session.clear_events()
        assert not session.pull_events()
        assert session.get_statistics()["total_samples"] == 1
    finally:
        session.stop()
    assert "eval" not in runtime.core.calls and "reset" not in runtime.core.calls


def test_unavailable_mlx_does_not_fall_back_to_cpu(monkeypatch):
    from stormlog.mlx.runtime import MLXMetalUnavailableError

    def fail(*args, **kwargs):
        raise MLXMetalUnavailableError("Metal unavailable")

    monkeypatch.setattr("stormlog.tui.mlx_adapter.MemoryTracker", fail)
    session = TrackerSession(framework="mlx")
    with pytest.raises(TrackerUnavailableError, match="Metal unavailable"):
        session.start()
    assert not session.is_active


def test_profile_adapter_no_runtime_for_empty_or_saved_profiles(tmp_path):
    clear_global_profiler()
    assert get_profile_summaries() == []
    assert fetch_mlx_profiles() == []
    profiler = MLXMemoryProfiler(runtime=make_runtime())
    with profiler.profile_context("unverified"):
        pass
    profile = tmp_path / "profile.json"
    write_profiles(profile, profiler.get_results())
    assert fetch_mlx_profiles(path=str(profile))[0].completion_verified is False
    set_global_profiler(profiler)
    assert fetch_mlx_profiles()[0].peak_mb == 100 / 1024**2
    assert clear_mlx_profiles()
    assert not fetch_mlx_profiles()
    clear_global_profiler()


def test_mlx_timeline_retains_gaps_and_distinct_cache_series(monkeypatch):
    canvas = TimelineCanvas()
    update = Mock()
    monkeypatch.setattr(canvas, "update", update)
    canvas.render_timeline(
        {
            "framework": "mlx",
            "allocated": [0, None, 1024**2],
            "cache": [None, 0, 2048**2],
        }
    )
    text = update.call_args.args[0]
    assert "Active allocator" in text and "Reusable cache" in text
    assert "?" in text and "Reserved" not in text
    _, active, cache = GPUMemoryProfilerTUI._timeline_plot_data(
        {
            "framework": "mlx",
            "timestamps": [1, 2, 3],
            "allocated": [0, None, 1024**3],
            "cache": [2 * 1024**3, None, 0],
        }
    )
    assert active == [0, None, 1] and cache == [2, None, 0]


def test_mlx_app_actions_tracking_and_cleanup(tmp_path, monkeypatch):
    async def scenario():
        monkeypatch.setattr("stormlog.mlx.tracker.MLXRuntime", make_runtime)
        app = GPUMemoryProfilerTUI(framework="mlx")
        async with app.run_test(size=(160, 55)):
            await app.start_live_tracking()
            assert app.tracker_session.is_active
            app.refresh_monitoring_panel()
            assert app.watchdog_button.disabled
            assert app.tracker_session.backend == "mlx"
            app.stop_live_tracking()
            await app.refresh_mlx_profiles()

    asyncio.run(scenario())


def test_mlx_only_startup_and_saved_artifact_browsing_block_all_frameworks(tmp_path):
    from tests.test_mlx_analysis import write_events

    events = tmp_path / "events.json"
    write_events(events, [0, None, 1024])
    code = """
import asyncio, importlib.abc, sys
attempts = []
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx','torch','jax','tensorflow'}:
            attempts.append(fullname)
            raise AssertionError('unexpected framework discovery: '+fullname)
sys.meta_path.insert(0, Block())
from stormlog.tui.app import GPUMemoryProfilerTUI
from textual.widgets import Input
async def check():
    app = GPUMemoryProfilerTUI(framework='mlx')
    async with app.run_test(size=(140,50)):
        app.query_one('#mlx-artifact-path', Input).value = sys.argv[1]
        await app.load_mlx_artifact()
        assert app._last_timeline['allocated'] == [0,None,1024]
        await app.refresh_mlx_profiles()
        await app.clear_mlx_profiles()
    assert not attempts, attempts
asyncio.run(check())
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(events)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
