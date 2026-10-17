import json
from dataclasses import replace

import pytest

from stormlog.mlx.analyzer import analyze
from stormlog.mlx.collector import MLXCollector
from stormlog.mlx.profile_artifact import write_profiles
from stormlog.mlx.profiler import MLXMemoryProfiler
from stormlog.mlx.tracker import MemoryTracker
from stormlog.session import create_session_summary
from stormlog.telemetry import load_telemetry_sessions
from tests.mlx_fakes import FakeCore, make_runtime


def write_events(path, values, *, session=None, timestamps=None):
    session = session or create_session_summary(source="test")
    collector = MLXCollector(make_runtime())
    previous = None
    records = []
    for i, value in enumerate(values):
        snapshot = replace(
            collector.capture_snapshot(),
            active_bytes=value,
            timestamp_ns=(timestamps[i] if timestamps else (i + 1) * 100_000_000),
        )
        records.append(
            collector.telemetry_record(
                snapshot,
                session,
                sampling_interval_ms=100,
                previous_active_bytes=previous,
            )
        )
        previous = value
    path.write_text(json.dumps(records))
    return records


def test_nullable_metrics_growth_windows_and_cache(tmp_path):
    source = tmp_path / "events.json"
    write_events(source, [0, 200, None, 1000, 1000, 1200])
    result = analyze(source)
    assert result["metrics"]["sampled_peak_bytes"] == 1200
    assert result["metrics"]["cache_peak_bytes"] == 20
    assert result["metrics"]["runtime_lifetime_peak_bytes"] == 1000
    assert result["metrics"]["unavailable_samples"] == 1
    for key in (
        "device_total_bytes",
        "device_free_bytes",
        "utilization_ratio",
        "fragmentation_ratio",
    ):
        assert result["metrics"][key] is None
    assert result["timeline"][2]["active_bytes"] is None
    assert [w["growth_bytes"] for w in result["growth_windows"]] == [200, 200]
    assert all("leak" not in f["kind"] for f in result["findings"])


def test_no_growth_across_missing_time_or_all_unavailable(tmp_path):
    source = tmp_path / "events.json"
    write_events(
        source,
        [10, 20, 1000, 1000],
        timestamps=[1, 100_000_001, 9_000_000_001, 9_100_000_001],
    )
    assert [w["growth_bytes"] for w in analyze(source)["growth_windows"]] == [10, 0]
    write_events(source, [None, None])
    result = analyze(source)
    assert result["metrics"]["sampled_peak_bytes"] is None
    assert result["growth_windows"] == []


def test_sessions_selected_without_merging(tmp_path):
    source = tmp_path / "events.json"
    a = create_session_summary(source="a")
    b = create_session_summary(source="b")
    first = write_events(source, [1, 2], session=a)
    second = write_events(
        source, [1000, 1001], session=b, timestamps=[1_000_000_000, 1_100_000_000]
    )
    source.write_text(json.dumps(first + second))
    assert (
        analyze(source, session_id=a.session_id)["metrics"]["sampled_peak_bytes"] == 2
    )
    assert analyze(source)["session"]["session_id"] == b.session_id
    with pytest.raises(ValueError, match="session_id not found"):
        analyze(source, session_id="missing")


def test_profile_analysis_preserves_scope_and_completion(tmp_path):
    profiler = MLXMemoryProfiler(runtime=make_runtime())
    with profiler.profile_context("host"):
        pass
    source = tmp_path / "profile.json"
    write_profiles(source, profiler.get_results())
    result = analyze(source)
    assert result["profiles"][0]["completion_verified"] is False
    assert result["findings"][0]["kind"] == "incomplete_compute"


def test_query_sessions_events_and_unknown_metrics(tmp_path):
    from stormlog import query

    source = tmp_path / "events.json"
    write_events(source, [0, 200, None])
    store = query.open([source])
    assert len(store.list_sessions()) == 1
    events = store.query_events()
    assert len(events) == 3
    assert events[2].event.allocator_allocated_bytes is None
    assert all(e.event.device_total_bytes is None for e in events)


@pytest.mark.parametrize("status", ["completed", "interrupted", "incomplete"])
@pytest.mark.parametrize("max_history", [1, 10])
def test_tracker_json_export_round_trips_through_offline_readers(
    tmp_path, status, max_history
):
    from stormlog import query

    core = FakeCore()
    tracker = MemoryTracker(
        runtime=make_runtime(core), sampling_interval=10, max_history=max_history
    )
    tracker.start_tracking()
    core.values["active_memory"] = RuntimeError("unavailable")
    tracker._sample()
    core.values["active_memory"] = 0
    tracker._sample()
    result = tracker.stop_tracking(status=status)
    source = tmp_path / "tracking.json"
    tracker.export(source)

    sessions = load_telemetry_sessions(source, permissive_legacy=False)
    assert len(sessions) == 1
    session = sessions[0]
    assert session.summary.session_id == result.session_summary.session_id
    assert session.summary.status == status
    assert [event.allocator_allocated_bytes for event in session.events] == [
        row["allocator_allocated_bytes"] for row in result.telemetry_events
    ]
    report = analyze(source)
    assert report["metrics"]["sampled_peak_bytes"] == (0 if max_history == 1 else 100)
    assert report["metrics"]["unavailable_samples"] == (0 if max_history == 1 else 1)
    assert report["timeline"][-1]["active_bytes"] == 0
    assert len(query.open([source]).query_events()) == len(result.telemetry_events)

    payload = json.loads(source.read_text())
    payload["session_summary"]["session_id"] = "another-session"
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="summary does not match"):
        load_telemetry_sessions(source, permissive_legacy=False)
