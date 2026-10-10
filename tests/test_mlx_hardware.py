"""Required Metal gate: select explicitly; unavailable Metal fails, never skips."""

import gc
import time

import pytest

pytestmark = pytest.mark.mlx_hardware


def test_native_snapshot_activity_cache_and_lazy_completion():
    import mlx.core as mx

    from stormlog.mlx import MLXMemoryProfiler

    profiler = MLXMemoryProfiler()
    # Explicit test harness cleanup, never collector behavior.
    mx.synchronize(mx.gpu)
    mx.clear_cache()
    baseline = profiler.capture_snapshot("baseline")
    x = mx.ones((256, 256), stream=mx.gpu)
    mx.eval(x)
    mx.synchronize(mx.gpu)
    allocated = profiler.capture_snapshot("allocated")
    assert allocated.active_bytes == mx.get_active_memory()
    assert allocated.cache_bytes == mx.get_cache_memory()
    assert allocated.active_bytes > baseline.active_bytes
    output = profiler.profile_function(lambda: x @ x.T, name="matmul")
    result = profiler.get_results()[-1]
    active_before_read = mx.get_active_memory()
    assert output[0, 0].item() == 256
    assert result.completion_verified
    assert mx.get_active_memory() == active_before_read
    assert result.sampled_peak_bytes >= baseline.active_bytes
    assert result.final.runtime_peak_bytes >= result.sampled_peak_bytes
    assert result.elapsed_ns > 0
    del output
    x = None
    gc.collect()
    mx.synchronize(mx.gpu)
    released = profiler.capture_snapshot("released")
    assert released.active_bytes < allocated.active_bytes
    assert released.cache_bytes > 0
    mx.clear_cache()
    assert profiler.capture_snapshot("cleared").cache_bytes == 0


def test_native_state_roots_and_declared_multistream_scope():
    import mlx.core as mx

    from stormlog.mlx import MLXMemoryProfiler

    streams = (mx.new_stream(mx.gpu), mx.new_stream(mx.gpu))
    profiler = MLXMemoryProfiler()
    x = mx.ones((128, 128), stream=mx.gpu)
    mx.eval(x)
    state = {}

    def workload():
        with mx.stream(streams[0]):
            output = x @ x
        with mx.stream(streams[1]):
            state["value"] = x + 2
        return {"output": output, "ordinary": "ok"}

    output = profiler.profile_function(
        workload, state_getter=lambda: state, streams=streams
    )
    assert output["output"][0, 0].item() == 128
    assert state["value"][0, 0].item() == 3
    result = profiler.get_results()[-1]
    assert result.metadata["stream_scope"] == [str(s) for s in streams]
    assert result.metadata["evaluation_count"] == 2
    assert result.completion_verified


def test_native_tracker_sink_session_roundtrip(tmp_path):
    import mlx.core as mx

    from stormlog.mlx import MemoryTracker
    from stormlog.telemetry import load_telemetry_sessions
    from stormlog.telemetry_sink import TelemetrySinkConfig

    tracker = MemoryTracker(
        sampling_interval=0.005,
        max_history=3,
        telemetry_sink_config=TelemetrySinkConfig(tmp_path, write_rollups=False),
    )
    tracker.start_tracking()
    try:
        x = mx.ones((128, 128), stream=mx.gpu)
        mx.eval(x)
        with tracker.phase("matrix"):
            for _ in range(20):
                y = x @ x.T
                mx.eval(y)
                time.sleep(0.002)
    finally:
        result = tracker.stop_tracking()
    assert result.valid_samples >= 3
    assert result.total_samples > len(result.samples)
    assert result.session_summary.status == "completed"
    sessions = load_telemetry_sessions(tmp_path)
    assert len(sessions) == 1
    assert len(sessions[0].events) == result.total_events
    assert all(e.device_total_bytes is None for e in sessions[0].events)


def test_native_compiled_callable_is_profiled_outside_transform():
    import mlx.core as mx

    from stormlog.mlx import MLXMemoryProfiler

    @mx.compile
    def matmul(x):
        return x @ x.T

    x = mx.ones((128, 128), stream=mx.gpu)
    mx.eval(x, matmul(x))  # compilation and warmup outside the measurement
    profiler = MLXMemoryProfiler()
    result = profiler.profile_function(matmul, x, name="compiled_matmul")
    assert result[0, 0].item() == 128
    assert profiler.get_results()[0].completion_verified
