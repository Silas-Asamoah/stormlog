import gc
import weakref

import pytest

from stormlog.mlx import MLXMemoryProfiler
from stormlog.mlx.profile_artifact import load_profiles
from tests.mlx_fakes import FakeCore, make_runtime


def test_transparent_output_state_and_order(tmp_path):
    core = FakeCore()
    profiler = MLXMemoryProfiler(runtime=make_runtime(core))
    output = {"lazy": object(), "ordinary": 3}
    state = [object()]
    count = []

    def workload():
        count.append(1)
        return output

    assert profiler.profile_function(workload, state_getter=lambda: state) is output
    assert len(count) == 1
    assert core.outputs == [(id(output),), (id(state),)]
    first_sync = core.calls.index(("sync", "default"))
    assert first_sync < core.calls.index("active_memory") < core.calls.index("eval")
    assert core.calls.index("eval") < len(core.calls) - 1
    assert "reset" not in core.calls
    result = profiler.get_results()[0]
    assert result.name == "workload" and result.completion_verified
    assert result.elapsed_ns > 0
    assert result.sampled_peak_bytes == 500
    assert result.final.runtime_peak_bytes == 1000
    profiler.export(str(tmp_path / "profile.json"))
    assert load_profiles(tmp_path / "profile.json")["profiles"][0]["name"] == "workload"


def test_does_not_retain_outputs():
    class Output:
        pass

    profiler = MLXMemoryProfiler(runtime=make_runtime())
    output = profiler.profile_function(Output)
    reference = weakref.ref(output)
    del output
    gc.collect()
    assert reference() is None


def test_context_completion_and_declared_streams():
    core = FakeCore()
    profiler = MLXMemoryProfiler(runtime=make_runtime(core))
    with profiler.profile_context("unevaluated", streams=("a", "b")):
        pass
    assert not profiler.get_results()[-1].completion_verified
    with profiler.profile_context("done") as region:
        region.evaluate({"loss": object()})
    assert profiler.get_results()[-1].completion_verified
    with pytest.raises(RuntimeError, match="closed"):
        region.evaluate(None)
    with profiler.profile_context("caller_completed", already_evaluated=True):
        pass
    assert profiler.get_results()[-1].completion_verified
    assert core.calls.count(("sync", "a")) == 2
    assert core.calls.count(("sync", "b")) == 2


@pytest.mark.parametrize(
    "exception", [ValueError("user"), SystemExit(7), KeyboardInterrupt()]
)
def test_original_exception_survives_cleanup_failure(exception):
    core = FakeCore()
    profiler = MLXMemoryProfiler(runtime=make_runtime(core))
    with pytest.raises(type(exception)) as found:
        with profiler.profile_context("failure"):
            core.sync_error = RuntimeError("cleanup failed")
            raise exception
    assert found.value is exception
    result = profiler.get_results()[-1]
    assert not result.completion_verified
    assert result.metadata["cleanup_errors"]
    assert result.status == (
        "interrupted" if isinstance(exception, KeyboardInterrupt) else "incomplete"
    )


def test_reset_lease_and_peak_semantics():
    core = FakeCore()
    profiler = MLXMemoryProfiler(runtime=make_runtime(core), peak_mode="reset")
    other = MLXMemoryProfiler(runtime=profiler.runtime)
    with profiler.profile_context("outer") as region:
        with pytest.raises(RuntimeError, match="overlapping"):
            with other.profile_context("nested", peak_mode="reset"):
                pass
        region.evaluate(object())
    result = profiler.get_results()[0]
    assert result.metadata["peak_reset_performed"]
    assert result.final.runtime_peak_bytes == 500
    assert result.metadata["reset_peak_including_baseline_bytes"] == 500
    with other.profile_context("next", peak_mode="reset"):
        pass
    assert other.get_results()[0].metadata["reset_peak_including_baseline_bytes"] == 500


def test_cleanup_failure_without_user_exception_is_raised():
    core = FakeCore()
    profiler = MLXMemoryProfiler(runtime=make_runtime(core))
    with pytest.raises(RuntimeError, match="cleanup"):
        with profiler.profile_context("bad"):
            core.sync_error = RuntimeError("cleanup")
    assert profiler.get_results()[0].status == "incomplete"


def test_custom_root_extractor_and_failed_evaluation():
    core = FakeCore()
    profiler = MLXMemoryProfiler(runtime=make_runtime(core))

    class Container:
        def __init__(self):
            self.root = object()

    output = profiler.profile_function(
        Container, root_extractor=lambda value: value.root
    )
    assert core.outputs[-1] == (id(output.root),)
    error = RuntimeError("eval failed")
    core.eval = lambda *roots: (_ for _ in ()).throw(error)
    with pytest.raises(RuntimeError) as found:
        profiler.profile_function(lambda: output)
    assert found.value is error
    assert not profiler.get_results()[-1].completion_verified


def test_final_snapshot_failure_keeps_original(monkeypatch):
    profiler = MLXMemoryProfiler(runtime=make_runtime())
    capture = profiler.capture_snapshot

    def fail_final(name="snapshot"):
        if name == "final":
            raise RuntimeError("final failed")
        return capture(name)

    monkeypatch.setattr(profiler, "capture_snapshot", fail_final)
    original = SystemExit(8)
    with pytest.raises(SystemExit) as found:
        with profiler.profile_context("failure"):
            raise original
    assert found.value is original
    assert profiler.get_results()[-1].final.active_bytes is None


def test_aggregate_peak_survives_bounded_history():
    core = FakeCore()
    profiler = MLXMemoryProfiler(
        runtime=make_runtime(core), sampling_interval=0.001, max_history=2
    )
    import time

    with profiler.profile_context("bounded", already_evaluated=True):
        core.values["active_memory"] = 900
        time.sleep(0.015)
        core.values["active_memory"] = 10
        time.sleep(0.015)
    result = profiler.get_results()[-1]
    assert result.sampled_peak_bytes == 900
    assert result.metadata["sample_count"] > result.metadata["retained_sample_count"]
    assert result.metadata["retained_sample_count"] == 2


def test_unavailable_active_makes_profile_collection_incomplete():
    core = FakeCore()
    core.values["active_memory"] = RuntimeError("allocator unavailable")
    profiler = MLXMemoryProfiler(runtime=make_runtime(core))
    with profiler.profile_context("no_metrics", already_evaluated=True):
        pass
    result = profiler.get_results()[0]
    assert result.status == "incomplete"
    assert result.sampled_peak_bytes is None
    assert result.metadata["unavailable_sample_count"] == 2


def test_reset_lease_released_after_setup_failure():
    core = FakeCore()
    core.sync_error = RuntimeError("setup failed")
    profiler = MLXMemoryProfiler(runtime=make_runtime(core))
    with pytest.raises(RuntimeError):
        with profiler.profile_context("bad", peak_mode="reset"):
            pass
    core.sync_error = None
    with profiler.profile_context("good", peak_mode="reset"):
        pass
