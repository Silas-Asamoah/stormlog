from stormlog.mlx import MLXMemoryProfiler
from stormlog.mlx.context_profiler import (
    clear_global_profiler,
    get_global_profiler,
    profile_context,
    profile_function,
    set_global_profiler,
)
from tests.mlx_fakes import make_runtime


def test_decorator_preserves_metadata_and_kwargs():
    profiler = MLXMemoryProfiler(runtime=make_runtime())

    @profile_function(profiler=profiler)
    def workload(name, state_getter=3):
        """A workload with kwargs resembling profiler options."""
        return name, state_getter

    assert workload(name="user", state_getter=7) == ("user", 7)
    assert workload.__name__ == "workload"
    assert workload.__doc__.startswith("A workload")
    assert profiler.get_results()[0].name == "workload"


def test_global_helpers_lifecycle():
    profiler = MLXMemoryProfiler(runtime=make_runtime())
    set_global_profiler(profiler)
    try:
        assert get_global_profiler() is profiler
        with profile_context("global") as region:
            region.evaluate(None)
        assert profiler.get_results()[0].name == "global"
    finally:
        clear_global_profiler()
