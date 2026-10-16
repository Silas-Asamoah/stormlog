"""Qualification observers preserve workload behavior and only read counters."""

import json
import os

import pytest

from examples.scenarios.mlx_inference_server import ReadOnlyCheckpoints


class GettersOnly:
    def __init__(self):
        self.calls = []

    def read_bytes(self, name):
        self.calls.append(name)
        return {"get_active_memory": 20, "get_cache_memory": 30, "get_peak_memory": 50}[
            name
        ]


def test_checkpoints_read_only_bounded_and_preserve_forward(tmp_path):
    runtime = GettersOnly()
    observer = ReadOnlyCheckpoints(tmp_path / "checkpoints.jsonl", runtime, maximum=2)
    marker = object()

    class Model:
        def __call__(self, value, *, setting):
            assert setting is marker
            return value

    model = Model()
    observer.instrument(model, "__call__")
    observer.instrument(model, "__call__")
    assert model(marker, setting=marker) is marker
    assert model(marker, setting=marker) is marker
    rows = [json.loads(line) for line in observer.output.read_text().splitlines()]
    assert len(rows) == 2
    assert [row["boundary"] for row in rows] == ["before_call", "after_call"]
    assert all(row["pid"] == os.getpid() for row in rows)
    assert (
        runtime.calls
        == ["get_active_memory", "get_cache_memory", "get_peak_memory"] * 2
    )


def test_checkpoint_generator_preserves_lazy_work_and_close(tmp_path):
    observer = ReadOnlyCheckpoints(tmp_path / "checkpoints.jsonl", GettersOnly())
    calls = []

    class Model:
        def generate(self, value):
            calls.append("start")
            try:
                yield value
                calls.append("second")
                yield value
            finally:
                calls.append("close")

    model = Model()
    observer.instrument(model, "generate")
    marker = object()
    result = model.generate(marker)
    assert calls == []
    assert next(result) is marker
    result.close()
    assert calls == ["start", "close"]


def test_checkpoint_preserves_primary_generator_error(tmp_path):
    observer = ReadOnlyCheckpoints(tmp_path / "checkpoints.jsonl", GettersOnly())
    problem = ValueError("original failure")

    class Model:
        def generate(self):
            yield 1
            raise problem

    model = Model()
    observer.instrument(model, "generate")
    result = model.generate()
    assert next(result) == 1
    with pytest.raises(ValueError) as caught:
        next(result)
    assert caught.value is problem
