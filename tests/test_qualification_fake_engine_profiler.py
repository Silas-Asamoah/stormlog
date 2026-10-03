"""The fake engine's profiler routes, traces and adversarial switches."""

from __future__ import annotations

import http.client
import urllib.error
from pathlib import Path

import pytest

from examples.qualification.fake_engine import FakeEngine, FakeEngineConfig
from stormlog.infer.trace_capture import TRACE_GLOB, TraceCaptureConfig
from stormlog.infer.trace_kineto import index_spans, link_gpu_event, load_kineto_trace
from tests.qualification_fake_engine_helpers import (
    chat,
    chats_in_background,
    join_all,
    of_type,
    post,
    records,
    run_profile,
    wait_until,
    words,
)


def _engine(tmp_path: Path, **changes: object) -> FakeEngine:
    config = FakeEngineConfig(step_seconds=0.001, trace_dir=tmp_path / "traces")
    engine = FakeEngine(config)
    for name, value in changes.items():
        setattr(engine.controls, name, value)
    return engine


def _traces(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "traces").glob(TRACE_GLOB))


def test_a_window_writes_a_trace_whose_gpu_work_links_to_steps(tmp_path: Path) -> None:
    with _engine(tmp_path) as engine:
        started, _ = post(f"{engine.base_url}/start_profile")
        chat(engine, words(8, "a"), max_tokens=3)
        stopped, _ = post(f"{engine.base_url}/stop_profile")
    (path,) = _traces(tmp_path)
    trace = load_kineto_trace(path)
    index_spans(trace)
    links = [link_gpu_event(trace, event) for event in trace.gpu_events]
    assert (started, stopped) == (200, 200)
    assert path.name.startswith("rank0.")
    assert links and all(link.iteration_ref is not None for link in links)


def test_repeated_starts_and_stops_answer_200_like_vllm(tmp_path: Path) -> None:
    with _engine(tmp_path) as engine:
        statuses = [
            post(f"{engine.base_url}/{route}")[0]
            for route in ("stop_profile", "start_profile", "start_profile")
        ]
        chat(engine, words(4, "a"), max_tokens=2)
        statuses += [
            post(f"{engine.base_url}/{route}")[0]
            for route in ("stop_profile", "stop_profile")
        ]
    assert statuses == [200] * 5
    assert len(_traces(tmp_path)) == 1


def test_the_stop_holds_the_step_loop(tmp_path: Path) -> None:
    config = FakeEngineConfig(
        step_seconds=0.001,
        decode_token_seconds=0.002,
        trace_dir=tmp_path / "traces",
    )
    with FakeEngine(config) as engine:
        engine.controls.stop_pause_seconds = 0.3
        post(f"{engine.base_url}/start_profile")
        threads = chats_in_background(engine, [words(4, "a")], max_tokens=400)
        assert wait_until(lambda: len(engine.engine.steps) > 5)
        post(f"{engine.base_url}/stop_profile")
        engine.engine.abort(next(iter(engine.engine.live.values())))
        join_all(threads)
        starts = [step.exec_start_ns for step in engine.engine.steps]
        profiler = engine.profiler
        assert profiler is not None
        ((stop_start, stop_end),) = profiler.stops
    # Judged on the loop's own clock: a client's bracket around the call also
    # holds the steps that ran before the call reached the loop.
    assert stop_end - stop_start >= 300_000_000
    assert not any(stop_start <= start <= stop_end for start in starts)


def test_a_server_without_a_profiler_refuses_both_routes(tmp_path: Path) -> None:
    with FakeEngine(FakeEngineConfig(step_seconds=0.001)) as engine:
        unconfigured = post(f"{engine.base_url}/start_profile")[0]
    with _engine(tmp_path, profiler_status=503) as engine:
        refused = (
            post(f"{engine.base_url}/start_profile")[0],
            post(f"{engine.base_url}/stop_profile")[0],
        )
    assert unconfigured == 404
    assert refused == (503, 503)


def test_a_start_whose_answer_is_lost_still_starts_the_profiler(
    tmp_path: Path,
) -> None:
    with _engine(tmp_path, drop_start_response=True) as engine:
        with pytest.raises((urllib.error.URLError, http.client.HTTPException, OSError)):
            post(f"{engine.base_url}/start_profile")
        profiler = engine.profiler
        assert profiler is not None and profiler.active
        chat(engine, words(4, "a"), max_tokens=2)
        stopped = post(f"{engine.base_url}/stop_profile")[0]
    assert stopped == 200
    assert len(_traces(tmp_path)) == 1


def test_a_stop_can_answer_200_and_write_no_trace(tmp_path: Path) -> None:
    with _engine(tmp_path, stop_writes_trace=False) as engine:
        post(f"{engine.base_url}/start_profile")
        chat(engine, words(4, "a"), max_tokens=2)
        stopped = post(f"{engine.base_url}/stop_profile")[0]
    assert stopped == 200
    assert _traces(tmp_path) == []


def test_max_iterations_ends_the_window_without_a_stop(tmp_path: Path) -> None:
    with _engine(tmp_path, profiler_max_iterations=3) as engine:
        post(f"{engine.base_url}/start_profile")
        chat(engine, words(4, "a"), max_tokens=6)
        assert wait_until(lambda: len(_traces(tmp_path)) == 1)
        profiler = engine.profiler
        assert profiler is not None and not profiler.active
        later = post(f"{engine.base_url}/stop_profile")[0]
    (path,) = _traces(tmp_path)
    assert later == 200
    assert len(load_kineto_trace(path).gpu_events) == 3


def test_a_delayed_or_foreign_trace_appears_on_its_own(tmp_path: Path) -> None:
    with _engine(tmp_path, trace_write_delay_seconds=0.2) as engine:
        post(f"{engine.base_url}/start_profile")
        chat(engine, words(4, "a"), max_tokens=2)
        post(f"{engine.base_url}/stop_profile")
        profiler = engine.profiler
        assert profiler is not None
        assert wait_until(lambda: len(profiler.written) == 1)
        ((_stop_start, stop_end),) = profiler.stops
        ((_path, written_ns),) = profiler.written
        profiler.drop_foreign_trace()
    # The write came the delay after the stop returned, on the engine's clock.
    assert written_ns - stop_end >= 190_000_000
    assert len(_traces(tmp_path)) == 2


def test_infer_profile_captures_and_imports_a_window(tmp_path: Path) -> None:
    hook = tmp_path / "hook"
    output = tmp_path / "infer.jsonl"
    config = FakeEngineConfig(
        step_seconds=0.001,
        trace_dir=tmp_path / "traces",
        hook_dir=hook,
        hook_seal_seconds=0.5,
    )
    with FakeEngine(config) as engine:
        run_profile(
            engine,
            output,
            vllm_execution_dir=hook,
            trace=TraceCaptureConfig(
                control_url=engine.base_url,
                trace_dir=tmp_path / "traces",
                settle_seconds=0.1,
                missing_grace_seconds=2.0,
            ),
        )
    items = records(output)
    (window,) = of_type(items, "infer.trace_window")
    activities = of_type(items, "infer.activity_ref")
    assert window["started"] and window["stop_status"] == 200
    assert len(window["trace_files"]) == 1
    assert activities
    assert {a["attribution_status"] for a in activities} == {"linked"}
    assert {a["context"]["device_uuid"] for a in activities} == {config.device_uuid}
