"""The fixture artifacts themselves: what each holds, and that Stormlog
reads it as the rest of the Studio suite assumes."""

from __future__ import annotations

import contextlib
import io
import json
import math
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any, get_args

import pytest

from stormlog.exit_codes import ExitCode
from stormlog.infer.audio_events import AudioRequestEvent
from stormlog.infer.cli import main as infer_main
from stormlog.infer.events import (
    InferenceRequestEvent,
    InferenceSystemSample,
    JsonlEventWriter,
)
from stormlog.infer.populations import case_populations
from stormlog.studio.contracts import ProblemCode
from stormlog.telemetry import validate_telemetry_record
from tests.studio.artifacts import (
    ANALYZE_FAILS,
    LARGE_GAPS,
    LARGE_REQUESTS,
    MULTI_SESSION,
    NO_TERMINAL_RECORD,
    PROBLEM_LINES,
    STANDARD_ARTIFACTS,
    UNSUPPORTED,
    CaseSpec,
    GrowingCopy,
    SamplePlan,
    decoded_records,
    last_line_start,
    profile_artifact,
    profile_config,
    rewrite_in_place,
    synthetic_artifact,
    truncated_copy,
    with_execution_import,
)
from tests.studio.conftest import StandardArtifacts
from tests.studio.fake_openai import Answer, CasePolicy, ServerState, fake_openai

NAMES = sorted(STANDARD_ARTIFACTS)


def _analyze(path: Path, output: Path) -> int:
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        args = ["analyze", str(path), "--format", "json", "--output", str(output)]
        return infer_main(args)


def _lines(path: Path) -> list[dict[str, Any]]:
    """Every complete JSON line, as written (contexts not expanded)."""
    records = []
    for line in path.read_bytes().split(b"\n"):
        with contextlib.suppress(ValueError):
            records.append(json.loads(line))
    return records


def _client_requests(path: Path) -> list[dict[str, Any]]:
    return [
        record
        for record in _lines(path)
        if record.get("event_type") == "infer.request"
        and record.get("schema_version", 1) == 1
    ]


def _measured(path: Path) -> list[dict[str, Any]]:
    return [r for r in _client_requests(path) if r["phase"] == "measured"]


def _sessions(path: Path) -> set[str]:
    kinds = {"infer.session", "infer.request"}
    return {
        str(record["session_id"])
        for record in _lines(path)
        if record.get("event_type") in kinds and record.get("schema_version", 1) == 1
    }


def _is_terminal(record: dict[str, Any]) -> bool:
    return record.get("event_type") == "infer.session" and record.get("status") not in (
        None,
        "running",
    )


@pytest.mark.parametrize("name", NAMES)
def test_infer_analyze_reads_each_artifact_unless_it_is_meant_to_fail(
    name: str, studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    code = _analyze(studio_artifacts[name], tmp_path / "report.json")

    if name in ANALYZE_FAILS:
        assert code == int(ExitCode.INVALID_INPUT)
    else:
        assert code == int(ExitCode.OK)
        assert json.loads((tmp_path / "report.json").read_text())["cases"] is not None


@pytest.mark.parametrize("name", NAMES)
def test_only_the_multi_session_artifact_holds_two_sessions(
    name: str, studio_artifacts: StandardArtifacts
) -> None:
    path = studio_artifacts[name]
    expected = 2 if name in MULTI_SESSION else 1

    assert len(_sessions(path)) == (0 if name in UNSUPPORTED else expected)


def test_the_multi_session_artifact_is_two_whole_runs_end_to_end(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["multi_session"]
    sessions = [r for r in _lines(path) if r.get("event_type") == "infer.session"]

    assert [r["status"] for r in sessions] == [
        "running",
        "completed",
        "running",
        "completed",
    ]
    first, second = sessions[0]["session_id"], sessions[2]["session_id"]
    assert first != second
    by_session = Counter(r["session_id"] for r in _measured(path))
    assert by_session == {first: 12, second: 12}


@pytest.mark.parametrize("name", NAMES)
def test_a_terminal_session_record_ends_only_the_finished_artifacts(
    name: str, studio_artifacts: StandardArtifacts
) -> None:
    terminal = [r for r in _lines(studio_artifacts[name]) if _is_terminal(r)]

    assert bool(terminal) == (name not in NO_TERMINAL_RECORD | set(UNSUPPORTED))


@pytest.mark.parametrize("cut", ["mid_line", "after_newline"])
def test_truncated_copies_cut_where_they_say(
    cut: Any, studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    source = studio_artifacts["tokens_exact"]
    data = source.read_bytes()
    start = last_line_start(data)

    copy = truncated_copy(source, tmp_path / "cut.jsonl", cut=cut).read_bytes()

    assert data.startswith(copy)
    if cut == "after_newline":
        assert copy == data[:start]
        assert copy.endswith(b"\n")
        assert _is_terminal(json.loads(data[start:]))
    else:
        tail = copy[start:]
        assert 0 < len(tail) < len(data) - 1 - start
        assert b"\n" not in tail
        assert copy[:start] == data[:start]


def test_the_standard_truncated_artifacts_are_cut_copies_of_their_source(
    studio_artifacts: StandardArtifacts,
) -> None:
    for name in ("truncated_mid_line", "truncated_after_newline"):
        path = studio_artifacts[name]
        source = path.with_name(f"{name}.src.jsonl").read_bytes()
        start = last_line_start(source)
        cut = path.read_bytes()
        assert source.startswith(cut)
        if name.endswith("newline"):
            assert cut == source[:start]
        else:
            assert start < len(cut) < len(source) - 1
            assert not cut.endswith(b"\n")


def test_rewrite_in_place_keeps_the_inode_and_changes_the_leading_bytes(
    tmp_path: Path,
) -> None:
    cases = [CaseSpec(1, 8, 4, 3)]
    path = profile_artifact(tmp_path, cases, warmup=0)
    before = path.stat()
    first_line = path.read_bytes().split(b"\n", 1)[0]

    rewrite_in_place(path, cases=[CaseSpec(2, 8, 4, 5)])

    after = path.stat()
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    # Line 0, the anchor C1 re-checks, is a new infer.session record.
    assert path.read_bytes()[: len(first_line)] != first_line
    old_session = json.loads(first_line)["session_id"]
    assert _sessions(path) == {_lines(path)[0]["session_id"]} != {old_session}
    assert len(_measured(path)) == 5


def test_the_execution_import_replaces_the_file_keeping_its_bytes_as_prefix(
    tmp_path: Path,
) -> None:
    path = profile_artifact(tmp_path, [CaseSpec(2, 8, 4, 6, fail_every=3)], warmup=0)
    before, inode = path.read_bytes(), path.stat().st_ino

    with_execution_import(path)

    assert path.stat().st_ino != inode
    assert path.read_bytes().startswith(before)
    kinds = Counter(r["event_type"] for r in decoded_records(path))
    assert kinds["infer.iteration"] == 8
    assert kinds["infer.membership"] == 8
    server_view = [
        r
        for r in decoded_records(path)
        if r["event_type"] == "infer.request" and "context" in r
    ]
    assert len(server_view) == 4


def test_the_execution_import_fixture_has_server_admitted_requests(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    assert _analyze(studio_artifacts["execution_import"], tmp_path / "r.json") == 0
    report = json.loads((tmp_path / "r.json").read_text())
    (case,) = report["cases"].values()

    assert case["population"]["server_admitted"] == 6
    assert case["population"]["failed"] == 2


def _field(name: str, studio: StandardArtifacts, key: str) -> Counter[Any]:
    return Counter(r[key] for r in _client_requests(studio[name]))


@pytest.mark.parametrize(
    ("name", "key", "expected"),
    [
        ("all_failed", "status", {"error": 10}),
        ("mixed", "status", {"ok": 49, "error": 7}),
        ("non_streaming", "stream", {False: 10}),
        ("tokens_exact", "output_token_source", {"server_usage": 10}),
        ("tokens_estimated", "output_token_source", {"estimated": 10}),
        ("tokens_missing", "output_token_source", {"unknown": 60}),
        ("tokens_missing", "output_tokens", {None: 60}),
        (
            "tokens_mixed",
            "output_token_source",
            {"server_usage": 18, "estimated": 18, "unknown": 24},
        ),
        ("warmup_measured", "phase", {"warmup": 4, "measured": 8}),
        ("empty", "phase", {}),
    ],
)
def test_each_state_fixture_holds_its_state(
    name: str,
    key: str,
    expected: dict[Any, int],
    studio_artifacts: StandardArtifacts,
) -> None:
    assert _field(name, studio_artifacts, key) == expected


def test_the_growing_artifact_stops_inside_its_measured_phase(
    studio_artifacts: StandardArtifacts,
) -> None:
    records = _lines(studio_artifacts["growing"])
    kinds = Counter(r["event_type"] for r in records)

    assert kinds["infer.phase_start"] == 2
    assert kinds["infer.phase_window"] == 1  # the warmup's; measured is open
    assert 0 < len(_measured(studio_artifacts["growing"])) < 12


def test_a_growing_copy_grows_on_its_inode_to_equal_its_source(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    source = studio_artifacts["tokens_exact"]
    data = source.read_bytes()
    growing = GrowingCopy(source, tmp_path / "g.jsonl", lines=3)
    inode = growing.target.stat().st_ino
    assert growing.target.read_bytes().count(b"\n") == 3

    growing.append_partial()
    partial = growing.target.read_bytes()
    assert not partial.endswith(b"\n") and data.startswith(partial)
    growing.append(2)
    assert growing.target.read_bytes().count(b"\n") == 5
    growing.finish()

    assert growing.target.read_bytes() == data
    assert growing.target.stat().st_ino == inode
    assert growing.remaining == 0


def test_synthetic_requests_equal_what_the_dataclass_writes(tmp_path: Path) -> None:
    path = synthetic_artifact(
        tmp_path, requests=40, gaps_per_request=5, fail_every=7, tokens="mixed"
    )
    names = {field.name for field in fields(InferenceRequestEvent)}

    for record in _measured(path):
        event = InferenceRequestEvent(**{k: v for k, v in record.items() if k in names})
        assert event.to_record() == record
        assert len(record["chunk_interarrival_ms"]) == (
            5 if record["status"] == "ok" else 0
        )


def test_a_seed_always_writes_the_same_requests(tmp_path: Path) -> None:
    def requests(directory: str, seed: int) -> list[dict[str, Any]]:
        (tmp_path / directory).mkdir()
        path = synthetic_artifact(
            tmp_path / directory, requests=50, gaps_per_request=3, seed=seed
        )
        return _measured(path)

    first, again, other = requests("a", 1), requests("b", 1), requests("c", 2)

    assert first == again
    assert first != other


def test_synthetic_requests_lie_inside_their_phase_window(tmp_path: Path) -> None:
    path = synthetic_artifact(tmp_path, requests=300, gaps_per_request=15)
    (window,) = [r for r in _lines(path) if r["event_type"] == "infer.phase_window"]
    requests = _measured(path)

    assert min(r["started_at_ns"] for r in requests) >= window["started_at_ns"]
    assert max(r["started_at_ns"] for r in requests) == window["window_ended_at_ns"]
    assert max(r["ended_at_ns"] for r in requests) == window["drained_at_ns"]


def test_the_fake_server_fails_every_nth_measured_request(tmp_path: Path) -> None:
    path = profile_artifact(tmp_path, [CaseSpec(3, 8, 4, 10, fail_every=3)], warmup=2)
    requests = _client_requests(path)

    assert Counter((r["phase"], r["status"]) for r in requests) == {
        ("warmup", "ok"): 2,
        ("measured", "ok"): 7,
        ("measured", "error"): 3,
    }


def _post(endpoint: str, request_id: str, body: dict[str, Any]) -> Any:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json", "X-Request-Id": request_id},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=10) as response:
        return response.read()


def test_the_fake_server_answers_by_case_and_phase() -> None:
    policies = {
        "c1_in8_out4": CasePolicy(all_fail=True),
        "c2_in8_out4": CasePolicy(usage=False),
    }
    with fake_openai(policies) as server:
        with pytest.raises(urllib.error.HTTPError):
            _post(server.endpoint, "stormlog-r-1-c1_in8_out4_warmup_0_0", {})
        quiet = json.loads(
            _post(server.endpoint, "stormlog-r-1-c2_in8_out4_measured_0_0", {})
        )
        normal = json.loads(
            _post(server.endpoint, "stormlog-r-1-c3_x_measured_0_0", {})
        )
        stream = _post(
            server.endpoint,
            "stormlog-r-1-c2_in8_out4_measured_0_1",
            {"stream": True, "stream_options": {"include_usage": True}},
        )

    assert "usage" not in quiet and "usage" in normal
    assert stream.endswith(b"data: [DONE]\n\n") and b"usage" not in stream
    assert server.state.failed == 1
    assert server.state.measured == {"c2_in8_out4": 2, "c3_x": 1}


@pytest.mark.parametrize(
    "cases",
    [
        [CaseSpec(1, 8, 4, 3), CaseSpec(2, 8, 4, 4)],
        [CaseSpec(1, 8, 4, 3), CaseSpec(2, 8, 4, 3, stream=False)],
        [CaseSpec(1, 8, 4, 3), CaseSpec(2, 8, 16, 3)],
        [CaseSpec(1, 8, 4, 3), CaseSpec(1, 8, 4, 3)],
        [],
    ],
)
def test_one_run_refuses_cases_it_cannot_hold(
    cases: list[CaseSpec], tmp_path: Path
) -> None:
    with pytest.raises(ValueError):
        profile_config(
            cases, endpoint="http://x", output=tmp_path / "a", warmup=0, seed=0
        )


def test_fresh_artifact_gives_a_private_copy(
    fresh_artifact: Callable[[str], Path], studio_artifacts: StandardArtifacts
) -> None:
    copy = fresh_artifact("empty")

    assert copy.read_bytes() == studio_artifacts["empty"].read_bytes()
    assert copy != studio_artifacts["empty"]


def _windows(path: Path) -> dict[str, dict[str, Any]]:
    """Each case's measured phase window."""
    return {
        r["case_id"]: r
        for r in _lines(path)
        if r.get("event_type") == "infer.phase_window" and r["phase"] == "measured"
    }


def _by_case(path: Path) -> Counter[str]:
    return Counter(r["case_id"] for r in _measured(path))


def test_the_open_loop_artifact_has_a_scheduled_rate_window_per_case(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["open_loop"]
    windows = _windows(path)
    populations = case_populations(decoded_records(path))

    assert _by_case(path) == {"fixed40_in8_out4": 12, "fixed40_in8_out16": 12}
    assert set(windows) == set(populations) == set(_by_case(path))
    for case_id, window in windows.items():
        assert window["arrival_mode"] == "fixed-rate"
        assert window["stopped_early"] is False
        assert window["scheduled_arrivals"] == 12
        assert window["scheduled_endpoint_offset_ns"] == 300_000_000  # 12 at 40/s
        rate = populations[case_id].intervals.rate
        assert rate is not None and rate.kind == "scheduled_window"
        # A window over the case's rate interval keeps every one of its rows.
        for request in _measured(path):
            if request["case_id"] == case_id:
                assert rate.started_at_ns <= request["intended_at_ns"]
                assert request["intended_at_ns"] < rate.ended_at_ns


def test_the_stopped_early_artifact_ends_its_window_at_the_stop(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["stopped_early"]
    (window,) = _windows(path).values()
    (population,) = case_populations(decoded_records(path)).values()
    rate = population.intervals.rate
    sessions = [r for r in _lines(path) if _is_terminal(r)]

    assert window["stopped_early"] is True
    assert 8 <= window["scheduled_arrivals"] < 40
    assert window["scheduled_endpoint_offset_ns"] < 2_000_000_000
    assert _by_case(path) == {"fixed20_in8_out4": window["scheduled_arrivals"]}
    assert rate is not None and rate.kind == "scheduled_window"
    assert rate.ended_at_ns - rate.started_at_ns == (
        window["scheduled_endpoint_offset_ns"]
    )
    assert [r["status"] for r in sessions] == ["completed"]


def test_the_interrupted_artifact_has_a_case_without_a_window(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["interrupted"]
    populations = case_populations(decoded_records(path))
    starts = [
        r["case_id"]
        for r in _lines(path)
        if r.get("event_type") == "infer.phase_start" and r["phase"] == "measured"
    ]
    whole, cut = "c2_in8_out4", "c2_in8_out16"

    assert [r["status"] for r in _lines(path) if _is_terminal(r)] == ["interrupted"]
    assert starts == [whole, cut]
    assert set(_windows(path)) == {whole}
    assert _by_case(path)[whole] == 12
    assert 5 <= _by_case(path)[cut] < 12
    assert populations[whole].intervals.rate is not None
    assert populations[cut].intervals.rate is None
    assert populations[cut].intervals.rate_reason == "phase_window_missing"


def test_the_fake_server_reads_open_loop_ids_and_stops_or_interrupts_once() -> None:
    state = ServerState(
        policies={"fixed4_in8_out4": CasePolicy(stop_after=2, interrupt_after=3)}
    )
    ids = [f"stormlog-r-1-fixed4_in8_out4_measured_{i}" for i in range(4)]
    warmup = state.answer("stormlog-r-1-fixed4_in8_out4_warmup_0")

    answers = [state.answer(request_id) for request_id in ids]

    assert warmup == Answer()
    assert [(a.stop, a.interrupt) for a in answers] == [
        (False, False),
        (True, False),
        (False, True),
        (False, False),
    ]
    assert state.measured == {"fixed4_in8_out4": 4}
    assert state.stopped and state.interrupted


@pytest.mark.parametrize(
    "options",
    [
        {"arrival": "closed", "rate_per_second": 5.0},
        {"arrival": "fixed-rate"},
        {"arrival": "burst", "rate_per_second": 5.0},
    ],
)
def test_profile_config_refuses_arrivals_it_cannot_run(
    options: dict[str, Any], tmp_path: Path
) -> None:
    with pytest.raises(ValueError):
        profile_config(
            [CaseSpec(1, 8, 4, 3)],
            endpoint="http://x",
            output=tmp_path / "a",
            warmup=0,
            seed=0,
            **options,
        )


def test_an_open_loop_run_has_one_in_flight_limit(tmp_path: Path) -> None:
    cases = [CaseSpec(1, 8, 4, 3), CaseSpec(2, 8, 4, 3)]
    with pytest.raises(ValueError, match="in-flight"):
        profile_config(
            cases,
            endpoint="http://x",
            output=tmp_path / "a",
            warmup=0,
            seed=0,
            arrival="poisson",
            rate_per_second=3.0,
        )

    config = profile_config(
        cases[:1],
        endpoint="http://x",
        output=tmp_path / "a",
        warmup=0,
        seed=0,
        arrival="poisson",
        rate_per_second=3.0,
    )
    assert [case.case_id for case in config.cases()] == [
        cases[0].run_case_id(config.arrival_specs()[0])
    ]


def test_the_timeouts_artifact_has_two_timed_out_requests(
    studio_artifacts: StandardArtifacts,
) -> None:
    measured = _measured(studio_artifacts["timeouts"])
    timed_out = [r for r in measured if r["status"] == "timeout"]

    assert Counter(r["status"] for r in measured) == {"ok": 6, "timeout": 2}
    for record in timed_out:
        assert record["http_status"] is None
        assert record["ttft_ms"] is None
        assert record["e2e_latency_ms"] >= 250.0


def test_the_nan_gap_artifact_writes_one_bare_nan(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["nan_gap"]
    lines = [line for line in path.read_bytes().split(b"\n") if b"NaN" in line]
    (line,) = lines
    record = json.loads(line)
    gaps = record["chunk_interarrival_ms"]

    assert b"NaN," in line or b"NaN]" in line
    assert record["request_index"] == 0
    assert [math.isnan(gap) for gap in gaps].count(True) == 1
    assert math.isnan(gaps[len(gaps) // 2])


def test_nan_gap_needs_a_gap(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        synthetic_artifact(tmp_path, requests=3, gaps_per_request=0, nan_gap=True)


def test_the_audio_artifact_is_two_audio_requests(
    studio_artifacts: StandardArtifacts,
) -> None:
    records = _lines(studio_artifacts["audio"])

    assert [r["event_type"] for r in records] == ["infer.audio.request"] * 2
    assert [r["status"] for r in records] == ["ok", "error"]
    for record in records:
        assert AudioRequestEvent.from_record(dict(record)).to_record() == record


def test_the_non_inference_artifact_is_memory_telemetry(
    studio_artifacts: StandardArtifacts,
) -> None:
    records = _lines(studio_artifacts["not_an_inference_artifact"])

    assert len(records) == 4
    for record in records:
        validate_telemetry_record(record)
        assert record["schema_version"] == 4
        assert record["collector"] == "stormlog.cpu_tracker"
        assert not str(record["event_type"]).startswith("infer.")


def test_the_unsupported_version_artifact_ends_with_a_v2_client_request(
    studio_artifacts: StandardArtifacts, tmp_path: Path
) -> None:
    path = studio_artifacts["unsupported_version"]
    records = _lines(path)
    last = records[-1]
    v1 = [r for r in _measured(path) if r["request_id"] == last["request_id"]]

    assert _is_terminal(records[-2])
    assert last["event_type"] == "infer.request"
    assert last["schema_version"] == 2
    assert "context" not in last and "context_id" not in last
    assert [dict(r, schema_version=2) for r in v1] == [last]
    with JsonlEventWriter(tmp_path / "refused.jsonl") as writer:
        with pytest.raises(ValueError):
            writer.append(last)


def test_problem_lines_name_problem_codes_of_the_contract() -> None:
    assert set(PROBLEM_LINES.values()) <= set(get_args(ProblemCode))
    assert set(PROBLEM_LINES) | set(UNSUPPORTED) <= set(STANDARD_ARTIFACTS)


def test_the_fake_server_holds_every_nth_measured_request() -> None:
    state = ServerState(
        policies={"c1_in8_out4": CasePolicy(slow_every=2, delay_seconds=0.5)}
    )
    warmup = state.answer("stormlog-r-1-c1_in8_out4_warmup_0_0")
    delays = [
        state.answer(f"stormlog-r-1-c1_in8_out4_measured_0_{i}").delay_seconds
        for i in range(4)
    ]

    assert warmup.delay_seconds == 0.0
    assert delays == [0.0, 0.5, 0.0, 0.5]
    with pytest.raises(ValueError):
        CasePolicy(delay_seconds=-1.0)


def _samples(path: Path) -> list[dict[str, Any]]:
    return [r for r in _lines(path) if r.get("event_type") == "infer.system_sample"]


def _series(path: Path) -> dict[tuple[str, int | None], list[int]]:
    """Each ``(sampler, device_id)`` series' timestamps, in file order."""
    series: dict[tuple[str, int | None], list[int]] = {}
    for sample in _samples(path):
        key = (sample["sampler"], sample["device_id"])
        series.setdefault(key, []).append(sample["timestamp_ns"])
    return series


def test_the_samples_artifact_has_client_rss_and_two_devices(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["samples"]
    series = _series(path)

    assert set(series) == {("psutil", None), ("nvidia-smi", 0), ("nvidia-smi", 1)}
    assert len(series[("psutil", None)]) == len(series[("nvidia-smi", 0)]) > 24
    for sample in _samples(path):
        assert sample["observation_scope"] == "client_local"
        if sample["sampler"] == "psutil":
            assert sample["process_rss_bytes"] > 0
            assert sample["device_used_bytes"] is None
        else:
            assert sample["process_rss_bytes"] is None
            assert sample["device_used_bytes"] + sample["device_free_bytes"] == (
                sample["device_total_bytes"]
            )


def test_the_samples_artifact_has_a_gap_and_a_stale_device(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["samples"]
    series = _series(path)
    steady, stale = series[("nvidia-smi", 0)], series[("nvidia-smi", 1)]
    plan = SamplePlan()
    step = plan.interval_ms * 1_000_000
    gaps = [later - earlier for earlier, later in zip(stale, stale[1:])]

    # More than observers.SCRAPE_GAP_FACTOR (2.0) times the median interval.
    assert max(gaps) == (plan.gap_ticks[1] - plan.gap_ticks[0] + 1) * step
    assert max(gaps) > 2 * sorted(gaps)[len(gaps) // 2]
    # The device goes quiet well before the run ends: its last value is stale.
    assert steady[-1] - stale[-1] >= 4 * step
    assert len(stale) == plan.stop_tick - (plan.gap_ticks[1] - plan.gap_ticks[0])


def test_samples_lie_inside_the_measured_phase_in_time_order(
    studio_artifacts: StandardArtifacts,
) -> None:
    path = studio_artifacts["samples"]
    (window,) = [r for r in _lines(path) if r["event_type"] == "infer.phase_window"]
    times = [sample["timestamp_ns"] for sample in _samples(path)]

    assert times == sorted(times)
    assert window["started_at_ns"] <= times[0]
    assert times[-1] <= window["drained_at_ns"]


def test_synthetic_samples_equal_what_the_dataclass_writes_and_repeat(
    tmp_path: Path,
) -> None:
    def samples(directory: str) -> list[dict[str, Any]]:
        (tmp_path / directory).mkdir()
        return _samples(
            synthetic_artifact(
                tmp_path / directory, requests=30, gaps_per_request=2, samples=True
            )
        )

    first = samples("a")
    names = {field.name for field in fields(InferenceSystemSample)}

    assert first == samples("b")
    for record in first:
        event = InferenceSystemSample(**{k: v for k, v in record.items() if k in names})
        assert event.to_record() == record


def test_the_real_psutil_samples_have_the_synthetic_shape(
    studio_artifacts: StandardArtifacts,
) -> None:
    real = _samples(studio_artifacts["samples_psutil"])
    synthetic = [
        s for s in _samples(studio_artifacts["samples"]) if s["sampler"] == "psutil"
    ]

    assert len(real) >= 2
    shape = {(k, type(v)) for k, v in synthetic[0].items()}
    for sample in real:
        assert {(k, type(v)) for k, v in sample.items()} == shape
        assert sample["process_rss_bytes"] > 0


def test_artifacts_without_samples_have_none(
    studio_artifacts: StandardArtifacts,
) -> None:
    for name in ("mixed", "synthetic", "empty"):
        assert _samples(studio_artifacts[name]) == []


@pytest.mark.studio_large
def test_the_large_artifact_holds_500000_requests_of_15_gaps(
    large_studio_artifact: Path,
) -> None:
    started = time.perf_counter()
    statuses: Counter[str] = Counter()
    gaps = Counter[int]()
    sessions = set()
    with large_studio_artifact.open("rb") as handle:
        for line in handle:
            if b'"infer.request"' not in line:
                continue
            record = json.loads(line)
            statuses[record["status"]] += 1
            gaps[len(record["chunk_interarrival_ms"])] += 1
            sessions.add(record["session_id"])
    print(
        f"read {sum(statuses.values())} requests in {time.perf_counter() - started:.1f} s"
    )

    assert statuses == {
        "ok": LARGE_REQUESTS - LARGE_REQUESTS // 100,
        "error": LARGE_REQUESTS // 100,
    }
    assert gaps == {LARGE_GAPS: statuses["ok"], 0: statuses["error"]}
    assert len(sessions) == 1
