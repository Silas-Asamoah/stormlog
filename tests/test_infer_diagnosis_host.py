"""Host stalls in the engine core, against a toy engine whose loop stalls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.diagnosis import DiagnoseOptions, diagnose_artifact
from stormlog.infer.diagnosis_context import Context
from stormlog.infer.diagnosis_host import assess_engine_core, engine_stalls
from stormlog.infer.diagnosis_inputs import read_input
from stormlog.infer.diagnosis_join import join
from stormlog.infer.diagnosis_loop import (
    pause_intervals,
    stalls_over_limit,
    steps_from_raw,
)
from stormlog.infer.diagnosis_selection import select
from tests.diagnosis_scenarios import (
    MS,
    OBSERVES,
    SESSION,
    Engine,
    SimRequest,
    build_run,
    poisson_free,
)
from tests.vllm_execution_helpers import SECOND, WALL_OFFSET

AT = 90 * SECOND


def _calm() -> list[SimRequest]:
    return poisson_free(140, 10 * SECOND, 500 * MS, prefix="a")


def _burst() -> list[SimRequest]:
    """Requests the engine has room for, 20 ms apart."""
    return poisson_free(120, AT, 20 * MS, prefix="b")


def _diagnose(
    tmp_path: Path,
    engine: Engine,
    requests: list[SimRequest] | None = None,
    **run: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    artifact = build_run(tmp_path, _calm() + (requests or _burst()), engine, **run)
    report = diagnose_artifact(artifact, options=DiagnoseOptions(generated_at_ns=1))
    details = report["payload"]["findings_detail"].values()
    by_kind = {f"{d['kind']}@{d['location']['component']}": d for d in details}
    return report, by_kind


def _context(
    tmp_path: Path,
    engine: Engine,
    requests: list[SimRequest] | None = None,
    **run: Any,
) -> Context:
    requests = _calm() + (requests or _burst())
    view = join(read_input(build_run(tmp_path, requests, engine, **run)))
    return Context(view, select(view))


def _alternatives(detail: dict[str, Any]) -> dict[str, str]:
    return {a["kind"]: a["status"] for a in detail["alternatives"]}


def test_a_stalled_loop_is_a_host_stall_at_the_engine_core(tmp_path: Path) -> None:
    """Capacity to spare, and the loop stops for 2 s between two steps
    while requests arrive: the host's stall, and the incident's fault."""
    engine = Engine(max_num_seqs=256, stall=(AT + 50 * MS, 2 * SECOND))

    report, by_kind = _diagnose(tmp_path, engine)

    stall = by_kind["host_stall@engine_core"]
    assert (stall["severity"], stall["claim"], stall["cause"]) == (
        "warning",
        "fault",
        "fault",
    )
    assert stall["location"]["engine_producer"]
    assert (stall["detail"]["form"], stall["detail"]["locus"]) == (
        "loop_stall",
        "between_steps",
    )
    assert stall["detail"]["attribution"] == "host"
    assert set(_alternatives(stall).values()) == {"ruled_out"}
    assert report["payload"]["coverage"]["host_stall"]["components"]["engine_core"] == (
        "assessed"
    )


def test_a_stall_a_pause_could_make_is_no_claim_without_pause_records(
    tmp_path: Path,
) -> None:
    engine = Engine(
        max_num_seqs=256,
        stall=(AT + 50 * MS, 2 * SECOND),
        observes=["cache_reset", "enqueued"],
    )

    _, by_kind = _diagnose(tmp_path, engine)

    stall = by_kind["host_stall@engine_core"]
    assert _alternatives(stall)["scheduler_paused"] == "untestable"
    assert stall["claim"] == "observation"


def test_a_slow_step_alone_is_no_host_claim(tmp_path: Path) -> None:
    """A step that took 2 s may be the GPU's: without a trace it is
    host_or_gpu, and the claim rests on host stalls alone."""
    engine = Engine(max_num_seqs=256, slow_steps=[(AT + 50 * MS, 2 * SECOND)])

    _, by_kind = _diagnose(tmp_path, engine)

    stall = by_kind["host_stall@engine_core"]
    assert stall["detail"]["attribution"] == "host_or_gpu"
    assert stall["detail"]["locus"] == "within_step"
    assert "host_attribution" in stall["eligibility"]["failed"]
    assert stall["claim"] == "observation"


def test_the_engine_s_own_profiler_call_is_cut_exactly(tmp_path: Path) -> None:
    """A 700 ms stall under way when the stop is picked up, then the stop's
    own 1 s: only the 700 ms is the host's. A plain capture is no stall.
    (The toy's sparse calm traffic leaves no baseline, so its stalls are
    judged by the 500 ms floor.)"""
    observes = [*OBSERVES, "profile"]
    both = Engine(
        max_num_seqs=256,
        observes=observes,
        stall=(AT + 50 * MS, 700 * MS),
        profiles=[(AT + 50 * MS, SECOND, False)],
    )
    plain = Engine(
        max_num_seqs=256, observes=observes, profiles=[(AT + 50 * MS, SECOND, False)]
    )

    counted = engine_stalls(*_engine(tmp_path / "both", both)).counted
    nothing = engine_stalls(*_engine(tmp_path / "plain", plain)).counted

    assert [(s.locus, round(s.duration / MS)) for s in counted] == [
        ("between_steps", 700)
    ]
    assert nothing == []


def _engine(
    tmp_path: Path, engine: Engine, requests: list[SimRequest] | None = None, **run: Any
) -> tuple[Context, str]:
    context = _context(tmp_path, engine, requests, **run)
    (producer,) = {e.producer for e in context.view.executions.values()}
    return context, producer


def _trace_window(start: int, stop: int) -> dict[str, Any]:
    """The client's record of a capture, its stop on the engine's monotonic
    times; it started 5 s before, in calm traffic."""
    return {
        "schema_version": 1,
        "event_type": "infer.trace_window",
        "session_id": SESSION,
        "case_id": "c1",
        "phase": "measured",
        "requested_at_ns": start + WALL_OFFSET - 5 * SECOND,
        "started": True,
        "started_at_ns": start + WALL_OFFSET - 5 * SECOND + MS,
        "stop_requested_at_ns": start + WALL_OFFSET,
        "stopped_at_ns": stop + WALL_OFFSET,
        "timestamp_ns": stop + WALL_OFFSET,
    }


def test_a_client_only_window_leaves_its_overlap_unresolved(tmp_path: Path) -> None:
    """F4a's pulses, one of them under a capture the hook did not record:
    that one's time is unresolved and judged by nothing, and the others
    still make an eligible finding."""
    burst = poisson_free(300, AT, 20 * MS, prefix="b")
    pulses = [(AT + 100 * MS + k * SECOND, 600 * MS) for k in range(4)]
    plain = engine_stalls(
        *_engine(
            tmp_path / "plain",
            Engine(max_num_seqs=256, stalls=list(pulses)),
            requests=burst,
        )
    ).counted
    third = plain[2]
    # A stop window over the third pulse, on the engine's clock.
    window = _trace_window(third.start - 5 * MS, third.end + 5 * MS)
    engine = Engine(max_num_seqs=256, stalls=list(pulses))

    _, by_kind = _diagnose(
        tmp_path / "window", engine, requests=burst, windows=[window]
    )

    stall = by_kind["host_stall@engine_core"]
    assert stall["eligibility"]["eligible"]
    assert stall["detail"]["unresolved_ms"] == pytest.approx(third.duration / MS, abs=1)
    assert _alternatives(stall)["capture_pause@profiler"] == "ruled_out"


def test_the_offline_stalls_are_the_trigger_s_with_the_same_cut(
    tmp_path: Path,
) -> None:
    """R11: the class counts what the trigger flags on the raw log, cutting
    the same pauses and the engine's own profiler calls."""
    engine = Engine(
        max_num_seqs=8,
        observes=[*OBSERVES, "profile"],
        stalls=[(AT + 50 * MS, 600 * MS), (AT + 900 * MS, 300 * MS)],
        profiles=[(AT + 900 * MS, 500 * MS, True)],
    )
    context, producer = _engine(tmp_path, engine)
    raw = [
        json.loads(line)
        for path in sorted((tmp_path / "hook").rglob("engine-*/*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    calls = [
        (r["start_wall_ns"], r["end_wall_after_ns"])
        for r in raw
        if r["kind"] == "engine_profile"
    ]

    online = stalls_over_limit(steps_from_raw(raw), [*pause_intervals(raw), *calls])
    offline = engine_stalls(context, producer).counted

    assert [(s.locus, s.duration) for s in offline] == [
        (s.locus, s.duration_ns) for s, _ in online
    ]
    assert online


@pytest.mark.parametrize("tail_ms", [700])
def test_a_capture_s_tail_is_the_capture_s(tmp_path: Path, tail_ms: int) -> None:
    """The engine's stop call is cut, but the loop stays stopped 700 ms after
    it returns: that residue began at the call's end, so the stall is the
    capture's secondary, and its cause is instrumentation, not a fault."""
    stop = AT + 50 * MS
    engine = Engine(
        max_num_seqs=256,
        observes=[*OBSERVES, "profile"],
        profiles=[(stop, SECOND, False)],
        profile_tail_ns=tail_ms * MS,
    )
    window = _trace_window(stop + 10 * MS, stop + SECOND + (tail_ms + 30) * MS)

    _, by_kind = _diagnose(tmp_path, engine, windows=[window])

    stall = by_kind["host_stall@engine_core"]
    capture = by_kind["capture_pause@profiler"]
    assert stall["role"] == "secondary" and stall["secondary_to"] == [capture["id"]]
    assert stall["cause"] == "instrumentation" and stall["claim"] == "condition"


def test_no_stall_while_the_subject_ran_is_not_observed(tmp_path: Path) -> None:
    context = _context(tmp_path, Engine(max_num_seqs=256))
    (subject,) = context.subjects()

    assessment = assess_engine_core(context, subject)

    assert (assessment.status, assessment.reasons) == ("assessed", ["not_observed"])
