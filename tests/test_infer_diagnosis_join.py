"""Reading an artifact by physical line, and joining its evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from stormlog.infer.diagnosis_inputs import read_input, record_id
from stormlog.infer.diagnosis_join import join
from tests.diagnosis_scenarios import (
    EPOCH,
    MS,
    OBSERVES,
    Engine,
    SimRequest,
    build_run,
    poisson_free,
)
from tests.vllm_execution_helpers import SECOND


def test_lines_are_numbered_from_zero_counting_blank_ones(tmp_path: Path) -> None:
    lines = [
        b'{"schema_version": 1, "event_type": "infer.request", "request_id": "r0"}',
        b"",
        b"not json",
        b'{"schema_version": 1, "event_type": "infer.vllm_scrape", "observed_at_ns": 5}',
    ]
    data = b"\n".join(lines) + b"\n"
    path = tmp_path / "infer.jsonl"
    path.write_bytes(data)

    source = read_input(path)

    assert [line.number for line in source.lines] == [0, 1, 2, 3]
    assert [line.record_id for line in source.lines] == [
        "infer.request/r0",
        None,
        None,
        "infer.vllm_scrape@5",
    ]
    assert source.lines[3].sha256 == hashlib.sha256(lines[3]).hexdigest()
    assert source.describe() == {
        "kind": "inference_jsonl",
        "path": str(path),
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "lines": 4,
    }
    assert [problem.split(":")[0] for problem in source.problems] == ["line 2"]


def test_each_record_type_has_its_own_id() -> None:
    def v1(event_type: str, **fields: object) -> dict[str, object]:
        return {"schema_version": 1, "event_type": event_type, **fields}

    assert record_id({"schema_version": 2, "event_type": "x", "event_id": "e1"}) == "e1"
    assert record_id(v1("infer.dispatch", request_id="r1")) == "infer.dispatch/r1"
    assert (
        record_id(v1("infer.phase_window", case_id="c1", phase="measured"))
        == "infer.phase_window/c1/measured"
    )
    assert (
        record_id(v1("infer.vllm_span", trace_id="t", span_id="s"))
        == "infer.vllm_span/t/s"
    )
    assert record_id(v1("infer.session", timestamp_ns=9)) == "infer.session@9"
    assert record_id(v1("infer.request")) is None  # nothing identifies it


def test_the_join_groups_a_request_s_client_and_engine_records(
    tmp_path: Path,
) -> None:
    requests = poisson_free(6, 10 * SECOND, 50 * MS, output=3)
    late = SimRequest("late", 10 * SECOND + 400 * MS, output=3)
    artifact = build_run(tmp_path, [*requests, late], Engine(max_num_seqs=2))
    lines = artifact.read_text().splitlines()
    # The last request is still in flight: only its dispatch was written.
    kept = [
        line
        for line in lines
        if not (
            '"request_id": "late"' in line
            and json.loads(line)["event_type"] != "infer.dispatch"
        )
    ]
    artifact.write_text("\n".join(kept) + "\n")

    view = join(read_input(artifact))

    first = view.client["r0"]
    assert (first.status, first.case_id, first.phase) == ("ok", "c1", "measured")
    assert first.sent_at_ns is not None and first.first_content_at_ns is not None
    assert (
        first.ended_at_ns is not None and first.first_content_at_ns < first.ended_at_ns
    )
    in_flight = view.client["late"]
    assert (in_flight.status, in_flight.sent_at_ns) == (
        None,
        late.sent_ns + 1_790_000_000_000_000_000,
    )
    (execution,) = view.executions_of("r0")
    assert execution.ownership == "run"
    assert execution.metadata["enqueued_mono_ns"] == requests[0].enqueued_ns
    steps = [m.iteration_ref.id for _, m in execution.memberships]
    assert steps == sorted(steps, key=int) and len(steps) == 3
    engine = view.engines[EPOCH]
    assert engine.config["max_num_seqs"] == 2
    assert engine.observes == OBSERVES
    assert engine.coverage is not None and engine.coverage["spans"]
    assert view.has_dispatch_records() and view.has_first_content_records()
    assert view.run_id == "run-1"


def test_a_request_s_executions_are_found_again_after_more_are_joined(
    tmp_path: Path,
) -> None:
    """The index by request is rebuilt when executions are added."""
    requests = poisson_free(3, SECOND, 50 * MS)
    view = join(read_input(build_run(tmp_path, requests, Engine())))
    (first,) = view.executions_of("r0")
    key = next(k for k, e in view.executions.items() if e is first)

    view.executions[replace(key, id=key.id + "-again")] = first

    assert view.executions_of("r0") == [first, first]
    assert view.executions_of("r1") and view.executions_of("nobody") == []
