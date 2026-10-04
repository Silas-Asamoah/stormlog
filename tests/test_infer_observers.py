"""Whether each observer was active and healthy over the compared phases."""

from __future__ import annotations

from typing import Any

import pytest

from stormlog.infer.observers import observer_lines, observer_states
from stormlog.infer.vllm_analysis import JoinedSpans

SECOND = 1_000_000_000
START = 100 * SECOND
END = START + 10 * SECOND


def _config(**changes: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "system_sampler": "psutil",
        "sample_interval_seconds": 1.0,
        "vllm_metrics": {
            "url": "http://127.0.0.1:8000/metrics",
            "interval_seconds": 1.0,
        },
        "vllm_spans": {"listen": "127.0.0.1:4318", "path": "/v1/traces"},
        "trace": {"mode": "vllm-torch", "phase": "measured"},
        "vllm_execution_dir": "/var/tmp/hook",
    }
    config.update(changes)
    return config


def _base(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [
        {"event_type": "infer.session", "config": config or _config()},
        {
            "event_type": "infer.phase_window",
            "phase": "measured",
            "case_id": "c1",
            "started_at_ns": START,
            "drained_at_ns": END,
        },
        {
            "event_type": "infer.phase_window",
            "phase": "warmup",
            "case_id": "c1",
            "started_at_ns": 0,
            "drained_at_ns": SECOND,
        },
    ]


def _samples(every: int = 1) -> list[dict[str, Any]]:
    return [
        {"event_type": "infer.system_sample", "timestamp_ns": START + i * SECOND}
        for i in range(0, 11, every)
    ]


def _scrapes(*, end: bool = True, gap: int = 1) -> list[dict[str, Any]]:
    def scrape(at: int, marker: str) -> dict[str, Any]:
        return {
            "event_type": "infer.vllm_scrape",
            "case_id": "c1",
            "phase": "measured",
            "marker": marker,
            "status": "ok",
            "observed_at_ns": at,
        }

    scrapes = [scrape(START, "phase_start")]
    scrapes += [scrape(START + i * SECOND, "interval") for i in range(gap, 10, gap)]
    if end:
        scrapes.append(scrape(END, "phase_end"))
    return scrapes


def _requests(count: int = 100) -> list[dict[str, Any]]:
    return [
        {
            "event_type": "infer.request",
            "phase": "measured",
            "case_id": "c1",
            "status": "ok",
            "x_request_id": f"x{i}",
        }
        for i in range(count)
    ]


def _span_capability(**counters: int) -> dict[str, Any]:
    return {
        "event_type": "infer.capabilities",
        "component": "vllm.spans",
        "metadata": {"decode_failures": 0, **counters},
    }


def _trace(*, imported: bool = True) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = [
        {
            "event_type": "infer.trace_window",
            "case_id": "c1",
            "phase": "measured",
            "started": True,
            "stopped_at_ns": END,
            "stop_error": None,
            "trace_files": ["host.12345.pt.trace.json.gz"],
        }
    ]
    if imported:
        records.append(
            {
                "event_type": "infer.capabilities",
                "component": "trace_collector",
                "metadata": {
                    "summary": {"traces": [{"file": "host.12345.pt.trace.json.gz"}]}
                },
            }
        )
    return records


def _execution(**epoch: Any) -> list[dict[str, Any]]:
    state = {"dropped": {}, "errors": [], "capped": False, **epoch}
    return [
        {
            "event_type": "infer.capabilities",
            "component": "engine_adapter",
            "metadata": {"summary": {"execution": {"epochs": {"e1": state}}}},
        },
        {
            "event_type": "infer.iteration",
            "metadata": {"start_wall_ns": START + SECOND},
        },
    ]


def _spans(joined: int) -> JoinedSpans:
    return JoinedSpans(by_request={f"x{i}": {} for i in range(joined)}, quarantined={})


def _healthy_run() -> list[dict[str, Any]]:
    return [
        *_base(),
        *_samples(),
        *_scrapes(),
        *_requests(),
        _span_capability(),
        *_trace(),
        *_execution(),
    ]


def test_every_observer_holds_up_over_the_compared_phase() -> None:
    states = observer_states(_healthy_run(), spans=_spans(100))
    observers = states["observers"]

    assert states["compared_phases"] == ["c1"]
    for name in ("system_sampler", "vllm_metrics", "vllm_spans", "trace"):
        assert (observers[name]["active"], observers[name]["healthy"]) == (True, True)
    # The artifact does not keep the hook's heartbeats: active, not judged healthy.
    execution = observers["execution"]
    assert (execution["active"], execution["healthy"]) == (True, None)
    assert execution["unjudged"] == ["heartbeat_gaps"]


def test_observers_nobody_asked_for_are_not_judged() -> None:
    config = _config(
        system_sampler="noop", vllm_metrics=None, vllm_spans=None, trace=None
    )
    config["vllm_execution_dir"] = None
    observers = observer_states(_base(config))["observers"]
    for state in observers.values():
        assert state["requested"] is False
        assert state["active"] is None and state["healthy"] is None


def test_too_few_samples_in_the_phase_is_unhealthy() -> None:
    records = [*_base(), *_samples(every=3)]
    sampler = observer_states(records)["observers"]["system_sampler"]
    assert sampler["active"] is True and sampler["healthy"] is False
    assert sampler["phases"]["c1"]["reasons"] == ["4 samples of 10 expected"]


def test_a_phase_too_short_for_one_sample_is_not_judged() -> None:
    records = _base()
    records[1]["drained_at_ns"] = START + SECOND // 20
    sampler = observer_states(records)["observers"]["system_sampler"]
    assert sampler["phases"]["c1"] == {
        "active": None,
        "healthy": None,
        "reasons": ["the phase is shorter than one sample interval"],
    }
    assert (sampler["active"], sampler["healthy"]) == (None, None)


def test_a_scraper_whose_window_did_not_resolve_is_unhealthy() -> None:
    vllm = {"cases": {"c1": {"state": "unresolved", "reasons": ["counter_reset"]}}}
    states = observer_states([*_base(), *_scrapes()], vllm=vllm)
    scraper = states["observers"]["vllm_metrics"]
    assert scraper["healthy"] is False
    assert scraper["phases"]["c1"]["reasons"] == ["window unresolved: counter_reset"]


def test_a_hook_running_on_the_server_is_requested_without_its_import() -> None:
    # An overhead baseline must run no observer, and the server's hook is
    # one whether or not the client imports its log.
    description = {"server": {"environ": {"STORMLOG_VLLM_HOOK_DIR": "/var/tmp/h"}}}
    records = [
        *_base(_config(vllm_execution_dir=None)),
        {"event_type": "infer.manifest", "role": "before", "description": description},
    ]
    execution = observer_states(records)["observers"]["execution"]
    assert execution["requested"] is True
    assert execution["settings"] == {"directory": None, "server_hook_dir": "/var/tmp/h"}
    assert execution["active"] is False


def test_a_scraper_that_missed_the_end_or_went_quiet_is_unhealthy() -> None:
    no_end = observer_states([*_base(), *_scrapes(end=False)])["observers"]
    quiet = observer_states([*_base(), *_scrapes(gap=4)])["observers"]
    assert no_end["vllm_metrics"]["phases"]["c1"]["reasons"] == [
        "no ok phase_end scrape"
    ]
    assert quiet["vllm_metrics"]["healthy"] is False
    assert (
        "gap between ok scrapes" in quiet["vllm_metrics"]["phases"]["c1"]["reasons"][0]
    )


def test_spans_must_cover_nearly_every_accepted_request_without_errors() -> None:
    records = [*_base(), *_requests(), _span_capability()]
    short = observer_states(records, spans=_spans(98))["observers"]["vllm_spans"]
    assert short["healthy"] is False
    assert short["phases"]["c1"]["reasons"] == [
        "spans joined for 98 of 100 accepted requests"
    ]
    broken = [*_base(), *_requests(), _span_capability(decode_failures=2)]
    failing = observer_states(broken, spans=_spans(100))["observers"]["vllm_spans"]
    assert failing["phases"]["c1"]["reasons"] == ["decode_failures: 2"]


def test_requests_that_never_reached_the_server_need_no_span() -> None:
    records = [*_base(), *_requests(99), _span_capability()]
    records.append(
        {
            "event_type": "infer.request",
            "phase": "measured",
            "case_id": "c1",
            "status": "unreachable",
            "x_request_id": "x-never",
        }
    )
    spans = observer_states(records, spans=_spans(99))["observers"]["vllm_spans"]
    assert spans["healthy"] is True


def test_a_trace_that_was_not_imported_is_unhealthy() -> None:
    records = [*_base(), *_trace(imported=False)]
    trace = observer_states(records)["observers"]["trace"]
    assert trace["active"] is True and trace["healthy"] is False
    assert trace["phases"]["c1"]["reasons"] == ["its trace was not imported"]


def test_a_hook_that_dropped_records_is_unhealthy() -> None:
    records = [*_base(), *_execution(dropped={"iterations": 3})]
    execution = observer_states(records)["observers"]["execution"]
    assert execution["healthy"] is False
    assert execution["phases"]["c1"]["reasons"] == ["epoch e1 dropped records"]


def _liveness(
    first: int = START - SECOND,
    last: int = END + SECOND,
    gaps: tuple[tuple[int, int], ...] = (),
) -> dict[str, Any]:
    """#218's liveness block: when the hook's writer was beating."""
    return {
        "basis": "heartbeat_gaps/1",
        "gap_ns": 5 * SECOND,
        "heartbeats": 12,
        "max_interval_ns": SECOND,
        "first": {"start_seq": 1, "start_mono_ns": 0, "start_wall_ns": first},
        "last": {"end_seq": 12, "end_mono_ns": 0, "end_wall_ns": last},
        "gaps": [
            {"start_wall_ns": start, "end_wall_ns": end, "start_seq": 3, "end_seq": 4}
            for start, end in gaps
        ],
    }


def test_a_hook_beating_through_the_phase_is_healthy() -> None:
    records = [*_base(), *_execution(liveness=_liveness())]
    execution = observer_states(records)["observers"]["execution"]
    assert execution["healthy"] is True
    assert execution["unjudged"] == []


@pytest.mark.parametrize(
    ("liveness", "reason"),
    [
        (_liveness(gaps=((START + 2 * SECOND, START + 8 * SECOND),)), "heartbeat"),
        (_liveness(last=END - 3 * SECOND), "heartbeat"),
        (_liveness(first=START + SECOND), "heartbeat"),
    ],
    ids=["gap_in_phase", "stopped_early", "started_late"],
)
def test_a_hook_that_did_not_beat_through_the_phase_is_unhealthy(
    liveness: dict[str, Any], reason: str
) -> None:
    records = [*_base(), *_execution(liveness=liveness)]
    execution = observer_states(records)["observers"]["execution"]
    assert execution["healthy"] is False
    assert any(reason in item for item in execution["phases"]["c1"]["reasons"])


def test_a_gap_outside_the_phase_does_not_count() -> None:
    gap = (START - 10 * SECOND, START - 4 * SECOND)
    records = [*_base(), *_execution(liveness=_liveness(first=0, gaps=(gap,)))]
    assert observer_states(records)["observers"]["execution"]["healthy"] is True


def test_an_observer_silent_in_the_phase_is_not_active() -> None:
    records = [*_base(), *_execution()]
    records[-1]["metadata"]["start_wall_ns"] = 0  # only during warmup
    execution = observer_states(records)["observers"]["execution"]
    assert execution["active"] is False and execution["healthy"] is False


def test_the_text_report_names_each_requested_observers_health() -> None:
    block = observer_states(_healthy_run(), spans=_spans(100))
    (line,) = observer_lines(block)
    assert line == (
        "Observers: system_sampler healthy, vllm_metrics healthy, "
        "vllm_spans healthy, trace healthy, execution unjudged"
    )
