"""The watch configuration: ``stormlog.infer.watch_config`` version 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.errors import InferInputError, InferUsageError
from stormlog.infer.watch.config import (
    CONFIG_FORMAT,
    DEFAULT_TRIGGERS,
    DEFAULTS_VERSION,
    load_watch_config,
    resolve_watch_config,
)
from stormlog.infer.watch.predicates import (
    FrozenExporter,
    ScrapeFailures,
    SignalExceeds,
)

BASE = "http://127.0.0.1:8000"


def _payload(**extra: Any) -> dict[str, Any]:
    return {
        "format": CONFIG_FORMAT,
        "version": 1,
        "server": {"base_url": BASE},
        **extra,
    }


def _trigger(**extra: Any) -> dict[str, Any]:
    trigger = {
        "id": "queue",
        "kind": "metric",
        "gauge": {"family": "vllm:num_requests_waiting", "at_least": 8},
    }
    trigger.update(extra)
    return trigger


def test_defaults_resolve_to_the_documented_settings() -> None:
    config = resolve_watch_config(_payload())
    assert config.tick_seconds == 1.0
    assert config.scrape_timeout_seconds == 1.0
    assert config.history_seconds == 600.0
    assert config.history_bytes == 32 * 1024 * 1024
    assert config.metrics_url == "auto"
    assert (config.incident.pre_seconds, config.incident.post_seconds) == (120.0, 60.0)
    assert config.incident.max_open_incidents == 2
    assert config.incident.max_incidents_per_hour == 30
    assert [spec.trigger_id for spec in config.triggers] == [
        t["id"] for t in DEFAULT_TRIGGERS
    ]
    predicates = [type(spec.predicate) for spec in config.triggers]
    assert predicates == [SignalExceeds, SignalExceeds, ScrapeFailures, FrozenExporter]
    health = [spec for spec in config.triggers if spec.kind == "health"]
    assert health and not any(spec.counts_toward_exit for spec in health)
    resolved = config.resolved()
    assert resolved["defaults"] == DEFAULTS_VERSION
    assert set(resolved["guarantees"]) == {t["id"] for t in DEFAULT_TRIGGERS}


def test_the_digest_follows_the_resolved_settings() -> None:
    first = resolve_watch_config(_payload())
    assert first.digest() == resolve_watch_config(_payload()).digest()
    assert first.digest() == resolve_watch_config(_payload(tick_seconds=1.0)).digest()
    assert first.digest() != resolve_watch_config(_payload(tick_seconds=2.0)).digest()


def test_guarantees_state_each_triggers_bounds() -> None:
    config = resolve_watch_config(
        _payload(triggers=[_trigger(window_seconds=30, hold_seconds=60)])
    )
    guarantee = config.resolved()["guarantees"]["queue"]
    assert guarantee["shortest_firing_violation_seconds"] == 30.0
    # W + Δ + ceil((F + j) / Δ)·Δ + j, with j the 1 s scrape timeout.
    assert guarantee["detection_bound_seconds"] == 30 + 1 + 61 + 1
    slow = resolve_watch_config(
        _payload(
            scrape_timeout_seconds=0.5,
            triggers=[_trigger(window_seconds=30, hold_seconds=60)],
        )
    )
    assert slow.resolved()["guarantees"]["queue"]["detection_bound_seconds"] == 92.5


def test_load_reads_a_file_and_applies_overrides(tmp_path: Path) -> None:
    path = tmp_path / "watch.json"
    path.write_text(json.dumps(_payload(tick_seconds=2.0)), encoding="utf-8")
    config = load_watch_config(
        path,
        overrides={
            "server.base_url": "http://10.0.0.2:8000",
            "tick_seconds": None,  # not given on the command line
        },
    )
    assert config.base_url == "http://10.0.0.2:8000"
    assert config.tick_seconds == 2.0


def test_no_file_needs_a_base_url() -> None:
    with pytest.raises(InferUsageError, match="base_url"):
        load_watch_config(None, overrides={})
    config = load_watch_config(None, overrides={"server.base_url": BASE})
    assert config.base_url == BASE


@pytest.mark.parametrize(
    "content",
    ["{not json", "[1, 2]", json.dumps({"format": CONFIG_FORMAT, "version": 2})],
)
def test_an_unreadable_or_foreign_file_is_an_input_error(
    tmp_path: Path, content: str
) -> None:
    path = tmp_path / "watch.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(InferInputError):
        load_watch_config(path)


def test_a_missing_file_is_an_input_error(tmp_path: Path) -> None:
    with pytest.raises(InferInputError, match="cannot read"):
        load_watch_config(tmp_path / "absent.json")


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"surprise": 1}, "unknown key"),
        ({"server": {"base_url": BASE, "port": 1}}, "unknown key"),
        ({"history": {"seconds": 60, "minutes": 1}}, "unknown key"),
        ({"tick_seconds": 0}, "tick_seconds"),
        ({"tick_seconds": True}, "tick_seconds"),
        ({"history": {"seconds": -1}}, "history.seconds"),
        ({"incident": {"max_open_incidents": 1.5}}, "integer"),
        ({"incident": {"max_incidents_per_hour": 0}}, "integer"),
        ({"incident": {"pre_seconds": 900}}, "pre_seconds"),
        ({"store": {"max_incidents": 0}}, "store"),
        ({"store": {"max_total_bytes": 10, "max_incident_bytes": 20}}, "store"),
        ({"triggers": {"id": "x"}}, "list"),
        ({"export": []}, "export"),
        ({"server": []}, "server"),
    ],
)
def test_settings_the_watcher_cannot_use_are_usage_errors(
    extra: dict[str, Any], message: str
) -> None:
    with pytest.raises(InferUsageError, match=message):
        resolve_watch_config(_payload(**extra))


@pytest.mark.parametrize(
    ("trigger", "message"),
    [
        (_trigger(hold_seconds=10, window_seconds=30), "F >= W"),
        (_trigger(window_seconds=900, hold_seconds=900), "longer than history"),
        (_trigger(colour="red"), "unknown key"),
        ({"id": "x", "kind": "metric"}, "exactly one predicate"),
        (
            _trigger(signal="queue_saturation"),
            "exactly one predicate",
        ),
        ({"id": "s", "kind": "signal", "signal": "warp_drive"}, "signal must be"),
        ({"id": "g", "kind": "metric", "gauge": {"family": "f"}}, "missing"),
        (_trigger(kind="slo"), "not available yet"),
        (_trigger(kind="test"), "not evaluated here"),
        (_trigger(action="deep_capture"), "deep capture is not available yet"),
        (_trigger(action="page_someone"), "unknown action"),
        (_trigger(deep_capture_when="sometimes"), "deep_capture_when"),
        (_trigger(deep_capture_when=1), "deep_capture_when must be a JSON string"),
        # Any non-empty string was true: "no" made a trigger count.
        (_trigger(counts_toward_exit="no"), "must be a JSON boolean"),
        (
            {
                "id": "failures",
                "kind": "health",
                "window_seconds": 3,
                "hold_seconds": 3,
                "scrape_failures": {},
                "counts_toward_exit": True,
            },
            "never count toward the exit code",
        ),
        ("not an object", "must be an object"),
    ],
)
def test_bad_triggers_are_usage_errors(trigger: Any, message: str) -> None:
    with pytest.raises(InferUsageError, match=message):
        resolve_watch_config(_payload(triggers=[trigger]))


def test_a_trigger_s_policy_defaults_to_its_kind_s() -> None:
    config = resolve_watch_config(
        _payload(
            triggers=[
                _trigger(),
                _trigger(id="quiet", counts_toward_exit=False),
                {
                    "id": "failures",
                    "kind": "health",
                    "window_seconds": 3,
                    "hold_seconds": 3,
                    "scrape_failures": {},
                },
            ]
        )
    )
    counts = {s.trigger_id: s.counts_toward_exit for s in config.triggers}
    assert counts == {"queue": True, "quiet": False, "failures": False}
    assert {s.deep_capture_when for s in config.triggers} == {"always"}


def test_trigger_ids_are_unique() -> None:
    with pytest.raises(InferUsageError, match="unique"):
        resolve_watch_config(_payload(triggers=[_trigger(), _trigger()]))


def test_completion_recorded_histograms_are_marked() -> None:
    config = resolve_watch_config(
        _payload(
            triggers=[
                {
                    "id": "e2e",
                    "kind": "metric",
                    "histogram_share": {
                        "family": "vllm:e2e_request_latency_seconds",
                        "above": 5.0,
                        "share": 0.05,
                    },
                },
                {
                    "id": "ttft",
                    "kind": "metric",
                    "histogram_share": {
                        "family": "vllm:time_to_first_token_seconds",
                        "above": 1.0,
                        "share": 0.05,
                    },
                },
            ]
        )
    )
    marked = {spec.trigger_id: spec.completion_recorded for spec in config.triggers}
    assert marked == {"e2e": True, "ttft": False}


def test_an_engine_is_named_for_the_server_or_per_trigger() -> None:
    config = resolve_watch_config(
        _payload(
            server={"base_url": BASE, "engine": "0"},
            triggers=[
                _trigger(),
                {"id": "q", "kind": "signal", "signal": "queue_saturation"},
                {**_trigger(id="other"), "engine": 1},
            ],
        )
    )
    engines = [
        getattr(spec.predicate, "engine", None)
        or getattr(getattr(spec.predicate, "config", None), "engine", None)
        for spec in config.triggers
    ]
    assert engines == ["0", "0", "1"]
    assert config.resolved()["server"]["engine"] == "0"
    unnamed = resolve_watch_config(_payload(triggers=[_trigger()]))
    assert getattr(unnamed.triggers[0].predicate, "engine") is None


@pytest.mark.parametrize("engine", ["", True, 1.5, ["0"]])
def test_an_engine_is_a_label_value(engine: Any) -> None:
    with pytest.raises(InferUsageError, match="engine label"):
        resolve_watch_config(_payload(server={"base_url": BASE, "engine": engine}))
    with pytest.raises(InferUsageError, match="engine label"):
        resolve_watch_config(_payload(triggers=[{**_trigger(), "engine": engine}]))


def test_export_is_passed_through_for_the_exporter() -> None:
    config = resolve_watch_config(_payload(export={"otlp": {"endpoint": "x"}}))
    assert config.export == {"otlp": {"endpoint": "x"}}
