"""The export contract fixture, which #221's harness reads, matches the code.

A change to a flag, a default, a reason or a key fails here until the
fixture changes with it, so a reader of the fixture is never out of date.
"""

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from stormlog._export import delivery, otlp_http, span_export
from stormlog._export.filesink import FILE_DISABLED, FILE_ERROR, FILE_FULL
from stormlog.infer import export as export_module
from stormlog.infer import profile as profile_module
from stormlog.infer.cli import build_parser
from stormlog.infer.export_config import ExportConfig, export_config_from_args
from stormlog.infer.export_metrics import ProfileLabels
from stormlog.infer.export_otlp import OtlpExport
from stormlog.infer.export_spans import CONTENT_ITEMS, SpanIdentity
from stormlog.infer.trace_context import POLICIES
from stormlog.scrub import KnownSecrets

FIXTURE = Path(__file__).parent / "fixtures" / "export" / "contract_v1.json"
CONTRACT: dict[str, Any] = json.loads(FIXTURE.read_text())


def _profile_actions() -> dict[str, argparse.Action]:
    parser = build_parser()
    subparsers = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    profile = subparsers.choices["profile"]
    return {
        option: action
        for action in profile._actions
        for option in action.option_strings
    }


def test_every_profile_flag_is_declared_and_parses_to_its_key() -> None:
    actions = _profile_actions()
    declared = {
        flag: spec
        for flag, spec in CONTRACT["flags"].items()
        if "profile" in spec["commands"]
    }
    exported = {
        option
        for option in actions
        if option.startswith(("--prometheus-", "--otlp-"))
        or option in ("--trace-context", "--server-trace-sampler", "--export-content")
    }
    assert exported == set(declared)
    defaults = ExportConfig()
    for flag, spec in declared.items():
        action = actions[flag]
        if "choices" in spec and action.choices is not None:
            assert list(action.choices) == spec["choices"], flag
        default = getattr(defaults, spec["key"])
        if isinstance(default, (tuple, frozenset)):
            default = sorted(default)
        if isinstance(default, Path):
            default = str(default)
        assert default == spec["default"], flag


def test_the_export_keys_are_the_config_fields() -> None:
    fields = [field.name for field in dataclasses.fields(ExportConfig)]
    assert fields == CONTRACT["export_keys"]
    assert {spec["key"] for spec in CONTRACT["flags"].values()} == set(fields)
    with pytest.raises(ValueError):
        ExportConfig.from_mapping({"trace_context": "preserve-engine"}, "watch")
    assert CONTRACT["watch_refuses"] == ["trace_context"]


def test_choices_match_the_code() -> None:
    flags = CONTRACT["flags"]
    assert flags["--trace-context"]["choices"] == list(POLICIES)
    assert flags["--export-content"]["choices"] == list(CONTENT_ITEMS)


def test_the_span_reasons_match_the_code() -> None:
    spans = CONTRACT["spans"]
    assert spans["transmission_kinds"] == list(delivery.TRANSMISSION_KINDS)
    assert spans["categories"] == list(otlp_http.CATEGORIES)
    assert spans["reasons"]["refused"] == list(delivery.REFUSED_REASONS)
    assert spans["reasons"]["dropped"] == list(delivery.DROPPED_REASONS)
    assert spans["reasons"]["unknown"] == list(delivery.UNKNOWN_REASONS)
    assert {FILE_FULL, FILE_ERROR, FILE_DISABLED} <= set(spans["reasons"]["dropped"])
    destination = CONTRACT["destination"]
    assert destination["transition_events"] == [
        delivery.FIRST_FAILURE,
        delivery.BREAKER_OPEN,
        delivery.FIRST_SUCCESS,
        delivery.BREAKER_CLOSED,
    ]
    assert destination["max_transitions"] == delivery.MAX_TRANSITIONS
    assert destination["breaker_threshold_batches"] == delivery.Breaker().threshold


def test_the_defaults_match_the_code() -> None:
    # Every default the fixture declares, so none can go stale unchecked.
    retry = delivery.RetryPolicy()
    assert CONTRACT["defaults"] == {
        "attempt_seconds": otlp_http.DEFAULT_ATTEMPT_SECONDS,
        "max_attempts": retry.max_attempts,
        "retry_budget_seconds": retry.budget_seconds,
        "backoff_initial_seconds": retry.initial_seconds,
        "backoff_max_seconds": retry.max_seconds,
        "schedule_delay_seconds": span_export.SCHEDULE_DELAY_SECONDS,
        "batch_spans": span_export.MAX_BATCH_SPANS,
        "batch_bytes": span_export.MAX_BATCH_BYTES,
        "span_queue_spans": span_export.SPAN_QUEUE_ITEMS,
        "span_queue_bytes": span_export.SPAN_QUEUE_BYTES,
        "metric_queue_records": export_module.METRIC_QUEUE_ITEMS,
        "metric_queue_bytes": export_module.METRIC_QUEUE_BYTES,
        "otlp_file_max_bytes": span_export.MAX_FILE_BYTES,
        "max_response_bytes": otlp_http.MAX_RESPONSE_BYTES,
        "interrupt_flush_timeout_seconds": (
            profile_module.EXPORT_INTERRUPT_CLOSE_SECONDS
        ),
    }


def _pipeline(tmp_path: Path) -> export_module.ExportPipeline:
    return export_module.ExportPipeline(
        ExportConfig(prometheus_textfile_dir=tmp_path),
        ProfileLabels(
            model="m",
            server="http://h",
            cases=(("c", "closed"),),
            run_id="r",
            session_id="s",
            version="0",
        ),
    )


def _rule_holds(rule: str, figures: dict[str, Any]) -> bool:
    """Evaluate a fixture rule: clauses ``a + b == 0`` joined by ``and``."""

    def value(name: str) -> int:
        node: Any = figures
        for part in name.split("."):
            node = node[part]
        return int(node)

    holds = True
    for clause in rule.split(" and "):
        terms, _, zero = clause.partition(" == ")
        assert zero == "0", clause
        holds = holds and sum(value(t) for t in terms.split(" + ")) == 0
    return holds


@pytest.mark.parametrize(
    "figure",
    [
        None,
        "dropped.queue_full",
        "dropped.closed",
        "dropped.shutdown",
        "dropped.error",
        "internal_errors.observe",
        "tokens_rejected",
    ],
)
def test_the_exact_rule_is_the_one_the_pipeline_applies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, figure: str | None
) -> None:
    # Each figure the summary reports is made non-zero in turn; the fixture's
    # rule, read as written, must agree with the pipeline's own `exact`.
    pipeline = _pipeline(tmp_path)
    pipeline.close(0.1)
    drops = dict.fromkeys(CONTRACT["metrics"]["dropped_reasons"], 0)
    if figure is not None and figure.startswith("dropped."):
        drops[figure.split(".", 1)[1]] = 1
    monkeypatch.setattr(pipeline, "_drop_counts", lambda _queue: dict(drops))
    if figure == "internal_errors.observe":
        pipeline._counts.internal_errors["observe"] = 1
    if figure == "tokens_rejected":
        pipeline.metrics.tokens_rejected = 1
    summary = pipeline.summary()
    figures = {**summary["records"], "internal_errors": summary["internal_errors"]}
    expected = _rule_holds(CONTRACT["metrics"]["exact"], figures)
    assert summary["records"]["exact"] is expected
    assert expected is (figure in (None, "dropped.closed"))


def test_the_summaries_carry_the_declared_keys(tmp_path: Path) -> None:
    config = ExportConfig(otlp_endpoint="http://127.0.0.1:9")
    otlp = OtlpExport(
        config,
        SpanIdentity(run_id="r", session_id="s", model="m", endpoint="http://h/v1"),
        host="h",
        version="0",
        secrets=KnownSecrets(),
        environ={},
    )
    otlp.close(0.1)
    accounting = otlp.accounting()
    assert sorted(accounting) == sorted(CONTRACT["spans"]["accounting_keys"])
    assert accounting["queued"] == accounting["in_flight"] == 0
    pipeline = _pipeline(tmp_path)
    pipeline.close(0.1)
    records = pipeline.summary()["records"]
    assert sorted(records) == sorted(CONTRACT["metrics"]["summary_keys"])
    assert sorted(records["dropped"]) == sorted(CONTRACT["metrics"]["dropped_reasons"])


def test_the_arms_parse() -> None:
    parser = build_parser()
    substitutes = {
        "HOST:PORT": "127.0.0.1:9900",
        "URL": "http://127.0.0.1:4318",
        "NAME[:ARG]": "parentbased_always_on",
    }
    for name, arm in CONTRACT["arms"].items():
        if arm["command"] != "profile":
            continue
        flags = [substitutes.get(flag, flag) for flag in arm["flags"]]
        args = parser.parse_args(
            [
                "profile",
                "--base-url",
                "http://h/v1",
                "--model",
                "m",
                "--output",
                "out.jsonl",
                *flags,
            ]
        )
        assert export_config_from_args(args).enabled, name
