"""The collector and Prometheus configs in examples/observability."""

from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.vllm_analysis import _join_spans
from stormlog.infer.vllm_spans import RawSpan, span_record
from stormlog.infer.vllm_telemetry import SPAN_SOURCE_RECEIVER

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "observability"
VLLM_MODULE = "vllm.instrumenting_module_name"
FILTER = 'resource.attributes["vllm.instrumenting_module_name"] == nil'


def _yaml(name: str) -> dict[str, Any]:
    yaml = pytest.importorskip("yaml")
    return dict(yaml.safe_load((EXAMPLES / name).read_text()))


def test_the_analysis_pipeline_only_selects_and_never_samples() -> None:
    config = _yaml("otelcol.yaml")
    pipelines = config["service"]["pipelines"]
    analysis = pipelines["traces/stormlog-analysis"]
    assert analysis["processors"] == ["memory_limiter", "filter/vllm-only"]
    assert analysis["exporters"] == ["otlphttp/stormlog"]
    assert config["processors"]["filter/vllm-only"]["traces"]["span"] == [FILTER]
    exporter = config["exporters"]["otlphttp/stormlog"]
    assert exporter["sending_queue"]["queue_size"] > 0
    assert exporter["retry_on_failure"]["max_elapsed_time"]
    backend = pipelines["traces/backend"]
    assert "tail_sampling" in backend["processors"]
    assert backend["receivers"] == analysis["receivers"] == ["otlp"]


def test_the_x1_collector_writes_straight_to_its_file() -> None:
    config = _yaml("otelcol-x1.yaml")
    (pipeline,) = config["service"]["pipelines"].values()
    assert pipeline["processors"] == [] and pipeline["exporters"] == ["file"]
    assert config["exporters"]["file"]["sending_queue"]["enabled"] is False


def test_prometheus_scrapes_vllm_and_stormlog_apart() -> None:
    config = _yaml("prometheus.yml")
    jobs = {job["job_name"] for job in config["scrape_configs"]}
    assert jobs == {"vllm", "stormlog"}


_IDS = iter(range(1, 1000))


def _raw(name: str, resource: dict[str, Any], **attributes: Any) -> RawSpan:
    index = next(_IDS)
    return RawSpan(
        name=name,
        trace_id=f"{index:032x}",
        span_id=f"{index:016x}",
        attributes=attributes,
        resource=resource,
    )


def _kept_by_filter(span: RawSpan) -> bool:
    """What the filter processor's condition keeps: spans whose resource has it."""
    return span.resource.get(VLLM_MODULE) is not None


def test_t21_the_filter_gives_the_analysis_what_a_direct_receiver_would() -> None:
    vllm_resource = {"service.name": "vllm", VLLM_MODULE: "vllm.v1.engine"}
    vllm = [
        _raw(
            "llm_request",
            vllm_resource,
            **{"gen_ai.request.id": f"chatcmpl-stormlog-r-c1_measured_{i}"},
        )
        for i in range(3)
    ]
    others = [
        _raw("stormlog.infer.request", {"service.name": "stormlog"}),
        # Stormlog renamed through OTEL_SERVICE_NAME, even to "vllm".
        _raw("stormlog.infer.request", {"service.name": "vllm"}),
        _raw("stormlog.infer.phase", {"service.name": "bench"}),
        # An unrelated service, one of whose spans has vLLM's span name.
        _raw("llm_request", {"service.name": "other-app"}),
        _raw("GET /health", {"service.name": "other-app"}),
    ]
    requests = [
        {"x_request_id": f"stormlog-r-c1_measured_{i}", "phase": "measured"}
        for i in range(3)
    ]

    def summary(raws: list[RawSpan]) -> dict[str, Any]:
        records = [
            span_record(
                raw,
                session_id="s",
                run_id="r",
                source=SPAN_SOURCE_RECEIVER,
                clock_domain="host/boot/unix_epoch_ns",
            )
            for raw in raws
        ]
        return _join_spans(records, requests).summary()

    direct = summary(vllm)
    through_collector = summary([s for s in vllm + others if _kept_by_filter(s)])
    assert through_collector == direct
    assert direct["joined"] == 3 and direct["unjoined_by_reason"] == {}
    # Without the filter, the analysis would count spans that are not vLLM's.
    assert summary(vllm + others) != direct
