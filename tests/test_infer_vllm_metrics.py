"""Prometheus text parsing and the vLLM 0.30.0 metric catalog."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from stormlog.infer.vllm_metrics import (
    CATALOG,
    DEPRECATED_ALIASES,
    MAX_LINE_CHARS,
    CompactScrape,
    HistogramValue,
    MetricFamily,
    Sample,
    ScrapeTooLarge,
    bucket_boundary,
    compact_scrape,
    created_family_for,
    decode_number,
    discover,
    family_group,
    parse_prometheus_text,
    resolve_name,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "vllm"
MODEL = (
    "/home/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/"
    "snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _families_of_kind(families: dict[str, MetricFamily], kind: str) -> list[str]:
    return sorted(
        name
        for name, family in families.items()
        if family.kind == kind and name.startswith("vllm:")
    )


class TestParser:
    def test_real_scrape_keeps_every_family_and_sample(self) -> None:
        text = _fixture("q05_c08_metrics_post.txt")
        families = parse_prometheus_text(text)
        sample_lines = [
            line for line in text.splitlines() if line and not line.startswith("#")
        ]
        assert sum(len(f.samples) for f in families.values()) == len(sample_lines)
        assert len(_families_of_kind(families, "histogram")) == 16
        assert len(_families_of_kind(families, "counter")) == 15
        prompt = families["vllm:prompt_tokens_total"]
        assert prompt.kind == "counter"
        assert prompt.help == "Number of prefill tokens processed."
        assert prompt.samples == (
            Sample(
                "vllm:prompt_tokens_total",
                (("engine", "0"), ("model_name", MODEL)),
                71680.0,
            ),
        )
        assert families["vllm:prompt_tokens_created"].kind == "gauge"
        assert "process_start_time_seconds" in families

    def test_histogram_samples_fold_under_their_family(self) -> None:
        families = parse_prometheus_text(_fixture("q05_c08_metrics_post.txt"))
        queue = families["vllm:request_queue_time_seconds"]
        names = {sample.name for sample in queue.samples}
        assert names == {
            "vllm:request_queue_time_seconds_bucket",
            "vllm:request_queue_time_seconds_sum",
            "vllm:request_queue_time_seconds_count",
        }
        assert sum(1 for s in queue.samples if s.name.endswith("_bucket")) == 22

    def test_label_escapes_and_untyped_samples(self) -> None:
        text = (
            '# TYPE demo gauge\ndemo{path="a\\\\b",quote="say \\"hi\\"",nl="x\\ny"} 1\n'
            'orphan_total{engine="1"} 3 1700000000000\n'
            "plain 2.5e3\n"
        )
        families = parse_prometheus_text(text)
        assert dict(families["demo"].samples[0].labels) == {
            "path": "a\\b",
            "quote": 'say "hi"',
            "nl": "x\ny",
        }
        assert families["orphan_total"].kind == "untyped"
        assert families["orphan_total"].samples[0].value == 3.0
        assert families["plain"].samples[0].value == 2500.0

    @pytest.mark.parametrize(
        "text",
        ["<html>not metrics</html>", "demo{bad labels} 1", "demo notanumber"],
    )
    def test_malformed_lines_are_errors(self, text: str) -> None:
        with pytest.raises(ValueError, match="line 1"):
            parse_prometheus_text(text)

    def test_blank_scrape_is_empty_not_an_error(self) -> None:
        assert parse_prometheus_text("\n\n") == {}


class TestParserBounds:
    """What parsing may hold, whatever an 8 MiB response contains."""

    @staticmethod
    def _peak(text: str, **limits: int) -> tuple[int, BaseException | None]:
        import tracemalloc

        tracemalloc.start()
        error: BaseException | None = None
        try:
            parse_prometheus_text(text, **limits)
        except ValueError as exc:
            error = exc
        finally:
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        return peak, error

    def test_a_line_over_the_cap_is_refused_before_the_label_regex(self) -> None:
        """A 4 MiB label under the 8 MiB cap peaked at 930 MB in the regex."""
        text = '# TYPE vllm:x gauge\nvllm:x{model_name="' + "x" * (4 << 20) + '"} 1\n'
        peak, error = self._peak(text)
        assert isinstance(error, ScrapeTooLarge)
        assert str(error) == f"line 2 is over the {MAX_LINE_CHARS}-character cap"
        assert peak < 2 * len(text)

    def test_a_long_label_within_the_cap_parses_in_bounded_memory(self) -> None:
        """A plain label is matched without a step per character: the old
        pattern peaked at 13 MB on one of 64 Ki characters, the new one at
        0.2 MB. An escape-heavy label costs a few megabytes either way."""
        value = "x" * (MAX_LINE_CHARS - 64)
        escapes = "\\\\" * ((MAX_LINE_CHARS - 64) // 2)
        for label, bound in ((value, 1 << 20), (escapes, 8 << 20)):
            text = f'vllm:x{{model_name="{label}"}} 1\n'
            peak, error = self._peak(text)
            assert error is None
            assert peak < bound

    def test_series_over_the_cap_stop_the_parse(self) -> None:
        text = "vllm:x 1\n" * 1_000_000  # 9 MB of samples
        peak, error = self._peak(text, max_series=20_000)
        assert isinstance(error, ScrapeTooLarge)
        assert str(error) == "over the 20000-series cap"
        assert peak < 16 * 1024 * 1024

    def test_families_count_toward_the_cap(self) -> None:
        text = "".join(f"# TYPE f{i} gauge\n" for i in range(50_000))
        _peak, error = self._peak(text, max_series=20_000)
        assert isinstance(error, ScrapeTooLarge)

    def test_lines_are_not_held_as_a_list(self) -> None:
        text = "\n" * (512 << 10)  # a list of them alone took 4 MiB
        peak, error = self._peak(text)
        assert error is None
        assert peak < 1024 * 1024


class TestCompactScrape:
    def test_round_trip_preserves_values_labels_and_boundaries(self) -> None:
        families = parse_prometheus_text(_fixture("q05_c08_metrics_post.txt"))
        compact = compact_scrape(families)
        record = compact.to_record()
        assert CompactScrape.from_record(record) == compact
        assert compact.families["vllm:request_queue_time_seconds"] == "histogram"
        (set_id,) = compact.series("vllm:prompt_tokens_total")
        assert compact.labels(set_id) == {"engine": "0", "model_name": MODEL}
        assert compact.scalar("vllm:prompt_tokens_total", set_id) == 71680.0
        queue = compact.series("vllm:request_queue_time_seconds")[set_id]
        assert isinstance(queue, HistogramValue)
        assert queue.buckets[0][0] == "0.3"
        assert queue.buckets[-1][0] == "+Inf"
        assert queue.boundaries[-1] == math.inf
        assert queue.count == queue.buckets[-1][1]
        assert len(queue.buckets) == 22

    def test_label_sets_are_shared_across_series(self) -> None:
        families = parse_prometheus_text(_fixture("q05_c08_metrics_post.txt"))
        compact = compact_scrape(families)
        # Every plain engine series shares one label set; the by-reason,
        # sleep-state, by-source and http/python families add a few more.
        (prompt_set,) = compact.series("vllm:prompt_tokens_total")
        (generation_set,) = compact.series("vllm:generation_tokens_total")
        (usage_set,) = compact.series("vllm:kv_cache_usage_perc")
        assert prompt_set == generation_set == usage_set
        assert len(compact.label_sets) < len(compact.values) / 2
        assert compact.label_values("engine") == ("0",)
        assert compact.label_values("model_name") == (MODEL,)
        reasons = {
            compact.labels(set_id)["reason"]
            for set_id in compact.series("vllm:num_requests_waiting_by_reason")
        }
        assert reasons == {"capacity", "deferred"}

    def test_summary_quantiles_fold_like_histograms(self) -> None:
        text = (
            "# TYPE http_request_size_bytes summary\n"
            'http_request_size_bytes{quantile="0.5"} 10\n'
            'http_request_size_bytes{quantile="0.9"} 20\n'
            "http_request_size_bytes_sum 30\n"
            "http_request_size_bytes_count 2\n"
        )
        compact = compact_scrape(parse_prometheus_text(text))
        (value,) = compact.series("http_request_size_bytes").values()
        assert isinstance(value, HistogramValue)
        assert value.buckets == (("0.5", 10.0), ("0.9", 20.0))
        assert (value.sum, value.count) == (30.0, 2.0)

    def test_bucket_boundary_parses_native_strings(self) -> None:
        assert bucket_boundary("0.3") == 0.3
        assert bucket_boundary("+Inf") == math.inf

    def test_a_missing_sum_or_count_stays_missing(self) -> None:
        # A partial exposition is kept as what it is: no component is
        # invented as 0.0, so nothing downstream can difference it.
        text = (
            "# TYPE vllm:request_queue_time_seconds histogram\n"
            'vllm:request_queue_time_seconds_bucket{engine="0",le="+Inf"} 10\n'
            'vllm:request_queue_time_seconds_count{engine="0"} 10\n'
            "# TYPE demo summary\n"
            'demo_sum{engine="0"} 3\n'
            'demo_count{engine="0"} 2\n'
        )
        compact = compact_scrape(parse_prometheus_text(text))
        (queue,) = compact.series("vllm:request_queue_time_seconds").values()
        assert isinstance(queue, HistogramValue)
        assert queue.sum is None and queue.count == 10.0
        # A quantile-free summary is kept too, with no buckets.
        (summary,) = compact.series("demo").values()
        assert isinstance(summary, HistogramValue)
        assert summary.buckets == () and (summary.sum, summary.count) == (3.0, 2.0)
        record = json.loads(json.dumps(compact.to_record(), allow_nan=False))
        restored = CompactScrape.from_record(record)
        assert restored == compact

    def test_non_finite_values_round_trip_as_strict_json(self) -> None:
        text = (
            "# TYPE demo_summary summary\n"
            'demo_summary{quantile="0.5"} NaN\n'
            "demo_summary_sum +Inf\n"
            "demo_summary_count 2\n"
            "# TYPE demo gauge\n"
            "demo -Inf\n"
        )
        compact = compact_scrape(parse_prometheus_text(text))
        payload = json.dumps(compact.to_record(), allow_nan=False)
        record = json.loads(payload)
        (quantiles,) = record["values"]["demo_summary"].values()
        assert quantiles == {"buckets": [["0.5", "NaN"]], "sum": "+Inf", "count": 2.0}
        assert list(record["values"]["demo"].values()) == ["-Inf"]
        restored = CompactScrape.from_record(record)
        (value,) = restored.series("demo_summary").values()
        assert isinstance(value, HistogramValue)
        assert math.isnan(value.buckets[0][1]) and value.sum == math.inf
        (gauge,) = restored.series("demo").values()
        assert gauge == -math.inf
        with pytest.raises(ValueError):
            decode_number("Infinity")


class TestCatalog:
    def test_catalog_names_are_unique_and_prefixed(self) -> None:
        assert all(name.startswith("vllm:") for name in CATALOG)
        fields = [entry.field for entry in CATALOG.values()]
        assert len(fields) == len(set(fields))
        assert not set(CATALOG) & set(DEPRECATED_ALIASES)

    def test_residency_series_are_never_called_execution_time(self) -> None:
        for name in (
            "vllm:request_inference_time_seconds",
            "vllm:request_prefill_time_seconds",
            "vllm:request_decode_time_seconds",
            "vllm:request_queue_time_seconds",
        ):
            meaning = CATALOG[name].meaning
            assert "residency" in meaning
            assert "not GPU time" in meaning

    def test_real_scrape_discovery(self) -> None:
        compact = compact_scrape(
            parse_prometheus_text(_fixture("q05_c08_metrics_post.txt"))
        )
        found = discover(compact)
        assert found.unknown == ()
        assert found.deprecated_present == ()
        assert found.removed_present == ()
        assert found.absent == ()
        assert "vllm:spec_decode_num_drafts_total" in found.optional_absent
        assert "vllm:kv_block_lifetime_seconds" in found.optional_absent
        assert found.optional_present == ()
        assert found.to_record()["optional_present"] == []
        assert "vllm:estimated_flops_per_gpu_total" in found.present
        assert found.engines == ("0",)
        assert found.model_names == (MODEL,)
        assert found.process_start_ns is not None
        assert found.process_start_ns > 1_700_000_000 * 10**9
        assert found.to_record()["verified_vllm_version"] == "0.30.0"

    def test_deprecated_and_removed_names_are_recognised(self) -> None:
        text = (
            "# TYPE vllm:gpu_cache_usage_perc gauge\n"
            'vllm:gpu_cache_usage_perc{engine="0"} 0.25\n'
            "# TYPE vllm:model_forward_time_milliseconds histogram\n"
            'vllm:model_forward_time_milliseconds_bucket{engine="0",le="+Inf"} 1\n'
            'vllm:model_forward_time_milliseconds_sum{engine="0"} 1\n'
            'vllm:model_forward_time_milliseconds_count{engine="0"} 1\n'
            "# TYPE vllm:brand_new_thing gauge\n"
            'vllm:brand_new_thing{engine="0"} 1\n'
            "# TYPE vllm:kv_offload_load_bytes_total counter\n"
            'vllm:kv_offload_load_bytes_total{engine="0"} 5\n'
        )
        found = discover(compact_scrape(parse_prometheus_text(text)))
        assert found.deprecated_present == ("vllm:gpu_cache_usage_perc",)
        assert found.removed_present == ("vllm:model_forward_time_milliseconds",)
        # A family of a known optional subsystem is not an unknown series.
        assert found.unknown == ("vllm:brand_new_thing",)
        assert found.optional_present == ("vllm:kv_offload_load_bytes_total",)
        assert resolve_name("vllm:gpu_cache_usage_perc") == (
            "vllm:kv_cache_usage_perc",
            "vllm:gpu_cache_usage_perc",
        )
        assert resolve_name("vllm:model_forward_time_milliseconds") == (
            "vllm:model_forward_time_milliseconds",
            None,
        )
        assert resolve_name("vllm:brand_new_thing") == ("vllm:brand_new_thing", None)

    def test_created_family_and_groups(self) -> None:
        assert created_family_for("vllm:prompt_tokens_total", "counter") == (
            "vllm:prompt_tokens_created"
        )
        assert created_family_for("vllm:request_queue_time_seconds", "histogram") == (
            "vllm:request_queue_time_seconds_created"
        )
        # Only a counter drops _total: a histogram named like one keeps it.
        assert created_family_for("vllm:iteration_tokens_total", "histogram") == (
            "vllm:iteration_tokens_total_created"
        )
        assert created_family_for("vllm:prompt_tokens_created", "counter") is None
        assert family_group("vllm:kv_cache_usage_perc") == "kv_cache"
        assert family_group("vllm:gpu_cache_usage_perc") == "kv_cache"
        assert family_group("vllm:kv_offload_load_bytes") == "kv_transfer"
        assert family_group("vllm:brand_new_thing") == "other"
        assert CATALOG["vllm:estimated_flops_per_gpu_total"].provenance == "estimated"
