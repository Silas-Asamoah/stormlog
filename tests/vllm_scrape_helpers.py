"""Build synthetic vLLM ``/metrics`` scrapes for window tests.

The exposition follows vLLM 0.30.0's names and the text format
prometheus_client writes (see ``tests/fixtures/vllm``); only the series a test
names are present, so a missing series is a deliberate choice of the test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from stormlog.infer.vllm_metrics import compact_scrape, discover, parse_prometheus_text
from stormlog.infer.vllm_telemetry import (
    MARKER_INTERVAL,
    SCRAPE_ERROR,
    SCRAPE_OK,
    VllmScrapeRecord,
)

T0 = 1_790_000_000_000_000_000
SECOND = 1_000_000_000
CLOCK = "client/boot/unix_epoch_ns"
LABELS = 'engine="0",model_name="m"'
START = 1_790_000_000.0
CREATED = 1_790_000_000.0

Buckets = Sequence[tuple[str, float]]


def exposition(
    *,
    gauges: Mapping[str, float | str] | None = None,
    counters: Mapping[str, float] | None = None,
    histograms: Mapping[str, tuple[Buckets, float]] | None = None,
    labelled: Mapping[str, Mapping[str, float]] | None = None,
    start: float | None = START,
    created: float = CREATED,
    extra: Sequence[str] = (),
) -> str:
    """Prometheus text for engine 0.

    ``labelled`` maps a gauge to ``{"reason=capacity": value}`` series;
    ``histograms`` maps a family to its cumulative buckets and sum; a counter
    named ``x_total`` gets the ``x_created`` stamp prometheus_client writes.
    """
    lines: list[str] = []
    if start is not None:
        lines += ["# TYPE process_start_time_seconds gauge"]
        lines += [f"process_start_time_seconds {start}"]
    for name, value in (gauges or {}).items():
        lines += [f"# TYPE {name} gauge", f"{name}{{{LABELS}}} {value}"]
    for name, by_label in (labelled or {}).items():
        lines.append(f"# TYPE {name} gauge")
        for pair, value in by_label.items():
            key, _, label = pair.partition("=")
            lines.append(f'{name}{{{LABELS},{key}="{label}"}} {value}')
    for name, total in (counters or {}).items():
        stamp = name.removesuffix("_total") + "_created"
        lines += [f"# TYPE {name} counter", f"{name}{{{LABELS}}} {total}"]
        lines += [f"# TYPE {stamp} gauge", f"{stamp}{{{LABELS}}} {created}"]
    for name, (buckets, total_sum) in (histograms or {}).items():
        lines.append(f"# TYPE {name} histogram")
        for le, count in buckets:
            lines.append(f'{name}_bucket{{{LABELS},le="{le}"}} {count}')
        lines.append(f"{name}_sum{{{LABELS}}} {total_sum}")
        lines.append(f"{name}_count{{{LABELS}}} {buckets[-1][1]}")
    return "\n".join([*lines, *extra]) + "\n"


def scrape(
    text: str | None, at_s: float, *, duration_ms: float = 4.0
) -> VllmScrapeRecord:
    """A scrape stamped ``at_s`` seconds after T0; ``None`` text is a failure."""
    observed_at_ns = T0 + round(at_s * SECOND)
    if text is None:
        return VllmScrapeRecord(
            session_id="session",
            run_id="run-1",
            observed_at_ns=observed_at_ns,
            source_url="http://127.0.0.1:8000/metrics",
            marker=MARKER_INTERVAL,
            interval_ms=1000,
            status=SCRAPE_ERROR,
            clock_domain=CLOCK,
            duration_ms=duration_ms,
            error="HTTP 503",
        )
    compact = compact_scrape(parse_prometheus_text(text))
    return VllmScrapeRecord(
        session_id="session",
        run_id="run-1",
        observed_at_ns=observed_at_ns,
        source_url="http://127.0.0.1:8000/metrics",
        marker=MARKER_INTERVAL,
        interval_ms=1000,
        status=SCRAPE_OK,
        clock_domain=CLOCK,
        duration_ms=duration_ms,
        scrape=compact,
        discovery=discover(compact),
    )


def series(
    texts: Sequence[str | None], *, step_s: float = 1.0
) -> list[VllmScrapeRecord]:
    """One scrape per text, ``step_s`` apart."""
    return [scrape(text, index * step_s) for index, text in enumerate(texts)]
