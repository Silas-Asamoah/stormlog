"""How long ``ExportPipeline.observe`` takes on the producer's thread.

Reported, never asserted: the numbers depend on the machine and its load.
The worker applies records and four threads scrape the shared render
meanwhile, as in a run; with ``--spans``, the span exporter also builds
every request's span and writes the batches to an OTLP JSON file. Under the GIL a producer can wait a whole switch
interval (``sys.getswitchinterval()``) whenever another thread is running
Python; the lock discipline only limits how often that happens.

    python scripts/benchmark_export_observe.py --records 100000
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

from stormlog.infer.export import ExportPipeline
from stormlog.infer.export_config import ExportConfig
from stormlog.infer.export_metrics import ProfileLabels, summarize_chunk_gaps
from stormlog.infer.export_spans import SpanIdentity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=int, default=100_000)
    parser.add_argument(
        "--rate",
        type=float,
        default=0.0,
        help="Records per second, as a run produces them; 0 is a tight loop, "
        "which keeps the worker busy and is the worst case for the GIL.",
    )
    parser.add_argument("--gaps", type=int, default=1024)
    parser.add_argument("--scrapers", type=int, default=4)
    parser.add_argument(
        "--scrape-interval",
        type=float,
        default=1.0,
        help="Seconds between one scraper's requests (Prometheus uses 15-60).",
    )
    parser.add_argument(
        "--spans",
        action="store_true",
        help="Export spans too, to an OTLP JSON file in a temporary directory.",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        spans_file = Path(directory) / "spans.jsonl" if args.spans else None
        pipeline = ExportPipeline(
            ExportConfig(prometheus_textfile_dir=Path(directory), otlp_file=spans_file),
            ProfileLabels(
                model="m",
                server="http://127.0.0.1:8000",
                cases=(("c1", "closed"),),
                run_id="r",
                session_id="s",
                version="0",
            ),
            spans=SpanIdentity(
                run_id="r",
                session_id="s",
                model="m",
                endpoint="http://127.0.0.1:8000/v1/chat/completions",
            ),
        )
        pipeline.start(started_at=time.time())
        gaps = [5.0] * args.gaps
        record: dict[str, object] = {
            "event_type": "infer.request",
            "case_id": "c1",
            "phase": "measured",
            "status": "ok",
            "request_id": "c1_measured_0",
            "x_request_id": "stormlog-r-c1_measured_0",
            "started_at_ns": time.time_ns(),
            "ended_at_ns": time.time_ns() + 120_000_000,
            "e2e_latency_ms": 120.0,
            "ttft_ms": 20.0,
            "arrival_mode": "closed",
            "dispatch_lag_ms": 0.5,
            "prompt_tokens": 512,
            "prompt_token_source": "server_usage",
            "output_tokens": 128,
            "output_token_source": "server_usage",
            "chunk_interarrival_ms": gaps,
        }
        extras = {"chunk_summary": summarize_chunk_gaps(gaps)}
        stop = threading.Event()

        def scrape() -> None:
            while not stop.is_set():
                generation = pipeline.renders.acquire()
                pipeline.renders.release(generation)
                stop.wait(args.scrape_interval)

        scrapers = [
            threading.Thread(target=scrape, daemon=True) for _ in range(args.scrapers)
        ]
        for thread in scrapers:
            thread.start()
        durations = []
        gap = 1.0 / args.rate if args.rate > 0 else 0.0
        next_at = time.perf_counter()
        for _ in range(args.records):
            if gap:
                next_at += gap
                time.sleep(max(0.0, next_at - time.perf_counter()))
            started = time.perf_counter_ns()
            pipeline.observe(record, extras)
            durations.append(time.perf_counter_ns() - started)
        stop.set()
        pipeline.close(5.0)
    durations.sort()

    def at(fraction: float) -> float:
        return durations[min(len(durations) - 1, int(fraction * len(durations)))] / 1e3

    print(
        f"observe() over {args.records} records "
        f"({'a tight loop' if not args.rate else f'{args.rate:g}/s'}), "
        f"{args.gaps} chunk gaps each, {'spans and ' if args.spans else ''}"
        f"{args.scrapers} scrapers every {args.scrape_interval:g} s, "
        f"switch interval {sys.getswitchinterval() * 1e3:g} ms"
    )
    print(
        f"  p50 {at(0.5):.1f} us  p99 {at(0.99):.1f} us  p99.9 {at(0.999):.1f} us  "
        f"max {durations[-1] / 1e3:.1f} us  mean {statistics.fmean(durations) / 1e3:.1f} us"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
