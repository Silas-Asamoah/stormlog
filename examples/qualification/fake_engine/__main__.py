"""Run the fake engine as its own process, for signals and the kill switch.

``python -m examples.qualification.fake_engine --port 0`` prints one line,
``FAKE_ENGINE_URL=<url> PID=<pid>``, once it serves, and stops on SIGTERM or
SIGINT. SIGSTOP and SIGCONT then reach it like any server process.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Sequence

from .config import FakeEngineConfig
from .server import FakeEngine


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m examples.qualification.fake_engine"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--model", default=FakeEngineConfig.model)
    for name in (
        "max-num-seqs",
        "max-num-batched-tokens",
        "num-gpu-blocks",
        "block-size",
    ):
        default = getattr(FakeEngineConfig, name.replace("-", "_"))
        parser.add_argument(f"--{name}", type=int, default=default)
    for name in ("step-seconds", "prefill-token-seconds", "decode-token-seconds"):
        default = getattr(FakeEngineConfig, name.replace("-", "_"))
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--no-request-id-randomization", action="store_true")
    parser.add_argument("--hook-dir", type=Path, default=None)
    parser.add_argument("--hook-seal-seconds", type=float, default=60.0)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--no-dump-cuda-time-total", action="store_true")
    parser.add_argument("--spans-endpoint", default=None)
    parser.add_argument("--span-export-seconds", type=float, default=0.25)
    parser.add_argument(
        "--span-encoding", choices=("protobuf", "json"), default="protobuf"
    )
    parser.add_argument("--span-export-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def config_from_args(argv: Sequence[str] | None = None) -> FakeEngineConfig:
    args = _parser().parse_args(argv)
    return FakeEngineConfig(
        model=args.model,
        host=args.host,
        port=args.port,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        num_gpu_blocks=args.num_gpu_blocks,
        block_size=args.block_size,
        enable_prefix_caching=not args.no_prefix_caching,
        step_seconds=args.step_seconds,
        prefill_token_seconds=args.prefill_token_seconds,
        decode_token_seconds=args.decode_token_seconds,
        request_id_randomization=not args.no_request_id_randomization,
        hook_dir=args.hook_dir,
        hook_seal_seconds=args.hook_seal_seconds,
        trace_dir=args.trace_dir,
        torch_profiler_dump_cuda_time_total=not args.no_dump_cuda_time_total,
        spans_endpoint=args.spans_endpoint,
        span_export_seconds=args.span_export_seconds,
        span_encoding=args.span_encoding,
        span_export_timeout_seconds=args.span_export_timeout_seconds,
        seed=args.seed,
        allow_kill=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_args: stop.set())
    engine = FakeEngine(config_from_args(argv)).start()
    print(f"FAKE_ENGINE_URL={engine.base_url} PID={os.getpid()}", flush=True)
    try:
        while not stop.wait(0.2):
            pass
    finally:
        engine.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
