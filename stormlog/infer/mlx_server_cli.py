"""Lazy same-process server entrypoint, separate from framework-free clients."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from .errors import InferUsageError


def add_mlx_serve_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "mlx-serve",
        help="Capture an explicitly launched MLX server in this interpreter",
    )
    parser.add_argument(
        "--engine", choices=["omlx", "mlx-lm", "mlx-audio"], required=True
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--max-history", type=int, default=1000)
    parser.add_argument("server_args", nargs=argparse.REMAINDER, metavar="SERVER_ARGS")


def cmd_mlx_serve(args: argparse.Namespace) -> int:
    from ..mlx.artifacts import CaptureOptions
    from ..mlx.runtime import MLXRuntimeError
    from .mlx_server import run_mlx_server

    try:
        options = CaptureOptions(interval=args.interval, max_history=args.max_history)
        _directory, outcome = run_mlx_server(
            engine=args.engine,
            run_id=args.run_id,
            output=args.output,
            args=args.server_args,
            options=options,
        )
    except (ValueError, ImportError, MLXRuntimeError) as exc:
        raise InferUsageError(str(exc)) from exc
    if outcome.error:
        print(
            f"Server failed: {outcome.error}; evidence: {_directory}", file=sys.stderr
        )
    return int(outcome.exit_code)
