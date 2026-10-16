"""Top-level Stormlog console entrypoint."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the `stormlog` console script."""
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    if not resolved_argv:
        from .tui import run_app

        run_app([])
        return 0

    command = resolved_argv[0]
    if command == "tui":
        return _run_tui(resolved_argv[1:])
    if command == "infer":
        from .infer.cli import main as infer_main

        # Run as the console script (no argv given), the process exits
        # right after, so signals a watch ignored stay ignored.
        return infer_main(resolved_argv[1:], restore_signals=argv is not None)
    if command == "query":
        from .query_cli import main as query_main

        return query_main(resolved_argv[1:])
    if command in {"-h", "--help"}:
        _build_parser().print_help()
        return 0

    parser = _build_parser()
    parser.error(f"unknown command: {command}")


def _run_tui(argv: list[str]) -> int:
    from .tui import run_app

    if argv:
        parser = argparse.ArgumentParser(
            prog="stormlog tui", description="Launch the Stormlog Textual TUI."
        )
        parser.add_argument(
            "--framework", choices=("pytorch", "mlx"), default="pytorch"
        )
        parser.parse_args(argv)
        run_app(argv)
    else:
        run_app([])
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stormlog",
        description=(
            "Stormlog command surface. Running `stormlog` with no arguments "
            "opens the Textual TUI."
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["tui", "query", "infer"],
        help="Command group. Omit to launch the TUI.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
