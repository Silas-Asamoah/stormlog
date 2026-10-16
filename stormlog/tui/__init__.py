"""Textual-based terminal UI and top-level Stormlog dispatcher."""

from __future__ import annotations

import argparse
import sys


def run_app(argv: list[str] | None = None) -> None:
    """Launch the TUI app or dispatch top-level Stormlog subcommands."""
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    if resolved_argv and resolved_argv[0] == "query":
        from stormlog.query_cli import main as query_main

        raise SystemExit(query_main(resolved_argv[1:]))
    if resolved_argv and resolved_argv[0] in {"-h", "--help"}:
        print("usage: stormlog [query ...]\n")
        print("Launches the Stormlog TUI with no arguments.")
        print("Use `stormlog query --help` to query local artifact directories.")
        return

    parser = argparse.ArgumentParser(prog="stormlog tui")
    parser.add_argument("--framework", choices=("pytorch", "mlx"), default="pytorch")
    args = parser.parse_args(resolved_argv)
    _launch_tui(args.framework)


def _launch_tui(framework: str) -> None:
    try:
        from .app import run_app as _run_app
    except ModuleNotFoundError as exc:
        if exc.name == "textual" or (
            isinstance(exc.name, str) and exc.name.startswith("textual.")
        ):
            raise SystemExit(
                "The Stormlog TUI requires optional dependencies. "
                "Install with `pip install 'stormlog[tui,torch]'`."
            ) from exc
        raise

    if framework == "pytorch":
        _run_app()
    else:
        _run_app(framework=framework)


__all__ = ["run_app"]
