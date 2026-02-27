"""Textual-based terminal UI for Stormlog."""


def run_app() -> None:
    """Launch the TUI app while keeping textual imports lazy."""
    from .app import run_app as _run_app

    _run_app()


__all__ = ["run_app"]
