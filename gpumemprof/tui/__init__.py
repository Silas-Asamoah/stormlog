"""Textual-based terminal UI for Stormlog."""


def run_app() -> None:
    """Launch the TUI app while keeping textual imports lazy."""
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

    _run_app()


__all__ = ["run_app"]
