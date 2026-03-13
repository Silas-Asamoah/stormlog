"""Markdown panel widgets used by the Textual TUI."""

from __future__ import annotations

from typing import Any, Callable

from textual.widgets import Markdown


class MarkdownPanel(Markdown):
    """Reusable Markdown panel with refresh support."""

    def __init__(self, builder: Callable[[], str], **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self.builder = builder

    def refresh_content(self) -> None:
        self.update(self.builder())

    def on_mount(self) -> None:
        self.refresh_content()
