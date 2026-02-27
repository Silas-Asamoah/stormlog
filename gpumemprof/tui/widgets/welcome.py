"""Welcome banner widget used by the Textual TUI."""

from __future__ import annotations

import logging
from typing import Any

from rich.text import Text
from textual.widgets import Static


class AsciiWelcome(Static):
    """Animated ASCII welcome banner, uses pyfiglet when available."""

    def __init__(
        self,
        messages: list[str],
        font: str = "Standard",
        interval: float = 3.0,
        figlet_cls: Any | None = None,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__("", **kwargs)
        self.messages = messages or ["Stormlog"]
        self.font_name = font
        self.interval = interval
        self._frame_index = 0
        self._figlet = None
        self._figlet_cls = figlet_cls
        self._logger = logger or logging.getLogger(__name__)

        if self._figlet_cls:
            try:
                self._figlet = self._figlet_cls(font=self.font_name)
            except Exception as exc:
                self._logger.debug("Figlet initialization failed: %s", exc)
                self._figlet = None

    def on_mount(self) -> None:
        self._render_frame()
        if len(self.messages) > 1:
            self.set_interval(self.interval, self._advance_frame)

    def _advance_frame(self) -> None:
        self._frame_index = (self._frame_index + 1) % len(self.messages)
        self._render_frame()

    def _render_frame(self) -> None:
        message = self.messages[self._frame_index]
        ascii_text = self._render_ascii(message)
        self.update(ascii_text)

    def _render_ascii(self, message: str) -> Text:
        if self._figlet:
            try:
                rendered = self._figlet.renderText(message)
                return Text(rendered.rstrip())
            except Exception as exc:
                self._logger.debug("Figlet render failed, using fallback: %s", exc)

        title = "Stormlog"
        width = max(28, len(title), len(message))
        border = f"+{'-' * (width + 2)}+"
        fallback = "\n".join(
            [
                border,
                f"| {title.center(width)} |",
                f"| {message.center(width)} |",
                border,
            ]
        )
        return Text(fallback)
