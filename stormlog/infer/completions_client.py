"""Raw text completions using the shared text transport and timing contract."""

from __future__ import annotations

from typing import Any

from .openai_client import OpenAIChatCompletionsClient


class OpenAICompletionsClient(OpenAIChatCompletionsClient):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs, task="completion")
