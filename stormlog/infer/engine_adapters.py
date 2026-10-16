"""Versioned endpoint descriptors; importing them never initializes a runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ADAPTER_VERSION = 1
TASK_PATHS = {
    "chat": "/chat/completions",
    "completion": "/completions",
    "tts": "/audio/speech",
    "stt": "/audio/transcriptions",
}


@dataclass(frozen=True)
class EngineAdapter:
    name: str
    tasks: tuple[str, ...]
    source_revision: str | None = None

    def validate(self, task: str) -> None:
        if task not in self.tasks:
            raise ValueError(
                f"Engine {self.name!r} does not support task {task!r}; choose {', '.join(self.tasks)}"
            )

    def to_record(self, task: str, server_version: str | None = None) -> dict[str, Any]:
        self.validate(task)
        return {
            "engine": self.name,
            "task": task,
            "adapter_version": ADAPTER_VERSION,
            "server_version": server_version,
            "server_version_source": "user_declared" if server_version else "unknown",
            "protocol": (
                "openai_text" if task in ("chat", "completion") else "mlx_audio_http"
            ),
            "source_revision": self.source_revision,
            "request_execution_attribution": "unavailable",
            "scheduler_attribution": "unavailable",
            "prefill_decode_attribution": "unavailable",
            "kernel_attribution": "unavailable",
            "native_memory_scope": "unavailable_without_server_evidence",
        }


ENGINES = {
    "generic": EngineAdapter("generic", ("chat", "completion")),
    "vllm": EngineAdapter("vllm", ("chat", "completion")),
    "omlx": EngineAdapter(
        "omlx", ("chat", "completion"), "013bc3f3a48fa9a4b7dffbfeb6cdc7c1face1c5d"
    ),
    "mlx-lm": EngineAdapter(
        "mlx-lm", ("chat", "completion"), "ba8a08fd2ed72bbc53de84f89933f91fe4743e4a"
    ),
    "mlx-audio": EngineAdapter(
        "mlx-audio", ("tts", "stt"), "e66ad2ff4eb96ce1149000187b30370cbcd707aa"
    ),
}


def get_adapter(engine: str, task: str) -> EngineAdapter:
    try:
        adapter = ENGINES[engine]
    except KeyError as exc:
        raise ValueError(f"Unknown inference engine {engine!r}") from exc
    adapter.validate(task)
    return adapter
