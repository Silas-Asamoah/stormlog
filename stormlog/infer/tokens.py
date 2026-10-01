"""Token counting helpers for inference profiling."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import logging
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenCount:
    """A token count plus provenance."""

    value: int
    source: str
    exact: bool


class TokenCounter(Protocol):
    """Counter interface used by workload generation and fallback accounting."""

    source: str
    exact: bool

    def count_text(self, text: str) -> TokenCount:
        """Count tokens in a text payload."""


class EstimatedTokenCounter:
    """A deterministic fallback counter based on whitespace-like chunks."""

    source = "estimated"
    exact = False

    def count_text(self, text: str) -> TokenCount:
        chunks = [chunk for chunk in text.replace("\n", " ").split(" ") if chunk]
        return TokenCount(value=max(1, len(chunks)), source=self.source, exact=False)

    def identity(self) -> dict[str, Any]:
        return {"source": self.source, "exact": False, "name": "whitespace words"}


class TiktokenCounter:
    """Token counter backed by tiktoken."""

    source = "tiktoken"
    exact = True

    def __init__(self, *, model: str | None, encoding_name: str | None) -> None:
        tiktoken = importlib.import_module("tiktoken")
        if encoding_name:
            self._encoding = tiktoken.get_encoding(encoding_name)
        elif model:
            self._encoding = tiktoken.encoding_for_model(model)
        else:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        self._version = getattr(tiktoken, "__version__", None)

    def count_text(self, text: str) -> TokenCount:
        return TokenCount(
            value=len(self._encoding.encode(text)),
            source=self.source,
            exact=True,
        )

    def identity(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "exact": True,
            "name": getattr(self._encoding, "name", None),
            "library_version": self._version,
        }


class TransformersTokenCounter:
    """Token counter backed by transformers.AutoTokenizer."""

    source = "transformers"
    exact = True

    def __init__(self, *, model: str) -> None:
        transformers = importlib.import_module("transformers")
        auto_tokenizer = transformers.AutoTokenizer
        self._tokenizer = auto_tokenizer.from_pretrained(model)
        self._model = model
        self._version = getattr(transformers, "__version__", None)

    def count_text(self, text: str) -> TokenCount:
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        return TokenCount(value=len(token_ids), source=self.source, exact=True)

    def identity(self) -> dict[str, Any]:
        init_kwargs = getattr(self._tokenizer, "init_kwargs", None) or {}
        name = getattr(self._tokenizer, "name_or_path", None) or self._model
        return {
            "source": self.source,
            "exact": True,
            "name": name,
            "revision": init_kwargs.get("_commit_hash") or _hub_revision(name),
            "library_version": self._version,
        }

    def chat_template_digest(self) -> str | None:
        """Digest of the local tokenizer's chat template, if it has one."""
        template = getattr(self._tokenizer, "chat_template", None)
        if not isinstance(template, str) or not template:
            return None
        return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


def _hub_revision(repo_id: str) -> str | None:
    """The commit of a Hugging Face Hub tokenizer in the local cache.

    transformers 5 no longer keeps the commit on the tokenizer, but the cache
    stores each download under ``snapshots/<commit>/``. A tokenizer loaded
    from a local directory has no revision.
    """
    try:
        hub = importlib.import_module("huggingface_hub")
        path = hub.try_to_load_from_cache(repo_id, "tokenizer_config.json")
    except Exception:
        return None
    if not isinstance(path, str):
        return None
    parts = PurePath(path).parts
    if "snapshots" not in parts or parts.index("snapshots") + 1 >= len(parts):
        return None
    return str(parts[parts.index("snapshots") + 1])


def build_token_counter(
    *,
    tokenizer: str,
    model: str,
    tokenizer_model: str | None = None,
    tiktoken_encoding: str | None = None,
    strict: bool = False,
) -> TokenCounter:
    """Build the requested token counter, falling back to estimates in auto mode."""
    normalized = tokenizer.strip().lower()
    resolved_model = tokenizer_model or model

    if normalized in {"none", "estimate", "estimated"}:
        return EstimatedTokenCounter()

    if normalized == "tiktoken":
        return TiktokenCounter(model=resolved_model, encoding_name=tiktoken_encoding)

    if normalized in {"transformers", "hf", "huggingface"}:
        return TransformersTokenCounter(model=resolved_model)

    if normalized != "auto":
        raise ValueError(
            "--tokenizer must be one of auto, none, tiktoken, transformers"
        )

    try:
        return TiktokenCounter(
            model=resolved_model,
            encoding_name=tiktoken_encoding,
        )
    except Exception as exc:
        logger.debug("tiktoken unavailable in auto mode: %s", exc)

    try:
        return TransformersTokenCounter(model=resolved_model)
    except Exception as exc:
        logger.debug("transformers unavailable in auto mode: %s", exc)

    if strict:
        raise RuntimeError("No configured tokenizer is available")
    return EstimatedTokenCounter()


def generate_prompt(target_tokens: int, counter: TokenCounter, *, seed: int) -> str:
    """Generate a deterministic prompt near the requested token count."""
    base_words = [
        "profile",
        "inference",
        "latency",
        "throughput",
        "memory",
        "tokens",
        "scheduler",
        "request",
        "streaming",
        "capacity",
    ]
    words: list[str] = []
    index = seed % len(base_words)
    while len(words) < target_tokens * 2:
        words.append(base_words[index % len(base_words)])
        candidate = " ".join(words)
        count = counter.count_text(candidate)
        if count.value >= target_tokens:
            return candidate
        index += 1
    return " ".join(words)
