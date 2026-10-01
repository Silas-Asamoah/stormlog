"""Deterministic prompts with controlled prefix sharing.

Serving engines cache the key/value state of prompt prefixes they have
seen. Whether two requests share a prefix therefore changes how much work
the second one needs, so a benchmark has to choose it on purpose:

- ``repeat`` sends one prompt for every request of a case, warmup included,
  as Stormlog always has. After the first request, most of each prompt can
  come from the cache.
- ``unique`` starts every request with its own nonce, so no two requests
  share a prefix beyond whatever the server's chat template adds.
- ``shared-prefix`` gives each request one of N seeded group prefixes that
  covers a set share of its tokens, then a request nonce and filler.

Nonces depend on the seed, the case and the phase, so a run can be repeated
exactly, and neither another case nor the warmup shares a prefix with the
measured requests.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .tokens import TokenCount, TokenCounter, generate_prompt

REPEAT = "repeat"
UNIQUE = "unique"
SHARED_PREFIX = "shared-prefix"
PROMPT_MODES = (REPEAT, UNIQUE, SHARED_PREFIX)
# Bump when the generated text changes, so digests from older runs are not
# mistaken for the same prompts.
GENERATOR_VERSION = 2

_FILLER = (
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
)


@dataclass(frozen=True)
class PromptSpec:
    """How the prompts of a run share their prefixes."""

    mode: str = REPEAT
    shared_prefix_ratio: float | None = None
    prefix_groups: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in PROMPT_MODES:
            raise ValueError(f"prompt mode must be one of {', '.join(PROMPT_MODES)}")
        shared = self.mode == SHARED_PREFIX
        if shared != (self.shared_prefix_ratio is not None):
            raise ValueError("a shared-prefix ratio goes with shared-prefix prompts")
        if self.prefix_groups is not None and not shared:
            raise ValueError("prefix groups go with shared-prefix prompts")
        ratio = self.shared_prefix_ratio
        if ratio is not None and not 0 < ratio < 1:
            raise ValueError("shared-prefix ratio must be between 0 and 1")
        if self.prefix_groups is not None and self.prefix_groups < 1:
            raise ValueError("prefix groups must be >= 1")

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"mode": self.mode}
        if self.mode == SHARED_PREFIX:
            record["shared_prefix_ratio"] = self.shared_prefix_ratio
            record["prefix_groups"] = self.groups
        return record

    @property
    def groups(self) -> int:
        return self.prefix_groups or 1


@dataclass(frozen=True)
class Prompt:
    """One request's prompt and what it shares with others."""

    text: str
    count: TokenCount
    prompt_id: str
    prefix_group: int | None = None

    @property
    def digest(self) -> str:
        return _digest(self.text)


class PromptSource:
    """The prompts of one case phase, generated on demand and cached."""

    def __init__(
        self,
        spec: PromptSpec,
        *,
        counter: TokenCounter,
        seed: int,
        case_id: str,
        phase: str,
        input_tokens: int,
    ) -> None:
        self.spec = spec
        self.counter = counter
        self.seed = seed
        self.input_tokens = input_tokens
        self._namespace = f"{seed}:{case_id}:{phase}"
        self._prompts: dict[int, Prompt] = {}
        self._prefixes: dict[int, str] = {}
        self._repeated: Prompt | None = None

    def prompt(self, index: int) -> Prompt:
        if index not in self._prompts:
            self._prompts[index] = self._build(index)
        return self._prompts[index]

    def prepare(self, indices: Iterable[int]) -> None:
        """Build prompts ahead of a schedule, so building them adds no delay."""
        for index in indices:
            self.prompt(index)

    def digest(self) -> str | None:
        """One digest over every prompt handed out, in index order."""
        if not self._prompts:
            return None
        digests = [self._prompts[index].digest for index in sorted(self._prompts)]
        return _digest("\n".join(digests))

    def _build(self, index: int) -> Prompt:
        if self.spec.mode == REPEAT:
            return self._repeat()
        if self.spec.mode == UNIQUE:
            nonce = self._nonce(f"request:{index}")
            text = _sized(f"[{nonce}] ", self.input_tokens, self.counter, index)
            return Prompt(text, self.counter.count_text(text), f"r-{nonce}")
        return self._shared(index)

    def _repeat(self) -> Prompt:
        # The same text Stormlog has always generated for this length.
        if self._repeated is None:
            text = generate_prompt(
                self.input_tokens, self.counter, seed=self.seed + self.input_tokens
            )
            self._repeated = Prompt(text, self.counter.count_text(text), REPEAT)
        return self._repeated

    def _shared(self, index: int) -> Prompt:
        group = self._group(index)
        prefix = self._prefix(group)
        nonce = self._nonce(f"request:{index}")
        text = _sized(f"{prefix} [{nonce}] ", self.input_tokens, self.counter, index)
        return Prompt(
            text, self.counter.count_text(text), f"g{group}-{nonce}", prefix_group=group
        )

    def _prefix(self, group: int) -> str:
        if group not in self._prefixes:
            ratio = self.spec.shared_prefix_ratio or 0.0
            tokens = max(1, round(ratio * self.input_tokens))
            head = f"[{self._nonce(f'prefix:{group}')}] "
            self._prefixes[group] = _sized(head, tokens, self.counter, group)
        return self._prefixes[group]

    def _group(self, index: int) -> int:
        draw = hashlib.sha256(f"{self._namespace}:group:{index}".encode()).digest()
        return int.from_bytes(draw[:8], "big") % self.spec.groups

    def _nonce(self, label: str) -> str:
        return _digest(f"{self._namespace}:{label}")[:12]


def _sized(head: str, target_tokens: int, counter: TokenCounter, start: int) -> str:
    """``head`` plus the fewest filler words that reach ``target_tokens``."""
    words = [_FILLER[(start + i) % len(_FILLER)] for i in range(target_tokens * 2)]

    def text(count: int) -> str:
        return (head + " ".join(words[:count])).rstrip()

    low, high = 0, len(words)
    while low < high:
        middle = (low + high) // 2
        if counter.count_text(text(middle)).value >= target_tokens:
            high = middle
        else:
            low = middle + 1
    return text(low)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
