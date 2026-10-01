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
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from .tokens import TokenCount, TokenCounter, generate_prompt

REPEAT = "repeat"
UNIQUE = "unique"
SHARED_PREFIX = "shared-prefix"
PROMPT_MODES = (REPEAT, UNIQUE, SHARED_PREFIX)
# A nonce takes about eight subword tokens, so shorter prompts overshoot
# their target and cannot hold a set prefix share.
MIN_CONTROLLED_TOKENS = 32
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
    """One request's prompt and what it shares with others.

    Its exact token count is only worked out when something reads it: most
    servers report the prompt's tokens themselves.
    """

    text: str
    prompt_id: str
    counter: TokenCounter = field(repr=False, compare=False)
    prefix_group: int | None = None
    shared_prefix_tokens: int | None = None

    @cached_property
    def count(self) -> TokenCount:
        return self.counter.count_text(self.text)

    @cached_property
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
        self._digests: dict[int, str] = {}
        # Each group's prefix text and its token count.
        self._prefixes: dict[int, tuple[str, int]] = {}
        # Filler text by token count, shared by every prompt that needs it.
        self._fillers: dict[int, str] = {}
        self._word_tokens: list[int] | None = None
        self._repeated: Prompt | None = None

    def prompt(self, index: int) -> Prompt:
        """Build, or return the built, prompt for one request."""
        prompt = self._prompts.get(index)
        if prompt is None:
            prompt = self._build(index)
            self._prompts[index] = prompt
            self._digests[index] = prompt.digest
        return prompt

    def take(self, index: int) -> Prompt:
        """The prompt a request is about to use; forget it once it is done."""
        return self.prompt(index)

    def forget(self, index: int) -> None:
        """Drop a used prompt's text; its digest stays for the phase digest."""
        self._prompts.pop(index, None)

    def warm(self, sample: int = 64) -> None:
        """Do a phase's one-off prompt work before its clock starts.

        That is the repeated prompt, every group prefix, and the filler for
        each length the first ``sample`` nonces leave room for. Each later
        prompt then only tokenizes its short nonce. The sample prompts are
        not handed out, so they are not part of the phase digest.
        """
        for index in range(sample):
            self._build(index)
        if self.spec.mode == SHARED_PREFIX:
            for group in range(self.spec.groups):
                self._prefix(group)

    def prepare(self, indices: Iterable[int]) -> None:
        """Build prompts ahead of use."""
        for index in indices:
            self.prompt(index)

    def digest(self) -> str | None:
        """One digest over every prompt handed out, in index order."""
        if not self._digests:
            return None
        digests = [self._digests[index] for index in sorted(self._digests)]
        return _digest("\n".join(digests))

    def _build(self, index: int) -> Prompt:
        if self.spec.mode == REPEAT:
            return self._repeat()
        if self.spec.mode == UNIQUE:
            nonce = self._nonce(f"request:{index}")
            text = self._extend(f"[{nonce}]", self.input_tokens)
            return Prompt(text, f"r-{nonce}", self.counter)
        return self._shared(index)

    def _repeat(self) -> Prompt:
        # The same text Stormlog has always generated for this length.
        if self._repeated is None:
            text = generate_prompt(
                self.input_tokens, self.counter, seed=self.seed + self.input_tokens
            )
            self._repeated = Prompt(text, REPEAT, self.counter)
        return self._repeated

    def _shared(self, index: int) -> Prompt:
        group = self._group(index)
        prefix, prefix_tokens = self._prefix(group)
        nonce = self._nonce(f"request:{index}")
        marker = f"[{nonce}]"
        used = prefix_tokens + self._count(marker)
        text = self._extend(f"{prefix} {marker}", self.input_tokens, used=used)
        return Prompt(
            text,
            f"g{group}-{nonce}",
            self.counter,
            prefix_group=group,
            shared_prefix_tokens=prefix_tokens,
        )

    def _prefix(self, group: int) -> tuple[str, int]:
        if group not in self._prefixes:
            ratio = self.spec.shared_prefix_ratio or 0.0
            tokens = max(1, round(ratio * self.input_tokens))
            text = self._extend(f"[{self._nonce(f'prefix:{group}')}]", tokens)
            self._prefixes[group] = (text, self._count(text))
        return self._prefixes[group]

    def _extend(self, head: str, target_tokens: int, used: int | None = None) -> str:
        """``head`` and then filler, sized so the whole has about the target.

        Only the head is tokenized for each prompt: the filler for each size
        is built once per phase. Token counts are treated as adding up across
        the space between them, which holds for whitespace estimates and,
        apart from the odd merge, for subword tokenizers.
        """
        if used is None:
            used = self._count(head)
        filler = self._filler(target_tokens - used)
        return f"{head} {filler}" if filler else head

    def _filler(self, tokens: int) -> str:
        """The fewest filler words that reach ``tokens``, built once per length.

        The word count is estimated from each filler word's own token count,
        which subword tokenizers add up across spaces, then checked against
        the counter: usually two counts, a few more for a counter whose
        counts do not add up.
        """
        if tokens <= 0:
            return ""
        if tokens not in self._fillers:
            if self._word_tokens is None:
                self._word_tokens = [
                    max(1, self._count(f" {word}")) for word in _FILLER
                ]
            words = _estimated_words(tokens, self._word_tokens)
            while words > 1 and self._count(_filler_words(words - 1)) >= tokens:
                words -= 1
            while self._count(_filler_words(words)) < tokens:
                words += 1
            self._fillers[tokens] = _filler_words(words)
        return self._fillers[tokens]

    def _count(self, text: str) -> int:
        return self.counter.count_text(text).value

    def _group(self, index: int) -> int:
        draw = hashlib.sha256(f"{self._namespace}:group:{index}".encode()).digest()
        return int.from_bytes(draw[:8], "big") % self.spec.groups

    def _nonce(self, label: str) -> str:
        return _digest(f"{self._namespace}:{label}")[:12]


def _estimated_words(tokens: int, word_tokens: list[int]) -> int:
    words = total = 0
    while total < tokens:
        total += word_tokens[words % len(word_tokens)]
        words += 1
    return words


def _filler_words(count: int) -> str:
    return " ".join(_FILLER[index % len(_FILLER)] for index in range(count))


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
