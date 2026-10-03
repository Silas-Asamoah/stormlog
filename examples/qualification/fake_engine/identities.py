"""The fake engine's generated identities, random or from a seed."""

from __future__ import annotations

import random
import secrets
import threading


class Identities:
    """Hex identities: request IDs, vLLM's request suffixes, span and trace
    IDs. Unseeded they are random, as vLLM's are; seeded they repeat from run
    to run for the same order of arrivals."""

    def __init__(self, seed: int | None = None) -> None:
        self._random = None if seed is None else random.Random(seed)
        self._lock = threading.Lock()

    def hex(self, nbytes: int) -> str:
        if self._random is None:
            return secrets.token_hex(nbytes)
        with self._lock:
            return self._random.getrandbits(nbytes * 8).to_bytes(nbytes, "big").hex()


__all__ = ["Identities"]
