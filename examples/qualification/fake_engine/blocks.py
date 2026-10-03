"""KV blocks and the prefix cache, after vLLM's ``BlockPool``.

Free blocks wait in an LRU queue; a freed block keeps its hash, so a later
request with the same prefix can reuse it until an allocation evicts it from
the queue's head. A request's last full blocks are freed first, as in vLLM, so
they are evicted first. Like vLLM's ``BlockHashToBlockMap``, a hash keeps every
block cached under it, so evicting one copy leaves the others' hits.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    block_hash: int | None = None


class BlockPool:
    """A fixed set of blocks with LRU eviction of cached free blocks."""

    def __init__(self, num_blocks: int, block_size: int, *, caching: bool) -> None:
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be positive")
        self.block_size = block_size
        self.caching = caching
        self.blocks = [Block(index) for index in range(num_blocks)]
        self._free: OrderedDict[int, None] = OrderedDict(
            (block.block_id, None) for block in self.blocks
        )
        # Each hash's cached blocks, in the order they were cached.
        self._cached: dict[int, dict[int, None]] = {}
        self.evictions = 0

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def free_count(self) -> int:
        return len(self._free)

    def usage(self) -> float:
        return 1.0 - self.free_count / self.num_blocks

    def held_count(self) -> int:
        return sum(1 for block in self.blocks if block.ref_count > 0)

    def block_hashes(self, tokens: Sequence[str]) -> list[int]:
        """Chained hashes of the full blocks of ``tokens``."""
        hashes: list[int] = []
        parent = 0
        size = self.block_size
        for start in range(0, len(tokens) - size + 1, size):
            parent = hash((parent, tuple(tokens[start : start + size])))
            hashes.append(parent)
        return hashes

    def lookup(self, hashes: Sequence[int], prompt_len: int) -> list[int]:
        """The cached blocks a prompt can reuse: at least one token is left to
        compute, as vLLM leaves the last prompt token for its first step."""
        if not self.caching:
            return []
        limit = (prompt_len - 1) // self.block_size
        found: list[int] = []
        for block_hash in hashes[:limit]:
            copies = self._cached.get(block_hash)
            if not copies:
                break
            found.append(next(iter(copies)))
        return found

    def fits(self, cached: Sequence[int], fresh: int) -> bool:
        """Whether a request can take its ``cached`` hits and ``fresh`` new
        blocks. An idle hit leaves the free queue too, so it counts against
        the free blocks, as in vLLM."""
        idle = sum(1 for block_id in cached if self.blocks[block_id].ref_count == 0)
        return fresh + idle <= self.free_count

    def touch(self, block_ids: Sequence[int]) -> None:
        """Take a reference on cached blocks, out of the free queue if idle."""
        for block_id in block_ids:
            block = self.blocks[block_id]
            if block.ref_count == 0:
                self._free.pop(block_id, None)
            block.ref_count += 1

    def allocate(self, count: int) -> list[int] | None:
        """``count`` fresh blocks from the queue's head, or None if too few."""
        if count > self.free_count:
            return None
        taken: list[int] = []
        for _ in range(count):
            block_id, _none = self._free.popitem(last=False)
            block = self.blocks[block_id]
            if block.block_hash is not None:
                self._evict(block)
            block.ref_count = 1
            taken.append(block_id)
        return taken

    def cache(self, block_id: int, block_hash: int) -> None:
        """Mark a full block reusable under ``block_hash``, beside any other
        block already cached under it."""
        if not self.caching:
            return
        self.blocks[block_id].block_hash = block_hash
        self._cached.setdefault(block_hash, {})[block_id] = None

    def _evict(self, block: Block) -> None:
        """Forget one block's hash; other blocks under it keep theirs."""
        assert block.block_hash is not None
        copies = self._cached.get(block.block_hash, {})
        copies.pop(block.block_id, None)
        if not copies:
            self._cached.pop(block.block_hash, None)
        block.block_hash = None
        self.evictions += 1

    def free(self, block_ids: Sequence[int]) -> None:
        """Drop references, last block first; idle blocks join the queue's tail."""
        for block_id in reversed(block_ids):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._free[block_id] = None

    def reset(self) -> bool:
        """vLLM's reset: refused while any block is held, else forget hashes."""
        if self.held_count():
            return False
        for block in self.blocks:
            block.block_hash = None
        self._cached.clear()
        return True


__all__ = ["Block", "BlockPool"]
