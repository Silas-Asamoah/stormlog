"""Inflate gzip bodies from peers without letting a small one grow without bound."""

from __future__ import annotations

import zlib


def gunzip_capped(body: bytes | bytearray, cap: int) -> bytes | None:
    """Inflate a gzip body, or None when its output would exceed ``cap``.

    A gzip member a few hundred kilobytes long can hold gigabytes of zeros,
    so the decoder is asked for at most ``cap + 1`` bytes: one byte over the
    cap, or input left unconsumed, refuses the body without inflating it
    whole. A stream cut before its trailer raises ``ValueError``; a
    malformed one raises ``zlib.error``.
    """
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out = decoder.decompress(body, cap + 1)
    if len(out) > cap or decoder.unconsumed_tail:
        return None
    if not decoder.eof:
        raise ValueError("truncated gzip body")
    return out
