"""Framework-free incremental parsing of PCM WAV segments and STT NDJSON."""

from __future__ import annotations

import json
import struct
from typing import Any


class WavSegments:
    """Count actual PCM frames across concatenated RIFF/WAVE containers.

    Data is consumed incrementally. The first playable timestamp requires a
    complete PCM frame; a header or an arbitrary HTTP read never counts.
    """

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.bytes_seen = 0
        self.pending = bytearray()
        self.remaining: int | None = None
        self.chunk: bytes | None = None
        self.chunk_remaining = 0
        self.chunk_size = 0
        self.padding = 0
        self.format: tuple[int, int, int] | None = None
        self.frames = 0
        self.data_bytes = 0
        self.segments: list[dict[str, Any]] = []
        self.first_audio_at: float | None = None

    def feed(self, data: bytes, observed_at: float) -> None:
        self.bytes_seen += len(data)
        if self.bytes_seen > self.max_bytes:
            raise ValueError("WAV exceeded byte limit")
        self.pending.extend(data)
        while self._advance(observed_at):
            pass

    def _advance(self, now: float) -> bool:
        if self.remaining is None:
            return self._header()
        if self.chunk is not None:
            return self._body(now)
        if self.padding:
            if not self.pending:
                return False
            self._take(1)
            self.padding = 0
            return True
        if self.remaining == 0:
            self._finish_segment()
            return True
        return self._next_chunk()

    def _next_chunk(self) -> bool:
        assert self.remaining is not None
        if self.remaining < 8:
            raise ValueError("Invalid WAV chunk header length")
        if len(self.pending) < 8:
            return False
        header = self._take(8)
        self.chunk, self.chunk_size = struct.unpack("<4sI", header)
        self.chunk_remaining = self.chunk_size
        self.padding = self.chunk_size % 2
        if self.chunk_size + self.padding > self.remaining:
            raise ValueError("WAV chunk exceeds RIFF bounds")
        self._validate_chunk()
        return True

    def _validate_chunk(self) -> None:
        if self.chunk == b"fmt " and (
            self.format is not None or self.chunk_size < 16 or self.chunk_size > 4096
        ):
            raise ValueError("Invalid or duplicate WAV fmt chunk")
        if self.chunk == b"data" and self.format is None:
            raise ValueError("WAV data precedes fmt")

    def _header(self) -> bool:
        if len(self.pending) < 12:
            return False
        magic, size, wave = struct.unpack("<4sI4s", self.pending[:12])
        if magic != b"RIFF" or wave != b"WAVE" or not 4 <= size <= self.max_bytes - 8:
            raise ValueError("Only bounded RIFF/WAVE PCM audio is supported")
        del self.pending[:12]
        self.remaining = size - 4
        self.format = None
        self.frames = self.data_bytes = 0
        return True

    def _take(self, size: int) -> bytes:
        data = bytes(self.pending[:size])
        del self.pending[:size]
        assert self.remaining is not None
        self.remaining -= size
        return data

    def _body(self, now: float) -> bool:
        if self.chunk == b"fmt ":
            if len(self.pending) < self.chunk_remaining:
                return False
            self._format(self._take(self.chunk_remaining))
            self.chunk_remaining = 0
        elif self.chunk_remaining:
            size = min(len(self.pending), self.chunk_remaining)
            if not size:
                return False
            self._take(size)
            self.chunk_remaining -= size
            if self.chunk == b"data":
                self._frames(size, now)
        if self.chunk_remaining == 0:
            if self.chunk == b"data" and self.format is not None:
                if self.chunk_size % self.format[2]:
                    raise ValueError("WAV data contains an incomplete PCM frame")
            self.chunk = None
        return True

    def _format(self, data: bytes) -> None:
        codec, channels, rate, byte_rate, align, bits = struct.unpack(
            "<HHIIHH", data[:16]
        )
        if (
            codec != 1
            or not 1 <= channels <= 32
            or not 1 <= rate <= 384000
            or bits not in {8, 16, 24, 32}
        ):
            raise ValueError("Unsupported WAV PCM format")
        if align != channels * (bits // 8) or byte_rate != rate * align:
            raise ValueError("Inconsistent WAV PCM frame layout")
        self.format = (channels, rate, align)

    def _frames(self, size: int, now: float) -> None:
        assert self.format is not None
        self.data_bytes += size
        self.frames = self.data_bytes // self.format[2]
        if self.frames and self.first_audio_at is None:
            self.first_audio_at = now

    def _finish_segment(self) -> None:
        if self.format is None or not self.frames:
            raise ValueError("WAV segment has no nonempty PCM frames")
        channels, rate, align = self.format
        self.segments.append(
            {
                "channels": channels,
                "sample_rate": rate,
                "frame_bytes": align,
                "frames": self.frames,
                "duration_seconds": self.frames / rate,
            }
        )
        self.remaining = None

    def finish(self) -> list[dict[str, Any]]:
        if self.remaining is not None or self.pending or not self.segments:
            raise ValueError("Truncated or empty WAV response")
        return self.segments

    @property
    def duration_seconds(self) -> float:
        return sum(float(segment["duration_seconds"]) for segment in self.segments)


class TranscriptStream:
    """Pinned MLX-Audio NDJSON semantics: text deltas, or full segment records.

    Records carrying segments/language contain the full transcription and
    replace accumulated deltas. This is not a generic cumulative-text guess.
    """

    def __init__(self, max_record_bytes: int = 1024**2) -> None:
        self.max_record_bytes = max_record_bytes
        self.pending = bytearray()
        self.text = ""
        self.records = 0
        self.first_transcript_at: float | None = None
        self.full_record = False

    def feed(self, data: bytes, observed_at: float) -> None:
        self.pending.extend(data)
        while b"\n" in self.pending:
            line, _, rest = self.pending.partition(b"\n")
            self.pending = bytearray(rest)
            self._record(bytes(line).rstrip(b"\r"), observed_at)
        if len(self.pending) > self.max_record_bytes:
            raise ValueError("NDJSON record exceeded byte limit")

    def _record(self, line: bytes, now: float) -> None:
        if not line.strip():
            return
        if len(line) > self.max_record_bytes:
            raise ValueError("NDJSON record exceeded byte limit")
        payload = json.loads(line.decode("utf-8"))
        self.accept(payload, now)

    def accept(self, payload: Any, now: float) -> None:
        if not isinstance(payload, dict):
            raise ValueError("Transcription must be a JSON object")
        if payload.get("error") is not None:
            # Never persist server transcript/error content as an error string.
            raise ValueError("Server returned a transcription error record")
        text = payload.get("text")
        if not isinstance(text, str):
            raise ValueError("Transcription record requires string text")
        full = "segments" in payload or "language" in payload
        if full:
            self.text = text
            self.full_record = True
        elif not self.full_record:
            self.text += text
        else:
            raise ValueError("Text delta after full transcription record")
        self.records += 1
        if text and self.first_transcript_at is None:
            self.first_transcript_at = now

    def finish(self, observed_at: float) -> str:
        if self.pending:
            self._record(bytes(self.pending), observed_at)
            self.pending.clear()
        if not self.records:
            raise ValueError("Empty transcription response")
        return self.text
