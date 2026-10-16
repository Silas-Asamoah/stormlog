"""Bounded MLX-Audio HTTP client; no audio or inference runtime imports."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any

from .audio_config import AudioProfileConfig
from .audio_stream import TranscriptStream, WavSegments
from .audio_workload import AudioFixture
from .http_transport import (
    BoundedResponse,
    EndpointHTTPError,
    inference_opener,
    json_headers,
)


@dataclass(frozen=True)
class AudioResult:
    started_at_ns: int
    ended_at_ns: int
    e2e_latency_ms: float
    first_response_byte_ms: float | None
    first_audio_ms: float | None
    first_transcript_ms: float | None
    input_audio_seconds: float | None
    output_audio_seconds: float | None
    rtf: float | None
    response_bytes: int
    response_digest: str
    audio_segments: list[dict[str, Any]] = field(default_factory=list)
    media: bytes | None = None


class AudioCallError(RuntimeError):
    def __init__(self, cause: Exception, observations: dict[str, Any]) -> None:
        super().__init__(f"Audio response failed: {type(cause).__name__}")
        self.cause = cause
        self.observations = observations


def multipart(fixture: AudioFixture, config: AudioProfileConfig) -> tuple[bytes, str]:
    boundary = "stormlog-" + uuid.uuid4().hex
    fields = {
        "model": config.model,
        "stream": config.stream,
        "response_format": config.response_format,
        **fixture.settings,
    }
    parts: list[bytes] = []
    for key, value in fields.items():
        encoded = (
            value if isinstance(value, str) else json.dumps(value, allow_nan=False)
        )
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
            + encoded.encode()
            + b"\r\n"
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="fixture.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
        + (fixture.audio or b"")
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class AudioClient:
    def __init__(self, config: AudioProfileConfig) -> None:
        self.config = config
        self._opener = inference_opener()

    def complete(self, fixture: AudioFixture, *, request_id: str) -> AudioResult:
        config = self.config
        if config.task == "tts":
            body = json.dumps(
                {
                    "model": config.model,
                    "input": fixture.text,
                    **fixture.settings,
                    "response_format": "wav",
                    "stream": config.stream,
                },
                allow_nan=False,
            ).encode()
            headers = json_headers(config.api_key, accept="audio/wav")
        else:
            body, content_type = multipart(fixture, config)
            headers = json_headers(
                config.api_key,
                accept="application/x-ndjson" if config.stream else "application/json",
            )
            headers["Content-Type"] = content_type
        headers["X-Request-Id"] = request_id
        request = urllib.request.Request(
            config.endpoint, data=body, headers=headers, method="POST"
        )
        return self._perform(request, fixture)

    def _perform(
        self, request: urllib.request.Request, fixture: AudioFixture
    ) -> AudioResult:
        started_ns, started = time.time_ns(), time.perf_counter()
        state = _AudioResponse(self.config, fixture, started_ns, started)
        try:
            with self._opener.open(request, timeout=self.config.timeout_seconds) as raw:
                with BoundedResponse(
                    raw,
                    deadline=started + self.config.timeout_seconds,
                    max_bytes=self.config.max_response_bytes,
                ) as response:
                    state.read(response)
            return state.finish()
        except urllib.error.HTTPError as exc:
            # Error content can contain input text/transcripts/credentials;
            # HTTP status suffices and reading a body cannot delay cleanup.
            status = exc.code
            exc.close()
            cause = EndpointHTTPError(status, "Audio endpoint rejected the request")
            raise AudioCallError(cause, state.observations()) from exc
        except Exception as exc:
            raise AudioCallError(exc, state.observations()) from exc


class _AudioResponse:
    def __init__(
        self,
        config: AudioProfileConfig,
        fixture: AudioFixture,
        started_ns: int,
        started: float,
    ) -> None:
        self.config, self.fixture = config, fixture
        self.started_ns, self.started = started_ns, started
        self.first_byte: float | None = None
        self.wav = WavSegments(config.max_response_bytes)
        self.transcript = TranscriptStream(min(1024**2, config.max_response_bytes))
        self.buffer = bytearray()
        self.hash = hashlib.sha256()
        self.bytes = 0

    def read(self, response: BoundedResponse) -> None:
        while True:
            data = response.read_chunk()
            now = time.perf_counter()
            if not data:
                break
            if self.first_byte is None:
                self.first_byte = now
            self.hash.update(data)
            self.bytes += len(data)
            if self.config.retain_media_dir is not None or (
                self.config.task == "stt" and not self.config.stream
            ):
                self.buffer.extend(data)
            if self.config.task == "tts":
                self.wav.feed(data, now)
            elif self.config.stream:
                self.transcript.feed(data, now)

    def observations(self) -> dict[str, Any]:
        return {
            "started_at_ns": self.started_ns,
            "ended_at_ns": time.time_ns(),
            "e2e_latency_ms": (time.perf_counter() - self.started) * 1000,
            "first_response_byte_ms": self._latency(self.first_byte),
            "first_audio_ms": self._latency(self.wav.first_audio_at),
            "first_transcript_ms": self._latency(self.transcript.first_transcript_at),
            "response_bytes": self.bytes,
            "input_audio_seconds": self.fixture.audio_seconds,
        }

    def _latency(self, value: float | None) -> float | None:
        return None if value is None else (value - self.started) * 1000

    def finish(self) -> AudioResult:
        duration: float | None
        output_duration: float | None
        if self.config.task == "tts":
            segments = self.wav.finish()
            output_duration = self.wav.duration_seconds
            if output_duration > self.config.max_audio_seconds:
                raise ValueError("Output audio exceeded duration limit")
            duration = output_duration
        else:
            if not self.config.stream:
                self.transcript.accept(
                    json.loads(self.buffer.decode("utf-8")), time.perf_counter()
                )
            self.transcript.finish(time.perf_counter())
            segments = []
            output_duration = None
            duration = self.fixture.audio_seconds
        observations = self.observations()
        elapsed = observations["e2e_latency_ms"] / 1000
        return AudioResult(
            **observations,
            output_audio_seconds=output_duration,
            rtf=elapsed / duration if duration else None,
            response_digest=self.hash.hexdigest(),
            audio_segments=segments,
            media=(
                bytes(self.buffer) if self.config.retain_media_dir is not None else None
            ),
        )
