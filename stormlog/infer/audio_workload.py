"""Bounded, deterministic audio fixtures and privacy-preserving identities."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio_config import AudioProfileConfig, validate_settings
from .audio_stream import WavSegments
from .errors import InferInputError


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def bounded_file(path: Path, limit: int) -> bytes:
    if not path.is_file():
        raise ValueError("Fixture must be a regular readable file")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Fixture file exceeded byte limit")
    return data


@dataclass(frozen=True)
class AudioFixture:
    fixture_id: str
    task: str
    settings: dict[str, Any]
    content_digest: str
    text: str | None = None
    audio: bytes | None = None
    audio_seconds: float | None = None
    audio_segments: tuple[dict[str, Any], ...] = ()

    def identity(self) -> dict[str, Any]:
        return {
            "id": self.fixture_id,
            "task": self.task,
            "content_digest": self.content_digest,
            "settings": {
                key: (
                    {"sha256": digest(value)}
                    if key in {"instruct", "context", "text"}
                    else value
                )
                for key, value in self.settings.items()
            },
            "input_audio_seconds": self.audio_seconds,
            "audio_segments": list(self.audio_segments),
        }


class AudioWorkload:
    def __init__(self, config: AudioProfileConfig) -> None:
        self.config = config
        path = Path(config.manifest_path)
        try:
            raw = bounded_file(path, config.max_manifest_bytes)
            self.fixtures = self._parse(raw, path.parent)
        except (OSError, ValueError, UnicodeError) as exc:
            raise InferInputError(
                f"Invalid audio manifest or fixture: {type(exc).__name__}"
            ) from exc
        if not self.fixtures:
            raise InferInputError("Audio manifest must contain fixtures")

    def _parse(self, raw: bytes, directory: Path) -> tuple[AudioFixture, ...]:
        fixtures = []
        ids: set[str] = set()
        retained = 0
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(
                line,
                object_pairs_hook=_unique_fields,
                parse_constant=lambda _: (_ for _ in ()).throw(
                    ValueError("Nonfinite JSON")
                ),
            )
            fixture = self._fixture(row, directory)
            if fixture.fixture_id in ids:
                raise ValueError("Duplicate fixture ID")
            ids.add(fixture.fixture_id)
            retained += len(fixture.audio or b"") + len((fixture.text or "").encode())
            if retained > self.config.max_total_fixture_bytes:
                raise ValueError("Manifest total fixture bytes exceeded limit")
            fixtures.append(fixture)
        return tuple(fixtures)

    def _row(self, row: Any) -> tuple[str, str, dict[str, Any]]:
        task = self.config.task
        field = "text" if task == "tts" else "audio"
        if not isinstance(row, dict) or set(row) - {
            "schema_version",
            "id",
            "task",
            field,
            "settings",
        }:
            raise ValueError("Invalid manifest fields")
        if (
            type(row.get("schema_version")) is not int
            or row["schema_version"] != 1
            or row.get("task") != task
        ):
            raise ValueError("Unknown manifest version or mixed task")
        fixture_id = _fixture_id(row)
        settings = validate_settings(task, row.get("settings", {}))
        content = row.get(field)
        if not isinstance(content, str) or not content:
            raise ValueError("Missing fixture content")
        return fixture_id, content, settings

    def _fixture(self, row: Any, directory: Path) -> AudioFixture:
        fixture_id, content, settings = self._row(row)
        if self.config.task == "tts":
            if len(content.encode()) > self.config.max_fixture_bytes:
                raise ValueError("Text fixture exceeded byte limit")
            return AudioFixture(
                fixture_id,
                "tts",
                settings,
                digest({"text": content, "settings": settings}),
                text=content,
            )
        return self._speech_fixture(fixture_id, content, settings, directory)

    def _speech_fixture(
        self, fixture_id: str, content: str, settings: dict[str, Any], directory: Path
    ) -> AudioFixture:
        task = "stt"
        data = bounded_file(directory / content, self.config.max_fixture_bytes)
        parser = WavSegments(self.config.max_fixture_bytes)
        parser.feed(data, 0)
        segments = parser.finish()
        if (
            len(segments) != 1
            or parser.duration_seconds > self.config.max_audio_seconds
        ):
            raise ValueError("Input requires one bounded PCM WAV segment")
        return AudioFixture(
            fixture_id,
            task,
            settings,
            digest(
                {"audio_sha256": hashlib.sha256(data).hexdigest(), "settings": settings}
            ),
            audio=data,
            audio_seconds=parser.duration_seconds,
            audio_segments=tuple(segments),
        )

    def select(self, index: int) -> AudioFixture:
        return self.fixtures[(index + self.config.seed) % len(self.fixtures)]

    def record(self, session_id: str) -> dict[str, Any]:
        config = self.config
        cases: list[dict[str, Any]] = [
            {
                "case_id": case.case_id,
                "arrival": case.arrival.to_record(),
                "concurrency": None if case.arrival.open_loop else case.concurrency,
                "max_in_flight": case.concurrency if case.arrival.open_loop else None,
            }
            for case in config.cases()
        ]
        # Replay source filenames are metadata, not input identity.
        for case in cases:
            trace = case["arrival"].get("trace")
            if trace:
                trace.pop("source", None)
        spec = {
            "task": config.task,
            "seed": config.seed,
            "selection": "cycle_index_plus_seed_v1",
            "fixtures": [fixture.identity() for fixture in self.fixtures],
            "cases": cases,
            "response_format": config.response_format,
            "stream": config.stream,
            "warmup": {"requests": config.warmup_requests, "fixtures": "same_cycle"},
            "measurement": {
                "request_count": config.request_count,
                "duration_seconds": config.duration_seconds,
                "overflow": config.overflow,
                "timeout_seconds": config.timeout_seconds,
                "drain_timeout_seconds": config.drain_timeout,
                "max_response_bytes": config.max_response_bytes,
                "max_audio_seconds": config.max_audio_seconds,
            },
        }
        return {
            "schema_version": 1,
            "event_type": "infer.audio.workload",
            "session_id": session_id,
            "engine": config.engine,
            "model": config.model,
            "workload_digest": digest(spec),
            **spec,
        }


def _fixture_id(row: dict[str, Any]) -> str:
    fixture_id = row.get("id")
    if (
        not isinstance(fixture_id, str)
        or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", fixture_id) is None
    ):
        raise ValueError("Invalid fixture ID")
    return fixture_id


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result
