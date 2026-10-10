"""Offline validation and serialization for stormlog.mlx.profile v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from stormlog.session import session_summary_from_dict

from .models import MemorySnapshot, ProfileResult
from .runtime import validate_bytes

FORMAT = "stormlog.mlx.profile"


def validate_profiles(payload: Any) -> None:
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("Expected stormlog.mlx.profile artifact")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("Unsupported MLX profile schema_version")
    _validate_keys(payload, {"format", "schema_version", "profiles"})
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("MLX profiles must be an array")
    for item in profiles:
        _validate_profile(item)


def _validate_profile(item: Any) -> None:
    if (
        not isinstance(item, dict)
        or not isinstance(item.get("name"), str)
        or not item["name"]
    ):
        raise ValueError("Profile must be an object with a nonempty name")
    _validate_keys(item, set(ProfileResult.__dataclass_fields__))
    session_summary_from_dict(item.get("session_summary", {}))
    for key in ("started_at_ns", "ended_at_ns", "elapsed_ns", "valid_sample_count"):
        validate_bytes(item.get(key), key)
    if item["ended_at_ns"] < item["started_at_ns"]:
        raise ValueError("Profile end precedes start")
    _validate_policy(item)
    for key in ("baseline", "final"):
        _validate_snapshot(item.get(key))
    if not isinstance(item.get("metadata"), dict):
        raise ValueError("Profile metadata must be an object")


def _validate_policy(item: dict[str, Any]) -> None:
    if type(item.get("completion_verified")) is not bool:
        raise ValueError("completion_verified must be boolean")
    if item.get("status") not in {"completed", "incomplete", "interrupted"}:
        raise ValueError("Unsupported profile status")
    if item.get("peak_mode") not in {"sampled", "reset"}:
        raise ValueError("Unsupported profile peak policy")
    if item.get("sampled_peak_bytes") is not None:
        validate_bytes(item["sampled_peak_bytes"], "sampled_peak_bytes")


def _validate_snapshot(item: Any) -> None:
    if not isinstance(item, dict):
        raise ValueError("Profile snapshot must be an object")
    _validate_keys(item, set(MemorySnapshot.__dataclass_fields__))
    if not isinstance(item["name"], str) or not item["name"]:
        raise ValueError("Snapshot name must be nonempty")
    for key in ("timestamp_ns", "collection_duration_ns"):
        validate_bytes(item.get(key), key)
    for key in (
        "active_bytes",
        "cache_bytes",
        "runtime_peak_bytes",
        "process_rss_bytes",
    ):
        if key not in item:
            raise ValueError(f"Missing snapshot {key}")
        if item[key] is not None:
            validate_bytes(item[key], key)
    if not isinstance(item.get("metadata"), dict) or not isinstance(
        item.get("unavailable"), dict
    ):
        raise ValueError("Snapshot metadata/unavailable must be objects")


def _validate_keys(item: dict[str, Any], expected: set[str]) -> None:
    if set(item) != expected:
        raise ValueError(
            f"Artifact fields differ: missing {expected - set(item)}, "
            f"unknown {set(item) - expected}"
        )


def write_profiles(path: str | Path, profiles: Sequence[ProfileResult]) -> None:
    payload = {
        "format": FORMAT,
        "schema_version": 1,
        "profiles": [p.to_dict() for p in profiles],
    }
    validate_profiles(payload)
    Path(path).write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def load_profiles(path: str | Path) -> dict[str, Any]:
    payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_profiles(payload)
    return dict(payload)
