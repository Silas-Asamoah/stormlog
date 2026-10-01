"""Internal strict parsing for the Stormlog run envelope v1 schema."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from ._run_catalog_models import (
    RUN_ENVELOPE_FORMAT,
    RUN_ENVELOPE_SCHEMA_VERSION,
    AttachmentStorage,
    CatalogRunAttachment,
    CatalogRunEnvelope,
    CatalogRunSessionRef,
)

_ENVELOPE_KEYS = {
    "schema_version",
    "format",
    "run_id",
    "title",
    "description",
    "job_id",
    "started_at_ns",
    "ended_at_ns",
    "created_at_utc",
    "updated_at_utc",
    "source_namespace",
    "source_ref",
    "tags",
    "sessions",
    "attachments",
    "metadata",
}
_SESSION_KEYS = {
    "session_id",
    "job_id",
    "rank",
    "local_rank",
    "world_size",
    "role",
    "source_namespace",
    "source_ref",
    "metadata",
}
_ATTACHMENT_KEYS = {
    "attachment_id",
    "title",
    "kind",
    "storage",
    "url",
    "path",
    "run_id",
    "session_id",
    "job_id",
    "rank",
    "local_rank",
    "world_size",
    "start_ns",
    "end_ns",
    "created_at_utc",
    "updated_at_utc",
    "source_namespace",
    "source_ref",
    "metadata",
}


def run_envelope_from_payload(
    payload: Mapping[str, Any],
    envelope_path: Path,
) -> CatalogRunEnvelope | None:
    """Parse a run envelope only when the complete v1 schema is satisfied."""
    if not _is_valid_run_envelope_payload(payload):
        return None

    run_id = cast(str, payload["run_id"])
    sessions = tuple(
        _run_session_ref_from_payload(cast(Mapping[str, Any], item))
        for item in cast(list[Any], payload.get("sessions", []))
    )
    attachments = tuple(
        _run_attachment_from_payload(
            cast(Mapping[str, Any], item),
            envelope_path,
            run_id,
        )
        for item in cast(list[Any], payload.get("attachments", []))
    )
    return CatalogRunEnvelope(
        run_id=run_id,
        path=envelope_path,
        title=_optional_string(payload, "title"),
        description=_optional_string(payload, "description", allow_empty=True),
        job_id=_optional_string(payload, "job_id"),
        started_at_ns=_optional_integer(payload, "started_at_ns"),
        ended_at_ns=_optional_integer(payload, "ended_at_ns"),
        created_at_utc=_optional_string(payload, "created_at_utc", allow_empty=True),
        updated_at_utc=_optional_string(payload, "updated_at_utc", allow_empty=True),
        source_namespace=_optional_string(payload, "source_namespace"),
        source_ref=_optional_string(payload, "source_ref"),
        tags=tuple(cast(list[str], payload.get("tags", []))),
        sessions=sessions,
        attachments=attachments,
        metadata=dict(cast(Mapping[str, Any], payload["metadata"])),
    )


def is_run_envelope(payload: Mapping[str, Any]) -> bool:
    """Return whether a payload advertises the exact run envelope v1 format."""
    schema_version = payload.get("schema_version")
    return (
        isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version == RUN_ENVELOPE_SCHEMA_VERSION
        and payload.get("format") == RUN_ENVELOPE_FORMAT
    )


def attachment_storage_or_none(value: Any) -> AttachmentStorage | None:
    """Parse an attachment storage value."""
    if value == "reference" or value == "copy":
        return cast(AttachmentStorage, value)
    return None


def attachment_storage_or_default(value: Any) -> AttachmentStorage:
    """Parse an attachment storage value, defaulting legacy sidecars."""
    return attachment_storage_or_none(value) or "reference"


def _is_valid_run_envelope_payload(payload: Mapping[str, Any]) -> bool:
    if not is_run_envelope(payload):
        return False
    if not _has_exact_shape(
        payload, {"schema_version", "format", "run_id", "metadata"}, _ENVELOPE_KEYS
    ):
        return False
    if not _is_nonempty_string(payload["run_id"]):
        return False
    if not isinstance(payload["metadata"], Mapping):
        return False
    if not _valid_nullable_string_fields(
        payload,
        ("title", "job_id", "source_namespace", "source_ref"),
    ):
        return False
    if not _valid_nullable_string_fields(
        payload,
        ("description", "created_at_utc", "updated_at_utc"),
        allow_empty=True,
    ):
        return False
    if not _valid_nullable_integer_fields(
        payload,
        ("started_at_ns", "ended_at_ns"),
        minimum=0,
    ):
        return False
    if not _valid_time_range(payload, "started_at_ns", "ended_at_ns"):
        return False
    return _valid_envelope_collections(payload)


def _valid_envelope_tags(tags: Any) -> bool:
    if not isinstance(tags, list):
        return False
    if any(not _is_nonempty_string(item) for item in tags):
        return False
    return len(tags) == len(set(cast(list[str], tags)))


def _valid_envelope_collections(payload: Mapping[str, Any]) -> bool:
    tags = payload.get("tags", [])
    if not _valid_envelope_tags(tags):
        return False

    sessions = payload.get("sessions", [])
    if not isinstance(sessions, list):
        return False
    if any(
        not isinstance(item, Mapping) or not _is_valid_session_payload(item)
        for item in sessions
    ):
        return False

    attachments = payload.get("attachments", [])
    if not isinstance(attachments, list):
        return False
    return all(
        isinstance(item, Mapping) and _is_valid_attachment_payload(item)
        for item in attachments
    )


def _is_valid_session_payload(payload: Mapping[str, Any]) -> bool:
    if not _has_exact_shape(payload, {"session_id", "metadata"}, _SESSION_KEYS):
        return False
    if not _is_nonempty_string(payload["session_id"]):
        return False
    if not isinstance(payload["metadata"], Mapping):
        return False
    if not _valid_nullable_string_fields(
        payload,
        ("job_id", "role", "source_namespace", "source_ref"),
    ):
        return False
    if not _valid_nullable_integer_fields(
        payload,
        ("rank", "local_rank"),
        minimum=0,
    ):
        return False
    return _valid_nullable_integer_fields(payload, ("world_size",), minimum=1)


def _is_valid_attachment_payload(payload: Mapping[str, Any]) -> bool:
    if not _has_exact_shape(
        payload,
        {"title", "kind", "storage", "metadata"},
        _ATTACHMENT_KEYS,
    ):
        return False
    if not _is_nonempty_string(payload["title"]):
        return False
    if not _is_nonempty_string(payload["kind"]):
        return False
    if attachment_storage_or_none(payload["storage"]) is None:
        return False
    if not isinstance(payload["metadata"], Mapping):
        return False
    if not _valid_attachment_location(payload):
        return False
    return _valid_attachment_optional_fields(payload)


def _valid_attachment_location(payload: Mapping[str, Any]) -> bool:
    if "url" not in payload and "path" not in payload:
        return False
    if "url" in payload and not _is_nonempty_string(payload["url"]):
        return False
    if "path" in payload and not _is_nonempty_string(payload["path"]):
        return False
    return True


def _valid_attachment_optional_fields(payload: Mapping[str, Any]) -> bool:
    if not _valid_nullable_string_fields(
        payload,
        (
            "attachment_id",
            "run_id",
            "session_id",
            "job_id",
            "source_namespace",
            "source_ref",
        ),
    ):
        return False
    if not _valid_nullable_string_fields(
        payload,
        ("created_at_utc", "updated_at_utc"),
        allow_empty=True,
    ):
        return False
    if not _valid_nullable_integer_fields(
        payload,
        ("rank", "local_rank", "start_ns", "end_ns"),
        minimum=0,
    ):
        return False
    if not _valid_nullable_integer_fields(payload, ("world_size",), minimum=1):
        return False
    return _valid_time_range(payload, "start_ns", "end_ns")


def _run_session_ref_from_payload(
    payload: Mapping[str, Any],
) -> CatalogRunSessionRef:
    return CatalogRunSessionRef(
        session_id=cast(str, payload["session_id"]),
        job_id=_optional_string(payload, "job_id"),
        rank=_optional_integer(payload, "rank"),
        local_rank=_optional_integer(payload, "local_rank"),
        world_size=_optional_integer(payload, "world_size"),
        role=_optional_string(payload, "role"),
        source_namespace=_optional_string(payload, "source_namespace"),
        source_ref=_optional_string(payload, "source_ref"),
        metadata=dict(cast(Mapping[str, Any], payload["metadata"])),
    )


def _run_attachment_from_payload(
    payload: Mapping[str, Any],
    envelope_path: Path,
    envelope_run_id: str,
) -> CatalogRunAttachment:
    raw_path = cast(str | None, payload.get("path"))
    return CatalogRunAttachment(
        title=cast(str, payload["title"]),
        kind=cast(str, payload["kind"]),
        storage=cast(AttachmentStorage, payload["storage"]),
        attachment_id=_optional_string(payload, "attachment_id"),
        url=cast(str | None, payload.get("url")),
        path=_resolve_optional_path(raw_path, envelope_path),
        run_id=_optional_string(payload, "run_id") or envelope_run_id,
        session_id=_optional_string(payload, "session_id"),
        job_id=_optional_string(payload, "job_id"),
        rank=_optional_integer(payload, "rank"),
        local_rank=_optional_integer(payload, "local_rank"),
        world_size=_optional_integer(payload, "world_size"),
        start_ns=_optional_integer(payload, "start_ns"),
        end_ns=_optional_integer(payload, "end_ns"),
        created_at_utc=_optional_string(payload, "created_at_utc", allow_empty=True),
        updated_at_utc=_optional_string(payload, "updated_at_utc", allow_empty=True),
        source_namespace=_optional_string(payload, "source_namespace"),
        source_ref=_optional_string(payload, "source_ref"),
        metadata=dict(cast(Mapping[str, Any], payload["metadata"])),
    )


def _has_exact_shape(
    payload: Mapping[str, Any],
    required: set[str],
    allowed: set[str],
) -> bool:
    keys = set(payload)
    return required <= keys and keys <= allowed


def _valid_nullable_string_fields(
    payload: Mapping[str, Any],
    fields: tuple[str, ...],
    *,
    allow_empty: bool = False,
) -> bool:
    for field in fields:
        if field not in payload or payload[field] is None:
            continue
        if not isinstance(payload[field], str):
            return False
        if not allow_empty and not payload[field]:
            return False
    return True


def _valid_nullable_integer_fields(
    payload: Mapping[str, Any],
    fields: tuple[str, ...],
    *,
    minimum: int,
) -> bool:
    for field in fields:
        if field not in payload or payload[field] is None:
            continue
        value = payload[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            return False
    return True


def _valid_time_range(
    payload: Mapping[str, Any],
    start_field: str,
    end_field: str,
) -> bool:
    start = payload.get(start_field)
    end = payload.get(end_field)
    return start is None or end is None or cast(int, end) >= cast(int, start)


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _optional_string(
    payload: Mapping[str, Any],
    field: str,
    *,
    allow_empty: bool = False,
) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, str) and (allow_empty or value):
        return value
    raise AssertionError(f"validated string field {field!r} became invalid")


def _optional_integer(payload: Mapping[str, Any], field: str) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise AssertionError(f"validated integer field {field!r} became invalid")


def _resolve_optional_path(raw_path: str | None, source_path: Path) -> str | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = source_path.parent / path
    return str(path.resolve())


__all__ = [
    "attachment_storage_or_default",
    "attachment_storage_or_none",
    "is_run_envelope",
    "run_envelope_from_payload",
]
