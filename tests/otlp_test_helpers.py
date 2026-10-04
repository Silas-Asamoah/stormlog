"""Build OTLP trace exports from the installed descriptors, field by field.

The wire scan reads its schema from the installed ``opentelemetry-proto``,
so the tests do too: every field a version defines is built, including
fields older versions lack, such as ``Resource.entity_refs``.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class SchemaField(NamedTuple):
    """A field of the schema, and the fields leading to its holder."""

    path: tuple[Any, ...]
    field: Any

    @property
    def name(self) -> str:
        return f"{self.field.containing_type.name}.{self.field.name}"


def repeated(field: Any) -> bool:
    """protobuf 7 dropped ``FieldDescriptor.label``; 6 added ``is_repeated``."""
    is_repeated = getattr(field, "is_repeated", None)
    if is_repeated is None:
        return bool(field.label == field.LABEL_REPEATED)
    return bool(is_repeated)


def request_class() -> Any | None:
    try:
        from opentelemetry.proto.collector.trace.v1 import trace_service_pb2
    except ImportError:
        return None
    return trace_service_pb2.ExportTraceServiceRequest


def schema_fields() -> list[SchemaField]:
    """Every field of every type an export can hold, each with the shortest
    path to a message of its type; none without the extra."""
    request = request_class()
    if request is None:
        return []
    paths: dict[str, tuple[Any, ...]] = {request.DESCRIPTOR.full_name: ()}
    queue = [request.DESCRIPTOR]
    found: list[SchemaField] = []
    for descriptor in queue:  # grows while it is walked
        for field in descriptor.fields:
            found.append(SchemaField(paths[descriptor.full_name], field))
            if field.type != field.TYPE_MESSAGE:
                continue
            child = field.message_type
            if child.full_name not in paths:
                paths[child.full_name] = (*paths[descriptor.full_name], field)
                queue.append(child)
    return found


def message_fields() -> list[SchemaField]:
    """Every message-typed field an export can hold."""
    return [f for f in schema_fields() if f.field.type == f.field.TYPE_MESSAGE]


def repeated_scalar_fields() -> list[SchemaField]:
    """Every repeated string, bytes or number field an export can hold
    (``EntityRef.id_keys`` from opentelemetry-proto 1.45)."""
    return [
        f
        for f in schema_fields()
        if f.field.type != f.field.TYPE_MESSAGE and repeated(f.field)
    ]


def holder(request: Any, path: tuple[Any, ...]) -> Any:
    """The message at the end of ``path``, created in ``request``."""
    message = request
    for field in path:
        container = getattr(message, field.name)
        if repeated(field):
            message = container.add()
        else:
            container.SetInParent()
            message = container
    return message


def export_with(target: SchemaField, count: int) -> Any:
    """A request holding ``count`` elements in ``target`` (one if the field
    is not repeated): empty messages, two-character strings or bytes, or
    ones."""
    request_type = request_class()
    assert request_type is not None
    request = request_type()
    message = holder(request, target.path)
    container = getattr(message, target.field.name)
    field = target.field
    if field.type != field.TYPE_MESSAGE:
        element: Any = 1
        if field.type == field.TYPE_STRING:
            element = "ab"
        elif field.type == field.TYPE_BYTES:
            element = b"ab"
        container.extend([element] * count)
    elif repeated(field):
        for _ in range(count):
            container.add()
    else:
        container.SetInParent()
    return request


def messages_in(message: Any) -> int:
    """Every message a parse built for ``message``, itself included."""
    total = 1
    for field, value in message.ListFields():
        if field.type != field.TYPE_MESSAGE:
            continue
        for child in value if repeated(field) else [value]:
            total += messages_in(child)
    return total


def elements_in(message: Any) -> int:
    """Every element of a repeated string, bytes or number field a parse
    built for ``message``."""
    total = 0
    for field, value in message.ListFields():
        if field.type != field.TYPE_MESSAGE:
            total += len(value) if repeated(field) else 0
            continue
        for child in value if repeated(field) else [value]:
            total += elements_in(child)
    return total


def spans_in(message: Any) -> int:
    return sum(
        len(scope_spans.spans)
        for resource_spans in message.resource_spans
        for scope_spans in resource_spans.scope_spans
    )
