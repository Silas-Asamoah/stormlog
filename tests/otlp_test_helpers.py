"""Build OTLP trace exports from the installed descriptors, field by field.

The wire scan reads its schema from the installed ``opentelemetry-proto``,
so the tests do too: every message-typed field a version defines is built,
including fields older versions lack, such as ``Resource.entity_refs``.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class MessageField(NamedTuple):
    """A message-typed field, and the fields leading to its holder."""

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


def message_fields() -> list[MessageField]:
    """Every message-typed field of every type an export can hold, each
    with the shortest path to a message of its type; none without the
    extra."""
    request = request_class()
    if request is None:
        return []
    paths: dict[str, tuple[Any, ...]] = {request.DESCRIPTOR.full_name: ()}
    queue = [request.DESCRIPTOR]
    found: list[MessageField] = []
    for descriptor in queue:  # grows while it is walked
        for field in descriptor.fields:
            if field.type != field.TYPE_MESSAGE:
                continue
            found.append(MessageField(paths[descriptor.full_name], field))
            child = field.message_type
            if child.full_name not in paths:
                paths[child.full_name] = (*paths[descriptor.full_name], field)
                queue.append(child)
    return found


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


def export_with(target: MessageField, count: int) -> Any:
    """A request holding ``count`` empty messages in ``target`` (one if the
    field is not repeated)."""
    request_type = request_class()
    assert request_type is not None
    request = request_type()
    message = holder(request, target.path)
    container = getattr(message, target.field.name)
    if repeated(target.field):
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


def spans_in(message: Any) -> int:
    return sum(
        len(scope_spans.spans)
        for resource_spans in message.resource_spans
        for scope_spans in resource_spans.scope_spans
    )
