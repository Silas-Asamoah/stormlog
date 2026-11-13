"""Types shared by Studio's components.

This module is the step-0 contract of Stormlog Studio's server layer: the
types every component (C1 to C5) produces or consumes, a Protocol for each
component the application context holds, the error classes and the wire
encoder. A change goes through the lead.

Units:

- ``*_ns`` is int nanoseconds since the Unix epoch on the client's clock. On
  the wire it is a decimal string (:func:`to_wire`), because JavaScript
  numbers lose integer precision above 2**53.
- ``*_ms`` is float milliseconds; plot times are ms from ``origin_ns``.
- ``*_s`` and ``*_seconds`` are float seconds.
- ``*_bytes``, offsets, lengths and sizes are int bytes.
- Line numbers are 0-based, blank lines counted, and lines split on ``b"\\n"``
  only (``docs/report_contract.md``).
- SHA-256 digests are 64 lower-case hex characters.
- None means "no value", never zero.

Imports: the standard library, :mod:`stormlog.infer.populations` and
:mod:`stormlog.infer.slo` only, so every component can import this module
and this module imports no component.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from abc import abstractmethod
from array import array
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, ClassVar, Literal, Protocol

from stormlog.infer.populations import MeasuredInterval, Membership
from stormlog.infer.slo import CriterionValue, client_values

__all__ = [
    "FORMAT_PREFIX",
    "SCHEMA_VERSION",
    "MAX_SAFE_INTEGER",
    "CLIENT_COLUMN_PREFIX",
    "LineRef",
    "ProblemCode",
    "LineProblem",
    "FileIdentity",
    "NotResident",
    "RequestRow",
    "DictRow",
    "RequestTable",
    "DictTable",
    "IndexStatus",
    "IndexFailure",
    "LimitHit",
    "SessionInfo",
    "IndexSnapshot",
    "Phase",
    "SortKey",
    "TimeWindow",
    "LatencyRange",
    "RequestFilter",
    "RateScope",
    "Selection",
    "Page",
    "HttpRequest",
    "HttpResponse",
    "Auth",
    "Route",
    "StudioError",
    "UsageError",
    "ConflictError",
    "NotFoundError",
    "InvalidInputError",
    "UnauthorizedError",
    "ForbiddenError",
    "BusyError",
    "NotReady",
    "to_wire",
    "finite_copy",
    "encode_json",
    "artifact_block",
    "envelope",
    "IndexRegistry",
    "WorkspaceRoot",
    "TokenStore",
    "WorkspaceScanner",
    "AnalysisState",
    "AnalysisCache",
    "DocumentCache",
    "Limits",
    "LaunchOptions",
    "StudioContext",
    "Composer",
]

# Every document's ``format`` is this prefix plus its name (§5.1).
FORMAT_PREFIX = "stormlog.studio."
# Every step-0 document is at version 1; a shape change is a new version.
SCHEMA_VERSION = 1
# The largest integer a JavaScript number holds exactly: 2**53 - 1.
MAX_SAFE_INTEGER = 2**53 - 1
# IndexSnapshot.column("client.ttft") and its siblings read RequestRow.client.
CLIENT_COLUMN_PREFIX = "client."

_DOCUMENT_NAME = re.compile(r"[a-z][a-z0-9_]*")
_ENVELOPE_KEYS = frozenset({"format", "schema_version", "artifact"})


# --- Lines and files -------------------------------------------------- (C1 produces)


@dataclass(frozen=True)
class LineRef:
    """Where one line of an artifact is, and what its bytes hash to."""

    line: int  # 0-based
    offset: int  # byte offset of the line's first byte
    length: int  # bytes, without the b"\n"
    sha256: str  # hex, of those bytes (diagnosis_inputs.line_digest)


ProblemCode = Literal[
    "not_json",
    "not_object",
    "undecodable",
    "oversized",
    "unsupported_version",
    "truncated_final_line",
]


@dataclass(frozen=True)
class LineProblem:
    """A line the reader or the index could not use, and why."""

    ref: LineRef
    code: ProblemCode
    detail: str  # scrubbed, at most 200 characters


@dataclass(frozen=True)
class FileIdentity:
    """What ``os.fstat`` said about the artifact at the last read."""

    device: int
    inode: int
    size: int
    mtime_ns: int


# --- Rows ------------------------------------------------------ (C1 implements, C2 reads)


class NotResident(Exception):
    """A field the index does not hold. Not a KeyError, on purpose.

    ``Mapping.get`` turns a KeyError into None, so a summary could count zero
    gaps for a request whose gaps were merely dropped from memory. This
    exception passes through ``get`` instead.
    """


class RequestRow(Mapping[str, Any]):
    """One client request, read-only (abstract).

    Keys are the decoded record's keys. ``chunk_interarrival_ms`` and
    ``error_message`` raise :class:`NotResident` when their pool is not
    resident, on ``row[key]``, ``row.get(key)`` and therefore on iterating
    the items (``dict(row)``, :func:`finite_copy`). Code that needs the whole
    record whatever the pools hold (request detail) builds it from the other
    keys plus :meth:`load_gaps` and :meth:`load_error_message`.

    The identity properties follow one rule for a record that lacks the
    field: ``str(record.get(key, "unknown"))``, as the analyzer groups
    ``case_id``. C1's ``index.ColumnRow`` and :class:`DictRow` implement it.

    Every attribute is an abstract read-only property, so a subclass may
    read it from a column store on access (C1) or keep it in a field
    (:class:`DictRow`), and mypy accepts both.
    """

    @property
    @abstractmethod
    def index(self) -> int:
        """The row's position in ``IndexSnapshot.requests``.

        Stable for the life of the index (``index_id``): columns are
        append-only, so ``snapshot.column(name)[row.index]`` is this row's
        value in this snapshot and in every later one.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def ref(self) -> LineRef:
        """Where the record's line is."""
        raise NotImplementedError

    @property
    @abstractmethod
    def session_id(self) -> str:
        """``str(record.get("session_id", "unknown"))``."""
        raise NotImplementedError

    @property
    @abstractmethod
    def request_id(self) -> str:
        """``str(record.get("request_id", "unknown"))``."""
        raise NotImplementedError

    @property
    @abstractmethod
    def record_id(self) -> str | None:
        """``diagnosis_inputs.record_id`` of the record."""
        raise NotImplementedError

    @property
    @abstractmethod
    def case_id(self) -> str:
        """``str(record.get("case_id", "unknown"))``."""
        raise NotImplementedError

    @property
    @abstractmethod
    def phase(self) -> str:
        """``str(record.get("phase", "unknown"))``."""
        raise NotImplementedError

    @property
    @abstractmethod
    def client(self) -> Mapping[str, CriterionValue]:
        """``slo.client_values(record)``, computed at index time."""
        raise NotImplementedError

    @property
    @abstractmethod
    def gap_count(self) -> int:
        """``len(chunk_interarrival_ms)`` as written; 0 without a list."""
        raise NotImplementedError

    @abstractmethod
    def load_gaps(self) -> Sequence[float]:
        """The chunk gaps in written order; one line read if not resident.

        An element that is not a number is NaN, so the length matches the
        record and ``report_stats.is_number`` still drops it.
        """
        raise NotImplementedError

    @abstractmethod
    def load_error_message(self) -> str | None:
        """The error message; one line read if not resident."""
        raise NotImplementedError


class DictRow(RequestRow):
    """A resident row over a plain dict, for C2's tests and for fixtures.

    ``ref`` is synthetic: offset 0, and the length and SHA-256 of the
    record's compact JSON (sorted keys), with ``line`` as given. ``index``
    is the row's position in the snapshot it is put in; the caller gives it.
    """

    def __init__(
        self, record: Mapping[str, Any], *, line: int = 0, index: int = 0
    ) -> None:
        self._record: dict[str, Any] = dict(record)
        encoded = json.dumps(
            self._record, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self._index = index
        self._ref = LineRef(
            line=line,
            offset=0,
            length=len(encoded),
            sha256=hashlib.sha256(encoded).hexdigest(),
        )
        self._session_id = str(record.get("session_id", "unknown"))
        self._request_id = str(record.get("request_id", "unknown"))
        self._record_id = _request_record_id(record)
        self._case_id = str(record.get("case_id", "unknown"))
        self._phase = str(record.get("phase", "unknown"))
        self._client = MappingProxyType(client_values(record))
        gaps = record.get("chunk_interarrival_ms")
        self._gap_count = len(gaps) if isinstance(gaps, list) else 0

    @property
    def index(self) -> int:
        return self._index

    @property
    def ref(self) -> LineRef:
        return self._ref

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def record_id(self) -> str | None:
        return self._record_id

    @property
    def case_id(self) -> str:
        return self._case_id

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def client(self) -> Mapping[str, CriterionValue]:
        return self._client

    @property
    def gap_count(self) -> int:
        return self._gap_count

    def __getitem__(self, key: str) -> Any:
        return self._record[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._record)

    def __len__(self) -> int:
        return len(self._record)

    def load_gaps(self) -> Sequence[float]:
        gaps = self._record.get("chunk_interarrival_ms")
        if not isinstance(gaps, list):
            return ()
        return tuple(_gap_value(gap) for gap in gaps)

    def load_error_message(self) -> str | None:
        message = self._record.get("error_message")
        return message if isinstance(message, str) else None


def _gap_value(gap: Any) -> float:
    if isinstance(gap, (int, float)) and not isinstance(gap, bool):
        return float(gap)
    return math.nan


def _request_record_id(record: Mapping[str, Any]) -> str | None:
    """``diagnosis_inputs.record_id`` for an ``infer.request`` record."""
    event_type = record.get("event_type")
    if not isinstance(event_type, str):
        return None
    if record.get("schema_version", 1) != 1:
        event_id = record.get("event_id")
        return event_id if isinstance(event_id, str) and event_id else None
    request_id = record.get("request_id")
    if event_type != "infer.request" or request_id is None:
        return None
    return f"{event_type}/{request_id}"


# --- Columns ------------------------------------------- (C1 implements, C2 reads; D-1)


class RequestTable(Protocol):
    """Column access to an index snapshot's client requests (Fable's D-1).

    The path C2 uses to filter, sort and aggregate on arrays, so a selection
    over 500,000 rows builds no row views: C2 builds :class:`RequestRow`
    views only for the rows it returns.

    ``column(name)`` has one element per row of ``IndexSnapshot.requests``,
    in the same order, and its elements equal what the rows give:

    - for a record key, ``column(name)[i] == requests[i].get(name)``: None
      where the record lacks the key, never a KeyError;
    - for ``"client." + metric`` (a ``latency_report.CLIENT_METRICS`` key),
      ``column(name)[i] == requests[i].client[name].value_ms``; an unknown
      metric raises KeyError.

    A column of a non-resident pool (``chunk_interarrival_ms``,
    ``error_message``) raises :class:`NotResident`. A column may be a list,
    tuple or array, built once per snapshot; callers must not mutate it.
    """

    def __len__(self) -> int: ...

    def column(self, name: str) -> Sequence[Any]: ...


class DictTable:
    """A :class:`RequestTable` over resident rows, for C2's tests and fixtures.

    It reads the rows one by one, so it is as slow as row access; C1's index
    keeps real columns.
    """

    def __init__(self, rows: Sequence[RequestRow]) -> None:
        self._rows = tuple(rows)
        self._columns: dict[str, tuple[Any, ...]] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def column(self, name: str) -> Sequence[Any]:
        cached = self._columns.get(name)
        if cached is None:
            cached = tuple(_cell(row, name) for row in self._rows)
            self._columns[name] = cached
        return cached


def _cell(row: RequestRow, name: str) -> Any:
    if name.startswith(CLIENT_COLUMN_PREFIX):
        return row.client[name].value_ms
    return row.get(name)


# --- Index snapshots --------------------------------------------------- (C1 produces)

IndexStatus = Literal["building", "ready", "partial", "failed"]
IndexFailure = Literal[
    "unreadable", "not_an_inference_artifact", "audio", "multiple_sessions"
]
LimitHit = Literal["rows", "gaps", "error_messages", "samples", "kept_records"]


@dataclass(frozen=True)
class SessionInfo:
    """The run's ``infer.session`` records, digested."""

    session_id: str
    status: str | None  # status of the last infer.session record
    started_at_ns: int | None  # timestamp_ns of the first infer.session record
    ended_at_ns: int | None  # timestamp_ns of the terminal record
    terminal: bool  # a record with a status other than "running" exists
    config: Mapping[str, Any]  # the first record's config, as written


@dataclass(frozen=True)
class IndexSnapshot:
    """An immutable view of one index after one batch (§4.4).

    Columns are append-only, so a snapshot shares them with later ones,
    bounded by its own length. ``requests`` gives row views; ``table`` and
    :meth:`column` give the same values by column (D-1).
    """

    path: str  # root-relative POSIX path
    index_id: str  # 32 hex characters, new per build
    generation: int  # 0 for the first snapshot of an index_id
    status: IndexStatus
    failure: IndexFailure | None
    identity: FileIdentity  # at the last read
    bytes_read: int  # complete lines consumed
    pending_tail_bytes: int
    prefix_sha256: str  # of bytes [0, bytes_read)
    sha256: str | None  # whole file; None unless bytes_read == identity.size
    growing: bool  # size changed within the last 5 s
    origin_ns: int | None
    session: SessionInfo | None
    requests: Sequence[RequestRow]  # client requests, file order, all phases
    table: RequestTable  # the same requests by column (D-1)
    by_request_id: Mapping[str, tuple[int, ...]]  # request_id -> indexes into requests
    samples: Sequence[Mapping[str, Any]]  # infer.system_sample rows, no metadata
    kept: Mapping[str, tuple[Mapping[str, Any], ...]]  # event_type -> small records
    request_links: Mapping[
        str, tuple[LineRef | None, LineRef | None]
    ]  # request_id -> (dispatch, first_content)
    refs: Mapping[str, tuple[LineRef, ...]]  # event_type -> first 1,000 lines
    inventory: Mapping[str, Mapping[int, int]]  # event_type -> version -> count
    problems: tuple[LineProblem, ...]  # the first 100
    problem_count: int
    limits_hit: frozenset[LimitHit]
    gaps_resident: bool
    error_messages_resident: bool

    def column(self, name: str) -> Sequence[Any]:
        """One request field for every row, in file order (D-1).

        The path C2 uses to filter and sort on arrays; see
        :class:`RequestTable` for the names and the guarantees.
        """
        return self.table.column(name)


# --- Filters and selections ------------------------------ (C2 produces, C4 passes on)

Phase = Literal["measured", "warmup", "all"]
SortKey = Literal[
    "artifact",
    "started_at",
    "client.e2e",
    "client.ttft",
    "first_chunk",
    "output_tokens",
    "status",
    "case",
    "request_id",
]


@dataclass(frozen=True)
class TimeWindow:
    """A half-open window on the client clock."""

    start_ns: int  # inclusive
    end_ns: int  # exclusive; end_ns > start_ns
    membership: Membership = "arrival"


@dataclass(frozen=True)
class LatencyRange:
    """A half-open condition on one client latency metric."""

    metric: str  # a latency_report.CLIENT_METRICS key
    min_ms: float  # inclusive, finite
    max_ms: float | None  # exclusive; None is unbounded


@dataclass(frozen=True)
class RequestFilter:
    """Conditions on client requests, in canonical form (§5.3)."""

    phase: Phase = "measured"
    cases: tuple[str, ...] = ()  # sorted, unique; () is every case
    statuses: tuple[str, ...] = ()  # from events.REQUEST_STATUSES, sorted
    stream: bool | None = None
    token_sources: tuple[str, ...] = ()  # output_token_source, tokens.TOKEN_SOURCES
    text: str | None = None  # q, casefolded, at most 256 characters
    window: TimeWindow | None = None
    completed_by_ns: int | None = None  # keeps ended_at_ns <= this
    latency: LatencyRange | None = None


@dataclass(frozen=True)
class RateScope:
    """Whether a selection has a rate, over which interval, or why not (§3.3)."""

    eligible: bool
    rule: Literal["case", "segment"] | None
    interval: MeasuredInterval | None
    reason: str | None  # a §3.3 reason or the case's CaseIntervals.rate_reason


@dataclass(frozen=True)
class Selection:
    """The client requests of one snapshot that match one filter."""

    filter: RequestFilter
    params: str  # canonical query string
    fingerprint: str  # "sha256:" + hex of (path, index_id, generation, params)
    path: str
    index_id: str
    generation: int
    rows: Sequence[int]  # indexes into IndexSnapshot.requests, file order
    cases: tuple[str, ...]  # distinct case_ids, sorted
    whole_case: str | None  # the case, when rows are exactly its measured requests
    rate: RateScope
    searched_fields: tuple[str, ...]


@dataclass(frozen=True)
class Page:
    """One keyset page of a selection."""

    rows: tuple[int, ...]  # indexes into IndexSnapshot.requests
    total: int
    next_cursor: str | None
    generation_moved: bool


# --- HTTP ------------------------------------------------- (C3 serves, C3 and C4 route)


@dataclass(frozen=True)
class HttpRequest:
    """A parsed request, as httpd hands it to a route's handler."""

    method: str
    path: str  # percent-decoded, no query
    query: Mapping[str, tuple[str, ...]]  # parse_qs, strict, blanks kept
    headers: Mapping[str, str]  # lower-cased names
    body: bytes  # at most 64 KiB


@dataclass(frozen=True)
class HttpResponse:
    """What a handler returns; httpd adds the security headers."""

    status: int
    content_type: str
    body: bytes | Iterable[bytes]  # an iterable is streamed, chunked
    headers: tuple[tuple[str, str], ...] = ()


Auth = Literal["public", "bearer", "bearer_or_ticket", "admin"]


@dataclass(frozen=True)
class Route:
    """One exact method and path, its auth and its deadline."""

    method: Literal["GET", "POST"]
    path: str  # exact; no path parameters in slice 1
    auth: Auth
    handler: Callable[[HttpRequest], HttpResponse]
    deadline_s: float | None = 30.0  # 600 for export and line; None for streams


# --- Errors --------------------------------------------------- (anyone raises, C3 renders)


class StudioError(Exception):
    """An error with an HTTP status and a ``stormlog.studio.error`` body (§5.2).

    ``verdict_status`` is the ``stormlog.exit_codes`` verdict the status
    pairs with, None for security rejections. ``message`` is shown to the
    user; C3 passes it through ``scrub_text(..., max_bytes=512)`` before
    rendering (§7.6).
    """

    http_status: ClassVar[int] = 500
    verdict_status: ClassVar[str | None] = "error"

    def __init__(
        self,
        code: str,
        message: str,
        *,
        field: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field
        self.details: dict[str, Any] = dict(details or {})

    def document(self) -> dict[str, Any]:
        """The ``stormlog.studio.error`` body, before :func:`to_wire`."""
        return envelope(
            "error",
            {
                "error": {
                    "code": self.code,
                    "message": self.message,
                    "http_status": self.http_status,
                    "verdict_status": self.verdict_status,
                    "field": self.field,
                    "details": dict(self.details),
                }
            },
        )


class UsageError(StudioError):
    """400: a bad, unknown or over-limit parameter."""

    http_status = 400
    verdict_status = "usage"


class ConflictError(StudioError):
    """409: a cursor, request ID or line that no longer fits the index."""

    http_status = 409
    verdict_status = "usage"


class NotFoundError(StudioError):
    """404: no such artifact, request or line."""

    http_status = 404
    verdict_status = "invalid_input"


class InvalidInputError(StudioError):
    """422: the artifact or report cannot be served."""

    http_status = 422
    verdict_status = "invalid_input"


class UnauthorizedError(StudioError):
    """401: a missing or bad token, bootstrap or ticket."""

    http_status = 401
    verdict_status = None


class ForbiddenError(StudioError):
    """403: a Host, Origin, Fetch Metadata or path check failed."""

    http_status = 403
    verdict_status = None


class BusyError(StudioError):
    """503: no capacity for this now."""

    http_status = 503
    verdict_status = None


class NotReady(Exception):
    """Still building; httpd answers 202 with `state` and Retry-After."""

    def __init__(self, state: Mapping[str, Any], retry_after_s: int = 1) -> None:
        super().__init__("not ready")
        self.state: dict[str, Any] = dict(state)
        self.retry_after_s = retry_after_s


# --- Wire ---------------------------------------------------------------- (lead)


def to_wire(value: Any) -> Any:
    """A JSON-ready copy of ``value`` under §5.1's number rules, recursively.

    An integer whose nearest enclosing key ends in ``_ns`` (alone, or in a
    list or nested list under that key) becomes a decimal string, and so
    does any integer beyond +/-(2**53 - 1). Booleans stay booleans. Mappings
    become dicts (an int key becomes its decimal string, as ``json`` would
    write it); lists, tuples and arrays become lists. Floats, NaN included,
    pass unchanged: :func:`encode_json` refuses non-finite numbers, and
    record fields go through :func:`finite_copy` first. Any other type raises
    TypeError.
    """
    return _wire(value, ns=False)


def _wire(value: Any, *, ns: bool) -> Any:
    if value is None or isinstance(value, (bool, str, float)):
        return value
    if isinstance(value, int):
        return str(value) if ns or abs(value) > MAX_SAFE_INTEGER else value
    if isinstance(value, Mapping):
        return _wire_mapping(value)
    if _is_list_like(value):
        return [_wire(item, ns=ns) for item in value]
    raise TypeError(f"no wire form for {type(value).__name__}")


def _wire_mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    wired: dict[str, Any] = {}
    for key, item in value.items():
        name = _wire_key(key)
        wired[name] = _wire(item, ns=name.endswith("_ns"))
    return wired


def _wire_key(key: Any) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, int) and not isinstance(key, bool):
        return str(key)
    raise TypeError(f"no wire form for a {type(key).__name__} key")


def _is_list_like(value: Any) -> bool:
    if isinstance(value, (list, tuple, array)):
        return True
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def finite_copy(record: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """A deep copy of ``record`` with NaN and infinities replaced by None.

    Returns the copy and the paths it changed, in record order, for the
    document's ``non_finite_fields``. A path is the key, then ``.key`` for a
    nested mapping and ``[i]`` for a list element:
    ``"chunk_interarrival_ms[3]"``, ``"usage.ttft_ms"``.
    """
    changed: list[str] = []
    copy = {
        str(key): _finite(value, str(key), changed) for key, value in record.items()
    }
    return copy, changed


def _finite(value: Any, path: str, changed: list[str]) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        changed.append(path)
        return None
    if isinstance(value, Mapping):
        return {
            str(key): _finite(item, f"{path}.{key}", changed)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_finite(item, f"{path}[{i}]", changed) for i, item in enumerate(value)]
    return value


def encode_json(document: Mapping[str, Any]) -> bytes:
    """The response body: :func:`to_wire`, then compact UTF-8 JSON.

    Raises ValueError on a NaN or an infinity (``allow_nan=False``).
    """
    text = json.dumps(
        to_wire(document), allow_nan=False, ensure_ascii=False, separators=(",", ":")
    )
    return text.encode("utf-8")


def artifact_block(snapshot: IndexSnapshot) -> dict[str, Any]:
    """The ``artifact`` block every artifact-scoped document carries (§5.1).

    ``terminal`` is False for a file with no session record: no terminal
    record exists. ``limits_hit`` is sorted.
    """
    session = snapshot.session
    return {
        "path": snapshot.path,
        "index_id": snapshot.index_id,
        "generation": snapshot.generation,
        "status": snapshot.status,
        "size": snapshot.identity.size,
        "bytes_read": snapshot.bytes_read,
        "sha256": snapshot.sha256,
        "growing": snapshot.growing,
        "session_id": None if session is None else session.session_id,
        "session_status": None if session is None else session.status,
        "terminal": False if session is None else session.terminal,
        "origin_ns": snapshot.origin_ns,
        "gaps_resident": snapshot.gaps_resident,
        "error_messages_resident": snapshot.error_messages_resident,
        "limits_hit": sorted(snapshot.limits_hit),
    }


def envelope(
    name: str, body: Mapping[str, Any], snapshot: IndexSnapshot | None = None
) -> dict[str, Any]:
    """``{"format": "stormlog.studio." + name, "schema_version": 1, ...body}``.

    With a snapshot, ``artifact`` is :func:`artifact_block` of it; without
    one the key is absent. The body must not carry ``format``,
    ``schema_version`` or ``artifact`` itself (ValueError), so a component
    cannot overwrite the envelope by accident.
    """
    if not _DOCUMENT_NAME.fullmatch(name):
        raise ValueError(f"not a document name: {name!r}")
    clash = sorted(_ENVELOPE_KEYS.intersection(body))
    if clash:
        raise ValueError(f"the body sets envelope keys: {', '.join(clash)}")
    document: dict[str, Any] = {
        "format": FORMAT_PREFIX + name,
        "schema_version": SCHEMA_VERSION,
    }
    if snapshot is not None:
        document["artifact"] = artifact_block(snapshot)
    document.update(body)
    return document


# --- Components the app context holds ------------------------------------------


class IndexRegistry(Protocol):
    """C1: ``index.ArtifactIndexes``."""

    def snapshot(self, rel: str) -> IndexSnapshot:
        """The current snapshot of ``rel``, without waiting for a build.

        Resolves ``rel`` (NotFoundError, ForbiddenError); raises NotReady
        until the first snapshot exists; schedules a refresh on the builder
        pool when the last ``stat`` is more than 1 s old.
        """
        ...

    def wait(self, rel: str, timeout_s: float) -> IndexSnapshot:
        """The first snapshot of ``rel``, waiting up to ``timeout_s``."""
        ...

    def read_line(self, rel: str, ref: LineRef) -> bytes:
        """One line's bytes, without the newline; ConflictError ``stale_line``
        when its SHA-256 no longer matches."""
        ...

    def read_lines(self, rel: str, refs: Iterable[LineRef]) -> Iterator[bytes]:
        """Each line's bytes with its b"\\n", read as the caller iterates."""
        ...

    def close(self) -> None:
        """Cancel builds and drop every index."""
        ...


class WorkspaceRoot(Protocol):
    """C3: ``security.RootDirectory`` (§7.4)."""

    path: Path  # the root's real path

    def resolve(self, rel: str) -> Path:
        """The real path of a regular file under the root; ForbiddenError or
        NotFoundError otherwise."""
        ...

    def open(self, rel: str) -> BinaryIO:
        """Resolve, open read-only and check the open file is the one resolved."""
        ...

    def relative(self, path: Path) -> str:
        """The root-relative POSIX path of ``path``."""
        ...


class TokenStore(Protocol):
    """C3: ``security.SessionTokens`` (§7.2)."""

    def issue_admin(self) -> str:
        """The admin token, once, for the state file."""
        ...

    def issue_bootstrap(self, *, ttl_s: float = 600.0) -> str:
        """A single-use bootstrap token."""
        ...

    def exchange(self, bootstrap: str) -> str:
        """A session token for a live bootstrap; UnauthorizedError otherwise."""
        ...

    def check(self, token: str, kind: Literal["session", "admin"]) -> bool:
        """Whether ``token`` is a live token of that kind (constant time)."""
        ...

    def issue_ticket(self, *, route: str, query: str, ttl_s: float = 30.0) -> str:
        """A single-use ticket bound to the route and canonical query."""
        ...

    def redeem_ticket(self, ticket: str, *, route: str, query: str) -> bool:
        """True once for a live ticket issued for this route and query."""
        ...


class WorkspaceScanner(Protocol):
    """C4: ``workspace.ArtifactScanner``."""

    def scan(self) -> None:
        """Queue a scan on the builder pool; BusyError while one runs."""
        ...

    def state(self) -> dict[str, Any]:
        """The body of ``stormlog.studio.workspace``."""
        ...

    def items(
        self, *, kind: str | None, q: str | None, cursor: str | None, limit: int
    ) -> dict[str, Any]:
        """The body of ``stormlog.studio.items``."""
        ...


@dataclass(frozen=True)
class AnalysisState:
    """Where the out-of-process analyzer is for one artifact (decision 12)."""

    status: Literal["pending", "running", "ready", "unavailable", "failed"]
    reason: str | None  # over_size_cap, growing, timeout, nonzero_exit, ...
    exit_code: int | None
    report: Mapping[str, Any] | None
    analysed_bytes: int | None  # the complete-line prefix analysed


class AnalysisCache(Protocol):
    """C4: ``analysis_cache.AnalyzerRunner``."""

    def get(self, rel: str, snapshot: IndexSnapshot) -> AnalysisState:
        """The analysis of this snapshot's file, starting one if none runs."""
        ...

    def close(self) -> None:
        """Kill the analyzer subprocess, if any."""
        ...


class DocumentCache(Protocol):
    """C4: ``cache.Documents``; keys hold path, index ID, generation, params."""

    def selection(
        self, key: tuple[str, ...], make: Callable[[], Selection]
    ) -> Selection:
        """The cached selection for ``key``, made once."""
        ...

    def document(
        self, key: tuple[str, ...], make: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        """The cached document for ``key``, made once."""
        ...


@dataclass(frozen=True)
class Limits:
    """The bounds of §8.1; sizes in bytes."""

    max_connections: int = 16
    head_bytes: int = 16 * 2**10
    body_bytes: int = 64 * 2**10
    deadline_s: float = 30.0
    long_deadline_s: float = 600.0
    open_indexes: int = 4
    index_memory_bytes: int = 768 * 2**20
    row_bytes: int = 256 * 2**20
    gap_bytes: int = 128 * 2**20
    error_message_bytes: int = 32 * 2**20
    sample_bytes: int = 32 * 2**20
    kept_bytes: int = 16 * 2**20
    kept_record_bytes: int = 64 * 2**10
    line_bytes: int = 64 * 2**20
    scan_depth: int = 8
    scan_entries: int = 50_000
    scan_seconds: float = 10.0
    analysis_bytes: int = 128 * 2**20
    analysis_seconds: float = 300.0
    report_bytes: int = 16 * 2**20
    page_rows: int = 1_000
    points: int = 5_000
    bins: int = 200
    chunk_page: int = 10_000
    text_chars: int = 256
    cached_documents: int = 64
    cached_selections: int = 16


@dataclass(frozen=True)
class LaunchOptions:
    """What C3's cli builds from argv."""

    root: Path
    open_path: str | None
    host: str
    port: int
    open_browser: bool
    state_dir: Path
    log_requests: bool
    debug: bool
    limits: Limits = field(default_factory=Limits)


@dataclass(frozen=True)
class StudioContext:
    """Everything a route needs; built only by ``app.build_context`` (C4)."""

    instance_id: str  # 32 hex characters
    version: str  # stormlog.__version__
    options: LaunchOptions
    root: WorkspaceRoot
    tokens: TokenStore
    indexes: IndexRegistry
    scanner: WorkspaceScanner
    analysis: AnalysisCache
    cache: DocumentCache
    pool: Executor  # the builder pool


# C3's cli.main(argv, *, compose=None) takes one; the default imports app.
Composer = Callable[[LaunchOptions], tuple[StudioContext, list[Route]]]
