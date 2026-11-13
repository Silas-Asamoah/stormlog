"""Filters, selections, rate scopes and pages (design §5.3, §3.3; C2).

The index (C1) is built in parallel, so these tests read artifacts through
two stand-ins kept here: ``snapshot_of`` gives an ``IndexSnapshot`` of
``DictRow``s and a ``DictTable``, as C1's index would hold them, and
``column_snapshot_of`` gives one over real columns with lazy row views,
for the 500,000-request timings. ``tests/studio/test_summaries.py`` and
``test_metric_catalog.py`` import them.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import random
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import parse_qs

import pytest

from stormlog.infer.correlation_codec import CorrelationRecordDecoder
from stormlog.infer.populations import (
    Segment,
    _belongs,
    populations_from_records,
)
from stormlog.infer.slo import CriterionValue, client_values
from stormlog.studio import selection
from stormlog.studio.contracts import (
    CLIENT_COLUMN_PREFIX,
    ConflictError,
    DictRow,
    DictTable,
    FileIdentity,
    IndexSnapshot,
    LatencyRange,
    LineRef,
    NotFoundError,
    NotResident,
    RequestFilter,
    RequestRow,
    SessionInfo,
    TimeWindow,
    UsageError,
    selection_fingerprint,
)
from stormlog.studio.selection import (
    canonical_params,
    case_facts,
    page,
    parse_filter,
    request_row,
    select,
    without_latency,
)
from tests.studio.conftest import StandardArtifacts

INDEX_ID = "0123456789abcdef0123456789abcdef"
KEPT_TYPES = frozenset(
    {
        "infer.session",
        "infer.workload",
        "infer.phase_window",
        "infer.phase_start",
        "infer.cache_state",
        "infer.slo",
        "infer.artifact",
        "infer.capabilities",
        "infer.trace_window",
        "infer.treatments",
        "infer.model_identity",
        "infer.run_state",
    }
)
KEPT_RECORD_BYTES = 64 * 2**10
SECOND = 1_000_000_000


# --- A stand-in for C1's index ----------------------------------------------------


class _Lines:
    """The decoded lines of one artifact, split as the reader splits them."""

    def __init__(self, data: bytes) -> None:
        self.items: list[tuple[LineRef, dict[str, Any]]] = []
        self.bytes_read = 0
        decoder = CorrelationRecordDecoder()
        offset = 0
        pieces = data.split(b"\n")
        for number, piece in enumerate(pieces[:-1]):
            ref = LineRef(number, offset, len(piece), hashlib.sha256(piece).hexdigest())
            offset += len(piece) + 1
            record = _decoded(decoder, piece)
            if record is not None:
                self.items.append((ref, record))
        self.bytes_read = offset


def _decoded(decoder: CorrelationRecordDecoder, piece: bytes) -> dict[str, Any] | None:
    if not piece.strip():
        return None
    try:
        record = json.loads(piece)
        return decoder.decode(record) if isinstance(record, dict) else None
    except ValueError:
        return None


def is_client_request(record: Mapping[str, Any]) -> bool:
    """§0: an infer.request with no context, at schema version 1 or none."""
    return (
        record.get("event_type") == "infer.request"
        and "context" not in record
        and record.get("schema_version", 1) == 1
    )


def snapshot_of(
    path: Path,
    *,
    rel: str | None = None,
    generation: int = 0,
    index_id: str = INDEX_ID,
    gaps_resident: bool = True,
    error_messages_resident: bool = True,
    kept_record_bytes: int = KEPT_RECORD_BYTES,
) -> IndexSnapshot:
    """An IndexSnapshot of ``path``, as C1's index would publish it."""
    data = path.read_bytes()
    lines = _Lines(data)
    rows: list[RequestRow] = []
    for ref, record in lines.items:
        if is_client_request(record):
            rows.append(DictRow(record, index=len(rows), ref=ref))
    return _assemble(
        lines,
        rows,
        DictTable(rows),
        rel=rel or path.name,
        generation=generation,
        index_id=index_id,
        size=len(data),
        gaps_resident=gaps_resident,
        error_messages_resident=error_messages_resident,
        kept_record_bytes=kept_record_bytes,
        whole=hashlib.sha256(data).hexdigest(),
    )


def snapshot_from_records(
    records: Sequence[Mapping[str, Any]], **options: Any
) -> IndexSnapshot:
    """A snapshot of records written one per line, without a file."""
    data = b"".join(
        json.dumps(record, separators=(",", ":")).encode() + b"\n" for record in records
    )
    lines = _Lines(data)
    rows: list[RequestRow] = []
    for ref, record in lines.items:
        if is_client_request(record):
            rows.append(DictRow(record, index=len(rows), ref=ref))
    return _assemble(
        lines,
        rows,
        DictTable(rows),
        rel=options.pop("rel", "run.jsonl"),
        generation=options.pop("generation", 0),
        index_id=options.pop("index_id", INDEX_ID),
        size=len(data),
        gaps_resident=options.pop("gaps_resident", True),
        error_messages_resident=options.pop("error_messages_resident", True),
        kept_record_bytes=options.pop("kept_record_bytes", KEPT_RECORD_BYTES),
        whole=hashlib.sha256(data).hexdigest(),
    )


def _assemble(
    lines: _Lines,
    rows: Sequence[RequestRow],
    table: Any,
    *,
    rel: str,
    generation: int,
    index_id: str,
    size: int,
    gaps_resident: bool,
    error_messages_resident: bool,
    kept_record_bytes: int,
    whole: str,
) -> IndexSnapshot:
    kept: dict[str, list[Mapping[str, Any]]] = {}
    refs: dict[str, list[LineRef]] = {}
    inventory: dict[str, Counter[int]] = {}
    links: dict[str, list[LineRef | None]] = {}
    samples: list[Mapping[str, Any]] = []
    for ref, record in lines.items:
        event_type = str(record.get("event_type"))
        inventory.setdefault(event_type, Counter())[
            int(record.get("schema_version", 1))
        ] += 1
        if len(refs.setdefault(event_type, [])) < 1000:
            refs[event_type].append(ref)
        if event_type in KEPT_TYPES and ref.length <= kept_record_bytes:
            kept.setdefault(event_type, []).append(MappingProxyType(record))
        if event_type == "infer.system_sample":
            samples.append({k: v for k, v in record.items() if k != "metadata"})
        if event_type in ("infer.dispatch", "infer.first_content"):
            pair = links.setdefault(str(record.get("request_id")), [None, None])
            pair[0 if event_type == "infer.dispatch" else 1] = ref
    by_id: dict[str, list[int]] = {}
    for row in rows:
        by_id.setdefault(row.request_id, []).append(row.index)
    sessions = [
        record
        for _, record in lines.items
        if record.get("event_type") == "infer.session"
    ]
    session = _session(sessions, rows)
    origin = sessions[0].get("timestamp_ns") if sessions else None
    if origin is None and rows:
        origin = rows[0].get("started_at_ns")
    return IndexSnapshot(
        path=rel,
        index_id=index_id,
        generation=generation,
        status="ready",
        failure=None,
        identity=FileIdentity(1, 2, size, 3),
        bytes_read=lines.bytes_read,
        pending_tail_bytes=size - lines.bytes_read,
        prefix_sha256="0" * 64,
        sha256=whole if lines.bytes_read == size else None,
        growing=False,
        origin_ns=origin if isinstance(origin, int) else None,
        session=session,
        requests=tuple(rows),
        table=table,
        by_request_id={key: tuple(value) for key, value in by_id.items()},
        samples=tuple(samples),
        kept={key: tuple(value) for key, value in kept.items()},
        request_links={key: (value[0], value[1]) for key, value in links.items()},
        refs={key: tuple(value) for key, value in refs.items()},
        inventory={key: dict(value) for key, value in inventory.items()},
        problems=(),
        problem_count=0,
        limits_hit=frozenset(),
        gaps_resident=gaps_resident,
        error_messages_resident=error_messages_resident,
    )


def _session(
    sessions: Sequence[Mapping[str, Any]], rows: Sequence[RequestRow]
) -> SessionInfo | None:
    if not sessions:
        return None
    first, last = sessions[0], sessions[-1]
    terminal = [s for s in sessions if s.get("status") not in (None, "running")]
    return SessionInfo(
        session_id=str(first.get("session_id", "unknown")),
        status=last.get("status"),
        started_at_ns=first.get("timestamp_ns"),
        ended_at_ns=terminal[-1].get("timestamp_ns") if terminal else None,
        terminal=bool(terminal),
        config=dict(first.get("config") or {}),
    )


# The fields the large test's columns hold; a row view reads only these.
COLUMNS = (
    "event_type",
    "schema_version",
    "timestamp_ns",
    "session_id",
    "request_id",
    "x_request_id",
    "request_index",
    "case_id",
    "phase",
    "status",
    "http_status",
    "error_type",
    "error_message",
    "stream",
    "started_at_ns",
    "ended_at_ns",
    "intended_at_ns",
    "ttft_ms",
    "first_chunk_latency_ms",
    "e2e_latency_ms",
    "chunk_interarrival_ms",
    "prompt_tokens",
    "prompt_token_source",
    "prompt_token_exact",
    "output_tokens",
    "output_token_source",
    "output_token_exact",
    "total_tokens",
    "finish_reason",
)


class ColumnTable:
    """Columns of the client requests, one Python list per field."""

    def __init__(self) -> None:
        self.columns: dict[str, list[Any]] = {name: [] for name in COLUMNS}
        self.client: list[Mapping[str, CriterionValue]] = []
        self.client_columns: dict[str, list[float | None]] = {}
        self.refs: list[LineRef] = []

    def append(self, record: Mapping[str, Any], ref: LineRef) -> None:
        for name, values in self.columns.items():
            values.append(record.get(name))
        values_of = client_values(record)
        self.client.append(values_of)
        for metric, value in values_of.items():
            self.client_columns.setdefault(metric, []).append(value.value_ms)
        self.refs.append(ref)

    def __len__(self) -> int:
        return len(self.refs)

    def column(self, name: str) -> Sequence[Any]:
        if name.startswith(CLIENT_COLUMN_PREFIX):
            return self.client_columns[name]
        return self.columns.get(name) or [None] * len(self.refs)


class ColumnRow(RequestRow):
    """A view over one row of a ColumnTable, as C1's index.ColumnRow is."""

    __slots__ = ("_table", "_i")

    def __init__(self, table: ColumnTable, i: int) -> None:
        self._table = table
        self._i = i

    @property
    def index(self) -> int:
        return self._i

    @property
    def ref(self) -> LineRef:
        return self._table.refs[self._i]

    @property
    def session_id(self) -> str:
        return _text_or_unknown(self._table.columns["session_id"][self._i])

    @property
    def request_id(self) -> str:
        return _text_or_unknown(self._table.columns["request_id"][self._i])

    @property
    def record_id(self) -> str | None:
        return f"infer.request/{self._table.columns['request_id'][self._i]}"

    @property
    def case_id(self) -> str:
        return _text_or_unknown(self._table.columns["case_id"][self._i])

    @property
    def phase(self) -> str:
        return _text_or_unknown(self._table.columns["phase"][self._i])

    @property
    def client(self) -> Mapping[str, CriterionValue]:
        return self._table.client[self._i]

    @property
    def gap_count(self) -> int:
        gaps = self._table.columns["chunk_interarrival_ms"][self._i]
        return len(gaps) if isinstance(gaps, list) else 0

    def __getitem__(self, key: str) -> Any:
        values = self._table.columns.get(key)
        if values is None or values[self._i] is None:
            raise KeyError(key)
        return values[self._i]

    def get(self, key: str, default: Any = None) -> Any:
        values = self._table.columns.get(key)
        value = None if values is None else values[self._i]
        return default if value is None else value

    def __iter__(self) -> Iterator[str]:
        return (k for k, v in self._table.columns.items() if v[self._i] is not None)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def load_gaps(self) -> Sequence[float]:
        gaps = self._table.columns["chunk_interarrival_ms"][self._i]
        return [
            float(g) if isinstance(g, (int, float)) else math.nan for g in gaps or []
        ]

    def load_error_message(self) -> str | None:
        message = self._table.columns["error_message"][self._i]
        return message if isinstance(message, str) else None


def _text_or_unknown(value: Any) -> str:
    return "unknown" if value is None else str(value)


class LazyRows(Sequence[RequestRow]):
    """``IndexSnapshot.requests`` of a ColumnTable: a view per access."""

    def __init__(self, table: ColumnTable) -> None:
        self._table = table

    def __len__(self) -> int:
        return len(self._table)

    def __getitem__(self, i: Any) -> Any:
        if isinstance(i, slice):
            return [ColumnRow(self._table, j) for j in range(*i.indices(len(self)))]
        return ColumnRow(self._table, i if i >= 0 else len(self) + i)


def column_snapshot_of(path: Path, *, rel: str | None = None) -> IndexSnapshot:
    """A snapshot over real columns and lazy row views, for the timings."""
    data = path.read_bytes()
    lines = _Lines(data)
    table = ColumnTable()
    for ref, record in lines.items:
        if is_client_request(record):
            table.append(record, ref)
    rows = LazyRows(table)
    snapshot = _assemble(
        lines,
        [],
        table,
        rel=rel or path.name,
        generation=0,
        index_id=INDEX_ID,
        size=len(data),
        gaps_resident=True,
        error_messages_resident=True,
        kept_record_bytes=KEPT_RECORD_BYTES,
        whole=hashlib.sha256(data).hexdigest(),
    )
    by_id: dict[str, list[int]] = {}
    for i, request_id in enumerate(table.columns["request_id"]):
        by_id.setdefault(str(request_id), []).append(i)
    return replace(
        snapshot,
        requests=rows,
        by_request_id={key: tuple(value) for key, value in by_id.items()},
        session=snapshot.session,
    )


# --- Small records for the rules ----------------------------------------------

SESSION = "s" * 32
ORIGIN = 1_790_000_000 * SECOND


def request(index: int, **overrides: Any) -> dict[str, Any]:
    """A measured, successful, streamed client request of case c1."""
    started = ORIGIN + index * SECOND // 10
    record: dict[str, Any] = {
        "schema_version": 1,
        "event_type": "infer.request",
        "timestamp_ns": started + SECOND // 2,
        "session_id": SESSION,
        "request_id": f"r{index}",
        "x_request_id": f"x{index}",
        "request_index": index,
        "case_id": "c1",
        "phase": "measured",
        "status": "ok",
        "stream": True,
        "started_at_ns": started,
        "ended_at_ns": started + SECOND // 2,
        "ttft_ms": 100.0 + index,
        "first_chunk_latency_ms": 100.0 + index,
        "e2e_latency_ms": 500.0,
        "chunk_interarrival_ms": [10.0, 20.0, 30.0],
        "output_tokens": 16,
        "output_token_source": "server_usage",
        "output_token_exact": True,
        "prompt_tokens": 8,
        "prompt_token_source": "server_usage",
        "prompt_token_exact": True,
        "total_tokens": 24,
    }
    record.update(overrides)
    return record


def phase_window(
    case_id: str, started: int, drained: int, **extra: Any
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event_type": "infer.phase_window",
        "session_id": SESSION,
        "case_id": case_id,
        "phase": "measured",
        "started_at_ns": started,
        "window_ended_at_ns": drained,
        "drained_at_ns": drained,
        **extra,
    }


def workload(*cases: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event_type": "infer.workload",
        "session_id": SESSION,
        "seed": 0,
        "measurement": {"request_count": 10},
        "cases": [{"case_id": case, "arrival": {"mode": "closed"}} for case in cases],
    }


def session_record() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event_type": "infer.session",
        "session_id": SESSION,
        "timestamp_ns": ORIGIN - SECOND,
        "status": "running",
        "config": {},
    }


def closed_case(
    count: int = 10, *, case_id: str = "c1", **overrides: Any
) -> list[dict[str, Any]]:
    """A closed-loop case whose window spans its requests, with a workload."""
    rows = [request(i, case_id=case_id, **overrides) for i in range(count)]
    start = ORIGIN
    drained = ORIGIN + count * SECOND // 10 + SECOND
    return [
        session_record(),
        workload(case_id),
        *rows,
        phase_window(case_id, start, drained),
    ]


def _query(text: str) -> dict[str, list[str]]:
    return parse_qs(text, keep_blank_values=True, strict_parsing=bool(text))


# --- parse_filter and the canonical form ------------------------------------------


def test_the_default_filter_is_the_measured_phase() -> None:
    f = parse_filter({}, origin_ns=None)
    assert f == RequestFilter()
    assert canonical_params(f) == "phase=measured"


def test_every_parameter_parses_into_its_canonical_field() -> None:
    query = _query(
        "case=b&case=a&case=b&status=timeout&status=error&stream=false"
        "&token_source=tiktoken&q=MeAsUrEd_05&window=10,20&window_membership=overlap"
        "&completed_by_ns=30&latency=client.e2e:100:250.5&phase=all"
    )
    f = parse_filter(query, origin_ns=0)
    assert f == RequestFilter(
        phase="all",
        cases=("a", "b"),
        statuses=("error", "timeout"),
        stream=False,
        token_sources=("tiktoken",),
        text="measured_05",
        window=TimeWindow(10, 20, "overlap"),
        completed_by_ns=30,
        latency=LatencyRange("client.e2e", 100.0, 250.5),
    )


@pytest.mark.parametrize(
    ("text", "code", "field"),
    [
        ("sort=x", "unknown_parameter", "sort"),
        ("phase=both", "bad_parameter", "phase"),
        ("phase=measured&phase=all", "bad_parameter", "phase"),
        ("status=eror", "bad_parameter", "status"),
        ("stream=yes", "bad_parameter", "stream"),
        ("token_source=guess", "bad_parameter", "token_source"),
        ("q=" + "x" * 257, "limit_exceeded", "q"),
        ("window=5", "bad_parameter", "window"),
        ("window=5,5", "bad_parameter", "window"),
        ("window=a,5", "bad_parameter", "window"),
        ("window=1,2&window_ms=1,2", "bad_parameter", "window_ms"),
        ("window_ms=1.1234567,2", "bad_parameter", "window_ms"),
        ("window_membership=overlap", "bad_parameter", "window_membership"),
        ("window=1,2&window_membership=both", "bad_parameter", "window_membership"),
        ("completed_by_ns=1&completed_by_ms=1", "bad_parameter", "completed_by_ms"),
        ("latency=client.e2e:1", "bad_parameter", "latency"),
        ("latency=server.e2e:1:2", "bad_parameter", "latency"),
        ("latency=client.e2e:nan:", "bad_parameter", "latency"),
        ("latency=client.e2e:inf:", "bad_parameter", "latency"),
        ("latency=client.e2e:5:5", "bad_parameter", "latency"),
        ("latency=client.e2e:1e999:", "bad_parameter", "latency"),
    ],
)
def test_a_bad_parameter_is_a_400_naming_its_field(
    text: str, code: str, field: str
) -> None:
    with pytest.raises(UsageError) as raised:
        parse_filter(_query(text), origin_ns=0)
    assert raised.value.code == code
    assert raised.value.field == field
    assert raised.value.http_status == 400


def test_route_parameters_and_artifact_are_not_unknown() -> None:
    query = _query("artifact=runs/a.jsonl&sort=case&case=c1")
    assert parse_filter(query, origin_ns=None, route_params={"sort"}).cases == ("c1",)


def test_window_ms_converts_exactly_from_the_origin() -> None:
    origin = 1_788_609_600_123_456_789
    f = parse_filter(
        _query("window_ms=-0.5,1000.000001&completed_by_ms=0.1"), origin_ns=origin
    )
    assert f.window == TimeWindow(origin - 500_000, origin + 1_000_000_001)
    assert f.completed_by_ns == origin + 100_000
    # 0.1 + 0.2 in floats is 0.30000000000000004; the decimal is exact.
    exact = parse_filter(_query("completed_by_ms=0.3"), origin_ns=0)
    assert exact.completed_by_ns == 300_000


def test_window_ms_needs_an_origin() -> None:
    with pytest.raises(UsageError):
        parse_filter(_query("window_ms=0,1"), origin_ns=None)


def test_empty_q_is_no_text_condition() -> None:
    assert parse_filter(_query("q="), origin_ns=None).text is None


def _random_filter(rng: random.Random) -> RequestFilter:
    window = None
    if rng.random() < 0.5:
        start = rng.randrange(-(10**19), 10**19)
        window = TimeWindow(
            start, start + rng.randrange(1, 10**12), rng.choice(["arrival", "overlap"])
        )
    latency = None
    if rng.random() < 0.5:
        low = rng.uniform(0, 1e4)
        latency = LatencyRange(
            rng.choice(selection.CLIENT_METRICS),
            low,
            rng.choice([None, low + rng.uniform(1e-9, 1e4)]),
        )
    return RequestFilter(
        phase=rng.choice(["measured", "warmup", "all"]),
        cases=tuple(
            sorted(
                {
                    rng.choice(["a", "b c", "c&d", "é/x", "%41"])
                    for _ in range(rng.randrange(3))
                }
            )
        ),
        statuses=tuple(
            sorted(
                {
                    rng.choice(selection.REQUEST_STATUSES)
                    for _ in range(rng.randrange(3))
                }
            )
        ),
        stream=rng.choice([None, True, False]),
        token_sources=tuple(
            sorted(
                {rng.choice(selection.TOKEN_SOURCES) for _ in range(rng.randrange(3))}
            )
        ),
        text=rng.choice([None, "ab", "a&b=c", "naïve\n"]),
        window=window,
        completed_by_ns=rng.choice([None, rng.randrange(-(10**19), 10**19)]),
        latency=latency,
    )


def test_canonical_params_round_trip_through_parse_filter() -> None:
    rng = random.Random(53)
    for _ in range(500):
        f = _random_filter(rng)
        params = canonical_params(f)
        assert "\n" not in params
        assert parse_filter(_query(params), origin_ns=None) == f, params
        assert canonical_params(parse_filter(_query(params), origin_ns=None)) == params


def test_without_latency_drops_only_the_latency_condition() -> None:
    f = RequestFilter(cases=("c1",), latency=LatencyRange("client.e2e", 1.0, None))
    assert without_latency(f) == RequestFilter(cases=("c1",))


# --- select -------------------------------------------------------------------------


def _ids(snapshot: IndexSnapshot, f: RequestFilter) -> list[str]:
    return [snapshot.requests[i].request_id for i in select(snapshot, f).rows]


def _mixed_snapshot() -> IndexSnapshot:
    records = [
        session_record(),
        request(0),
        request(
            1,
            status="error",
            stream=False,
            ttft_ms=None,
            error_type="HTTPError",
            error_message="Server Said NO",
            output_token_source="estimated",
        ),
        request(2, phase="warmup"),
        request(3, case_id="c2", status="timeout", e2e_latency_ms=None),
        request(4, intended_at_ns=ORIGIN - SECOND),
        request(5, e2e_latency_ms=900.0),
    ]
    return snapshot_from_records(records)


def test_select_keeps_file_order_and_each_condition() -> None:
    snapshot = _mixed_snapshot()
    assert _ids(snapshot, RequestFilter()) == ["r0", "r1", "r3", "r4", "r5"]
    assert _ids(snapshot, RequestFilter(phase="all")) == [
        "r0",
        "r1",
        "r2",
        "r3",
        "r4",
        "r5",
    ]
    assert _ids(snapshot, RequestFilter(phase="warmup")) == ["r2"]
    assert _ids(snapshot, RequestFilter(cases=("c2",))) == ["r3"]
    assert _ids(snapshot, RequestFilter(statuses=("error", "timeout"))) == ["r1", "r3"]
    assert _ids(snapshot, RequestFilter(stream=False)) == ["r1"]
    assert _ids(snapshot, RequestFilter(token_sources=("estimated",))) == ["r1"]
    assert _ids(snapshot, RequestFilter(text="said no")) == ["r1"]
    assert _ids(snapshot, RequestFilter(text="x5")) == ["r5"]
    late = RequestFilter(latency=LatencyRange("client.e2e", 600.0, None))
    assert _ids(snapshot, late) == ["r5"]
    band = RequestFilter(latency=LatencyRange("client.e2e", 0.0, 900.0))
    assert _ids(snapshot, band) == ["r0", "r1", "r4"]
    cut = RequestFilter(completed_by_ns=ORIGIN + SECOND // 10 + SECOND // 2)
    assert _ids(snapshot, cut) == ["r0", "r1"]


def test_q_skips_error_messages_that_are_not_resident() -> None:
    snapshot = replace(_mixed_snapshot(), error_messages_resident=False)
    sel = select(snapshot, RequestFilter(text="said no"))
    assert list(sel.rows) == []
    assert "error_message" not in sel.searched_fields
    assert (
        select(_mixed_snapshot(), RequestFilter(text="x")).searched_fields
        == selection.TEXT_FIELDS
    )
    assert select(_mixed_snapshot(), RequestFilter()).searched_fields == ()


def test_window_membership_matches_populations_belongs() -> None:
    rng = random.Random(11)
    records = []
    for i in range(300):
        started = ORIGIN + rng.randrange(0, 50 * SECOND)
        extra: dict[str, Any] = {
            "started_at_ns": started,
            "ended_at_ns": started + rng.randrange(0, 5 * SECOND),
        }
        roll = rng.random()
        if roll < 0.3:
            extra["intended_at_ns"] = started - rng.randrange(0, SECOND)
        elif roll < 0.4:
            extra["started_at_ns"] = None
        elif roll < 0.45:
            extra["ended_at_ns"] = float("nan")
        records.append(request(i, **extra))
    snapshot = snapshot_from_records(records)
    for _ in range(50):
        begin = ORIGIN + rng.randrange(-SECOND, 50 * SECOND)
        end = begin + rng.randrange(1, 20 * SECOND)
        for membership in ("arrival", "overlap"):
            got = select(
                snapshot, RequestFilter(window=TimeWindow(begin, end, membership))
            )
            expected = [
                row.index
                for row in snapshot.requests
                if _belongs(row, begin, end, membership)
            ]
            assert list(got.rows) == expected


def test_the_selection_carries_the_contract_fingerprint() -> None:
    snapshot = _mixed_snapshot()
    sel = select(snapshot, RequestFilter(cases=("c1",)))
    assert sel.params == "case=c1&phase=measured"
    assert sel.fingerprint == selection_fingerprint(
        "run.jsonl", INDEX_ID, 0, sel.params
    )
    assert sel.cases == ("c1",)


# --- rate scope (§3.3, D-2, D-3, D-4) ----------------------------------------------


def test_a_whole_case_has_the_case_rate_and_its_interval() -> None:
    snapshot = snapshot_from_records(closed_case())
    sel = select(snapshot, RequestFilter(cases=("c1",)))
    population = case_facts(snapshot).populations["c1"]
    assert sel.whole_case == "c1"
    assert sel.rate.eligible and sel.rate.rule == "case"
    assert sel.rate.interval == population.intervals.rate
    # No case parameter, one case in the run: the same cohort.
    assert select(snapshot, RequestFilter()).whole_case == "c1"


@pytest.mark.parametrize(
    ("f", "reason"),
    [
        (RequestFilter(phase="all", cases=("c1",)), "not_measured_phase"),
        (RequestFilter(statuses=("ok",)), "outcome_filter"),
        (RequestFilter(stream=True), "outcome_filter"),
        (RequestFilter(token_sources=("server_usage",)), "outcome_filter"),
        (RequestFilter(text="r"), "outcome_filter"),
        (RequestFilter(completed_by_ns=ORIGIN + 10 * SECOND), "outcome_filter"),
        (
            RequestFilter(latency=LatencyRange("client.e2e", 0.0, None)),
            "outcome_filter",
        ),
        (RequestFilter(cases=("c1", "c2")), "multiple_cases"),
        (RequestFilter(cases=("nowhere",)), "empty_selection"),
        (
            RequestFilter(
                cases=("c1",), window=TimeWindow(ORIGIN, ORIGIN + SECOND, "overlap")
            ),
            "overlap_membership",
        ),
        (
            RequestFilter(
                cases=("c1",), window=TimeWindow(ORIGIN - 9 * SECOND, ORIGIN - SECOND)
            ),
            "window_outside_case",
        ),
    ],
)
def test_a_selection_that_is_no_cohort_of_one_case_has_no_rate(
    f: RequestFilter, reason: str
) -> None:
    snapshot = snapshot_from_records([*closed_case(), *closed_case(case_id="c2")[2:]])
    sel = select(snapshot, f)
    assert not sel.rate.eligible
    assert sel.rate.reason == reason
    assert sel.rate.interval is None and sel.rate.rule is None
    assert sel.whole_case is None


def test_two_cases_without_a_case_parameter_are_pooled_without_a_rate() -> None:
    snapshot = snapshot_from_records([*closed_case(), *closed_case(case_id="c2")[2:]])
    sel = select(snapshot, RequestFilter())
    assert sel.cases == ("c1", "c2")
    assert sel.rate.reason == "multiple_cases"


def test_an_interrupted_case_has_no_rate_even_with_a_window() -> None:
    records = closed_case()[:-1]  # the phase window is never written
    snapshot = snapshot_from_records(records)
    for f in (
        RequestFilter(),
        RequestFilter(window=TimeWindow(ORIGIN, ORIGIN + SECOND)),
    ):
        sel = select(snapshot, f)
        assert sel.rate.reason == "phase_window_missing"
        assert not sel.rate.eligible
        # Without a window the rows are still the case's whole cohort.
        assert sel.whole_case == (None if f.window else "c1")


def test_an_artifact_older_than_phase_windows_uses_the_request_span() -> None:
    records = [r for r in closed_case() if r["event_type"] == "infer.request"]
    snapshot = snapshot_from_records(records)
    sel = select(snapshot, RequestFilter())
    assert sel.rate.eligible and sel.rate.rule == "case"
    assert sel.rate.reason == "no_phase_window"
    assert sel.rate.interval is not None and sel.rate.interval.kind == "request_span"


def test_a_window_is_clamped_to_the_rate_interval_and_matches_a_segment() -> None:
    records = closed_case(40)
    snapshot = snapshot_from_records(records)
    facts = case_facts(snapshot)
    rate = facts.populations["c1"].intervals.rate
    assert rate is not None
    begin, end = ORIGIN - 5 * SECOND, ORIGIN + 2 * SECOND
    sel = select(snapshot, RequestFilter(window=TimeWindow(begin, end)))
    assert sel.rate.eligible and sel.rate.rule == "segment"
    assert sel.whole_case is None
    assert sel.rate.interval is not None
    assert (sel.rate.interval.started_at_ns, sel.rate.interval.ended_at_ns) == (
        rate.started_at_ns,
        end,
    )
    # populations' own segment over the same clamped window gives the same
    # interval and the same members.
    segment = Segment("probe", 0, end - rate.started_at_ns)
    rows = [snapshot.requests[i] for i in facts.measured]
    windows = [r for r in records if r["event_type"] == "infer.phase_window"]
    reference = populations_from_records(rows, windows, records[1], segments=[segment])
    expected = reference["c1"].segments["probe"]
    assert sel.rate.interval == expected.interval
    assert [snapshot.requests[i].request_id for i in sel.rows] == list(
        expected.request_ids
    )


def test_a_window_covering_the_rate_interval_is_the_whole_case() -> None:
    """D-2: whole case by the interval the window clamps to, not by rows."""
    records = closed_case(40)
    snapshot = snapshot_from_records(records)
    rate = case_facts(snapshot).populations["c1"].intervals.rate
    assert rate is not None
    whole = select(snapshot, RequestFilter())
    covering = select(
        snapshot,
        RequestFilter(window=TimeWindow(rate.started_at_ns - 1, rate.ended_at_ns + 1)),
    )
    assert covering.whole_case == "c1"
    assert covering.rate == whole.rate
    # A window that keeps every row but not the whole interval is a segment
    # over its part of the interval: no jump to the case's interval.
    last_arrival = max(
        int(r["started_at_ns"]) for r in records if r["event_type"] == "infer.request"
    )
    rows_only = select(
        snapshot, RequestFilter(window=TimeWindow(rate.started_at_ns, last_arrival + 1))
    )
    assert list(rows_only.rows) == list(whole.rows)
    assert rows_only.whole_case is None
    assert rows_only.rate.rule == "segment"
    assert rows_only.rate.interval is not None
    assert rows_only.rate.interval.ended_at_ns == last_arrival + 1


def test_a_row_arriving_outside_the_rate_interval_has_no_segment_rate() -> None:
    """D-3: the numerator's rows and the clamped interval must agree."""
    records = closed_case(20)
    records[3]["intended_at_ns"] = ORIGIN - 2 * SECOND  # arrived before the phase
    snapshot = snapshot_from_records(records)
    sel = select(
        snapshot, RequestFilter(window=TimeWindow(ORIGIN - 3 * SECOND, ORIGIN + SECOND))
    )
    assert records[3]["request_id"] in [
        snapshot.requests[i].request_id for i in sel.rows
    ]
    assert sel.rate.reason == "arrival_outside_interval"
    # A window that covers the interval but drops that row is not the case.
    rate = case_facts(snapshot).populations["c1"].intervals.rate
    assert rate is not None
    covering = select(
        snapshot, RequestFilter(window=TimeWindow(rate.started_at_ns, rate.ended_at_ns))
    )
    assert len(covering.rows) == 19
    assert covering.rate.reason == "arrival_outside_interval"
    assert covering.whole_case is None


def test_a_kept_record_over_the_limit_makes_rates_unavailable_not_wrong() -> None:
    """D-4: the analyzer reads the workload; without it a rate would change."""
    records = closed_case()
    records[1]["padding"] = "x" * 400
    snapshot = snapshot_from_records(records, kept_record_bytes=350)
    assert "infer.workload" not in snapshot.kept
    facts = case_facts(snapshot)
    assert facts.missing_kept == ("infer.workload",)
    sel = select(snapshot, RequestFilter())
    assert sel.rate.reason == "kept_record_over_limit"
    assert not sel.rate.eligible and sel.whole_case is None


class _Recording(dict[str, Any]):
    """A record that notes every key read from it."""

    read: set[str] = set()

    def get(self, key: str, default: Any = None) -> Any:
        _Recording.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key: str) -> Any:
        _Recording.read.add(key)
        return super().__getitem__(key)


def test_populations_read_only_the_fields_case_facts_hands_them() -> None:
    _Recording.read = set()
    records = closed_case(30)
    records[4]["request_index"] = 4.5
    records[5]["intended_at_ns"] = ORIGIN
    rows = [_Recording(r) for r in records if r["event_type"] == "infer.request"]
    windows = [r for r in records if r["event_type"] == "infer.phase_window"]
    segment = Segment("probe", 0, SECOND)
    for membership in ("arrival", "overlap"):
        populations_from_records(
            rows, windows, records[1], segments=[segment], membership=membership
        )
        populations_from_records(rows, [], None)
        populations_from_records(rows, [], records[1])
    assert _Recording.read <= set(selection.POPULATION_FIELDS)


def test_case_facts_equal_populations_over_the_full_records(
    studio_artifacts: StandardArtifacts,
) -> None:
    for name in ("mixed", "all_failed", "tokens_mixed", "execution_import", "growing"):
        snapshot = snapshot_of(studio_artifacts[name])
        rows = [row for row in snapshot.requests if row.phase == "measured"]
        windows = [
            w
            for w in snapshot.kept.get("infer.phase_window", ())
            if w.get("phase") == "measured"
        ]
        workload = (snapshot.kept.get("infer.workload") or [None])[0]
        assert case_facts(snapshot).populations == populations_from_records(
            rows, windows, workload
        ), name


def test_case_facts_are_computed_once_per_snapshot() -> None:
    snapshot = snapshot_from_records(closed_case())
    assert case_facts(snapshot) is case_facts(snapshot)
    other = snapshot_from_records(closed_case())
    assert case_facts(other) is not case_facts(snapshot)


# --- paging -------------------------------------------------------------------------


def _paging_snapshot() -> IndexSnapshot:
    rng = random.Random(5)
    records = [
        request(
            i,
            e2e_latency_ms=rng.choice(
                [None, 100.0, 200.0, float(rng.randrange(50, 900))]
            ),
            status=rng.choice(["ok", "ok", "error", "timeout"]),
            case_id=rng.choice(["b", "a", "c"]),
            output_tokens=rng.choice([None, 1, 2, 3, "junk"]),
        )
        for i in range(137)
    ]
    return snapshot_from_records(records)


def _all_pages(
    snapshot: IndexSnapshot, sel: Any, sort: Any, order: Any, limit: int
) -> list[int]:
    seen: list[int] = []
    cursor = None
    while True:
        got = page(snapshot, sel, sort=sort, order=order, cursor=cursor, limit=limit)
        assert got.total == len(sel.rows)
        seen.extend(got.rows)
        cursor = got.next_cursor
        if cursor is None:
            return seen


def _expected_order(
    snapshot: IndexSnapshot, rows: Sequence[int], sort: str, order: str
) -> list[int]:
    if sort == "artifact":
        return list(rows) if order == "asc" else list(rows)[::-1]
    values = snapshot.column(selection.SORT_KEYS[sort])
    text = sort in ("status", "case", "request_id")

    def value(i: int) -> Any:
        v = values[i]
        if text:
            return None if v is None else str(v)
        return (
            v
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(v)
            else None
        )

    present = [i for i in rows if value(i) is not None]
    nulls = [i for i in rows if value(i) is None]
    if order == "asc":
        present.sort(key=lambda i: (value(i), i))
    else:
        present.sort(key=lambda i: i)
        present.sort(key=value, reverse=True)
    return present + nulls


@pytest.mark.parametrize("sort", sorted(selection.SORT_KEYS))
@pytest.mark.parametrize("order", ["asc", "desc"])
def test_pages_walk_every_row_once_in_sort_order_with_nulls_last(
    sort: str, order: str
) -> None:
    snapshot = _paging_snapshot()
    sel = select(snapshot, RequestFilter())
    for limit in (1, 7, 1000):
        assert _all_pages(snapshot, sel, sort, order, limit) == _expected_order(
            snapshot, sel.rows, sort, order
        )


def test_a_cursor_from_another_index_filter_or_sort_is_a_409() -> None:
    snapshot = _paging_snapshot()
    sel = select(snapshot, RequestFilter())
    cursor = page(
        snapshot, sel, sort="client.e2e", order="asc", cursor=None, limit=5
    ).next_cursor
    rebuilt = replace(snapshot, index_id="f" * 32)
    cases = [
        (rebuilt, select(rebuilt, RequestFilter()), "client.e2e", "index_rebuilt"),
        (
            snapshot,
            select(snapshot, RequestFilter(cases=("a",))),
            "client.e2e",
            "other_filter",
        ),
        (snapshot, sel, "status", "other_sort"),
    ]
    for snap, other, sort, reason in cases:
        with pytest.raises(ConflictError) as raised:
            page(snap, other, sort=sort, order="asc", cursor=cursor, limit=5)  # type: ignore[arg-type]
        assert raised.value.code == "cursor_mismatch"
        assert raised.value.details == {"reason": reason}
    with pytest.raises(ConflictError):
        page(snapshot, sel, sort="client.e2e", order="desc", cursor=cursor, limit=5)


@pytest.mark.parametrize("cursor", ["", "!!", "eyJ2IjoyfQ", "bm90IGpzb24"])
def test_a_cursor_this_server_did_not_issue_is_a_400(cursor: str) -> None:
    snapshot = _paging_snapshot()
    sel = select(snapshot, RequestFilter())
    with pytest.raises(UsageError) as raised:
        page(
            snapshot, sel, sort="artifact", order="asc", cursor=cursor or "AA", limit=5
        )
    assert raised.value.field == "cursor"


def _forged(cursor: str, **changes: Any) -> str:
    padded = cursor + "=" * (-len(cursor) % 4)
    payload = json.loads(base64.urlsafe_b64decode(padded))
    payload.update(changes)
    text = json.dumps(payload, separators=(",", ":"))
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


# Fable's F-6 probe: keys that typed only by shape reached a comparison
# and raised TypeError (a 500). Each is (sort, key).
FORGED_KEYS: list[tuple[str, list[Any]]] = [
    ("client.e2e", [0, {"a": 1}, 3]),
    ("client.e2e", [0, [1, 2], 3]),
    ("client.e2e", [0, "abc", 3]),
    ("client.e2e", [0, None, 3]),
    ("client.e2e", [0, True, 3]),
    ("client.e2e", [0, float("inf"), 3]),
    ("client.e2e", [1, 12.5, 3]),
    ("client.e2e", [0, 12.5, -7]),
    ("client.e2e", [0, 12.5, 10**30]),
    ("client.e2e", [0, 12.5, 137]),
    ("client.e2e", [True, 12.5, 3]),
    ("status", [0, 12.5, 3]),
    ("status", [0, ["ok"], 3]),
    ("artifact", [0, "3", 3]),
    ("artifact", [0, 4, 3]),
    ("artifact", [1, None, 3]),
]


@pytest.mark.parametrize(("sort", "key"), FORGED_KEYS)
def test_a_forged_cursor_key_is_a_400_never_a_crash(sort: str, key: list[Any]) -> None:
    snapshot = _paging_snapshot()
    sel = select(snapshot, RequestFilter())
    issued = page(snapshot, sel, sort=sort, order="asc", cursor=None, limit=5)  # type: ignore[arg-type]
    assert issued.next_cursor is not None
    with pytest.raises(UsageError) as raised:
        page(
            snapshot,
            sel,
            sort=sort,  # type: ignore[arg-type]
            order="asc",
            cursor=_forged(issued.next_cursor, key=key),
            limit=5,
        )
    assert raised.value.code == "bad_parameter"
    assert raised.value.field == "cursor"


@pytest.mark.parametrize("sort", sorted(selection.SORT_KEYS))
def test_every_issued_cursor_key_still_fits_its_sort(sort: str) -> None:
    snapshot = _paging_snapshot()
    sel = select(snapshot, RequestFilter())
    for order in ("asc", "desc"):
        assert _all_pages(snapshot, sel, sort, order, 3) == _expected_order(
            snapshot, sel.rows, sort, order
        )


@pytest.mark.parametrize("limit", [0, -1, 1001])
def test_a_page_limit_outside_1_to_1000_is_a_400(limit: int) -> None:
    snapshot = _paging_snapshot()
    sel = select(snapshot, RequestFilter())
    with pytest.raises(UsageError):
        page(snapshot, sel, sort="artifact", order="asc", cursor=None, limit=limit)


def test_a_cursor_of_an_older_generation_still_pages_and_says_so() -> None:
    snapshot = _paging_snapshot()
    sel = select(snapshot, RequestFilter())
    first = page(snapshot, sel, sort="artifact", order="asc", cursor=None, limit=10)
    newer = replace(snapshot, generation=3)
    later = page(
        newer,
        select(newer, RequestFilter()),
        sort="artifact",
        order="asc",
        cursor=first.next_cursor,
        limit=10,
    )
    assert later.generation_moved
    assert later.rows == tuple(sel.rows[10:20])
    same = page(
        snapshot, sel, sort="artifact", order="asc", cursor=first.next_cursor, limit=10
    )
    assert not same.generation_moved


def test_the_cursor_is_opaque_url_safe_text() -> None:
    snapshot = _paging_snapshot()
    sel = select(snapshot, RequestFilter())
    cursor = page(
        snapshot, sel, sort="client.ttft", order="desc", cursor=None, limit=3
    ).next_cursor
    assert cursor is not None
    assert set(cursor) <= set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    )


# --- request_row ------------------------------------------------------------------


def test_request_row_finds_one_row_or_says_why_not() -> None:
    records = [request(0), request(1), request(2, request_id="r1")]
    snapshot = snapshot_from_records(records)
    assert request_row(snapshot, "r0", None) == 0
    with pytest.raises(ConflictError) as raised:
        request_row(snapshot, "r1", None)
    assert raised.value.code == "request_ambiguous"
    assert raised.value.details == {"lines": [1, 2]}
    assert request_row(snapshot, "r1", 2) == 2
    with pytest.raises(NotFoundError):
        request_row(snapshot, "r1", 0)
    with pytest.raises(NotFoundError):
        request_row(snapshot, "nope", None)


# --- the standard artifacts ----------------------------------------------------------


def test_every_standard_case_is_a_whole_case_with_the_analyzers_cohort(
    studio_artifacts: StandardArtifacts,
) -> None:
    snapshot = snapshot_of(studio_artifacts["mixed"])
    facts = case_facts(snapshot)
    assert len(facts.populations) == 4
    for case in facts.populations:
        sel = select(snapshot, RequestFilter(cases=(case,)))
        assert sel.whole_case == case
        assert sel.rate.eligible


def test_a_non_resident_column_is_never_read_by_select() -> None:
    class Guarded(DictTable):
        def column(self, name: str) -> Sequence[Any]:
            if name in ("chunk_interarrival_ms", "error_message"):
                raise NotResident(name)
            return super().column(name)

    snapshot = _mixed_snapshot()
    guarded = replace(
        snapshot, table=Guarded(snapshot.requests), error_messages_resident=False
    )
    assert list(select(guarded, RequestFilter(text="r1")).rows) == [1]


# --- 500,000 requests ------------------------------------------------------------------


@pytest.mark.studio_large
def test_select_and_a_page_take_under_2_s_on_500000_requests(
    large_studio_artifact: Path,
) -> None:
    snapshot = column_snapshot_of(large_studio_artifact)
    timings: dict[str, float] = {}
    # Cohorts are computed once per snapshot: on the builder pool when C4
    # warms them as a snapshot is published, else by the first select.
    started = time.perf_counter()
    case_facts(snapshot)
    cold = time.perf_counter() - started
    started = time.perf_counter()
    sel = select(snapshot, RequestFilter())
    timings["select"] = time.perf_counter() - started
    assert len(sel.rows) == 500_000 and sel.whole_case is not None
    started = time.perf_counter()
    got = page(snapshot, sel, sort="client.e2e", order="desc", cursor=None, limit=1000)
    timings["page client.e2e"] = time.perf_counter() - started
    started = time.perf_counter()
    page(
        snapshot,
        sel,
        sort="client.e2e",
        order="desc",
        cursor=got.next_cursor,
        limit=1000,
    )
    timings["page client.e2e, cursor"] = time.perf_counter() - started
    print(json.dumps({"case_facts, once per snapshot": cold, **timings}))
    assert max(timings.values()) < 2.0, timings
