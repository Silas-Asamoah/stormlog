"""Studio's bounded, columnar index of inference artifacts (design §4.4).

:class:`ArtifactIndexes` implements the contract's ``IndexRegistry``. It
keeps one index per artifact, builds and refreshes it on the builder pool,
and publishes an immutable :class:`IndexSnapshot` after every batch, so a
request thread never waits for a build and never reads more than one line
of the file (§2.1's I/O rule).

An index keeps:

- client requests as rows of a column store (:class:`ColumnRow` is one row,
  :class:`ColumnTable` gives whole columns, as Fable's D-1 asked): typed
  arrays for numbers and booleans, a code per row into a string table for
  repeated strings, one byte pool for unique strings, and the record's JSON
  text in a side table for any value that fits none of them. Each row also
  holds ``slo.client_values`` of its record, computed once here;
- chunk gaps and error messages in two pools, each dropped whole when its
  budget runs out, after which those fields raise :class:`NotResident`;
- system samples in a second column store, without ``metadata``;
- the small records of §4.4 whole, through ``finite_copy``; the lines of
  every request's dispatch and first-content records; and, per event type,
  counts by schema version and the first 1,000 line references.

Budgets come from ``contracts.Limits`` (§8.1). Past the row budget the index
stops reading at the first request it could not keep: the snapshot is
``partial`` with ``limits_hit`` naming ``rows``, ``bytes_read`` is where it
stopped, and every number is about the lines before that cut. Past the
sample budget it keeps reading but keeps no more samples (``partial``,
``samples``). Past the context budget the reader refuses new context
definitions, so the lines that refer to them are problems (``partial``,
``contexts``); the definitions it holds count in :meth:`memory`. An oversized or over-budget kept record leaves its line
reference only (``kept_records``), and so does one nested too deep to
copy, as an ``undecodable`` problem; a field nested too deep for the side
table reads as None, with the same problem.

Only one builder task runs per artifact at a time; request threads read
snapshots, whose columns the builder only ever appends to.
"""

from __future__ import annotations

import ctypes
import json
import logging
import math
import secrets
import sys
import threading
import time
from array import array
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field, replace
from typing import Any, BinaryIO, Union, overload

from stormlog.infer.artifact_reader import (
    BATCH_BYTES,
    TAIL_GRACE_S,
    ArtifactReader,
    ReadItem,
)
from stormlog.infer.diagnosis_inputs import line_digest, record_id
from stormlog.infer.report_stats import is_number
from stormlog.infer.slo import CriterionValue, client_values
from stormlog.studio.contracts import (
    CLIENT_COLUMN_PREFIX,
    ConflictError,
    GapPool,
    IndexFailure,
    IndexSnapshot,
    IndexStatus,
    LimitHit,
    Limits,
    LineProblem,
    LineRef,
    NotResident,
    RequestRow,
    SessionInfo,
    StudioError,
    WorkspaceRoot,
    encode_json,
    finite_copy,
    identity_text,
)

__all__ = [
    "trim_heap",
    "CLIENT_METRICS",
    "GAPS",
    "ERROR_MESSAGE",
    "KEPT_EVENT_TYPES",
    "REFS_PER_TYPE",
    "PROBLEMS_KEPT",
    "ArtifactIndexes",
    "ColumnRow",
    "ColumnTable",
]

_LOG = logging.getLogger(__name__)

# The client metrics, in slo.client_values' order; the same keys as
# latency_report.CLIENT_METRICS, read without importing the quantile code.
CLIENT_METRICS: tuple[str, ...] = tuple(client_values({}))
GAPS = "chunk_interarrival_ms"
ERROR_MESSAGE = "error_message"
# §4.4: the records an index keeps whole (each at most kept_record_bytes).
KEPT_EVENT_TYPES = frozenset(
    "infer." + name
    for name in (
        "session",
        "workload",
        "phase_window",
        "phase_start",
        "cache_state",
        "slo",
        "artifact",
        "capabilities",
        "trace_window",
        "treatments",
        "model_identity",
        "run_state",
    )
)
_LINK_TYPES = {"infer.dispatch": 0, "infer.first_content": 1}
# String fields that are unique per request: a byte pool, not a table.
_UNIQUE_KEYS = frozenset(
    {
        "request_id",
        "x_request_id",
        "trace_id",
        "span_id",
        "response_id",
        "prompt_digest",
    }
)
REFS_PER_TYPE = 1_000
PROBLEMS_KEPT = 100
# A repeated-string table past this many entries becomes a byte pool.
_TABLE_LIMIT = 1 << 16
# Rough cost of one Python object entry (a str and its dict slot).
_ENTRY_BYTES = 96
# Rows between two exact measurements of the row store.
_MEASURE_EVERY = 256
_GROWING_WINDOW_S = 5.0
_REFRESH_INTERVAL_S = 1.0
# How many identity checks one refresh task makes before yielding the pool.
_REFRESH_ROUNDS = 3


# --- Columns --------------------------------------------------------------


def _pad_bytes(itemsize: int, count: int) -> bytes:
    return bytes(itemsize * count)


class _Numbers:
    """Ints, floats or booleans of one key: a typed array and a null mask.

    The mask is made only once a row lacks a value. Ints start as 32-bit
    and widen to 64-bit; an int past 64 bits does not fit (side table).
    """

    def __init__(self, typecode: str, kind: type, row: int) -> None:
        self.kind = kind
        self.data: array[Any] = array(typecode)
        self.nulls: bytearray | None = None
        self._pad(row)

    def _pad(self, row: int) -> None:
        missing = row - len(self.data)
        if missing <= 0:
            return
        if self.nulls is None:
            self.nulls = bytearray(len(self.data))
        self.data.frombytes(_pad_bytes(self.data.itemsize, missing))
        self.nulls.extend(b"\x01" * missing)

    def append(self, row: int, value: Any) -> bool:
        if type(value) is not self.kind:
            return False
        if len(self.data) != row:
            self._pad(row)
        if not self._push(value):
            return False
        if self.nulls is not None:
            self.nulls.append(0)
        return True

    def _push(self, value: Any) -> bool:
        """Append, widening 32-bit ints to 64-bit; False past 64 bits."""
        try:
            self.data.append(value)
            return True
        except OverflowError:
            if self.data.typecode != "i" or not -(2**63) <= value < 2**63:
                return False
        self.data = array("q", self.data)
        self.data.append(value)
        return True

    def get(self, row: int) -> Any:
        if row >= len(self.data) or (self.nulls is not None and self.nulls[row]):
            return None
        value = self.data[row]
        return bool(value) if self.kind is bool else value

    def values(self, n: int) -> Sequence[Any]:
        data = self.data[:n]
        nulls = bytes(self.nulls[:n]) if self.nulls is not None else b""
        if self.kind is not bool and len(data) == n and 1 not in nulls:
            return data
        out: list[Any] = [bool(v) for v in data] if self.kind is bool else list(data)
        out.extend([None] * (n - len(out)))
        position = nulls.find(1)
        while position >= 0:
            out[position] = None
            position = nulls.find(1, position + 1)
        return out

    def nbytes(self) -> int:
        masked = 0 if self.nulls is None else len(self.nulls)
        return len(self.data) * self.data.itemsize + masked


class _Interned:
    """A repeated string per row: a code into a table (code 0 is null)."""

    def __init__(self, row: int) -> None:
        self.table: list[str | None] = [None]
        self.index: dict[str, int] = {}
        self.codes: array[int] = array("B", bytes(row))
        self.text_bytes = 0
        self.full = False

    def append(self, row: int, value: Any) -> bool:
        if type(value) is not str:
            return False
        code = self.index.get(value)
        if code is None:
            if len(self.table) > _TABLE_LIMIT:
                self.full = True
                return False
            code = len(self.table)
            self.table.append(value)
            self.index[value] = code
            self.text_bytes += len(value)
        codes = self.codes
        if len(codes) != row:
            codes.frombytes(_pad_bytes(codes.itemsize, row - len(codes)))
        try:
            codes.append(code)
        except OverflowError:
            self.codes = codes = array("H" if codes.typecode == "B" else "I", codes)
            codes.append(code)
        return True

    def get(self, row: int) -> Any:
        codes = self.codes
        return self.table[codes[row]] if row < len(codes) else None

    def values(self, n: int) -> Sequence[Any]:
        table = self.table
        out = [table[code] for code in self.codes[:n]]
        out.extend([None] * (n - len(out)))
        return out

    def nbytes(self) -> int:
        size = len(self.codes) * self.codes.itemsize
        return size + self.text_bytes + _ENTRY_BYTES * len(self.table)


class _Pooled:
    """A unique string per row: UTF-8 bytes in one pool, an end offset each."""

    def __init__(self, row: int) -> None:
        self.data = bytearray()
        self.ends: array[int] = array("q")
        self.nulls: bytearray | None = None
        self.ascii = True
        self._pad(row)

    def _pad(self, row: int) -> None:
        missing = row - len(self.ends)
        if missing <= 0:
            return
        if self.nulls is None:
            self.nulls = bytearray(len(self.ends))
        self.ends.extend(array("q", [len(self.data)]) * missing)
        self.nulls.extend(b"\x01" * missing)

    def append(self, row: int, value: Any) -> bool:
        if type(value) is not str:
            return False
        encoded = value.encode("utf-8", "surrogatepass")
        self.ascii = self.ascii and len(encoded) == len(value)
        if len(self.ends) != row:
            self._pad(row)
        self.data += encoded
        self.ends.append(len(self.data))
        if self.nulls is not None:
            self.nulls.append(0)
        return True

    def get(self, row: int) -> Any:
        ends = self.ends
        if row >= len(ends) or (self.nulls is not None and self.nulls[row]):
            return None
        start = ends[row - 1] if row else 0
        return bytes(self.data[start : ends[row]]).decode("utf-8", "surrogatepass")

    def values(self, n: int) -> Sequence[Any]:
        ends = self.ends[:n]
        if not self.ascii:
            out: list[Any] = [self.get(row) for row in range(len(ends))]
        else:
            text = bytes(self.data[: ends[-1] if ends else 0]).decode("ascii")
            starts = [0, *ends[:-1]]
            out = [text[start:end] for start, end in zip(starts, ends)]
            nulls = bytes(self.nulls[: len(ends)]) if self.nulls is not None else b""
            position = nulls.find(1)
            while position >= 0:
                out[position] = None
                position = nulls.find(1, position + 1)
        out.extend([None] * (n - len(out)))
        return out

    def nbytes(self) -> int:
        masked = 0 if self.nulls is None else len(self.nulls)
        return len(self.data) + 8 * len(self.ends) + masked

    @classmethod
    def from_interned(cls, column: _Interned) -> _Pooled:
        pooled = cls(0)
        for row, code in enumerate(column.codes):
            value = column.table[code]
            if value is not None:
                pooled.append(row, value)
        pooled._pad(len(column.codes))
        return pooled


class _Opaque:
    """A key whose values are lists or objects: every value is side text."""

    def append(self, row: int, value: Any) -> bool:
        return False

    def get(self, row: int) -> Any:
        return None

    def values(self, n: int) -> Sequence[Any]:
        return [None] * n

    def nbytes(self) -> int:
        return 0


_Column = Union[_Numbers, _Interned, _Pooled, _Opaque]


def _new_column(key: str, value: Any, row: int) -> _Column:
    kind = type(value)
    if kind is bool:
        return _Numbers("b", bool, row)
    if kind is int:
        return _Numbers("i", int, row)
    if kind is float:
        return _Numbers("d", float, row)
    if kind is str:
        return _Pooled(row) if key in _UNIQUE_KEYS else _Interned(row)
    return _Opaque()


class _RecordStore:
    """Records of one kind as columns: one per key, plus the keys of each row.

    A row's keys, in record order, are one entry of a table of key tuples
    (most rows of a file share one). ``special`` keys are in the key tuple
    but stored by the caller. A value no column can hold exactly (an int
    in a float column, a list, a bool among strings) is kept as JSON text
    in a side table, so every row gives back exactly what was written.
    """

    def __init__(self, special: frozenset[str] = frozenset()) -> None:
        self.special = special
        self.keysets: list[tuple[tuple[str, ...], frozenset[str]]] = []
        self._keyset_codes: dict[tuple[str, ...], int] = {}
        self.row_keysets: array[int] = array("H")
        self.columns: dict[str, _Column] = {}
        self.side: dict[str, dict[int, str]] = {}
        self.side_bytes = 0
        self.rows = 0
        # keys whose value had no JSON text (nested too deep); the caller
        # records the problem and clears the list
        self.dropped: list[str] = []

    def add(self, record: Mapping[str, Any]) -> int:
        row = self.rows
        self._add_keyset(tuple(record))
        columns = self.columns
        special = self.special
        for key, value in record.items():
            if value is None or key in special:
                continue
            column = columns.get(key)
            if column is None:
                column = columns[key] = _new_column(key, value, row)
            if not column.append(row, value):
                self._misfit(key, row, value)
        self.rows = row + 1
        return row

    def _add_keyset(self, keys: tuple[str, ...]) -> None:
        code = self._keyset_codes.get(keys)
        if code is None:
            code = self._keyset_codes[keys] = len(self.keysets)
            self.keysets.append((keys, frozenset(keys)))
        try:
            self.row_keysets.append(code)
        except OverflowError:
            self.row_keysets = array("I", self.row_keysets)
            self.row_keysets.append(code)

    def _misfit(self, key: str, row: int, value: Any) -> None:
        column = self.columns[key]
        if isinstance(column, _Interned) and column.full:
            pooled = _Pooled.from_interned(column)
            self.columns[key] = pooled
            if pooled.append(row, value):
                return
        self.put_side(key, row, value)

    def put_side(self, key: str, row: int, value: Any) -> None:
        """Keep ``value`` as JSON text; one ``json.loads`` accepted but
        ``json.dumps`` refuses (nested too deep) reads as None, its key
        noted in ``dropped``."""
        try:
            text = json.dumps(value)
        except (RecursionError, ValueError):
            self.dropped.append(key)
            return
        self.side.setdefault(key, {})[row] = text
        self.side_bytes += len(text) + _ENTRY_BYTES

    def keys(self, row: int) -> tuple[tuple[str, ...], frozenset[str]]:
        return self.keysets[self.row_keysets[row]]

    def side_value(self, key: str, row: int) -> tuple[bool, Any]:
        side = self.side.get(key)
        text = None if side is None else side.get(row)
        return (False, None) if text is None else (True, json.loads(text))

    def value(self, key: str, row: int) -> Any:
        found, value = self.side_value(key, row)
        if found:
            return value
        column = self.columns.get(key)
        return None if column is None else column.get(row)

    def values(self, key: str, n: int) -> Sequence[Any]:
        column = self.columns.get(key)
        base = [None] * n if column is None else column.values(n)
        side = self.side.get(key)
        if not side:
            return base
        out = list(base)
        for row, text in side.copy().items():
            if row < n:
                out[row] = json.loads(text)
        return out

    def nbytes(self) -> int:
        size = sum(column.nbytes() for column in self.columns.values())
        size += len(self.row_keysets) * self.row_keysets.itemsize + self.side_bytes
        return size + sum(_ENTRY_BYTES * len(keys) for keys, _ in self.keysets)


class _Refs:
    """Line references as arrays: line, offset, length and 32 hash bytes."""

    def __init__(self) -> None:
        self.line: array[int] = array("q")
        self.offset: array[int] = array("q")
        self.length: array[int] = array("q")
        self.sha = bytearray()

    def __len__(self) -> int:
        return len(self.line)

    def append(self, ref: LineRef) -> int:
        self.line.append(ref.line)
        self.offset.append(ref.offset)
        self.length.append(ref.length)
        self.sha += bytes.fromhex(ref.sha256)
        return len(self.line) - 1

    def get(self, i: int) -> LineRef:
        sha = bytes(self.sha[32 * i : 32 * i + 32]).hex()
        return LineRef(self.line[i], self.offset[i], self.length[i], sha)

    def nbytes(self) -> int:
        return 24 * len(self.line) + len(self.sha)


class _IdIndex:
    """Rows by request ID: a chained hash table in arrays, no str objects.

    ``table`` is replaced whole when it grows, so a reader that took it
    once walks a consistent table; within one table a row is linked before
    its slot points at it.
    """

    def __init__(self) -> None:
        self.hashes: array[int] = array("q")
        self.table: tuple[array[int], array[int]] = (
            array("i", [-1]) * 1024,
            array("i"),
        )

    def add(self, key: str, row: int) -> None:
        digest = hash(key)
        self.hashes.append(digest)
        slots, chain = self.table
        if row >= len(slots) // 2:
            slots, chain = self._grown(row, len(slots) * 4)
        bucket = digest & (len(slots) - 1)
        chain.append(slots[bucket])
        slots[bucket] = row

    def _grown(self, rows: int, size: int) -> tuple[array[int], array[int]]:
        slots: array[int] = array("i", [-1]) * size
        chain: array[int] = array("i")
        mask = size - 1
        hashes = self.hashes
        for row in range(rows):
            bucket = hashes[row] & mask
            chain.append(slots[bucket])
            slots[bucket] = row
        self.table = (slots, chain)
        return slots, chain

    def lookup(self, key: str, n: int, key_of: Callable[[int], str]) -> tuple[int, ...]:
        slots, chain = self.table
        digest = hash(key)
        row = slots[digest & (len(slots) - 1)]
        found = []
        while row >= 0:
            if row < n and self.hashes[row] == digest and key_of(row) == key:
                found.append(row)
            row = chain[row] if row < len(chain) else -1
        return tuple(reversed(found))

    def nbytes(self) -> int:
        slots, chain = self.table
        return 8 * len(self.hashes) + 4 * (len(slots) + len(chain))


class _Links:
    """Per row, the index of a dispatch or first-content line, or -1."""

    def __init__(self) -> None:
        self.data: array[int] = array("i")

    def set(self, row: int, value: int) -> None:
        missing = row - len(self.data)
        if missing > 0:
            self.data.extend(array("i", [-1]) * missing)
        self.data.append(value)

    def get(self, row: int) -> int:
        return self.data[row] if row < len(self.data) else -1


def _gap_value(gap: Any) -> float:
    if isinstance(gap, (int, float)) and not isinstance(gap, bool):
        return float(gap)
    return math.nan


def _non_numeric(gaps: list[Any]) -> int:
    """How many of ``gaps`` are not finite numbers (``is_number``).

    Floats whose sum is finite are all finite, so the usual list takes one
    pass in C; a sum that overflows only sends it to the exact count.
    """
    if all(type(gap) is float for gap in gaps) and math.isfinite(sum(gaps)):
        return 0
    return sum(1 for gap in gaps if not is_number(gap))


def _as_gaps(gaps: list[Any]) -> list[float]:
    if all(type(gap) is float for gap in gaps):
        return gaps
    return [_gap_value(gap) for gap in gaps]


class _Requests:
    """The client requests of one index: rows, client values, pools, IDs."""

    def __init__(self, gap_budget: int, error_budget: int) -> None:
        self.store = _RecordStore(special=frozenset({GAPS, ERROR_MESSAGE}))
        self.refs = _Refs()
        self.count = 0
        self.client: dict[str, array[float]] = {m: array("d") for m in CLIENT_METRICS}
        self.reasons: dict[str, array[int]] = {m: array("H") for m in CLIENT_METRICS}
        self.reason_table: list[tuple[str | None, bool]] = [(None, False)]
        self._reason_codes: dict[tuple[str | None, bool], int] = {(None, False): 0}
        self.gap_count: array[int] = array("i")
        # row -> its gaps that are not finite numbers, for the few rows
        # with any: the analyzer counts chunks from the numbers only
        self.non_numeric_gaps: dict[int, int] = {}
        self.gaps: array[float] | None = array("d")
        self.gap_start: array[int] | None = array("q")
        # 1 where the row has no text message (null, absent or a side
        # value); kept when the pool is dropped, as gap_count is for gaps
        self.error_null = bytearray()
        self.errors: bytearray | None = bytearray()
        self.error_start: array[int] | None = array("q")
        self.error_length: array[int] | None = array("i")
        self.ids = _IdIndex()
        self.dispatch = _Links()
        self.first_content = _Links()
        self.gap_budget = gap_budget
        self.error_budget = error_budget

    def add(self, record: Mapping[str, Any], ref: LineRef) -> int:
        row = self.store.add(record)
        self.refs.append(ref)
        self._add_client(client_values(record))
        self._add_gaps(row, record.get(GAPS))
        self._add_error(row, record.get(ERROR_MESSAGE))
        self.ids.add(identity_text(record.get("request_id")), row)
        self.count = row + 1
        return row

    def _add_client(self, values: Mapping[str, CriterionValue]) -> None:
        for metric in CLIENT_METRICS:
            value = values[metric]
            number = value.value_ms
            self.client[metric].append(math.nan if number is None else number)
            self.reasons[metric].append(self._reason(value))

    def _reason(self, value: CriterionValue) -> int:
        key = (value.reason, value.not_applicable)
        code = self._reason_codes.get(key)
        if code is None:
            code = self._reason_codes[key] = len(self.reason_table)
            self.reason_table.append(key)
        return code

    def _add_gaps(self, row: int, gaps: Any) -> None:
        if type(gaps) is not list:
            self.gap_count.append(-1)
            if gaps is not None:
                self.store.put_side(GAPS, row, gaps)
            if self.gaps is not None and self.gap_start is not None:
                self.gap_start.append(len(self.gaps))
            return
        self.gap_count.append(len(gaps))
        missing = _non_numeric(gaps)
        if missing:
            self.non_numeric_gaps[row] = missing
        if self.gaps is None or self.gap_start is None:
            return
        if 8 * (len(self.gaps) + len(gaps)) + 8 * (row + 1) > self.gap_budget:
            self.gaps = self.gap_start = None
            return
        self.gap_start.append(len(self.gaps))
        self.gaps.extend(_as_gaps(gaps))

    def _add_error(self, row: int, message: Any) -> None:
        self.error_null.append(type(message) is not str)
        if type(message) is not str:
            if message is not None:
                self.store.put_side(ERROR_MESSAGE, row, message)
            self._append_error(-1, -1)
            return
        if self.errors is None:
            return
        encoded = message.encode("utf-8", "surrogatepass")
        if len(self.errors) + len(encoded) + 12 * (row + 1) > self.error_budget:
            self.errors = self.error_start = self.error_length = None
            return
        self._append_error(len(self.errors), len(encoded))
        self.errors += encoded

    def _append_error(self, start: int, length: int) -> None:
        if self.error_start is not None and self.error_length is not None:
            self.error_start.append(start)
            self.error_length.append(length)

    def row_bytes(self) -> int:
        """Bytes of the rows, without the gap and error-message pools."""
        client = sum(9 * len(values) for values in self.client.values())
        size = self.store.nbytes() + self.refs.nbytes() + client
        size += 4 * len(self.gap_count) + len(self.error_null) + self.ids.nbytes()
        size += sys.getsizeof(self.non_numeric_gaps)
        return size + 4 * (len(self.dispatch.data) + len(self.first_content.data))

    def gap_bytes(self) -> int:
        if self.gaps is None or self.gap_start is None:
            return 0
        return 8 * (len(self.gaps) + len(self.gap_start))

    def error_bytes(self) -> int:
        if self.errors is None or self.error_start is None:
            return 0
        return len(self.errors) + 12 * len(self.error_start)


# --- Views a snapshot holds ----------------------------------------------


class _RequestView:
    """What one snapshot sees of the request store: its first ``n`` rows,
    and the pools as they were when it was published."""

    def __init__(
        self,
        requests: _Requests,
        link_refs: _Refs,
        pending: Mapping[str, tuple[int, int]],
        read_line: Callable[[LineRef], bytes],
    ) -> None:
        self.requests = requests
        self.store = requests.store
        self.n = requests.count
        self.link_refs = link_refs
        self.link_count = len(link_refs)
        self.pending = pending
        self.read_line = read_line
        self.gaps = requests.gaps
        self.gap_start = requests.gap_start
        self.errors = requests.errors
        self.error_start = requests.error_start
        self.error_length = requests.error_length

    @property
    def gaps_resident(self) -> bool:
        return self.gaps is not None

    @property
    def errors_resident(self) -> bool:
        return self.errors is not None

    def request_id(self, row: int) -> str:
        _, keys = self.store.keys(row)
        return identity_text(self.store.value("request_id", row))

    def rows_of(self, request_id: str) -> tuple[int, ...]:
        return self.requests.ids.lookup(request_id, self.n, self.request_id)

    def gaps_of(self, row: int) -> list[float] | None:
        count = self.requests.gap_count[row]
        if count < 0:
            return None
        if self.gaps is None or self.gap_start is None:
            raise NotResident(GAPS)
        start = self.gap_start[row]
        return list(self.gaps[start : start + count])

    def error_of(self, row: int) -> str | None:
        if self.requests.error_null[row]:
            return None  # no text: needs no pool, so no line read (F-11)
        if self.errors is None or self.error_start is None:
            raise NotResident(ERROR_MESSAGE)
        assert self.error_length is not None
        length = self.error_length[row]
        if length < 0:
            return None
        start = self.error_start[row]
        return bytes(self.errors[start : start + length]).decode(
            "utf-8", "surrogatepass"
        )

    def client(self, row: int) -> dict[str, CriterionValue]:
        requests = self.requests
        out = {}
        for metric in CLIENT_METRICS:
            number = requests.client[metric][row]
            reason, not_applicable = requests.reason_table[
                requests.reasons[metric][row]
            ]
            value = None if math.isnan(number) else number
            out[metric] = CriterionValue(value, reason, not_applicable)
        return out

    def link(self, links: _Links, row: int) -> LineRef | None:
        index = links.get(row)
        if index < 0 or index >= self.link_count:
            return None
        return self.link_refs.get(index)

    def stored_record(self, row: int) -> dict[str, Any]:
        """The row's record as written, read from its line."""
        line = self.read_line(self.requests.refs.get(row))
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ConflictError("stale_line", "the line is no longer a record")
        return record


def _stored_gaps(record: Mapping[str, Any]) -> tuple[float, ...]:
    gaps = record.get(GAPS)
    if not isinstance(gaps, list):
        return ()
    return tuple(_gap_value(gap) for gap in gaps)


class ColumnRow(RequestRow):
    """One client request of an index snapshot: a view over its columns.

    Every attribute is read from the columns on access, so making a row
    costs one object. ``row[key]`` gives the record's value for ``key``
    (the decoded record's keys, in record order). ``chunk_interarrival_ms``
    is a list of floats, NaN where the record had a non-number, and with
    ``error_message`` raises :class:`NotResident` when its pool is not
    resident; :meth:`load_gaps` and :meth:`load_error_message` then read
    the row's line, once, and raise ``ConflictError("stale_line")`` if its
    bytes changed. A null gaps list or error message is known without the
    pool, so it reads as None and needs no line read.
    """

    __slots__ = ("_view", "_row")

    def __init__(self, view: _RequestView, row: int) -> None:
        self._view = view
        self._row = row

    @property
    def index(self) -> int:
        return self._row

    @property
    def ref(self) -> LineRef:
        return self._view.requests.refs.get(self._row)

    def _identity(self, key: str) -> str:
        _, keys = self._view.store.keys(self._row)
        return identity_text(self[key]) if key in keys else "unknown"

    @property
    def session_id(self) -> str:
        return self._identity("session_id")

    @property
    def request_id(self) -> str:
        return self._identity("request_id")

    @property
    def record_id(self) -> str | None:
        minimal = {
            key: self[key]
            for key in ("event_type", "schema_version", "request_id")
            if key in self
        }
        return record_id(minimal)

    @property
    def case_id(self) -> str:
        return self._identity("case_id")

    @property
    def phase(self) -> str:
        return self._identity("phase")

    @property
    def client(self) -> Mapping[str, CriterionValue]:
        return self._view.client(self._row)

    @property
    def gap_count(self) -> int:
        return max(0, self._view.requests.gap_count[self._row])

    @property
    def numeric_gap_count(self) -> int:
        requests = self._view.requests
        missing = requests.non_numeric_gaps.get(self._row, 0)
        return max(0, requests.gap_count[self._row]) - missing

    def __getitem__(self, key: str) -> Any:
        view = self._view
        row = self._row
        _, keys = view.store.keys(row)
        if key not in keys:
            raise KeyError(key)
        found, value = view.store.side_value(key, row)
        if found:
            return value
        if key == GAPS:
            return view.gaps_of(row)
        if key == ERROR_MESSAGE:
            return view.error_of(row)
        return view.store.value(key, row)

    def __contains__(self, key: object) -> bool:
        _, keys = self._view.store.keys(self._row)
        return key in keys

    def __iter__(self) -> Iterator[str]:
        ordered, _ = self._view.store.keys(self._row)
        return iter(ordered)

    def __len__(self) -> int:
        ordered, _ = self._view.store.keys(self._row)
        return len(ordered)

    def __repr__(self) -> str:
        return f"ColumnRow(index={self._row}, request_id={self.request_id!r})"

    def load_gaps(self) -> Sequence[float]:
        try:
            gaps = self.get(GAPS)
        except NotResident:
            return _stored_gaps(self._view.stored_record(self._row))
        if not isinstance(gaps, list):
            return ()
        return tuple(_gap_value(gap) for gap in gaps)

    def load_error_message(self) -> str | None:
        try:
            message = self.get(ERROR_MESSAGE)
        except NotResident:
            message = self._view.stored_record(self._row).get(ERROR_MESSAGE)
        return message if isinstance(message, str) else None


class _Rows(Sequence[RequestRow]):
    """``IndexSnapshot.requests``: a row view per index, made on access."""

    def __init__(self, view: _RequestView) -> None:
        self._view = view

    def __len__(self) -> int:
        return self._view.n

    @overload
    def __getitem__(self, index: int) -> RequestRow: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[RequestRow]: ...

    def __getitem__(self, index: int | slice) -> RequestRow | Sequence[RequestRow]:
        if isinstance(index, slice):
            return [ColumnRow(self._view, i) for i in range(*index.indices(len(self)))]
        n = self._view.n
        position = index + n if index < 0 else index
        if not 0 <= position < n:
            raise IndexError("row index out of range")
        return ColumnRow(self._view, position)


class _RefColumn(Sequence[LineRef]):
    """``ColumnTable.line_refs``: the first ``n`` line references, each
    made on access from the reference arrays (rows below ``n`` never
    change, and nothing exports the arrays' buffers)."""

    def __init__(self, refs: _Refs, n: int) -> None:
        self._refs = refs
        self._n = n

    def __len__(self) -> int:
        return self._n

    @overload
    def __getitem__(self, index: int) -> LineRef: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[LineRef]: ...

    def __getitem__(self, index: int | slice) -> LineRef | Sequence[LineRef]:
        if isinstance(index, slice):
            return [self._refs.get(i) for i in range(*index.indices(self._n))]
        position = index + self._n if index < 0 else index
        if not 0 <= position < self._n:
            raise IndexError("line reference index out of range")
        return self._refs.get(position)


class ColumnTable:
    """``IndexSnapshot.table``: whole columns of one snapshot (D-1).

    A column is built once per snapshot and cached. A numeric column with a
    value in every row is an ``array`` (compact; elements are Python
    numbers when read); any other column is a list with None where a row
    lacks a value. ``client.<metric>`` gives each row's ``value_ms``.

    It is the contract's ``ColumnarTable`` too (Fable's F-2): the flat
    arrays below are copies of the first ``n`` rows of the index's arrays,
    built once per table and cached, so ``numpy.frombuffer`` on them never
    blocks the builder's appends (an exported buffer would).
    """

    def __init__(self, view: _RequestView) -> None:
        self._view = view
        self._columns: dict[str, Sequence[Any]] = {}
        self._arrays: dict[tuple[str, str], Any] = {}

    def __len__(self) -> int:
        return self._view.n

    def _cached(self, key: tuple[str, str], make: Callable[[], Any]) -> Any:
        value = self._arrays.get(key)
        if value is None:
            value = self._arrays[key] = make()
        return value

    def client_values(self, metric: str) -> Sequence[float]:
        values = self._view.requests.client[metric]  # KeyError: unknown
        result: Sequence[float] = self._cached(
            ("client", metric), lambda: values[: self._view.n]
        )
        return result

    def client_reasons(
        self, metric: str
    ) -> tuple[Sequence[int], tuple[tuple[str | None, bool], ...]]:
        requests = self._view.requests
        codes = requests.reasons[metric]  # KeyError: unknown
        result: tuple[Sequence[int], tuple[tuple[str | None, bool], ...]] = (
            self._cached(
                ("reasons", metric),
                lambda: (codes[: self._view.n], tuple(requests.reason_table)),
            )
        )
        return result

    def line_refs(self) -> Sequence[LineRef]:
        return _RefColumn(self._view.requests.refs, self._view.n)

    def line_lengths(self) -> Sequence[int]:
        lengths = self._view.requests.refs.length
        result: Sequence[int] = self._cached(
            ("refs", "length"), lambda: lengths[: self._view.n]
        )
        return result

    def gap_counts(self) -> Sequence[int]:
        counts = self._view.requests.gap_count
        result: Sequence[int] = self._cached(
            ("gaps", "count"),
            lambda: array("i", [max(0, c) for c in counts[: self._view.n]]),
        )
        return result

    def gap_pool(self) -> GapPool:
        view = self._view
        if view.gaps is None or view.gap_start is None:
            raise NotResident(GAPS)
        gaps, starts = view.gaps, view.gap_start
        counts = self.gap_counts()

        def make() -> GapPool:
            n = view.n
            end = starts[n - 1] + counts[n - 1] if n else 0
            return GapPool(gaps[:end], starts[:n], counts)

        pool: GapPool = self._cached(("gaps", "pool"), make)
        return pool

    def column(self, name: str) -> Sequence[Any]:
        cached = self._columns.get(name)
        if cached is None:
            cached = self._columns[name] = self._build(name)
        return cached

    def _build(self, name: str) -> Sequence[Any]:
        view = self._view
        n = view.n
        if name.startswith(CLIENT_COLUMN_PREFIX):
            if name not in CLIENT_METRICS:
                raise KeyError(name)
            values = view.requests.client[name][:n]
            return [None if math.isnan(value) else value for value in values]
        if name in (GAPS, ERROR_MESSAGE):
            return self._pooled(name, n)
        return view.store.values(name, n)

    def _pooled(self, name: str, n: int) -> Sequence[Any]:
        view = self._view
        read = view.gaps_of if name == GAPS else view.error_of
        out = []
        for row in range(n):
            found, value = view.store.side_value(name, row)
            out.append(value if found else read(row))
        return out


class _IdsView(Mapping[str, tuple[int, ...]]):
    """``IndexSnapshot.by_request_id`` over the index's hash table."""

    def __init__(self, view: _RequestView) -> None:
        self._view = view
        self._order: tuple[str, ...] | None = None

    def __getitem__(self, key: str) -> tuple[int, ...]:
        rows = self._view.rows_of(key) if isinstance(key, str) else ()
        if not rows:
            raise KeyError(key)
        return rows

    def _ids(self) -> tuple[str, ...]:
        if self._order is None:
            seen = dict.fromkeys(
                self._view.request_id(row) for row in range(self._view.n)
            )
            self._order = tuple(seen)
        return self._order

    def __iter__(self) -> Iterator[str]:
        return iter(self._ids())

    def __len__(self) -> int:
        return len(self._ids())


_LinkPair = tuple[Union[LineRef, None], Union[LineRef, None]]


class _LinksView(Mapping[str, _LinkPair]):
    """``IndexSnapshot.request_links``: dispatch and first-content lines."""

    def __init__(self, view: _RequestView) -> None:
        self._view = view

    def __getitem__(self, key: str) -> _LinkPair:
        view = self._view
        for row in view.rows_of(key) if isinstance(key, str) else ():
            pair = (
                view.link(view.requests.dispatch, row),
                view.link(view.requests.first_content, row),
            )
            if pair != (None, None):
                return pair
        pending = view.pending.get(key)
        if pending is None:
            raise KeyError(key)
        return self._ref(pending[0]), self._ref(pending[1])

    def _ref(self, index: int) -> LineRef | None:
        view = self._view
        return view.link_refs.get(index) if 0 <= index < view.link_count else None

    def _keys(self) -> list[str]:
        view = self._view
        keys = dict.fromkeys(
            view.request_id(row)
            for row in range(view.n)
            if view.requests.dispatch.get(row) >= 0
            or view.requests.first_content.get(row) >= 0
        )
        keys.update(dict.fromkeys(view.pending))
        return list(keys)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys())

    def __len__(self) -> int:
        return len(self._keys())


class _StoreRow(Mapping[str, Any]):
    """One row of a plain record store (a system sample)."""

    __slots__ = ("_store", "_row")

    def __init__(self, store: _RecordStore, row: int) -> None:
        self._store = store
        self._row = row

    def __getitem__(self, key: str) -> Any:
        _, keys = self._store.keys(self._row)
        if key not in keys:
            raise KeyError(key)
        return self._store.value(key, self._row)

    def __iter__(self) -> Iterator[str]:
        ordered, _ = self._store.keys(self._row)
        return iter(ordered)

    def __len__(self) -> int:
        ordered, _ = self._store.keys(self._row)
        return len(ordered)


class _StoreRows(Sequence[Mapping[str, Any]]):
    """``IndexSnapshot.samples``: the first ``n`` sample rows."""

    def __init__(self, store: _RecordStore, n: int) -> None:
        self._store = store
        self._n = n

    def __len__(self) -> int:
        return self._n

    @overload
    def __getitem__(self, index: int) -> Mapping[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Mapping[str, Any]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> Mapping[str, Any] | Sequence[Mapping[str, Any]]:
        if isinstance(index, slice):
            return [_StoreRow(self._store, i) for i in range(*index.indices(self._n))]
        position = index + self._n if index < 0 else index
        if not 0 <= position < self._n:
            raise IndexError("sample index out of range")
        return _StoreRow(self._store, position)


# --- One index -----------------------------------------------------------


def trim_heap() -> bool:
    """Hand the C allocator's free pages back to the OS; True if it ran.

    A build grows many arrays by reallocation, and glibc keeps the freed
    blocks: Fable measured 591 MiB RSS after a 500,000-row build against
    275 MiB traced, and 416 MiB after ``malloc_trim(0)`` (F-12). Linux
    only, through ctypes, best effort: it never raises.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim(0)
    except Exception:  # no glibc (musl), no symbol, no ctypes: nothing to do
        return False
    return True


def _event_name(record: Mapping[str, Any]) -> str:
    name = record.get("event_type")
    return name if isinstance(name, str) else "unknown"


def _version(record: Mapping[str, Any]) -> int:
    """The decoded ``schema_version``: 1 when absent, 0 when not an int."""
    version = record.get("schema_version", 1)
    return version if type(version) is int else 0


def _int_or_none(value: Any) -> int | None:
    return value if type(value) is int else None


def _finite_record(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """``finite_copy`` of a record to keep whole, or None when it is
    nested too deep to copy or to write as JSON again (Fable's F-4)."""
    try:
        copied = finite_copy(record)[0]
        encode_json(copied)  # what a route does with it
    except (RecursionError, ValueError):
        return None
    return copied


@dataclass
class _Session:
    """What the ``infer.session`` records said, as they arrive."""

    session_id: str
    started_at_ns: int | None
    config: Mapping[str, Any]
    status: str | None = None
    ended_at_ns: int | None = None
    terminal: bool = False

    def info(self) -> SessionInfo:
        return SessionInfo(
            session_id=self.session_id,
            status=self.status,
            started_at_ns=self.started_at_ns,
            ended_at_ns=self.ended_at_ns,
            terminal=self.terminal,
            config=self.config,
        )


@dataclass
class _State:
    """Everything one index ID has taken in."""

    limits: Limits
    index_id: str = field(default_factory=lambda: secrets.token_hex(16))
    generation: int = 0
    requests: _Requests = field(init=False)
    samples: _RecordStore = field(default_factory=_RecordStore)
    link_refs: _Refs = field(default_factory=_Refs)
    pending_links: dict[str, list[int]] = field(default_factory=dict)
    kept: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    kept_bytes: int = 0
    refs: dict[str, list[LineRef]] = field(default_factory=dict)
    inventory: dict[str, dict[int, int]] = field(default_factory=dict)
    problems: list[LineProblem] = field(default_factory=list)
    problem_count: int = 0
    problem_counts: dict[str, int] = field(default_factory=dict)
    limits_hit: set[LimitHit] = field(default_factory=set)
    sessions: set[str] = field(default_factory=set)
    session: _Session | None = None
    origin_ns: int | None = None
    failure: IndexFailure | None = None
    cut: LineRef | None = None  # the first request a budget left out
    deep_ref: LineRef | None = None  # the last line found too deep to keep
    stopped: bool = False  # no more lines are read (a cut, or a failure)
    built: bool = False  # the first build reached the end of the file
    row_bytes: int = 0
    measured_at: int = 0

    def __post_init__(self) -> None:
        self.requests = _Requests(
            self.limits.gap_bytes, self.limits.error_message_bytes
        )


_SNAPSHOT_REFRESH_KEYS = (
    "growing",
    "identity",
    "problems",
    "problem_count",
    "problem_counts",
    "status",
)


class _Index:
    """One artifact's index: the reader, the state and the latest snapshot.

    Only the builder task touches the reader and the state; it runs on the
    registry's pool, one at a time per index. Request threads read
    ``snapshot``, which the builder replaces whole.
    """

    def __init__(
        self,
        path: str,
        opener: Callable[[], BinaryIO],
        read_line: Callable[[LineRef], bytes],
        limits: Limits,
        options: _Options,
    ) -> None:
        self.path = path
        self.limits = limits
        self.options = options
        self.read_line = read_line
        self.reader = ArtifactReader(
            opener,
            max_line_bytes=limits.line_bytes,
            max_context_bytes=limits.context_bytes,
            tail_grace_s=options.tail_grace_s,
            clock=options.clock,
        )
        self.cancelled = threading.Event()
        self.scheduled = False
        self.complete = False
        self.last_stat = -math.inf
        self.last_growth: float | None = None
        self.memory_bytes = 0
        self.state = _State(limits)
        self.snapshot = self._snapshot()

    # --- Building -----------------------------------------------------

    def refresh(self) -> None:
        """Take in whatever the file gained since the last refresh.

        When a build (a first build, or a rebuild after ``changed``)
        completes, :func:`trim_heap` returns the build's freed memory.
        """
        state, was_built = self.state, self.state.built
        self._rounds()
        self.complete = self.complete or self.state.built or bool(self.state.failure)
        if self.state.built and (self.state is not state or not was_built):
            trim_heap()

    def _rounds(self) -> None:
        try:
            for _ in range(_REFRESH_ROUNDS):
                if self.cancelled.is_set() or not self._round():
                    break
        except (OSError, StudioError) as exc:
            self._unreadable(exc)
        except Exception:
            _LOG.exception("indexing %s failed", self.path)
            self._internal()

    def _round(self) -> bool:
        """One identity check and the reading it calls for; True if the
        file changed while this round read it."""
        before = self.reader.identity
        change = self.reader.identity_change()
        self.last_stat = self.options.clock()
        if self.state.failure == "unreadable":
            self.state.failure = None
        if change == "changed":
            self._restart()
            change = self.reader.identity_change()
        after = self.reader.identity
        if before is not None and after is not None and before.size != after.size:
            self.last_growth = self.last_stat
        progressed = self._read_all()
        if self.cancelled.is_set():
            return False  # a cancelled build is not built (F-14)
        self.state.built = True
        self._publish(new_lines=False)
        return change != "same" or progressed

    def _restart(self) -> None:
        self.reader.reset()
        self.state = _State(self.limits)
        self.last_growth = self.options.clock()
        self._publish(new_lines=False)

    def _read_all(self) -> bool:
        progressed = False
        while not self.state.stopped and not self.cancelled.is_set():
            before = self.reader.offset
            items = self.reader.read_batch(max_bytes=self.options.batch_bytes)
            self._ingest(items)
            if self.reader.contexts_limited:
                self.state.limits_hit.add("contexts")
            if self.reader.offset == before:
                break
            progressed = True
            self._publish(new_lines=True)
        return progressed

    def _unreadable(self, exc: BaseException) -> None:
        """The file could not be read.

        A file never read (no identity) fails as ``unreadable``; one read
        before keeps its snapshot, retried at the next refresh.
        """
        _LOG.info("cannot read %s: %s", self.path, type(exc).__name__)
        if self.reader.identity is None:
            self.state.failure = "unreadable"
            self.state.built = True
        self._publish(new_lines=False)

    def _internal(self) -> None:
        """Indexing raised an unexpected error (logged by the caller).

        The index fails for good as ``internal``, so it never stays
        ``building``; only a ``changed`` file rebuilds it, under a new
        index ID.
        """
        self._fail("internal")
        self.state.built = True
        self._publish(new_lines=False)

    # --- Taking in lines ----------------------------------------------

    def _ingest(self, items: Sequence[ReadItem]) -> None:
        for item in items:
            if self.state.stopped:
                return
            if item.kind == "record" and item.record is not None:
                self._record(item.record, item.ref)
            elif item.kind == "problem" and item.problem is not None:
                self._problem(item.problem)
            elif item.kind == "context":
                self._count("infer.context", 4, item.ref)

    def _problem(self, problem: LineProblem) -> None:
        if problem.code == "truncated_final_line":
            return  # the reader keeps it while the tail is pending
        state = self.state
        state.problem_count += 1
        state.problem_counts[problem.code] = (
            state.problem_counts.get(problem.code, 0) + 1
        )
        if len(state.problems) < PROBLEMS_KEPT:
            state.problems.append(problem)

    def _record(self, record: dict[str, Any], ref: LineRef) -> None:
        name = _event_name(record)
        version = _version(record)
        if name == "infer.request":
            if not self._request(record, ref, version):
                return
        elif name == "infer.session":
            self._session(record, ref)
        elif name == "infer.system_sample":
            self._sample(record, ref)
        elif name in _LINK_TYPES:
            self._link(name, record, ref)
        if name in KEPT_EVENT_TYPES:
            self._keep(name, record, ref)
        self._count(name, version, ref)

    def _count(self, name: str, version: int, ref: LineRef) -> None:
        counts = self.state.inventory.setdefault(name, {})
        counts[version] = counts.get(version, 0) + 1
        refs = self.state.refs.setdefault(name, [])
        if len(refs) < REFS_PER_TYPE:
            refs.append(ref)

    def _request(self, record: dict[str, Any], ref: LineRef, version: int) -> bool:
        """Add a client request; False when the line must not be counted."""
        if "context" in record:
            return True  # a server-view request: counted, never a row
        if version != 1:
            detail = f"client infer.request at schema_version {version}"
            self._problem(LineProblem(ref, "unsupported_version", detail))
            return True
        if self._rows_full():
            self.reader.rewind(ref)
            self.state.limits_hit.add("rows")
            self.state.cut = ref
            self.state.stopped = True
            return False
        self._note_session(identity_text(record.get("session_id")))
        if self.state.failure is not None:
            return False
        self._add_request(record, ref)
        return True

    def _rows_full(self) -> bool:
        state = self.state
        count = state.requests.count
        per_row = state.row_bytes / state.measured_at if state.measured_at else 1024
        estimate = state.row_bytes + per_row * (count - state.measured_at)
        if (
            count - state.measured_at >= _MEASURE_EVERY
            or estimate >= state.limits.row_bytes
        ):
            state.row_bytes = self._row_bytes()
            state.measured_at = count
        return state.row_bytes >= state.limits.row_bytes

    def _row_bytes(self) -> int:
        state = self.state
        pending = _ENTRY_BYTES * 2 * len(state.pending_links)
        return state.requests.row_bytes() + state.link_refs.nbytes() + pending

    def _add_request(self, record: dict[str, Any], ref: LineRef) -> None:
        state = self.state
        requests = state.requests
        row = requests.add(record, ref)
        self._dropped(requests.store, ref)
        if requests.gaps is None:
            state.limits_hit.add("gaps")
        if requests.errors is None:
            state.limits_hit.add("error_messages")
        if state.origin_ns is None:
            state.origin_ns = _int_or_none(record.get("started_at_ns"))
        request_id = identity_text(record.get("request_id"))
        links = state.pending_links.pop(request_id, None)
        if links is not None:
            if links[0] >= 0:
                requests.dispatch.set(row, links[0])
            if links[1] >= 0:
                requests.first_content.set(row, links[1])

    def _note_session(self, session_id: str) -> None:
        if session_id == "unknown" or self.state.failure is not None:
            return
        self.state.sessions.add(session_id)
        if len(self.state.sessions) > 1:
            self._fail("multiple_sessions")

    def _fail(self, failure: IndexFailure) -> None:
        """Stop for good: drop every row, keep counts and problems."""
        state = self.state
        state.failure = failure
        state.stopped = True
        state.requests = _Requests(
            self.limits.gap_bytes, self.limits.error_message_bytes
        )
        state.samples = _RecordStore()
        state.link_refs = _Refs()
        state.pending_links = {}

    def _session(self, record: Mapping[str, Any], ref: LineRef) -> None:
        state = self.state
        session_id = identity_text(record.get("session_id"))
        self._note_session(session_id)
        timestamp = _int_or_none(record.get("timestamp_ns"))
        if state.session is None:
            config = record.get("config")
            copied = _finite_record(config) if isinstance(config, Mapping) else {}
            if copied is None:
                self._too_deep(ref, "the session's config")
            state.session = _Session(session_id, timestamp, copied or {})
            if state.origin_ns is None:
                state.origin_ns = timestamp
        status = record.get("status")
        state.session.status = status if isinstance(status, str) else None
        if state.session.status not in (None, "running"):
            state.session.terminal = True
            state.session.ended_at_ns = timestamp

    def _too_deep(self, ref: LineRef, what: str) -> None:
        """One ``undecodable`` problem per line nested too deep to keep."""
        if self.state.deep_ref == ref:
            return
        self.state.deep_ref = ref
        detail = f"{what} is nested too deeply to keep; its line is kept"
        self._problem(LineProblem(ref, "undecodable", detail))

    def _dropped(self, store: _RecordStore, ref: LineRef) -> None:
        if store.dropped:
            self._too_deep(ref, "field " + ", ".join(sorted(set(store.dropped))))
            store.dropped.clear()

    def _sample(self, record: Mapping[str, Any], ref: LineRef) -> None:
        state = self.state
        if "samples" in state.limits_hit:
            return
        if state.samples.nbytes() >= self.limits.sample_bytes:
            state.limits_hit.add("samples")
            return
        state.samples.add({k: v for k, v in record.items() if k != "metadata"})
        self._dropped(state.samples, ref)

    def _link(self, name: str, record: Mapping[str, Any], ref: LineRef) -> None:
        state = self.state
        request_id = identity_text(record.get("request_id"))
        pending = state.pending_links.setdefault(request_id, [-1, -1])
        slot = _LINK_TYPES[name]
        if pending[slot] < 0:
            pending[slot] = state.link_refs.append(ref)

    def _keep(self, name: str, record: Mapping[str, Any], ref: LineRef) -> None:
        state = self.state
        over_one = ref.length > self.limits.kept_record_bytes
        if over_one or state.kept_bytes + ref.length > self.limits.kept_bytes:
            state.limits_hit.add("kept_records")
            return
        copied = _finite_record(record)
        if copied is None:
            state.limits_hit.add("kept_records")  # its line reference only
            self._too_deep(ref, "the record")
            return
        state.kept.setdefault(name, []).append(copied)
        state.kept_bytes += ref.length

    # --- Publishing ---------------------------------------------------

    def _status(self) -> tuple[IndexStatus, IndexFailure | None]:
        state = self.state
        if state.failure is not None:
            return "failed", state.failure
        if not state.built:
            return "building", None
        inventory = state.inventory
        if "infer.request" not in inventory and any(
            name.startswith("infer.audio.") for name in inventory
        ):
            return "failed", "audio"
        if "infer.session" not in inventory and "infer.request" not in inventory:
            return "failed", "not_an_inference_artifact"
        if state.limits_hit & {"rows", "samples", "contexts"}:
            return "partial", None
        return "ready", None

    def _growing(self) -> bool:
        if self.last_growth is None:
            return False
        return self.options.clock() - self.last_growth < self.options.growing_window_s

    def _problems(self) -> tuple[tuple[LineProblem, ...], int, dict[str, int]]:
        problems = list(self.state.problems)
        count = self.state.problem_count
        counts = dict(self.state.problem_counts)
        tail = self.reader.tail_problem
        if tail is not None:
            count += 1
            counts[tail.code] = counts.get(tail.code, 0) + 1
            if len(problems) < PROBLEMS_KEPT:
                problems.append(tail)
        return tuple(problems), count, counts

    def memory(self) -> int:
        state = self.state
        requests = state.requests
        size = self._row_bytes() + requests.gap_bytes() + requests.error_bytes()
        size += self.reader.context_bytes  # the decoder's definitions
        return size + state.samples.nbytes() + state.kept_bytes

    def _snapshot(self) -> IndexSnapshot:
        state = self.state
        reader = self.reader
        pending = {key: (pair[0], pair[1]) for key, pair in state.pending_links.items()}
        view = _RequestView(state.requests, state.link_refs, pending, self.read_line)
        status, failure = self._status()
        identity = reader.identity
        problems, problem_count, problem_counts = self._problems()
        whole = identity is not None and reader.offset == identity.size
        return IndexSnapshot(
            path=self.path,
            index_id=state.index_id,
            generation=state.generation,
            status=status,
            failure=failure,
            identity=identity,
            bytes_read=reader.offset,
            pending_tail_bytes=reader.pending_tail_bytes,
            prefix_sha256=reader.prefix_sha256,
            sha256=reader.prefix_sha256 if whole else None,
            growing=self._growing(),
            origin_ns=state.origin_ns,
            session=None if state.session is None else state.session.info(),
            requests=_Rows(view),
            table=ColumnTable(view),
            by_request_id=_IdsView(view),
            samples=_StoreRows(state.samples, state.samples.rows),
            kept={name: tuple(records) for name, records in state.kept.items()},
            request_links=_LinksView(view),
            refs={name: tuple(refs) for name, refs in state.refs.items()},
            inventory={name: dict(counts) for name, counts in state.inventory.items()},
            problems=problems,
            problem_count=problem_count,
            limits_hit=frozenset(state.limits_hit),
            gaps_resident=view.gaps_resident,
            error_messages_resident=view.errors_resident,
            cut=state.cut,
            line_count=reader.next_line,
            problem_counts=problem_counts,
        )

    def _publish(self, *, new_lines: bool) -> None:
        """Make a new snapshot current; nothing once the index is cancelled
        (evicted or closed), so a cancelled first build never says ready."""
        if self.cancelled.is_set():
            return
        if new_lines:
            self.state.generation += 1
        current = self.snapshot
        fresh = self._snapshot()
        if not new_lines and _same_data(current, fresh):
            if all(
                getattr(current, key) == getattr(fresh, key)
                for key in _SNAPSHOT_REFRESH_KEYS
            ):
                return
            changes = {key: getattr(fresh, key) for key in _SNAPSHOT_REFRESH_KEYS}
            fresh = replace(current, **changes)
        self.memory_bytes = self.memory()
        self.snapshot = fresh
        self.options.on_publish(self)


def _same_data(a: IndexSnapshot, b: IndexSnapshot) -> bool:
    """Whether two snapshots hold the same lines of the same index."""
    return (
        a.index_id == b.index_id
        and a.generation == b.generation
        and a.bytes_read == b.bytes_read
        and a.limits_hit == b.limits_hit
        and a.failure == b.failure
    )


@dataclass(frozen=True)
class _Options:
    clock: Callable[[], float]
    tail_grace_s: float
    batch_bytes: int
    growing_window_s: float
    on_publish: Callable[[_Index], None]


# --- The registry --------------------------------------------------------


def _stale_reason(data: bytes, ref: LineRef) -> str | None:
    """Why ``data`` (``ref.length + 1`` bytes read at ``ref.offset``) is not
    ``ref``'s line and newline, in C4a's ``export`` terms; None if it is."""
    if not data:
        return "missing"  # the file now ends before the line
    if len(data) != ref.length + 1:
        return "length"
    if data[-1:] != b"\n":
        return "newline"
    if line_digest(data[:-1]) != ref.sha256:
        return "sha256"
    return None


def _read_ref(handle: BinaryIO, ref: LineRef) -> bytes:
    handle.seek(ref.offset)
    data = handle.read(ref.length + 1)
    reason = _stale_reason(data, ref)
    if reason is not None:
        raise ConflictError(
            "stale_line",
            f"line {ref.line} no longer has the bytes the index read ({reason})",
            details={"line": ref.line, "reason": reason},
        )
    return data[:-1]


class ArtifactIndexes:
    """C1's ``IndexRegistry``: one index per artifact, built on ``pool``.

    ``root.open(rel)`` is each reader's opener. At most
    ``limits.open_indexes`` indexes stay open, and together they hold at
    most ``limits.index_memory_bytes``; past either bound the least
    recently used index that is not building is evicted, and a later
    :meth:`snapshot` rebuilds it under a new index ID.

    ``clock``, ``tail_grace_s``, ``refresh_interval_s``, ``batch_bytes``
    and ``growing_window_s`` exist for tests; Studio uses the defaults.
    """

    def __init__(
        self,
        root: WorkspaceRoot,
        pool: Executor,
        limits: Limits,
        *,
        clock: Callable[[], float] = time.monotonic,
        tail_grace_s: float = TAIL_GRACE_S,
        refresh_interval_s: float = _REFRESH_INTERVAL_S,
        batch_bytes: int = BATCH_BYTES,
        growing_window_s: float = _GROWING_WINDOW_S,
    ) -> None:
        self._root = root
        self._pool = pool
        self._limits = limits
        self._clock = clock
        self._refresh_interval_s = refresh_interval_s
        self._options = _Options(
            clock, tail_grace_s, batch_bytes, growing_window_s, self._published
        )
        self._cond = threading.Condition()
        self._entries: OrderedDict[str, _Index] = OrderedDict()
        self._futures: set[Future[None]] = set()
        self._listeners: list[Callable[[IndexSnapshot], None]] = []
        self._closed = False

    # --- IndexRegistry ------------------------------------------------

    def snapshot(self, rel: str) -> IndexSnapshot:
        """The current snapshot of ``rel``; schedules a build or refresh.

        Never raises NotReady: a new index publishes a zero-row
        ``building`` snapshot before its build is scheduled. Raises the
        root's NotFoundError or ForbiddenError for ``rel``.
        """
        return self._entry(rel).snapshot

    def wait(self, rel: str, timeout_s: float) -> IndexSnapshot:
        """The newest snapshot once the first build completes, or at the
        timeout, whichever is first; never raises on a timeout."""
        entry = self._entry(rel)
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._cond:
            while not entry.complete and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(min(remaining, 0.25))
        return entry.snapshot

    def read_line(self, rel: str, ref: LineRef) -> bytes:
        """One line's bytes without its newline; ``stale_line`` (409) when
        its bytes no longer hash to ``ref.sha256``."""
        with self._root.open(rel) as handle:
            return _read_ref(handle, ref)

    def read_lines(self, rel: str, refs: Iterable[LineRef]) -> Iterator[bytes]:
        """Each line with its newline, read as the caller iterates.

        The file is opened now, so a missing or forbidden path raises here;
        a line whose bytes changed raises ``stale_line`` when reached, which
        ends the stream.
        """
        handle = self._root.open(rel)
        return self._lines(handle, refs)

    @staticmethod
    def _lines(handle: BinaryIO, refs: Iterable[LineRef]) -> Iterator[bytes]:
        with handle:
            for ref in refs:
                yield _read_ref(handle, ref) + b"\n"

    def add_publish_listener(self, fn: Callable[[IndexSnapshot], None]) -> None:
        """Call ``fn(snapshot)`` on the builder pool after each publish.

        The contract's ``PublishingRegistry`` hook (Fable's F-7): after
        every new generation and every ``replace`` republish of an open
        index, once the snapshot is current. Not while a first build is
        still ``building``, so a listener that warms a cache does not redo
        it for every batch of a large file; its first call is the first
        complete snapshot. A listener that raises is logged and skipped.
        """
        with self._cond:
            self._listeners.append(fn)

    def close(self) -> None:
        """Cancel builds and drop every index."""
        with self._cond:
            self._closed = True
            for entry in self._entries.values():
                entry.cancelled.set()
            for future in list(self._futures):
                future.cancel()
            self._entries.clear()
            self._cond.notify_all()

    # --- Entries ------------------------------------------------------

    def _entry(self, rel: str) -> _Index:
        path = self._root.relative(self._root.resolve(rel))
        with self._cond:
            entry = self._entries.get(path)
            if entry is None:
                entry = self._open(path)
            else:
                self._entries.move_to_end(path)
            self._schedule(entry)
            return entry

    def _open(self, path: str) -> _Index:
        entry = _Index(
            path,
            lambda: self._root.open(path),
            lambda ref: self.read_line(path, ref),
            self._limits,
            self._options,
        )
        if not self._closed:
            self._entries[path] = entry
            self._evict(keep=entry, count=self._limits.open_indexes)
        return entry

    def _schedule(self, entry: _Index) -> None:
        if self._closed or entry.scheduled:
            return
        if (
            entry.complete
            and self._clock() - entry.last_stat < self._refresh_interval_s
        ):
            if not entry.snapshot.growing or entry._growing():
                return
        entry.scheduled = True
        try:
            future = self._pool.submit(self._run, entry)
        except RuntimeError:  # the pool is shut down
            entry.scheduled = False
            return
        self._futures.add(future)
        future.add_done_callback(self._forget)

    def _forget(self, future: Future[None]) -> None:
        with self._cond:
            self._futures.discard(future)

    def _run(self, entry: _Index) -> None:
        try:
            if not entry.cancelled.is_set():
                entry.refresh()
        finally:
            with self._cond:
                entry.scheduled = False
                self._cond.notify_all()

    def _published(self, entry: _Index) -> None:
        with self._cond:
            current = self._entries.get(entry.path) is entry
            if current:
                # The count too: _open let building indexes exceed it, and
                # only a publish can bring it back within the limit.
                self._evict(keep=entry, count=self._limits.open_indexes)
            self._cond.notify_all()
            listeners = tuple(self._listeners)
        snapshot = entry.snapshot
        if not current or snapshot.status == "building":
            return
        for listener in listeners:
            try:
                listener(snapshot)
            except Exception:
                _LOG.exception("a publish listener failed for %s", entry.path)

    def _evict(self, *, keep: _Index, count: int | None) -> None:
        """Evict least recently used indexes, never ``keep`` and never one
        still building, until within ``count`` (when given) and memory."""
        for path, entry in list(self._entries.items()):
            over_count = count is not None and len(self._entries) > count
            if not over_count and self._memory() <= self._limits.index_memory_bytes:
                return
            if entry is keep or entry.snapshot.status == "building":
                continue
            entry.cancelled.set()
            del self._entries[path]

    def _memory(self) -> int:
        return sum(entry.memory_bytes for entry in self._entries.values())

    def stats(self) -> dict[str, Any]:
        """Open indexes and the bytes each holds, for tests and the
        server document."""
        with self._cond:
            return {
                path: {
                    "index_id": entry.snapshot.index_id,
                    "status": entry.snapshot.status,
                    "memory_bytes": entry.memory_bytes,
                }
                for path, entry in self._entries.items()
            }
