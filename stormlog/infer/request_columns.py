"""Some requests read by column, for the analyzer's columnar summaries.

Studio's index keeps a run's client requests as columns (its
``ColumnarTable``); building a mapping per request to summarize 500,000
of them costs more than the numbers. :class:`ColumnRows` names the rows
of such a table that a summary is of, and the analyzer functions that
take ``columns=`` (``latency_summary``, ``streaming_summary``,
``token_coverage``, ``successful_throughput``, ``request_window``) read
it instead of records.

They give the same floats as over the records, by construction: each
value passes the same test (``is_number``, ``int_value``, ``== "ok"``)
the record path applies, in the same order. Only the sorts run in numpy,
and a float64 sort of finite values is the same list ``sorted`` gives
(see :func:`sorted_floats`).
"""

from __future__ import annotations

from array import array
from collections.abc import Callable, Iterable, Sequence, Sized
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .report_stats import int_value, is_number

if TYPE_CHECKING:
    import numpy.typing as npt

# Whether row ``i`` of the table lacks key ``name`` (rather than holding
# null): a column gives None for both.
Lacks = Callable[[int, str], bool]


class GapArrays(Protocol):
    """Every row's chunk gaps as one flat float sequence (Studio's GapPool).

    Row ``i``'s gaps are ``values[starts[i] : starts[i] + counts[i]]``, in
    written order, NaN where the element was not a number.
    """

    @property
    def values(self) -> Sequence[float]: ...

    @property
    def starts(self) -> Sequence[int]: ...

    @property
    def counts(self) -> Sequence[int]: ...


class RequestColumns(Protocol):
    """Requests by column, one element per request, in one order.

    - ``column(name)[i]`` is request ``i``'s ``record.get(name)``: None
      where the record lacks the key or holds null.
    - ``client_values(metric)[i]`` is
      ``slo.client_values(record)[metric].value_ms`` as a float, NaN for
      None.
    - ``gap_counts()[i]`` is the length of the request's
      ``chunk_interarrival_ms`` when that is a list, else 0;
      ``gap_pool()`` its elements as floats (NaN for a non-number), and
      may raise when the gaps are not held.
    """

    def __len__(self) -> int: ...

    def column(self, name: str) -> Sequence[Any]: ...

    def client_values(self, metric: str) -> Sequence[float]: ...

    def gap_counts(self) -> Sequence[int]: ...

    def gap_pool(self) -> GapArrays: ...


class ColumnRows:
    """The rows ``rows`` of ``table``, in that order: what a summary is of.

    ``lacks`` tells a missing key from a null one where a rule needs it
    (``token_coverage`` reads a missing source as ``"unknown"``, a null
    one as ``"None"``, as ``str(record.get(key, "unknown"))`` does); it is
    asked only for rows whose cell is None. Without it a None cell is
    read as a missing key.

    Columns picked for the rows are cached, so give each summary its own.
    """

    def __init__(
        self,
        table: RequestColumns,
        rows: Sequence[int],
        *,
        lacks: Lacks | None = None,
    ) -> None:
        self.table = table
        self.rows = rows
        self._lacks = lacks
        self._whole = _covers(rows, len(table))
        self._columns: dict[str, Sequence[Any]] = {}
        self._successful: list[int] | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def column(self, name: str) -> Sequence[Any]:
        """``column(name)`` of the table, for these rows."""
        picked = self._columns.get(name)
        if picked is None:
            values = self.table.column(name)
            picked = values if self._whole else _pick(values, self.rows)
            self._columns[name] = picked
        return picked

    def positions(self) -> npt.NDArray[np.int64]:
        """These rows' indexes into the table, as an int64 array."""
        if isinstance(self.rows, range):
            return np.arange(
                self.rows.start, self.rows.stop, self.rows.step, dtype=np.int64
            )
        return as_ints(self.rows)

    def floats(self, metric: str) -> npt.NDArray[np.float64]:
        """``client_values(metric)`` of the table, for these rows."""
        values = as_floats(self.table.client_values(metric))
        return values if self._whole else values[self.positions()]

    def subset(self, chosen: Sequence[int]) -> ColumnRows:
        """The rows at positions ``chosen`` (ascending) of these rows."""
        picked = self.positions()[np.asarray(chosen, dtype=np.int64)]
        rows = array("q", picked.tobytes())
        return ColumnRows(self.table, rows, lacks=self._lacks)

    def lacks(self, position: int, name: str) -> bool:
        """Whether the row at ``position`` lacks ``name`` (its cell is None)."""
        return self._lacks is None or self._lacks(self.rows[position], name)

    def successful(self) -> list[int]:
        """Positions of the rows whose status is ``"ok"``, as the analyzer
        tests ``record.get("status") == "ok"``."""
        if self._successful is None:
            statuses = self.column("status")
            self._successful = [k for k, s in enumerate(statuses) if s == "ok"]
        return self._successful


def _pick(values: Sequence[Any], rows: Sequence[int]) -> Sequence[Any]:
    """``[values[i] for i in rows]``, in one C call (``itemgetter``)."""
    if len(rows) < 2:
        return [values[i] for i in rows]
    picked: Sequence[Any] = itemgetter(*rows)(values)
    return picked


def pick(values: Sequence[Any], positions: Sequence[int]) -> Sequence[Any]:
    """``[values[k] for k in positions]``."""
    return _pick(values, positions)


_INTS = frozenset({int, type(None)})
_REALS = frozenset({int, float, type(None)})


def numbers_at(cells: Sequence[Any]) -> list[int]:
    """The positions ``k`` where ``is_number(cells[k])``.

    Columns of JSON ints, floats and nulls take a path without a call per
    cell: ``v - v == 0`` holds for an int and for a finite float, and not
    for an infinity or NaN, which is ``is_number``'s rule for those types
    (``type(v) is int`` already leaves bools out).
    """
    kinds = set(map(type, cells))
    if kinds <= _INTS:
        return [k for k, v in enumerate(cells) if v is not None]
    if kinds <= _REALS:
        return [k for k, v in enumerate(cells) if v is not None and v - v == 0]
    return [k for k, v in enumerate(cells) if is_number(v)]


def int_sum(cells: Sequence[Any]) -> int:
    """``sum(int_value(v) for v in cells)``; without a call per cell for a
    column of JSON ints and nulls (a null adds 0, as ``int_value`` reads it)."""
    if set(map(type, cells)) <= _INTS:
        return sum(filter(None, cells))
    return sum(int_value(v) for v in cells)


def no_records(requests: Iterable[Any]) -> None:
    """Refuse records beside ``columns=``: a summary reads one or the other."""
    if isinstance(requests, Sized) and len(requests):
        raise ValueError("give requests or columns, not both")


def _covers(rows: Sequence[int], n: int) -> bool:
    """Whether ``rows`` is every row ``0 .. n-1`` in order."""
    if len(rows) != n:
        return False
    if isinstance(rows, range):
        return rows.start == 0 and rows.step == 1
    return n == 0 or (rows[0] == 0 and rows[-1] == n - 1 and _ascending(rows))


def _ascending(rows: Sequence[int]) -> bool:
    ints = as_ints(rows)
    return bool(np.all(ints[1:] > ints[:-1]))


def as_floats(values: Sequence[float]) -> npt.NDArray[np.float64]:
    """``values`` as a float64 array, without a copy for an ``array('d')``."""
    if isinstance(values, array) and values.typecode == "d":
        return np.frombuffer(values, dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def as_ints(values: Sequence[int]) -> npt.NDArray[np.int64]:
    """``values`` as an int64 array."""
    if isinstance(values, array) and values.typecode == "q":
        return np.frombuffer(values, dtype=np.int64)
    return np.asarray(values, dtype=np.int64)


def sorted_floats(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """``values`` (no NaN) in ascending order, as a stable sort leaves them.

    A float64 sort of values without NaN gives ``sorted``'s order but for
    equal values, and equal floats have the same bits except ``0.0`` and
    ``-0.0``. So this sorts with numpy's default (faster) sort and then
    writes the zeros back in their given order, as a stable sort leaves
    them: the result is bit for bit ``sorted(values)``.
    """
    ordered = np.sort(values)
    zeros = values[values == 0]
    if zeros.size:
        start = int(np.searchsorted(ordered, 0.0, side="left"))
        ordered[start : start + zeros.size] = zeros
    return ordered


__all__ = [
    "ColumnRows",
    "GapArrays",
    "Lacks",
    "RequestColumns",
    "as_floats",
    "as_ints",
    "int_sum",
    "no_records",
    "numbers_at",
    "pick",
    "sorted_floats",
]
