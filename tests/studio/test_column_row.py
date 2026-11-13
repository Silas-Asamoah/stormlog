"""A column-backed RequestRow, as C1's index.ColumnRow will be (F-1, F-7).

The pre-commit mypy hook checks this file, so it fails if the contract
declares a row attribute that a read-only property cannot override.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest

from stormlog.infer.slo import CriterionValue, client_values
from stormlog.studio.contracts import (
    DictRow,
    LineRef,
    NotResident,
    RequestRow,
    RequestTable,
)


class ColumnStore:
    """Append-only columns, one element per request."""

    def __init__(self) -> None:
        self.refs: list[LineRef] = []
        self.strings: dict[str, list[str]] = {
            "session_id": [],
            "request_id": [],
            "case_id": [],
            "phase": [],
        }
        self.client: list[Mapping[str, CriterionValue]] = []
        self.gap_count: array[int] = array("q")

    def append(self, record: Mapping[str, Any], ref: LineRef) -> None:
        self.refs.append(ref)
        for key, values in self.strings.items():
            values.append(str(record.get(key, "unknown")))
        self.client.append(client_values(record))
        gaps = record.get("chunk_interarrival_ms")
        self.gap_count.append(len(gaps) if isinstance(gaps, list) else 0)

    def __len__(self) -> int:
        return len(self.refs)

    def column(self, name: str) -> Sequence[Any]:
        if name == "chunk_interarrival_ms":
            raise NotResident(name)
        return self.strings[name]


class ColumnRow(RequestRow):
    """A view over one row of a column store; every attribute is computed."""

    def __init__(self, store: ColumnStore, i: int) -> None:
        self._store = store
        self._i = i

    @property
    def index(self) -> int:
        return self._i

    @property
    def ref(self) -> LineRef:
        return self._store.refs[self._i]

    @property
    def session_id(self) -> str:
        return self._store.strings["session_id"][self._i]

    @property
    def request_id(self) -> str:
        return self._store.strings["request_id"][self._i]

    @property
    def record_id(self) -> str | None:
        return f"infer.request/{self.request_id}"

    @property
    def case_id(self) -> str:
        return self._store.strings["case_id"][self._i]

    @property
    def phase(self) -> str:
        return self._store.strings["phase"][self._i]

    @property
    def client(self) -> Mapping[str, CriterionValue]:
        return self._store.client[self._i]

    @property
    def gap_count(self) -> int:
        return self._store.gap_count[self._i]

    @property
    def numeric_gap_count(self) -> int:
        return self._store.gap_count[self._i]  # this store's gaps are numbers

    def __getitem__(self, key: str) -> Any:
        if key == "chunk_interarrival_ms":
            raise NotResident(key)
        return self._store.strings[key][self._i]

    def __iter__(self) -> Iterator[str]:
        return iter(self._store.strings)

    def __len__(self) -> int:
        return len(self._store.strings)

    def load_gaps(self) -> Sequence[float]:
        return array("d")

    def load_error_message(self) -> str | None:
        return None


REF = LineRef(line=3, offset=120, length=40, sha256="0" * 64)


def _store() -> ColumnStore:
    store = ColumnStore()
    store.append({"request_id": "a", "case_id": "c1", "ttft_ms": 10.0}, REF)
    store.append({"request_id": "b", "phase": "warmup"}, REF)
    return store


def test_a_column_row_reads_its_attributes_from_the_columns() -> None:
    store = _store()
    rows = [ColumnRow(store, i) for i in range(len(store))]
    table: RequestTable = store
    assert [row.index for row in rows] == [0, 1]
    assert [row.request_id for row in rows] == list(table.column("request_id"))
    assert rows[1].phase == table.column("phase")[rows[1].index] == "warmup"
    assert rows[0].session_id == "unknown" and rows[0].ref == REF
    assert rows[0].client["client.ttft"].value_ms == 10.0
    assert rows[0].gap_count == 0
    with pytest.raises(NotResident):
        rows[0].get("chunk_interarrival_ms")


def test_a_row_index_stays_valid_as_columns_grow() -> None:
    store = _store()
    row = ColumnRow(store, 1)
    store.append({"request_id": "c"}, REF)
    assert store.column("request_id")[row.index] == row.request_id == "b"


def test_every_row_attribute_is_an_abstract_property() -> None:
    names = {
        "index",
        "ref",
        "session_id",
        "request_id",
        "record_id",
        "case_id",
        "phase",
        "client",
        "gap_count",
        "numeric_gap_count",
    }
    for name in names:
        assert isinstance(vars(RequestRow)[name], property), name
    assert names | {"load_gaps", "load_error_message"} <= set(
        RequestRow.__abstractmethods__
    )


def test_a_subclass_missing_a_property_cannot_be_built() -> None:
    abstract = vars(RequestRow)["gap_count"]
    incomplete = type("Incomplete", (ColumnRow,), {"gap_count": abstract})
    with pytest.raises(TypeError, match="gap_count"):
        incomplete(_store(), 0)


def test_dict_row_attributes_are_read_only() -> None:
    row = DictRow({"request_id": "a"}, line=4, index=9)
    assert row.index == 9
    with pytest.raises(AttributeError):
        row.request_id = "b"  # type: ignore[misc]
