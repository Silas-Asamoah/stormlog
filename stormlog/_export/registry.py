"""A metric registry whose size is fixed before the first sample.

Every family is declared up front. Its labels are either closed enums, whose
values are listed, or bounded by configuration (a model, a case, a server),
whose values the run's configuration names. When the run starts, every
known series is created at 0 and its text prefix rendered once, so the exact
number of samples and the largest possible exposition are known before
anything is sent, and a run over its budget is refused.

Past a family's cap, a counter or histogram keeps its enum labels and has
its configuration-bounded labels replaced by ``__overflow__``, so totals over
the enums stay exact; a gauge's extra label set is rejected. Every redirect
and rejection is counted.

Updates run under one lock, which ``apply`` holds for a whole record so it
lands atomically, and which a scrape holds only to copy values. After
``freeze`` nothing changes: the final values are what the run reports.
"""

from __future__ import annotations

import bisect
import hashlib
import math
import re
import threading
from array import array
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Literal

METRIC_PREFIX = "stormlog_"
OVERFLOW = "__overflow__"
# Configuration-bounded label values longer than this keep a prefix and a
# digest, so two long values never collapse into one series.
MAX_LABEL_VALUE = 64
_DIGEST_CHARS = 8
# An upper bound on one rendered sample value, such as -1.2345678901234567e-308.
VALUE_BYTES = 24
DEFAULT_MAX_SAMPLES = 50_000
DEFAULT_MAX_BYTES = 16 * 1024 * 1024
DEFAULT_HEADROOM = 64

_NAME = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*\Z")
_LABEL = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*\Z")
Kind = Literal["counter", "gauge", "histogram"]
LabelValues = tuple[str, ...]


class BudgetExceeded(ValueError):
    """The declared families would exceed the sample or byte budget."""

    def __init__(self, samples: int, size: int, max_samples: int, max_bytes: int):
        self.samples = samples
        self.size = size
        self.max_samples = max_samples
        self.max_bytes = max_bytes
        super().__init__(
            f"the metrics need {samples} samples and up to {size} bytes per "
            f"scrape; the limits are {max_samples} samples and {max_bytes} bytes"
        )


@dataclass(frozen=True)
class FamilySpec:
    """One metric family: its name, kind, unit, labels and buckets.

    ``enums`` lists the allowed values of each closed-enum label. Every
    other label is bounded by configuration.
    """

    name: str
    kind: Kind
    help: str
    unit: str = ""
    labels: tuple[str, ...] = ()
    enums: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    buckets: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _check_name(self.name, self.kind, self.unit)
        _check_labels(self.labels, self.enums, self.kind)
        _check_buckets(self.buckets, self.kind)

    @property
    def bounded(self) -> tuple[str, ...]:
        return tuple(label for label in self.labels if label not in self.enums)


@dataclass(frozen=True)
class Budget:
    """Samples and largest exposition size, including headroom and overflow."""

    samples: int
    size: int


@dataclass
class FamilyStats:
    series: int = 0
    overflow_redirects: int = 0
    rejected: int = 0


@dataclass
class _Series:
    prefixes: tuple[bytes, ...]
    # Counter or gauge: one value. Histogram: per-bucket counts (the last
    # is +Inf), then the sum and the count are kept apart.
    values: list[float]
    total: float = 0.0
    count: int = 0


# Per family: each series' rendered prefixes (shared, not copied) and its
# values, flattened into one array of floats. A counter or gauge has one
# value per series; a histogram has its bucket counts, its sum and its count.
Snapshot = list[tuple["Family", list[tuple[bytes, ...]], array]]


class Family:
    """A declared family. Its update methods are called inside ``Registry.apply``."""

    def __init__(
        self,
        registry: "Registry",
        spec: FamilySpec,
        known: Sequence[Mapping[str, str]],
        headroom: int,
        precreate: bool = True,
    ) -> None:
        self.registry = registry
        self.spec = spec
        self.stats = FamilyStats()
        self._series: dict[LabelValues, _Series] = {}
        self._overflow_keys: set[LabelValues] = set()
        # Series other than overflow ones; the cap applies to these.
        self._regular = 0
        expected = _known_label_sets(spec, known)
        if precreate:
            for values in expected:
                self._create(values)
        self.cap = len(expected) + (headroom if spec.bounded else 0)

    # ------------------------------------------------------------- updates
    # Each update takes the registry's lock itself; inside Registry.apply the
    # lock is already held, so a record's updates land together.
    def inc(self, labels: LabelValues, amount: float = 1.0) -> None:
        with self.registry._lock:
            series = self._target(labels)
            if series is not None:
                series.values[0] += amount

    def set(self, labels: LabelValues, value: float) -> None:
        with self.registry._lock:
            series = self._target(labels)
            if series is not None:
                series.values[0] = value

    def observe(self, labels: LabelValues, value: float) -> None:
        with self.registry._lock:
            if not math.isfinite(value):
                self.stats.rejected += 1
                return
            series = self._target(labels)
            if series is None:
                return
            series.values[bisect.bisect_left(self.spec.buckets, value)] += 1
            series.total += value
            series.count += 1

    def observe_counts(
        self, labels: LabelValues, counts: Sequence[int], total: float
    ) -> None:
        """Add pre-counted observations: one count per bucket, the last +Inf."""
        if len(counts) != len(self.spec.buckets) + 1:
            raise ValueError("one count per bucket, plus +Inf, is needed")
        with self.registry._lock:
            if not math.isfinite(total):
                self.stats.rejected += 1
                return
            series = self._target(labels)
            if series is None:
                return
            for index, count in enumerate(counts):
                series.values[index] += count
            series.total += total
            series.count += sum(counts)

    # ------------------------------------------------------------- internals
    def _target(self, labels: LabelValues) -> _Series | None:
        if not self.registry._writable():
            return None
        return self._resolve(labels)

    def _resolve(self, labels: LabelValues) -> _Series | None:
        values = self._normalise(labels)
        if values is None:
            self.stats.rejected += 1
            return None
        series = self._series.get(values)
        if series is not None:
            return series
        if self._regular < self.cap:
            return self._create(values)
        return self._overflow(values)

    def _normalise(self, labels: LabelValues) -> LabelValues | None:
        spec = self.spec
        if len(labels) != len(spec.labels):
            return None
        normalised = []
        for name, value in zip(spec.labels, labels):
            allowed = spec.enums.get(name)
            if allowed is not None and value not in allowed:
                return None
            normalised.append(value if allowed is not None else bounded_value(value))
        return tuple(normalised)

    def _overflow(self, values: LabelValues) -> _Series | None:
        if self.spec.kind == "gauge":
            self.stats.rejected += 1
            return None
        key = tuple(
            value if name in self.spec.enums else OVERFLOW
            for name, value in zip(self.spec.labels, values)
        )
        self.stats.overflow_redirects += 1
        series = self._series.get(key)
        if series is None:
            series = self._create(key, overflow=True)
        return series

    def _create(self, values: LabelValues, *, overflow: bool = False) -> _Series:
        spec = self.spec
        prefixes = _prefixes(spec, self.registry.const_labels, values)
        slots = len(spec.buckets) + 1 if spec.kind == "histogram" else 1
        series = _Series(prefixes=prefixes, values=[0.0] * slots)
        self._series[values] = series
        if overflow:
            self._overflow_keys.add(values)
        else:
            self._regular += 1
        self.stats.series = len(self._series)
        return series

    def _copy(self) -> tuple[list[tuple[bytes, ...]], array]:
        prefixes = []
        values = array("d")
        histogram = self.spec.kind == "histogram"
        for series in self._series.values():
            prefixes.append(series.prefixes)
            values.extend(series.values)
            if histogram:
                values.append(series.total)
                values.append(series.count)
        return prefixes, values


class Registry:
    """Families, their budget, and one lock for every update and copy."""

    def __init__(
        self,
        *,
        const_labels: Mapping[str, str] | None = None,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        max_bytes: int = DEFAULT_MAX_BYTES,
        headroom: int = DEFAULT_HEADROOM,
    ) -> None:
        self.const_labels = dict(const_labels or {})
        for name in self.const_labels:
            if not _LABEL.match(name) or name.startswith("__"):
                raise ValueError(f"invalid constant label name {name!r}")
        self.max_samples = max_samples
        self.max_bytes = max_bytes
        self.headroom = headroom
        self.families: list[Family] = []
        self._names: set[str] = set()
        self._lock = threading.RLock()
        self._frozen = False
        self.late_updates = 0

    def add(
        self,
        spec: FamilySpec,
        known: Iterable[Mapping[str, str]] = (),
        *,
        precreate: bool = True,
    ) -> Family:
        """Declare a family and create its known series at 0.

        ``known`` lists the configured values of the bounded labels, one
        mapping per combination; each is paired with every enum value. With
        ``precreate`` off, the known series are budgeted but appear only
        once they have a value, for sources whose missing values must not
        read as 0.
        """
        for name in _sample_names(spec):
            if name in self._names:
                raise ValueError(f"metric name {name!r} is already declared")
        family = Family(self, spec, list(known), self.headroom, precreate)
        with self._lock:
            self.families.append(family)
            self._names.update(_sample_names(spec))
        return family

    def budget(self) -> Budget:
        """Samples and the largest exposition, with headroom and overflow."""
        samples = 0
        size = 0
        for family in self.families:
            family_samples, family_size = _family_budget(family, self.const_labels)
            samples += family_samples
            size += family_size
        return Budget(samples=samples, size=size)

    def check_budget(self) -> Budget:
        """The budget, or ``BudgetExceeded`` when it is over a limit."""
        budget = self.budget()
        if budget.samples > self.max_samples or budget.size > self.max_bytes:
            raise BudgetExceeded(
                budget.samples, budget.size, self.max_samples, self.max_bytes
            )
        return budget

    def apply(self, update: Callable[[], None]) -> bool:
        """Run ``update`` under the lock, whole, unless the registry is frozen."""
        with self._lock:
            if self._frozen:
                self.late_updates += 1
                return False
            update()
            return True

    def freeze(self, final: Callable[[], None] | None = None) -> None:
        """Stop every later update; the values now are the final ones.

        ``final`` runs under the same lock just before, so values computed
        from counts that only this lock keeps steady (such as how many
        queued records were never applied) land in the frozen state.
        """
        with self._lock:
            if final is not None and not self._frozen:
                final()
            self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    def snapshot(self) -> Snapshot:
        """Copy every value under the lock; render the copy outside it.

        Only the values are copied, into one array of floats per family; the
        prefixes are shared, since they never change once rendered.
        """
        with self._lock:
            return [(family, *family._copy()) for family in self.families]

    def _writable(self) -> bool:
        # Called with the lock held, through apply, or directly by a caller
        # that wants a single update; either way a frozen registry refuses.
        with self._lock:
            if self._frozen:
                self.late_updates += 1
                return False
            return True


def render(snapshot: Snapshot) -> bytes:
    """The text exposition (format 0.0.4) of a snapshot.

    A render is the largest object an exporter holds, so it is built in a
    buffer of exactly its size: the pieces are measured in one pass and
    copied in a second, rather than grown in place (which leaves up to an
    eighth spare) or copied into a ``bytes`` afterwards (which doubles it).
    """
    size = sum(len(piece) for piece in _pieces(snapshot))
    out = bytearray(size)
    view = memoryview(out)
    position = 0
    for piece in _pieces(snapshot):
        view[position : position + len(piece)] = piece
        position += len(piece)
    return out


def _pieces(snapshot: Snapshot) -> Iterator[bytes]:
    for family, prefixes_list, values in snapshot:
        spec = family.spec
        yield f"# HELP {spec.name} {_escape_help(spec.help)}\n".encode()
        yield f"# TYPE {spec.name} {spec.kind}\n".encode()
        if spec.kind == "histogram":
            stride = len(spec.buckets) + 3
            for index, prefixes in enumerate(prefixes_list):
                start = index * stride
                yield from _histogram_pieces(prefixes, values[start : start + stride])
        else:
            for prefixes, value in zip(prefixes_list, values):
                yield prefixes[0]
                yield format_value(value).encode() + b"\n"


def bounded_value(value: str) -> str:
    """A configuration value as a label value: long ones keep a digest."""
    if len(value) <= MAX_LABEL_VALUE:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]
    return f"{value[: MAX_LABEL_VALUE - _DIGEST_CHARS - 1]}~{digest}"


def format_value(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# ------------------------------------------------------------------ helpers
def _check_name(name: str, kind: Kind, unit: str) -> None:
    if not name.startswith(METRIC_PREFIX) or not _NAME.match(name):
        raise ValueError(f"metric names must be valid and start with stormlog_: {name}")
    stem = name
    if kind == "counter":
        if not name.endswith("_total"):
            raise ValueError(f"a counter's name must end in _total: {name}")
        stem = name[: -len("_total")]
    if unit and not stem.endswith(f"_{unit}"):
        raise ValueError(f"{name} must end in its unit, _{unit}")


def _check_labels(
    labels: tuple[str, ...], enums: Mapping[str, tuple[str, ...]], kind: Kind
) -> None:
    if len(set(labels)) != len(labels):
        raise ValueError("label names must be unique")
    for label in labels:
        if not _LABEL.match(label) or label.startswith("__"):
            raise ValueError(f"invalid label name {label!r}")
    if kind == "histogram" and "le" in labels:
        raise ValueError("a histogram cannot use the label le")
    _check_enums(labels, enums)


def _check_enums(labels: tuple[str, ...], enums: Mapping[str, tuple[str, ...]]) -> None:
    for label, values in enums.items():
        if label not in labels:
            raise ValueError(f"enum {label!r} is not a label of the family")
        if not values or len(set(values)) != len(values):
            raise ValueError(f"enum {label!r} needs distinct values")


def _check_buckets(buckets: tuple[float, ...], kind: Kind) -> None:
    if kind != "histogram":
        if buckets:
            raise ValueError("only a histogram has buckets")
        return
    if not buckets or any(not math.isfinite(bound) for bound in buckets):
        raise ValueError("a histogram needs finite bucket bounds")
    if any(later <= earlier for earlier, later in zip(buckets, buckets[1:])):
        raise ValueError("bucket bounds must increase")


def _sample_names(spec: FamilySpec) -> tuple[str, ...]:
    if spec.kind == "histogram":
        return tuple(f"{spec.name}{suffix}" for suffix in ("_bucket", "_sum", "_count"))
    return (spec.name,)


def _known_label_sets(
    spec: FamilySpec, known: Sequence[Mapping[str, str]]
) -> list[LabelValues]:
    combos: list[Mapping[str, str]] = list(known) if spec.bounded else [{}]
    enum_axes = [spec.enums[label] for label in spec.labels if label in spec.enums]
    sets: list[LabelValues] = []
    for combo in combos:
        _check_combo(spec, combo)
        for enum_values in product(*enum_axes):
            sets.append(_merge(spec, combo, enum_values))
    return list(dict.fromkeys(sets))


def _check_combo(spec: FamilySpec, combo: Mapping[str, str]) -> None:
    missing = [label for label in spec.bounded if label not in combo]
    if missing:
        raise ValueError(f"known values lack labels {missing}")
    if any(bounded_value(combo[label]) == OVERFLOW for label in spec.bounded):
        raise ValueError(f"{OVERFLOW} is reserved and cannot be configured")


def _merge(
    spec: FamilySpec, combo: Mapping[str, str], enum_values: Iterable[str]
) -> LabelValues:
    enums = iter(enum_values)
    return tuple(
        next(enums) if label in spec.enums else bounded_value(combo[label])
        for label in spec.labels
    )


def _prefixes(
    spec: FamilySpec, const_labels: Mapping[str, str], values: LabelValues
) -> tuple[bytes, ...]:
    pairs = list(const_labels.items()) + list(zip(spec.labels, values))
    if spec.kind != "histogram":
        return (_prefix(spec.name, pairs),)
    bounds = [repr(float(bound)) for bound in spec.buckets] + ["+Inf"]
    buckets = tuple(
        _prefix(f"{spec.name}_bucket", pairs + [("le", bound)]) for bound in bounds
    )
    return buckets + (
        _prefix(f"{spec.name}_sum", pairs),
        _prefix(f"{spec.name}_count", pairs),
    )


def _prefix(name: str, pairs: Sequence[tuple[str, str]]) -> bytes:
    if not pairs:
        return f"{name} ".encode()
    labels = ",".join(f'{key}="{escape_label_value(value)}"' for key, value in pairs)
    return f"{name}{{{labels}}} ".encode()


def _histogram_pieces(
    prefixes: tuple[bytes, ...], values: Sequence[float]
) -> Iterator[bytes]:
    """One histogram series: its bucket counts, then its sum and count."""
    cumulative = 0.0
    for prefix, bucket_count in zip(prefixes[:-2], values[:-2]):
        cumulative += bucket_count
        yield prefix
        yield format_value(cumulative).encode() + b"\n"
    yield prefixes[-2]
    yield format_value(values[-2]).encode() + b"\n"
    yield prefixes[-1]
    yield format_value(values[-1]).encode() + b"\n"


def _family_budget(family: Family, const_labels: Mapping[str, str]) -> tuple[int, int]:
    spec = family.spec
    per_series = len(spec.buckets) + 3 if spec.kind == "histogram" else 1
    header = len(f"# HELP {spec.name} {_escape_help(spec.help)}\n".encode())
    header += len(f"# TYPE {spec.name} {spec.kind}\n")
    known = list(family._series.values())
    size = header + sum(
        len(prefix) + VALUE_BYTES + 1 for series in known for prefix in series.prefixes
    )
    extra = _extra_series(family)
    worst = _worst_series_size(spec, const_labels)
    return (len(known) + extra) * per_series, size + extra * worst


def _extra_series(family: Family) -> int:
    """Series that may still appear: headroom, and overflow for counters and
    histograms, beyond those created already."""
    spec = family.spec
    # Known series not created yet, and headroom.
    headroom = max(0, family.cap - family._regular)
    if not spec.bounded:
        return headroom
    overflow = 0
    if spec.kind != "gauge":
        overflow = math.prod(len(values) for values in spec.enums.values())
        overflow -= len(family._overflow_keys)
    return headroom + max(0, overflow)


def _worst_series_size(spec: FamilySpec, const_labels: Mapping[str, str]) -> int:
    # A character takes at most 4 bytes of UTF-8, more than any escape (2).
    widest = "\U0001f600" * MAX_LABEL_VALUE
    worst_values = tuple(
        max(spec.enums[label], key=len) if label in spec.enums else widest
        for label in spec.labels
    )
    prefixes = _prefixes(spec, const_labels, worst_values)
    return sum(len(prefix) + VALUE_BYTES + 1 for prefix in prefixes)


def _escape_help(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\n", "\\n")
