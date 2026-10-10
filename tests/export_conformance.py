"""A strict checker for the Prometheus text format (0.0.4) the exporters write.

Stormlog's own Prometheus parser is permissive by design (it reads what vLLM
writes), so it cannot vouch for what Stormlog writes. This checker enforces
what a consumer relies on, and then hands the text to ``prometheus_client``'s
parser as an independent reader.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

# An independent reader of the text format; test-only.
from prometheus_client.parser import (  # type: ignore[import-not-found, unused-ignore]
    text_string_to_metric_families,
)

_SAMPLE = re.compile(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})? (\S+)\Z")
_HISTOGRAM_SUFFIXES = ("_bucket", "_sum", "_count")
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


@dataclass
class Exposition:
    types: dict[str, str] = field(default_factory=dict)
    samples: dict[tuple[str, tuple[tuple[str, str], ...]], float] = field(
        default_factory=dict
    )

    def value(self, name: str, **labels: str) -> float:
        """The one sample of ``name`` whose labels include ``labels``."""
        found = self.matching(name, **labels)
        assert len(found) == 1, f"{len(found)} samples of {name} match {labels}"
        return found[0]

    def matching(self, name: str, **labels: str) -> list[float]:
        return [
            value
            for (sample_name, sample_labels), value in self.samples.items()
            if sample_name == name and _subset(labels, sample_labels)
        ]


def check_exposition(text: str) -> Exposition:
    """Parse ``text`` strictly; raise AssertionError on any violation."""
    assert text == "" or text.endswith("\n"), "the exposition must end with a newline"
    exposition = Exposition()
    helps: set[str] = set()
    for number, line in enumerate(text.split("\n")[:-1], 1):
        if line.startswith("# HELP "):
            name = line.split(" ", 3)[2]
            assert name not in helps, f"line {number}: second HELP for {name}"
            helps.add(name)
        elif line.startswith("# TYPE "):
            _, _, name, kind = line.split(" ")
            assert name not in exposition.types, f"line {number}: second TYPE"
            assert kind in {"counter", "gauge", "histogram"}, kind
            exposition.types[name] = kind
        else:
            _add_sample(exposition, line, number)
    _check_histograms(exposition)
    list(text_string_to_metric_families(text))
    return exposition


def _add_sample(exposition: Exposition, line: str, number: int) -> None:
    match = _SAMPLE.match(line)
    assert match, f"line {number}: not a sample: {line!r}"
    name, raw_labels, raw_value = match.groups()
    family = _family_of(name, exposition.types)
    assert family in exposition.types, f"line {number}: {name} before its TYPE"
    labels = tuple(_parse_labels(raw_labels or ""))
    key = (name, labels)
    assert key not in exposition.samples, f"line {number}: duplicate sample {key}"
    exposition.samples[key] = float(raw_value)


def _family_of(name: str, types: dict[str, str]) -> str:
    for suffix in _HISTOGRAM_SUFFIXES:
        base = name[: -len(suffix)]
        if name.endswith(suffix) and types.get(base) == "histogram":
            return base
    return name


def _parse_labels(raw: str) -> list[tuple[str, str]]:
    labels: list[tuple[str, str]] = []
    position = 0
    while position < len(raw):
        equals = raw.index("=", position)
        name = raw[position:equals]
        assert re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", name), name
        assert raw[equals + 1] == '"', raw
        value, position = _read_quoted(raw, equals + 2)
        labels.append((name, value))
        if position < len(raw):
            assert raw[position] == ",", raw
            position += 1
    names = [name for name, _ in labels]
    assert len(names) == len(set(names)), f"repeated label in {raw!r}"
    return labels


def _read_quoted(raw: str, position: int) -> tuple[str, int]:
    out: list[str] = []
    while True:
        char = raw[position]
        if char == '"':
            return "".join(out), position + 1
        if char == "\\":
            escaped = raw[position + 1]
            assert escaped in {"\\", '"', "n"}, f"bad escape \\{escaped}"
            out.append("\n" if escaped == "n" else escaped)
            position += 2
            continue
        assert char != "\n"
        out.append(char)
        position += 1


def _check_histograms(exposition: Exposition) -> None:
    buckets: dict[tuple[str, tuple], list[tuple[float, float]]] = defaultdict(list)
    for (name, labels), value in exposition.samples.items():
        if name.endswith("_bucket") and _family_of(name, exposition.types) != name:
            rest = tuple(item for item in labels if item[0] != "le")
            bound = dict(labels)["le"]
            buckets[(name[: -len("_bucket")], rest)].append(
                (math.inf if bound == "+Inf" else float(bound), value)
            )
    for (base, rest), series in buckets.items():
        bounds = [bound for bound, _ in series]
        assert bounds == sorted(bounds), f"{base}: le out of order"
        assert bounds[-1] == math.inf, f"{base}: no +Inf bucket"
        counts = [count for _, count in series]
        assert counts == sorted(counts), f"{base}: buckets not cumulative"
        assert exposition.samples[(f"{base}_count", rest)] == counts[-1]
        assert (f"{base}_sum", rest) in exposition.samples


def _subset(wanted: dict[str, str], labels: tuple[tuple[str, str], ...]) -> bool:
    have = dict(labels)
    return all(have.get(key) == value for key, value in wanted.items())
