"""The closed vocabularies of infer.request records, named in one place.

Exporters pre-create one series per status, phase and token source, so a
value the profiler can write but these lists lack would be dropped there.
"""

import ast
from pathlib import Path

import stormlog.infer.profile as profile
import stormlog.infer.trace_capture as trace_capture
from stormlog.infer.events import REQUEST_PHASES, REQUEST_STATUSES
from stormlog.infer.export_metrics import TRACE_STOP_REASONS
from stormlog.infer.tokens import (
    TOKEN_SOURCES,
    EstimatedTokenCounter,
    TiktokenCounter,
    TransformersTokenCounter,
)


def _keyword_literals(name: str, call: str | None = None) -> set[str]:
    """Strings passed as ``name=`` in profile.py, to ``call`` if given.

    A literal, or a module-level string constant named instead of one.
    """
    tree = ast.parse(Path(profile.__file__).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (call is None or getattr(node.func, "id", None) == call)
    ]
    values = set()
    for node in calls:
        for keyword in node.keywords:
            if keyword.arg != name:
                continue
            value = keyword.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                values.add(value.value)
            elif isinstance(value, ast.Name):
                constant = getattr(profile, value.id, None)
                if isinstance(constant, str):
                    values.add(constant)
    return values


def _classified_statuses() -> set[str]:
    tree = ast.parse(Path(profile.__file__).read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "classify_failure"
    )
    return {
        node.value.elts[0].value
        for node in ast.walk(function)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Tuple)
        and isinstance(node.value.elts[0], ast.Constant)
    }


def test_every_status_the_profiler_writes_is_listed() -> None:
    written = _keyword_literals("status") | _classified_statuses()
    assert written and written <= set(REQUEST_STATUSES)


def test_every_phase_the_profiler_writes_is_listed() -> None:
    written = _keyword_literals("phase")
    assert written and written <= set(REQUEST_PHASES)


def test_every_token_source_is_listed() -> None:
    sources = {
        EstimatedTokenCounter.source,
        TiktokenCounter.source,
        TransformersTokenCounter.source,
    }
    assert sources | _keyword_literals("source", "TokenCount") <= set(TOKEN_SOURCES)
    assert {"server_usage", "unknown"} <= set(TOKEN_SOURCES)


def _trace_stop_reasons() -> set[str]:
    """The reasons trace_capture.py gives a window's stop: each string passed
    to ``_stop`` or ``_close``, assigned to a ``reason``, or to a window's
    ``stop_reason``."""
    tree = ast.parse(Path(trace_capture.__file__).read_text(encoding="utf-8"))
    reasons = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) in (
            "_stop",
            "_close",
        ):
            reasons |= {
                arg.value
                for arg in node.args[1:2]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            }
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                name = getattr(target, "id", None) or getattr(target, "attr", None)
                if name in ("reason", "stop_reason") and isinstance(
                    node.value.value, str
                ):
                    reasons.add(node.value.value)
    return reasons


def test_every_trace_stop_reason_is_listed() -> None:
    written = _trace_stop_reasons()
    assert {"phase_end", "time_bound", "start_unknown"} <= written
    assert written <= set(TRACE_STOP_REASONS)
