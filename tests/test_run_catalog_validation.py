"""Boundary characterization for run-envelope parsing and catalog filters."""

from pathlib import Path
from typing import Any

import pytest

from stormlog._run_catalog_parser import run_envelope_from_payload


@pytest.mark.parametrize(
    "fields,valid",
    [
        ({}, True),
        ({"tags": []}, True),
        ({"tags": ["a", "b"]}, True),
        ({"tags": ["a", "a"]}, False),
        ({"tags": [{}]}, False),
        ({"tags": None}, False),
        ({"sessions": [None]}, False),
        ({"attachments": [None]}, False),
        ({"started_at_ns": True}, False),
        ({"started_at_ns": 2, "ended_at_ns": 1}, False),
        ({"started_at_ns": 2, "ended_at_ns": 2}, True),
        ({"title": None, "description": ""}, True),
        ({"title": ""}, False),
    ],
)
def test_envelope_validation_boundaries(fields: dict[str, Any], valid: bool) -> None:
    payload = {
        "schema_version": 1,
        "format": "stormlog.run_envelope",
        "run_id": "run-a",
        "metadata": {},
        **fields,
    }
    assert (run_envelope_from_payload(payload, Path("run.json")) is not None) is valid


@pytest.mark.parametrize(
    "fields,valid",
    [
        ({}, False),
        ({"path": "trace.json"}, True),
        ({"url": "https://example.test/trace"}, True),
        ({"path": "trace.json", "url": None}, False),
        ({"path": ""}, False),
        ({"path": "trace.json", "rank": True}, False),
        ({"path": "trace.json", "world_size": 0}, False),
        ({"path": "trace.json", "start_ns": 2, "end_ns": 1}, False),
        ({"path": "trace.json", "start_ns": 2, "end_ns": 2}, True),
        ({"path": "trace.json", "created_at_utc": ""}, True),
        ({"path": "trace.json", "run_id": ""}, False),
        ({"path": "trace.json", "unknown": 1}, False),
    ],
)
def test_attachment_validation_boundaries(fields: dict[str, Any], valid: bool) -> None:
    attachment = {
        "title": "Trace",
        "kind": "profiler_trace",
        "storage": "reference",
        "metadata": {},
        **fields,
    }
    payload = {
        "schema_version": 1,
        "format": "stormlog.run_envelope",
        "run_id": "run-a",
        "metadata": {},
        "attachments": [attachment],
    }
    assert (run_envelope_from_payload(payload, Path("run.json")) is not None) is valid
