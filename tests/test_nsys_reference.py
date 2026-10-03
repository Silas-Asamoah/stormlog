"""Nsight launch attribution tests for asynchronous CUDA work."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.native_probes.nsys_reference import measured_kernels


def _export(path: Path, *, duplicate: bool = False, missing: bool = False) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME
              (start INTEGER, end INTEGER, correlationId INTEGER, nameId INTEGER, returnValue INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL
              (start INTEGER, end INTEGER, correlationId INTEGER, shortName INTEGER);
            INSERT INTO StringIds VALUES (1, 'cudaLaunchKernel'), (2, 'work_kernel');
            INSERT INTO NVTX_EVENTS VALUES (100, 200, 'stormlog-native-probe-measured', NULL);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (110, 120, 7, 1, 0);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (210, 220, 7, 2);
            """
        )
        if duplicate:
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (221, 230, 7, 2)"
            )
        if missing:
            connection.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")


def test_host_launch_selects_async_kernel_after_nvtx_end(tmp_path: Path) -> None:
    path = tmp_path / "report.sqlite"
    _export(path)
    result = measured_kernels(path)
    assert result["status"] == "valid_export"
    assert result["measured_launches"] == 1
    assert result["kernel_counts_by_name"] == {"work_kernel": 1}
    assert result["kernels_executing_after_range_end"] == 1
    assert result["external_capture_complete"] is None


def test_duplicate_or_missing_correlation_is_partial(tmp_path: Path) -> None:
    for condition in ("duplicate", "missing"):
        path = tmp_path / f"{condition}.sqlite"
        _export(path, **{condition: True})
        assert measured_kernels(path)["status"] == "partial"


def test_missing_export_and_missing_range_are_unavailable(tmp_path: Path) -> None:
    path = tmp_path / "missing.sqlite"
    assert measured_kernels(path)["status"] == "unavailable"
    _export(path)
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM NVTX_EVENTS")
    assert measured_kernels(path)["status"] == "unavailable"
