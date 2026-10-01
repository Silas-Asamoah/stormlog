from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import stormlog.cuda_native_debug as native_debug


def test_module_name_index_prioritizes_modules_and_tolerates_broken_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tensor:
        def __init__(self, pointer: int | None, device: int = 0, cuda: bool = True):
            self.pointer = pointer
            self.is_cuda = cuda
            self.device = SimpleNamespace(index=device)

    class Module:
        def named_parameters(self, *, recurse: bool) -> list[tuple[str, Tensor]]:
            assert recurse is True
            return [
                ("layer.weight", Tensor(100)),
                ("cpu.weight", Tensor(200, cuda=False)),
                ("other.weight", Tensor(300, device=1)),
                ("missing.weight", Tensor(None)),
            ]

        def named_buffers(self, *, recurse: bool) -> list[tuple[str, Tensor]]:
            assert recurse is True
            return [("layer.running_mean", Tensor(400))]

    class BrokenModule(Module):
        def named_buffers(self, *, recurse: bool) -> list[tuple[str, Tensor]]:
            raise RuntimeError("partially constructed module")

    objects = [
        {
            "python_weight": Tensor(100),
            "python_buffer": Tensor(400),
            "python_only": Tensor(500),
            "__hidden": Tensor(600),
            7: Tensor(700),
            "cpu": Tensor(800, cuda=False),
            "other": Tensor(900, device=1),
            "missing": Tensor(None),
            "not_tensor": object(),
        },
        BrokenModule(),
        Module(),
    ]
    monkeypatch.setattr(
        native_debug,
        "torch",
        SimpleNamespace(nn=SimpleNamespace(Module=Module), Tensor=Tensor),
    )
    monkeypatch.setattr(native_debug.gc, "get_objects", lambda: objects)
    monkeypatch.setattr(
        native_debug, "_safe_storage_ptr", lambda tensor: tensor.pointer
    )

    assert native_debug._collect_module_name_index(0) == {
        100: {"layer.weight"},
        400: {"layer.running_mean"},
        500: {"python_only"},
    }


def test_start_and_stop_cuda_memory_history_use_expected_torch_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _record_memory_history(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            memory=SimpleNamespace(
                _record_memory_history=_record_memory_history,
                _snapshot=lambda device=None: {},
            ),
        )
    )

    monkeypatch.setattr(native_debug, "torch", fake_torch)

    native_debug.start_cuda_memory_history(device=0, trace_alloc_max_entries=123)
    native_debug.stop_cuda_memory_history(device=0)

    assert calls[0][0] == (True,)
    assert calls[0][1]["trace_alloc_max_entries"] == 123
    assert calls[0][1]["trace_alloc_record_context"] is True
    assert calls[0][1]["record_context"] is True
    assert calls[0][1]["device"] == 0
    assert calls[1][0] == (False,)
    assert calls[1][1]["device"] == 0


def test_build_snapshot_allocation_attribution_matches_storage_pointers() -> None:
    snapshot = {
        "segments": [
            {
                "address": 4096,
                "stream": 7,
                "blocks": [
                    {
                        "size": 64,
                        "history": [
                            {
                                "addr": 8192,
                                "real_size": 32,
                            }
                        ],
                    }
                ],
            }
        ],
        "device_traces": [],
    }
    tensor_index = {
        "device_index": 0,
        "storage_pointer_count": 1,
        "attributed_storage_pointers": [
            {
                "storage_ptr": hex(8192),
                "storage_ptr_int": 8192,
                "names": ["model.linear.weight"],
                "tensors": [{"shape": [8, 8], "dtype": "torch.float32"}],
            }
        ],
    }

    summary = native_debug.build_snapshot_allocation_attribution(snapshot, tensor_index)

    assert summary["attributed_allocation_count"] == 1
    assert summary["attributed_allocations"][0]["storage_ptr"] == hex(8192)
    assert summary["attributed_allocations"][0]["names"] == ["model.linear.weight"]
    assert summary["attributed_allocations"][0]["segment_address"] == hex(4096)


def test_build_snapshot_allocation_attribution_uses_block_fallback_without_history() -> (
    None
):
    snapshot = {
        "segments": [
            {
                "address": 4096,
                "stream": 7,
                "blocks": [
                    {
                        "address": 12288,
                        "size": 64,
                        "history": [],
                    }
                ],
            }
        ],
        "device_traces": [],
    }
    tensor_index = {
        "device_index": 0,
        "storage_pointer_count": 1,
        "attributed_storage_pointers": [
            {
                "storage_ptr": hex(12288),
                "storage_ptr_int": 12288,
                "names": ["model.linear.bias"],
                "tensors": [{"shape": [8], "dtype": "torch.float32"}],
            }
        ],
    }

    summary = native_debug.build_snapshot_allocation_attribution(snapshot, tensor_index)

    assert summary["attributed_allocation_count"] == 1
    assert summary["attributed_allocations"][0]["storage_ptr"] == hex(12288)
    assert summary["attributed_allocations"][0]["size_bytes"] == 64
    assert summary["attributed_allocations"][0]["names"] == ["model.linear.bias"]


def test_write_cuda_snapshot_artifacts_writes_expected_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = {
        "segments": [
            {
                "address": 4096,
                "segment_type": "large",
                "total_size": 128,
                "allocated_size": 64,
                "active_size": 64,
                "blocks": [
                    {
                        "address": 8192,
                        "size": 64,
                        "state": "active_allocated",
                        "frames": [
                            {"name": "forward", "filename": "linear.py", "line": 12}
                        ],
                    }
                ],
            }
        ],
        "device_traces": [
            [
                {
                    "action": "alloc",
                    "addr": 8192,
                    "size": 64,
                    "time_us": 100,
                    "frames": [
                        {"name": "forward", "filename": "linear.py", "line": 12}
                    ],
                }
            ]
        ],
    }
    tensor_index = {
        "device_index": 0,
        "storage_pointer_count": 1,
        "attributed_storage_pointers": [
            {
                "storage_ptr": hex(8192),
                "storage_ptr_int": 8192,
                "names": ["model.linear.weight"],
                "tensors": [{"shape": [8, 8], "dtype": "torch.float32"}],
            }
        ],
    }

    fake_memory_viz = SimpleNamespace(
        segsum=lambda data: "segment-summary",
        trace=lambda data: "trace-summary",
        trace_plot=lambda data, device=None: "<html>allocator-state-history</html>",
    )
    monkeypatch.setattr(native_debug, "_load_memory_viz", lambda: fake_memory_viz)

    files_written = native_debug.write_cuda_snapshot_artifacts(
        tmp_path,
        snapshot,
        tensor_index,
        history_recorded=True,
        device=0,
    )

    assert native_debug.SNAPSHOT_PICKLE_FILENAME in files_written
    assert native_debug.TENSOR_ATTRIBUTION_FILENAME in files_written
    assert native_debug.ALLOCATION_ATTRIBUTION_FILENAME in files_written
    assert native_debug.SEGMENT_SUMMARY_FILENAME in files_written
    assert native_debug.TRACE_SUMMARY_FILENAME in files_written
    assert native_debug.TRACE_HTML_FILENAME in files_written
    assert native_debug.TRACE_HTML_ANNOTATED_FILENAME in files_written
    assert native_debug.DEBUG_METADATA_FILENAME in files_written
    assert (tmp_path / native_debug.TRACE_HTML_FILENAME).read_text(
        encoding="utf-8"
    ) == "<html>allocator-state-history</html>"
    metadata = (tmp_path / native_debug.DEBUG_METADATA_FILENAME).read_text(
        encoding="utf-8"
    )
    annotated_html = (tmp_path / native_debug.TRACE_HTML_ANNOTATED_FILENAME).read_text(
        encoding="utf-8"
    )
    assert (tmp_path / native_debug.TRACE_HTML_ANNOTATED_FILENAME).exists()
    assert '"annotated_trace_html_written": true' in metadata
    assert "Timeline Trace" in annotated_html
    assert "Segment Explorer" in annotated_html
    assert "Active Memory Table" in annotated_html
    assert "model.linear.weight" in annotated_html


def test_capture_cuda_snapshot_artifacts_collects_heap_once_before_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(native_debug, "_require_cuda_history_support", lambda: None)
    monkeypatch.setattr(
        native_debug.gc,
        "collect",
        lambda: calls.append(("gc.collect", None)),
    )

    def _snapshot(device: object = None) -> dict[str, object]:
        calls.append(("snapshot", device))
        return {"segments": [], "device_traces": []}

    monkeypatch.setattr(
        native_debug,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                memory=SimpleNamespace(
                    _snapshot=_snapshot,
                )
            )
        ),
    )

    def _fake_build_tensor_index(
        device: object = None,
        *,
        skip_gc: bool = False,
    ) -> dict[str, object]:
        calls.append(("tensor_index", (device, skip_gc)))
        return {
            "device_index": 0,
            "storage_pointer_count": 0,
            "attributed_storage_pointers": [],
        }

    monkeypatch.setattr(
        native_debug,
        "build_cuda_tensor_attribution_index",
        _fake_build_tensor_index,
    )
    monkeypatch.setattr(
        native_debug,
        "write_cuda_snapshot_artifacts",
        lambda *args, **kwargs: ["cuda_allocator_snapshot.pickle"],
    )

    files_written = native_debug.capture_cuda_snapshot_artifacts(
        tmp_path,
        device=0,
        history_recorded=True,
    )

    assert files_written == ["cuda_allocator_snapshot.pickle"]
    assert calls == [
        ("gc.collect", None),
        ("snapshot", 0),
        ("tensor_index", (0, True)),
    ]


def test_write_cuda_snapshot_artifacts_writes_annotated_html_when_trace_plot_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot: dict[str, Any] = {
        "segments": [],
        "device_traces": [[]],
    }
    tensor_index = {
        "device_index": 0,
        "storage_pointer_count": 0,
        "attributed_storage_pointers": [],
    }

    fake_memory_viz = SimpleNamespace(
        segsum=lambda data: "segment-summary",
        trace=lambda data: "trace-summary",
        trace_plot=lambda data, device=None: (_ for _ in ()).throw(
            RuntimeError("trace_plot boom")
        ),
    )
    monkeypatch.setattr(native_debug, "_load_memory_viz", lambda: fake_memory_viz)

    files_written = native_debug.write_cuda_snapshot_artifacts(
        tmp_path,
        snapshot,
        tensor_index,
        history_recorded=True,
        device=0,
    )

    assert native_debug.TRACE_HTML_FILENAME not in files_written
    assert native_debug.TRACE_HTML_ANNOTATED_FILENAME in files_written
    metadata = (tmp_path / native_debug.DEBUG_METADATA_FILENAME).read_text(
        encoding="utf-8"
    )
    assert "trace_plot boom" in metadata
    assert '"annotated_trace_html_written": true' in metadata
