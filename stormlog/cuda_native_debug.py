"""CUDA-native allocator history capture and attribution helpers."""

from __future__ import annotations

import gc
import json
import logging
import pickle
import warnings as _warnings
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Union, cast

try:
    import torch as _torch
except ModuleNotFoundError:
    _torch = cast(Any, None)

torch: Any = _torch

logger = logging.getLogger(__name__)

DEFAULT_TRACE_ALLOC_MAX_ENTRIES = 100_000
SNAPSHOT_PICKLE_FILENAME = "cuda_allocator_snapshot.pickle"
TRACE_HTML_FILENAME = "cuda_allocator_state_history.html"
TRACE_HTML_ANNOTATED_FILENAME = "cuda_allocator_state_history_annotated.html"
SEGMENT_SUMMARY_FILENAME = "cuda_allocator_segments.txt"
TRACE_SUMMARY_FILENAME = "cuda_allocator_trace.txt"
TENSOR_ATTRIBUTION_FILENAME = "cuda_tensor_attribution.json"
ALLOCATION_ATTRIBUTION_FILENAME = "cuda_allocation_attribution.json"
DEBUG_METADATA_FILENAME = "cuda_native_debug_metadata.json"


def cuda_memory_history_supported() -> bool:
    """Return whether the current PyTorch runtime exposes CUDA history APIs."""
    if torch is None:
        return False
    memory_api = getattr(getattr(torch, "cuda", None), "memory", None)
    return bool(
        torch.cuda.is_available()
        and memory_api is not None
        and hasattr(memory_api, "_record_memory_history")
        and hasattr(memory_api, "_snapshot")
    )


def _resolve_device_index(device: Optional[Union[int, torch.device]]) -> int:
    if isinstance(device, int):
        return device
    if torch is None:
        raise RuntimeError("PyTorch is required for native CUDA memory history.")
    if device is None:
        return int(torch.cuda.current_device())
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _require_cuda_history_support() -> None:
    if not cuda_memory_history_supported():
        raise RuntimeError(
            "Native CUDA memory history is unavailable in this PyTorch runtime."
        )


def start_cuda_memory_history(
    device: Optional[Union[int, torch.device]] = None,
    trace_alloc_max_entries: int = DEFAULT_TRACE_ALLOC_MAX_ENTRIES,
) -> None:
    """Enable CUDA allocator history recording for the selected device."""
    _require_cuda_history_support()
    torch.cuda.memory._record_memory_history(
        True,
        record_context=True,
        trace_alloc_max_entries=trace_alloc_max_entries,
        trace_alloc_record_context=True,
        device=device,
    )


def stop_cuda_memory_history(
    device: Optional[Union[int, torch.device]] = None,
) -> None:
    """Disable CUDA allocator history recording for the selected device."""
    _require_cuda_history_support()
    torch.cuda.memory._record_memory_history(False, device=device)


@contextmanager
def cuda_memory_history(
    device: Optional[Union[int, torch.device]] = None,
    trace_alloc_max_entries: int = DEFAULT_TRACE_ALLOC_MAX_ENTRIES,
) -> Iterator[None]:
    """Context manager that records CUDA allocator history for a block."""
    start_cuda_memory_history(
        device=device,
        trace_alloc_max_entries=trace_alloc_max_entries,
    )
    try:
        yield
    finally:
        stop_cuda_memory_history(device=device)


def _normalize_snapshot(snapshot: Any) -> dict[str, Any]:
    if isinstance(snapshot, list):
        return {"segments": snapshot, "device_traces": []}
    if isinstance(snapshot, dict):
        return {
            "segments": list(snapshot.get("segments", [])),
            "device_traces": list(snapshot.get("device_traces", [])),
        }
    raise TypeError(f"Unsupported snapshot payload type: {type(snapshot).__name__}")


def _safe_storage_ptr(tensor: torch.Tensor) -> Optional[int]:
    try:
        if hasattr(tensor, "untyped_storage"):
            return int(tensor.untyped_storage().data_ptr())
        return int(tensor.storage().data_ptr())
    except Exception:
        try:
            return int(tensor.data_ptr())
        except Exception:
            return None


def _tensor_size_bytes(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.nelement() * tensor.element_size())
    except Exception:
        return 0


def _index_named_cuda_tensors(
    pointer_to_names: dict[int, set[str]],
    tensors: Iterable[tuple[str, torch.Tensor]],
    device_index: int,
) -> None:
    for name, tensor in tensors:
        if not tensor.is_cuda or tensor.device.index != device_index:
            continue
        storage_ptr = _safe_storage_ptr(tensor)
        if storage_ptr is not None:
            pointer_to_names[storage_ptr].add(name)


def _index_python_cuda_names(
    pointer_to_names: dict[int, set[str]],
    namespace: dict[Any, Any],
    device_index: int,
) -> None:
    for key, value in namespace.items():
        if not isinstance(key, str) or key.startswith("__"):
            continue
        if not isinstance(value, torch.Tensor) or not value.is_cuda:
            continue
        if value.device.index != device_index:
            continue
        storage_ptr = _safe_storage_ptr(value)
        if storage_ptr is not None:
            pointer_to_names[storage_ptr].add(key)


def _collect_module_name_index(device_index: int) -> dict[int, set[str]]:
    pointer_to_names: dict[int, set[str]] = defaultdict(set)
    pointer_to_python_names: dict[int, set[str]] = defaultdict(set)

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=FutureWarning)
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.nn.Module):
                    _index_named_cuda_tensors(
                        pointer_to_names,
                        obj.named_parameters(recurse=True),
                        device_index,
                    )
                    _index_named_cuda_tensors(
                        pointer_to_names, obj.named_buffers(recurse=True), device_index
                    )
                elif isinstance(obj, dict):
                    _index_python_cuda_names(pointer_to_python_names, obj, device_index)
            except Exception:
                continue

    for storage_ptr, names in pointer_to_python_names.items():
        if storage_ptr not in pointer_to_names:
            pointer_to_names[storage_ptr].update(names)

    return pointer_to_names


def build_cuda_tensor_attribution_index(
    device: Optional[Union[int, torch.device]] = None,
    *,
    skip_gc: bool = False,
) -> dict[str, Any]:
    """Build a best-effort index from CUDA storage pointers to live tensors."""
    _require_cuda_history_support()
    device_index = _resolve_device_index(device)

    if not skip_gc:
        gc.collect()
    pointer_to_names = _collect_module_name_index(device_index)
    pointer_to_tensors: dict[int, list[dict[str, Any]]] = defaultdict(list)

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=FutureWarning)
        for obj in gc.get_objects():
            try:
                if not isinstance(obj, torch.Tensor) or not obj.is_cuda:
                    continue
                if obj.device.index != device_index:
                    continue
            except Exception:
                continue

            storage_ptr = _safe_storage_ptr(obj)
            if storage_ptr is None:
                continue

            pointer_to_tensors[storage_ptr].append(
                {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "device": str(obj.device),
                    "size_bytes": _tensor_size_bytes(obj),
                    "requires_grad": bool(getattr(obj, "requires_grad", False)),
                    "is_leaf": bool(getattr(obj, "is_leaf", False)),
                }
            )

    attributed_pointers: list[dict[str, Any]] = []
    for storage_ptr in sorted(pointer_to_tensors):
        tensor_entries = pointer_to_tensors[storage_ptr]
        names = sorted(
            pointer_to_names.get(storage_ptr, set()),
            key=lambda value: (value.count("."), len(value)),
            reverse=True,
        )
        attributed_pointers.append(
            {
                "storage_ptr": hex(storage_ptr),
                "storage_ptr_int": storage_ptr,
                "names": names,
                "tensor_count": len(tensor_entries),
                "tensors": tensor_entries,
            }
        )

    return {
        "device_index": device_index,
        "storage_pointer_count": len(attributed_pointers),
        "attributed_storage_pointers": attributed_pointers,
    }


def build_snapshot_allocation_attribution(
    snapshot: Any,
    tensor_index: dict[str, Any],
) -> dict[str, Any]:
    """Cross-reference allocator addresses against live tensor storage pointers."""
    snapshot_dict = _normalize_snapshot(snapshot)
    pointer_map = {
        int(entry["storage_ptr_int"]): entry
        for entry in tensor_index.get("attributed_storage_pointers", [])
    }

    allocations: dict[int, dict[str, Any]] = {}

    for segment in snapshot_dict.get("segments", []):
        segment_address = int(segment.get("address", 0) or 0)
        for block in segment.get("blocks", []):
            candidate_allocations = _block_allocations(block)

            for addr, size_bytes in candidate_allocations:
                tensor_entry = pointer_map.get(addr)
                if tensor_entry is None:
                    continue
                allocations[addr] = {
                    "storage_ptr": hex(addr),
                    "storage_ptr_int": addr,
                    "size_bytes": size_bytes,
                    "segment_address": hex(segment_address),
                    "stream": segment.get("stream"),
                    "names": list(tensor_entry.get("names", [])),
                    "tensors": list(tensor_entry.get("tensors", [])),
                }

    return {
        "device_index": tensor_index.get("device_index"),
        "storage_pointer_count": tensor_index.get("storage_pointer_count", 0),
        "attributed_allocation_count": len(allocations),
        "attributed_allocations": [
            allocations[address] for address in sorted(allocations)
        ],
    }


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _load_memory_viz() -> Any:
    from torch.cuda import _memory_viz as memory_viz

    return memory_viz


def write_cuda_snapshot_artifacts(
    output_dir: Path,
    snapshot: Any,
    tensor_index: dict[str, Any],
    *,
    history_recorded: bool,
    device: Optional[Union[int, torch.device]] = None,
) -> list[str]:
    """Write snapshot, attribution, and best-effort visualization artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dict = _normalize_snapshot(snapshot)
    files_written: list[str] = []
    warnings: list[str] = []

    snapshot_path = output_dir / SNAPSHOT_PICKLE_FILENAME
    with snapshot_path.open("wb") as handle:
        pickle.dump(snapshot_dict, handle)
    files_written.append(SNAPSHOT_PICKLE_FILENAME)

    attribution_summary = build_snapshot_allocation_attribution(
        snapshot_dict,
        tensor_index,
    )

    tensor_index_path = output_dir / TENSOR_ATTRIBUTION_FILENAME
    tensor_index_path.write_text(json.dumps(tensor_index, indent=2), encoding="utf-8")
    files_written.append(TENSOR_ATTRIBUTION_FILENAME)

    allocation_path = output_dir / ALLOCATION_ATTRIBUTION_FILENAME
    allocation_path.write_text(
        json.dumps(attribution_summary, indent=2),
        encoding="utf-8",
    )
    files_written.append(ALLOCATION_ATTRIBUTION_FILENAME)

    try:
        memory_viz = _load_memory_viz()
    except Exception as exc:
        warnings.append(f"memory_viz load: {exc}")
    else:
        try:
            _write_text(
                output_dir / SEGMENT_SUMMARY_FILENAME,
                memory_viz.segsum(snapshot_dict),
            )
            files_written.append(SEGMENT_SUMMARY_FILENAME)
        except Exception as exc:
            warnings.append(f"segment summary: {exc}")

        try:
            _write_text(
                output_dir / TRACE_SUMMARY_FILENAME, memory_viz.trace(snapshot_dict)
            )
            files_written.append(TRACE_SUMMARY_FILENAME)
        except Exception as exc:
            warnings.append(f"trace summary: {exc}")

        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", category=FutureWarning)
                trace_html = memory_viz.trace_plot(
                    snapshot_dict,
                    device=_resolve_device_index(device),
                )
            _write_text(output_dir / TRACE_HTML_FILENAME, trace_html)
            files_written.append(TRACE_HTML_FILENAME)
        except Exception as exc:
            warnings.append(str(exc))

    try:
        from .attributed_viz import render_attributed_html

        attributed_html = render_attributed_html(
            snapshot_dict,
            tensor_index,
            device=_resolve_device_index(device),
        )
        _write_text(output_dir / TRACE_HTML_ANNOTATED_FILENAME, attributed_html)
        files_written.append(TRACE_HTML_ANNOTATED_FILENAME)
    except Exception as attr_exc:
        warnings.append(f"attributed HTML: {attr_exc}")

    metadata_path = output_dir / DEBUG_METADATA_FILENAME
    metadata_path.write_text(
        json.dumps(
            {
                "history_recorded": history_recorded,
                "warning_count": len(warnings),
                "warnings": warnings,
                "trace_html_written": TRACE_HTML_FILENAME in files_written,
                "annotated_trace_html_written": TRACE_HTML_ANNOTATED_FILENAME
                in files_written,
                "storage_pointer_count": tensor_index.get("storage_pointer_count", 0),
                "attributed_allocation_count": attribution_summary.get(
                    "attributed_allocation_count",
                    0,
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    files_written.append(DEBUG_METADATA_FILENAME)

    return files_written


def capture_cuda_snapshot_artifacts(
    output_dir: Path,
    *,
    device: Optional[Union[int, torch.device]] = None,
    history_recorded: bool,
) -> list[str]:
    """Capture the current CUDA allocator snapshot and write debug artifacts."""
    _require_cuda_history_support()
    gc.collect()
    snapshot = torch.cuda.memory._snapshot(device=device)
    tensor_index = build_cuda_tensor_attribution_index(device=device, skip_gc=True)
    return write_cuda_snapshot_artifacts(
        output_dir,
        snapshot,
        tensor_index,
        history_recorded=history_recorded,
        device=device,
    )


def _block_allocations(block: dict[str, Any]) -> list[tuple[int, int]]:
    history_entries = list(block.get("history", []))
    if history_entries:
        candidate_allocations = [
            (
                int(history_entry.get("addr", 0) or 0),
                int(history_entry.get("real_size", 0) or 0),
            )
            for history_entry in history_entries
        ]
    else:
        candidate_allocations = [
            (
                int(block.get("address", 0) or 0),
                int(block.get("size", 0) or 0),
            )
        ]

    return candidate_allocations


__all__ = [
    "ALLOCATION_ATTRIBUTION_FILENAME",
    "DEBUG_METADATA_FILENAME",
    "DEFAULT_TRACE_ALLOC_MAX_ENTRIES",
    "SEGMENT_SUMMARY_FILENAME",
    "SNAPSHOT_PICKLE_FILENAME",
    "TENSOR_ATTRIBUTION_FILENAME",
    "TRACE_HTML_ANNOTATED_FILENAME",
    "TRACE_HTML_FILENAME",
    "TRACE_SUMMARY_FILENAME",
    "build_cuda_tensor_attribution_index",
    "build_snapshot_allocation_attribution",
    "capture_cuda_snapshot_artifacts",
    "cuda_memory_history",
    "cuda_memory_history_supported",
    "start_cuda_memory_history",
    "stop_cuda_memory_history",
    "write_cuda_snapshot_artifacts",
]
