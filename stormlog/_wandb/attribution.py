"""Attribution-specific W&B export helpers."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

from ..attributed_viz import render_attributed_wandb_preview_html
from ..cuda_native_debug import (
    ALLOCATION_ATTRIBUTION_FILENAME,
    DEBUG_METADATA_FILENAME,
    SNAPSHOT_PICKLE_FILENAME,
    TENSOR_ATTRIBUTION_FILENAME,
    TRACE_HTML_ANNOTATED_FILENAME,
    TRACE_HTML_FILENAME,
)
from .core import read_json_if_exists


def log_attribution_outputs(
    wandb: Any,
    run: Any,
    *,
    root: Path,
    session_slug: str,
    allow_artifact_logging: bool = False,
) -> dict[str, Any]:
    summary_fields: dict[str, Any] = {}
    existing_files = attribution_files(root)

    if allow_artifact_logging and existing_files:
        artifact = wandb.Artifact(
            name=f"stormlog-attribution-{session_slug}",
            type="stormlog-attribution",
        )
        for path in existing_files:
            artifact.add_file(local_path=str(path), name=path.name)
        run.log_artifact(artifact)

    html_path = root / TRACE_HTML_ANNOTATED_FILENAME
    if html_path.exists():
        preview_html = build_attribution_preview_html(root)
        run.log({"stormlog_attribution_html": wandb.Html(preview_html)})
        summary_fields["stormlog_attribution_html_file"] = html_path.name

    tensor_rows = tensor_attribution_rows(root / TENSOR_ATTRIBUTION_FILENAME)
    if tensor_rows:
        run.log(
            {
                "stormlog_tensor_attribution": wandb.Table(
                    columns=[
                        "name",
                        "storage_ptr",
                        "tensor_count",
                        "total_size_bytes",
                        "shape",
                        "dtype",
                    ],
                    data=tensor_rows[:200],
                )
            }
        )
        summary_fields["stormlog_tensor_attribution_rows"] = len(tensor_rows)

    metadata = read_json_if_exists(root / DEBUG_METADATA_FILENAME)
    if isinstance(metadata, Mapping):
        history_recorded = metadata.get("history_recorded")
        if isinstance(history_recorded, bool):
            summary_fields["stormlog_attribution_history_recorded"] = history_recorded

    return summary_fields


def build_attribution_preview_html(root: Path) -> str:
    snapshot_path = root / SNAPSHOT_PICKLE_FILENAME
    tensor_index = read_json_if_exists(root / TENSOR_ATTRIBUTION_FILENAME)
    if snapshot_path.exists() and isinstance(tensor_index, Mapping):
        try:
            with snapshot_path.open("rb") as handle:
                snapshot = pickle.load(handle)
        except (
            OSError,
            EOFError,
            pickle.UnpicklingError,
            ValueError,
            TypeError,
            AttributeError,
        ):
            pass
        else:
            return render_attributed_wandb_preview_html(
                snapshot,
                dict(tensor_index),
            )
    return render_compact_attribution_summary_html(
        tensor_attribution_rows(root / TENSOR_ATTRIBUTION_FILENAME),
        read_json_if_exists(root / DEBUG_METADATA_FILENAME),
    )


def render_compact_attribution_summary_html(
    tensor_rows: list[list[Any]],
    metadata: Any,
) -> str:
    history_recorded = (
        metadata.get("history_recorded") if isinstance(metadata, Mapping) else None
    )
    storage_pointer_count = (
        metadata.get("storage_pointer_count") if isinstance(metadata, Mapping) else None
    )
    offenders = tensor_rows[:12]
    offender_rows = "".join(
        (
            "<tr>"
            f"<td>{_escape_html(str(row[0]))}</td>"
            f"<td>{_escape_html(str(row[3]))}</td>"
            f"<td>{_escape_html(str(row[4]) or '—')}</td>"
            f"<td>{_escape_html(str(row[5]) or '—')}</td>"
            "</tr>"
        )
        for row in offenders
    )
    note = (
        "Snapshot data was unavailable for this inline preview. "
        "Download the attribution artifact for the full interactive explorer."
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stormlog GPU Attribution Preview</title>
<style>
body {{ margin: 0; padding: 20px; background: #0d1117; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
.panel {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 16px; }}
.stats {{ display: flex; gap: 12px; margin: 16px 0; }}
.stat {{ background: #0d1117; border: 1px solid #30363d; border-radius: 10px; padding: 10px 12px; min-width: 110px; }}
.stat-value {{ color: #58a6ff; font-family: ui-monospace, SFMono-Regular, monospace; font-weight: 700; font-size: 18px; }}
.stat-label {{ color: #8b949e; font-size: 11px; margin-top: 4px; text-transform: uppercase; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th, td {{ text-align: left; padding: 8px 0; border-top: 1px solid #21262d; }}
th {{ border-top: none; color: #8b949e; }}
</style>
</head>
<body>
  <div class="panel">
    <h1 style="margin:0;font-size:20px">Stormlog GPU Attribution Preview</h1>
    <p style="margin:8px 0 0;color:#8b949e">{note}</p>
    <div class="stats">
      <div class="stat"><div class="stat-value">{_escape_html(str(storage_pointer_count if storage_pointer_count is not None else len(tensor_rows)))}</div><div class="stat-label">Tensors</div></div>
      <div class="stat"><div class="stat-value">{_escape_html(str(history_recorded if history_recorded is not None else 'unknown'))}</div><div class="stat-label">History</div></div>
      <div class="stat"><div class="stat-value">{len(offenders)}</div><div class="stat-label">Shown</div></div>
    </div>
    <table>
      <thead><tr><th>Name</th><th>Total Size (bytes)</th><th>Shape</th><th>Dtype</th></tr></thead>
      <tbody>{offender_rows}</tbody>
    </table>
  </div>
</body>
</html>"""


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def tensor_attribution_rows(path: Path) -> list[list[Any]]:
    payload = read_json_if_exists(path)
    if not isinstance(payload, Mapping):
        return []
    entries = payload.get("attributed_storage_pointers")
    if not isinstance(entries, list):
        return []

    rows: list[list[Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        tensors, total_size, shape, dtype = _tensor_details(entry)

        names = entry.get("names")
        name = "<unnamed>"
        if isinstance(names, list) and names:
            name = ", ".join(str(value) for value in names[:3])

        rows.append(
            [
                name,
                str(entry.get("storage_ptr", "")),
                int(entry.get("tensor_count", len(tensors))),
                total_size,
                shape,
                dtype,
            ]
        )
    rows.sort(key=lambda row: int(row[3]), reverse=True)
    return rows


def attribution_files(root: Path) -> list[Path]:
    """Return existing attribution artifacts in export order."""
    files_to_attach = [
        root / TRACE_HTML_ANNOTATED_FILENAME,
        root / TRACE_HTML_FILENAME,
        root / TENSOR_ATTRIBUTION_FILENAME,
        root / ALLOCATION_ATTRIBUTION_FILENAME,
        root / DEBUG_METADATA_FILENAME,
    ]
    return [path for path in files_to_attach if path.exists() and path.is_file()]


def _tensor_details(
    entry: Mapping[str, Any],
) -> tuple[list[Any], int, str, str]:
    tensors = entry.get("tensors")
    if not isinstance(tensors, list):
        tensors = []
    total_size = 0
    shape = ""
    dtype = ""
    if tensors:
        first_tensor = tensors[0] if isinstance(tensors[0], Mapping) else {}
        shape = str(first_tensor.get("shape", ""))
        dtype = str(first_tensor.get("dtype", ""))
        for tensor in tensors:
            if isinstance(tensor, Mapping):
                size_bytes = tensor.get("size_bytes", 0)
                if isinstance(size_bytes, int):
                    total_size += size_bytes

    return tensors, total_size, shape, dtype
