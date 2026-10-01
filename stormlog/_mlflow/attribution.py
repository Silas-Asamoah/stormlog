"""Attribution-specific MLflow export helpers (mirror of the W&B exporter).

The HTML preview rendering and tensor-table extraction are backend-agnostic and
shared with the W&B exporter (see ``stormlog._wandb.attribution``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .._wandb.attribution import (
    attribution_files,
    build_attribution_preview_html,
    tensor_attribution_rows,
)
from .._wandb.core import read_json_if_exists
from ..cuda_native_debug import (
    DEBUG_METADATA_FILENAME,
    TENSOR_ATTRIBUTION_FILENAME,
    TRACE_HTML_ANNOTATED_FILENAME,
)


def log_attribution_outputs(
    mlflow: Any,
    *,
    root: Path,
    artifact_prefix: str,
    session_slug: str,
    allow_artifact_logging: bool = False,
) -> dict[str, Any]:
    summary_fields: dict[str, Any] = {}
    existing_files = attribution_files(root)

    if allow_artifact_logging and existing_files:
        for path in existing_files:
            mlflow.log_artifact(
                local_path=str(path),
                artifact_path=f"stormlog-attribution-{session_slug}",
            )

    html_path = root / TRACE_HTML_ANNOTATED_FILENAME
    if html_path.exists():
        preview_html = build_attribution_preview_html(root)
        mlflow.log_text(preview_html, artifact_file="stormlog_attribution_preview.html")
        summary_fields["stormlog_attribution_html_file"] = html_path.name

    tensor_rows = tensor_attribution_rows(root / TENSOR_ATTRIBUTION_FILENAME)
    if tensor_rows:
        _log_tensor_table(mlflow, tensor_rows, artifact_prefix)
        summary_fields["stormlog_tensor_attribution_rows"] = len(tensor_rows)

    metadata = read_json_if_exists(root / DEBUG_METADATA_FILENAME)
    if isinstance(metadata, Mapping):
        history_recorded = metadata.get("history_recorded")
        if isinstance(history_recorded, bool):
            summary_fields["stormlog_attribution_history_recorded"] = history_recorded

    return summary_fields


def _log_tensor_table(
    mlflow: Any, tensor_rows: list[list[Any]], artifact_prefix: str
) -> None:
    mlflow.log_table(
        data={
            "name": [str(row[0]) for row in tensor_rows[:200]],
            "storage_ptr": [str(row[1]) for row in tensor_rows[:200]],
            "tensor_count": [int(row[2]) for row in tensor_rows[:200]],
            "total_size_bytes": [int(row[3]) for row in tensor_rows[:200]],
            "shape": [str(row[4]) for row in tensor_rows[:200]],
            "dtype": [str(row[5]) for row in tensor_rows[:200]],
        },
        artifact_file=f"{artifact_prefix}/stormlog_tensor_attribution.json",
    )
