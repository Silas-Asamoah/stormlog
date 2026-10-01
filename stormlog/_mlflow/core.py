"""Core helpers shared by Stormlog's optional MLflow exporter.

The W&B exporter modules host a number of backend-agnostic helpers (session
summary fields, path coercion, JSON reads).  The MLflow exporter reuses those
pure helpers so both integrations stay in sync; this module only implements the
MLflow-specific surface: config, CLI flags, import guard, run lifecycle, and
summary/metric/tag writing.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .._wandb.core import default_group, session_summary_fields
from ..session import SessionSummary

MLFLOW_INSTALL_GUIDANCE = (
    "MLflow integration requires optional dependencies. "
    "Install with `pip install 'stormlog[mlflow]'`."
)

_IDENTITY_TAG_KEYS = frozenset(
    {"stormlog_rank", "stormlog_local_rank", "stormlog_world_size"}
)


@dataclass(frozen=True)
class MlflowExportConfig:
    """Runtime configuration for optional MLflow exports."""

    enabled: bool = False
    tracking_uri: str | None = None
    experiment: str | None = None
    run_id: str | None = None
    run_name: str | None = None
    group: str | None = None
    job_type: str | None = None
    log_tables: bool = True
    log_artifacts: bool = False
    log_attribution: bool = False


def mlflow_config_from_namespace(args: Any) -> MlflowExportConfig:
    """Build an MLflow export config from CLI args or a similar namespace."""
    return MlflowExportConfig(
        enabled=bool(getattr(args, "mlflow", False)),
        tracking_uri=_normalized_optional_string(
            getattr(args, "mlflow_tracking_uri", None)
        ),
        experiment=_normalized_optional_string(
            getattr(args, "mlflow_experiment", None)
        ),
        run_id=_normalized_optional_string(getattr(args, "mlflow_run_id", None)),
        run_name=_normalized_optional_string(getattr(args, "mlflow_run_name", None)),
        group=_normalized_optional_string(getattr(args, "mlflow_group", None)),
        job_type=_normalized_optional_string(getattr(args, "mlflow_job_type", None)),
        log_artifacts=bool(getattr(args, "mlflow_log_artifacts", False)),
        log_attribution=bool(getattr(args, "mlflow_log_attribution", False)),
    )


def add_mlflow_arguments(parser: Any) -> None:
    """Attach shared optional MLflow flags to a CLI parser."""
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log Stormlog summaries to MLflow",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking server URI or local file path (default: MLFLOW_TRACKING_URI env)",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name (default: stormlog)",
    )
    parser.add_argument(
        "--mlflow-run-id",
        type=str,
        default=None,
        help="Existing MLflow run id to resume or attach to",
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="Explicit MLflow run name",
    )
    parser.add_argument(
        "--mlflow-group",
        type=str,
        default=None,
        help="MLflow group tag override (default: Stormlog job id)",
    )
    parser.add_argument(
        "--mlflow-job-type",
        type=str,
        default=None,
        help="MLflow job type tag override (default: Stormlog command name)",
    )
    parser.add_argument(
        "--mlflow-log-artifacts",
        action="store_true",
        help="Upload Stormlog output bundles as MLflow artifacts",
    )
    parser.add_argument(
        "--mlflow-log-attribution",
        action="store_true",
        help="Log attribution HTML and top offenders to MLflow when available",
    )


def ensure_mlflow_available(config: MlflowExportConfig) -> None:
    """Fail fast when the MLflow feature is enabled without dependencies installed."""
    if config.enabled:
        import_mlflow()


def import_mlflow() -> Any:
    try:
        return import_module("mlflow")
    except ModuleNotFoundError as exc:
        if exc.name == "mlflow":
            raise ImportError(MLFLOW_INSTALL_GUIDANCE) from exc
        raise


def resolve_run(
    config: MlflowExportConfig,
    *,
    command_name: str,
    session_summary: SessionSummary | None,
) -> tuple[Any, Any, bool]:
    """Return (mlflow module, run object, managed) like the W&B exporter."""
    mlflow = import_mlflow()
    tags: dict[str, Any] = dict(session_summary_fields(session_summary))
    tags["stormlog_command"] = command_name
    group = config.group or default_group(session_summary)
    if group is not None:
        tags["stormlog_group"] = group
    tags["stormlog_job_type"] = config.job_type or command_name

    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.set_tags(tags)
        return mlflow, active_run, False

    _configure_tracking_uri(mlflow, config.tracking_uri)

    start_kwargs: dict[str, Any] = {}
    if config.run_id is not None:
        start_kwargs["run_id"] = config.run_id
        if config.run_name is not None:
            start_kwargs["run_name"] = config.run_name
    else:
        experiment = config.experiment or "stormlog"
        mlflow.set_experiment(experiment)
        start_kwargs["run_name"] = config.run_name or _default_run_name(
            command_name, session_summary
        )

    run = mlflow.start_run(**start_kwargs)
    mlflow.set_tags(tags)
    return mlflow, run, True


def new_export_artifact_prefix(session_slug: str) -> str:
    """Return a unique artifact prefix so retries never append to prior tables."""
    return f"stormlog-exports/{session_slug}/{uuid4().hex}"


def update_summary(mlflow: Any, payload: Mapping[str, Any]) -> None:
    """Mirror ``wandb`` summary updates: numbers become metrics, the rest tags."""
    if not payload:
        return
    for key, value in payload.items():
        if value is None:
            continue
        if key in _IDENTITY_TAG_KEYS:
            mlflow.set_tag(key, str(value))
        elif isinstance(value, bool):
            mlflow.set_tag(key, "true" if value else "false")
        elif isinstance(value, (int, float)):
            mlflow.log_metric(key, float(value))
        else:
            mlflow.set_tag(key, str(value))


def log_file_artifact(
    mlflow: Any,
    *,
    artifact_path: str,
    path: Path,
) -> None:
    mlflow.log_artifact(local_path=str(path), artifact_path=artifact_path)


def log_directory_artifact(
    mlflow: Any,
    *,
    artifact_path: str,
    path: Path,
) -> None:
    mlflow.log_artifacts(local_dir=str(path), artifact_path=artifact_path)


def materialize_html_file(
    *,
    html_text: str,
    file_name: str,
    output_root: Path | None,
) -> Path:
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        target_path = output_root / file_name
        target_path.write_text(html_text, encoding="utf-8")
        return target_path

    temp_dir = Path(tempfile.mkdtemp(prefix="stormlog-mlflow-"))
    target_path = temp_dir / file_name
    target_path.write_text(html_text, encoding="utf-8")
    target_path.chmod(0o600)
    return target_path


def _default_run_name(
    command_name: str,
    session_summary: SessionSummary | None,
) -> str:
    if session_summary is None:
        return command_name
    return f"{command_name}-{session_summary.session_id[:8]}"


def _normalized_optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _configure_tracking_uri(mlflow: Any, tracking_uri: str | None) -> None:
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
        if tracking_uri.startswith("file:"):
            print(
                "Hint: a file: tracking URI requires MLFLOW_ALLOW_FILE_STORE=true "
                "on MLflow >= 3.13.0; consider sqlite:///mlflow.db for offline storage.",
                file=sys.stderr,
            )
