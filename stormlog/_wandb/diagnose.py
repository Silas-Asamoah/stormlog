"""Diagnose-bundle W&B export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .attribution import log_attribution_outputs
from .core import (
    WandbExportConfig,
    log_directory_artifact,
    read_json_if_exists,
    resolve_run,
    session_slug,
    session_summary_fields,
    session_summary_from_manifest,
    update_summary,
)


def export_diagnose_bundle_to_wandb(
    config: WandbExportConfig,
    *,
    command_name: str,
    artifact_dir: str | Path,
) -> None:
    """Export one diagnose bundle directory to W&B."""
    if not config.enabled:
        return

    bundle_dir = Path(artifact_dir)
    if not bundle_dir.exists() or not bundle_dir.is_dir():
        raise FileNotFoundError(
            f"Diagnose artifact directory not found: {artifact_dir}"
        )

    manifest = read_json_if_exists(bundle_dir / "manifest.json")
    diagnostic_summary = read_json_if_exists(bundle_dir / "diagnostic_summary.json")
    session_summary = session_summary_from_manifest(manifest)

    wandb, run, managed = resolve_run(
        config,
        command_name=command_name,
        session_summary=session_summary,
    )
    try:
        update_summary(
            run,
            diagnose_metrics(diagnostic_summary, manifest)
            | session_summary_fields(session_summary)
            | diagnose_summary_fields(bundle_dir, manifest),
        )

        if config.log_tables:
            log_suggestions_table(wandb, run, diagnostic_summary)

        if config.log_artifacts:
            log_directory_artifact(
                wandb,
                run,
                artifact_name=f"stormlog-diagnose-{session_slug(session_summary)}",
                artifact_type="stormlog-diagnose",
                path=bundle_dir,
            )

        if config.log_attribution:
            update_summary(
                run,
                log_attribution_outputs(
                    wandb,
                    run,
                    root=bundle_dir,
                    session_slug=session_slug(session_summary),
                    allow_artifact_logging=config.log_artifacts,
                ),
            )
    finally:
        if managed:
            run.finish()


def diagnose_metrics(
    diagnostic_summary: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(diagnostic_summary, Mapping):
        return {}

    metrics: dict[str, Any] = {}
    for source_key, target_key in (
        ("allocated_bytes", "stormlog_allocated_bytes"),
        ("reserved_bytes", "stormlog_reserved_bytes"),
        ("peak_bytes", "stormlog_peak_bytes"),
        ("total_bytes", "stormlog_total_bytes"),
        ("utilization_ratio", "stormlog_utilization_ratio"),
        ("fragmentation_ratio", "stormlog_fragmentation_ratio"),
        ("num_ooms", "stormlog_num_ooms"),
    ):
        value = diagnostic_summary.get(source_key)
        if isinstance(value, (int, float, bool)) and not isinstance(value, complex):
            metrics[target_key] = value

    risk_flags = diagnostic_summary.get("risk_flags")
    if isinstance(risk_flags, Mapping):
        for key, value in risk_flags.items():
            if isinstance(value, bool):
                metrics[f"stormlog_risk_{key}"] = value

    metrics.update(_manifest_metrics(manifest))

    return metrics


def diagnose_summary_fields(
    artifact_dir: Path,
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {"stormlog_artifact_dir": artifact_dir.name}
    if isinstance(manifest, Mapping):
        for source_key, target_key in (
            ("risk_detected", "stormlog_risk_detected"),
            ("native_history_enabled", "stormlog_native_history_enabled"),
            ("session_status", "stormlog_session_status"),
        ):
            value = manifest.get(source_key)
            if value is not None:
                fields[target_key] = value
    return fields


def log_suggestions_table(
    wandb: Any,
    run: Any,
    diagnostic_summary: Mapping[str, Any] | None,
) -> None:
    if not isinstance(diagnostic_summary, Mapping):
        return
    suggestions = diagnostic_summary.get("suggestions")
    if not isinstance(suggestions, list) or not suggestions:
        return
    rows = [
        [index + 1, str(suggestion)] for index, suggestion in enumerate(suggestions)
    ]
    run.log(
        {
            "stormlog_diagnostic_suggestions": wandb.Table(
                columns=["index", "suggestion"],
                data=rows,
            )
        }
    )


def _manifest_metrics(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if isinstance(manifest, Mapping):
        risk_detected = manifest.get("risk_detected")
        exit_code = manifest.get("exit_code")
        if isinstance(risk_detected, bool):
            metrics["stormlog_risk_detected"] = risk_detected
        if isinstance(exit_code, int):
            metrics["stormlog_exit_code"] = exit_code

    return metrics
