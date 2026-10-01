"""Per-case prompt and length summaries for inference reports."""

from __future__ import annotations

from typing import Any

from .report_stats import number_values, percentile


def prompt_summary(
    requests: list[dict[str, Any]], window: dict[str, Any] | None
) -> dict[str, Any]:
    """Which prompts a case sent and how much they share."""
    sent = [record for record in requests if record.get("status") != "dropped"]
    # Artifacts written before prompt modes existed repeated one prompt.
    modes = {str(record.get("prompt_mode", "repeat")) for record in sent}
    return {
        "mode": _one_value(modes),
        "distinct_prompts": _distinct(sent, "prompt_digest"),
        "prefix_groups_used": _distinct(sent, "prefix_group"),
        "prompts_digest": (window or {}).get("prompts_digest"),
    }


def _one_value(values: set[str]) -> str | None:
    if not values:
        return None
    return values.pop() if len(values) == 1 else "mixed"


def _distinct(records: list[dict[str, Any]], field: str) -> int | None:
    values = {record.get(field) for record in records} - {None}
    return len(values) or None


def prompt_lines(prompts: Any) -> list[str]:
    """A text-report line for prompts other than one repeated prompt."""
    if not isinstance(prompts, dict) or prompts.get("mode") in {None, "repeat"}:
        return []
    line = f"  prompts: {prompts['mode']}, {prompts.get('distinct_prompts')} distinct"
    if prompts.get("prefix_groups_used"):
        line += f" over {prompts['prefix_groups_used']} prefix groups"
    return [line]


def length_summary(requests: list[dict[str, Any]]) -> dict[str, Any]:
    """Prompt and output token counts of the completed requests."""
    return {
        field: _distribution(number_values(requests, field))
        for field in ("prompt_tokens", "output_tokens")
    }


def _distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "min": min(values, default=None),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values, default=None),
    }
