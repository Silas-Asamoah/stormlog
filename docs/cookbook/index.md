[← Back to main docs](../index.md)

# Production Cookbook

This cookbook packages Stormlog's profiling, tracking, diagnostic, and TUI flows
into task-oriented recipes for production-facing work.

Use these pages when you already know the tool is installed and want the
shortest path to a reliable operational workflow.

Audience: operators, ML engineers, release owners.
Difficulty: intermediate.

## Before you choose a recipe

- Read the [Installation Guide](../installation.md) first if the environment is
  not already set up.
- Use the [Command Line Guide](../cli.md) if you need option-by-option
  reference instead of a task recipe.
- If you installed from PyPI, use the pip-safe CLI commands on each page.
- If you are working from a source checkout, you can also use the maintained
  `examples/` and `benchmark_harness` flows for qualification.
- If you need API signatures or option-by-option reference, go back to the
  [Usage Guide](../usage.md), [Command Line Guide](../cli.md), or generated API
  reference.

## Choose the right recipe

| Goal | Start here |
| --- | --- |
| keep long-running artifacts bounded | [Always-on Tracking](always_on.md) |
| respond to a PyTorch incident quickly | [PyTorch Production Recipes](pytorch.md) |
| respond to a TensorFlow incident quickly | [TensorFlow Production Recipes](tensorflow.md) |
| profile MLX training or local generation | [MLX Recipes](mlx.md) |
| compare ranks or rebuild distributed timelines | [Distributed Diagnostics Recipes](distributed.md) |
| triage OOM or hidden-memory-gap findings | [Incident Playbooks](incidents.md) |
| qualify operational behavior in CI or before release | [CI and Release Qualification](ci_release.md) |

## Recipes

### Always-on tracking and bounded artifact budgets

Use the [Always-on Tracking](always_on.md) recipe when you want a long-running
tracking session with append-only sink files, retention limits, and explicit
guidance for degraded collectors.

### PyTorch production profiling and OOM capture

Use the [PyTorch Production Recipes](pytorch.md) page when you need to move
from a live PyTorch issue to a saved telemetry or OOM artifact quickly.

### TensorFlow production profiling and diagnosis

Use the [TensorFlow Production Recipes](tensorflow.md) page when the workload is
owned by TensorFlow and you need track, analyze, and diagnose guidance that
matches the current `tfmemprof` behavior.

### Distributed and rank-aware diagnosis

Use the [Distributed Diagnostics Recipes](distributed.md) page when you need to
track multiple ranks, preserve rank identity in artifacts, and rebuild
rank-aware diagnostics later in the TUI.

### Incident triage playbooks

Use the [Incident Playbooks](incidents.md) page when the main question is what
to do next after an OOM, hidden-memory-gap result, degraded collector, or
always-on retention issue.

### CI and release qualification

Use the [CI and Release Qualification](ci_release.md) page when you need one
place for source-checkout smoke commands, benchmark harness gates, and artifact
archival guidance.

## Suggested reading order

### New production deployment

1. [Always-on Tracking](always_on.md)
2. [Incident Playbooks](incidents.md)
3. [CI and Release Qualification](ci_release.md)

### PyTorch incident response

1. [PyTorch Production Recipes](pytorch.md)
2. [Incident Playbooks](incidents.md)
3. [Distributed Diagnostics Recipes](distributed.md) if more than one rank is involved

### TensorFlow incident response

1. [TensorFlow Production Recipes](tensorflow.md)
2. [Incident Playbooks](incidents.md)
3. [Distributed Diagnostics Recipes](distributed.md) if more than one rank is involved

---

[← Back to main docs](../index.md)
