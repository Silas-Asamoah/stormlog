from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_BANNED_DOC_SNIPPETS = {
    "docs/pytorch_testing_guide.md": [
        "alert_threshold_mb=2000",
        "tracker.get_tracking_results()",
        "results['memory_samples']",
        "python -m gpumemprof.cli analyze --input",
        "--detect-leaks --visualize",
        "#### CPU Only:\n\n```bash\npython -m examples.cli.quickstart",
    ],
    "docs/article.md": [
        "MemoryTracker(alert_threshold_mb=",
        "results = tracker.stop_tracking()",
        "tfmemprof diagnose --duration 0 --output ./tf_diag\nstormlog",
    ],
    "docs/tensorflow_testing_guide.md": [
        "from tfmemprof import MemoryTracker",
        "enable_alerts=True",
        "TFCPUMemoryTracker",
        "#### CPU Only:\n\n```bash\npython -m examples.cli.quickstart",
    ],
    "docs/cpu_compatibility.md": [
        'CUDA_VISIBLE_DEVICES="" gpumemprof info\npython -m examples.cli.quickstart',
        "## Getting Started (CPU Only)\n\n1. **Quick Start:**\n\n```bash\npython -m examples.cli.quickstart",
    ],
    "docs/gpu_setup.md": [
        "`python -m examples.cli.quickstart` plus `pytest tests/test_utils.py`",
    ],
    "docs/testing.md": [
        "Need a fast signal on CPU-only machines? Run both steps below:",
        "set CUDA_VISIBLE_DEVICES=\n    python -m examples.cli.quickstart",
        "export CUDA_VISIBLE_DEVICES=\n    python -m examples.cli.quickstart",
    ],
    "docs/examples/test_guides/README.md": [
        "# Run the CLI quickstart (also part of CI)",
        "# Step 1: force CPU mode and run the CLI walkthrough",
    ],
    "docs/usage.md": [
        "from gpumemprof import CPUMemoryTracker, MemoryTracker",
    ],
}

_REQUIRED_DOC_SNIPPETS = {
    "docs/index.md": [
        "## Important: Pip vs Source Checkout",
        "The `examples/` package is **not** included.",
    ],
    "docs/installation.md": [
        "### About the examples",
    ],
    "docs/examples.md": [
        "## CLI-only validation for pip users",
        "## Python snippets for pip users",
    ],
    "docs/testing.md": [
        "### CLI-only validation (pip install)",
    ],
    "docs/tui.md": [
        "### Capability Matrix or OOM scenario button fails with `ModuleNotFoundError`",
        "python -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated --skip-tui",
        "python -m examples.scenarios.oom_flight_recorder_scenario --mode simulated",
    ],
    "docs/examples/test_guides/README.md": [
        "Treat this as a non-CUDA\nsmoke test rather than a strict CPU-only force.",
    ],
    "docs/usage.md": [
        'If you installed the PyTorch extra (`pip install "stormlog[torch]"`)',
    ],
    "docs/article.md": [
        '# Optional TUI (requires the textual extra)\npip install "stormlog[tui]"',
    ],
}

_PARAMS = [
    (relative_path, snippet)
    for relative_path, snippets in _BANNED_DOC_SNIPPETS.items()
    for snippet in snippets
]

_REQUIRED_PARAMS = [
    (relative_path, snippet)
    for relative_path, snippets in _REQUIRED_DOC_SNIPPETS.items()
    for snippet in snippets
]


@pytest.mark.parametrize(("relative_path", "snippet"), _PARAMS)  # type: ignore[misc, unused-ignore]
def test_docs_do_not_reintroduce_known_stale_api_snippets(
    relative_path: str, snippet: str
) -> None:
    content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    assert (
        snippet not in content
    ), f"{relative_path} contains stale docs snippet: {snippet!r}"


@pytest.mark.parametrize(("relative_path", "snippet"), _REQUIRED_PARAMS)  # type: ignore[misc, unused-ignore]
def test_docs_keep_issue_82_pip_vs_source_disclaimers(
    relative_path: str, snippet: str
) -> None:
    content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    assert (
        snippet in content
    ), f"{relative_path} is missing required docs snippet: {snippet!r}"
