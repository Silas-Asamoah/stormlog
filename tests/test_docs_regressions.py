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
    ],
    "docs/article.md": [
        "MemoryTracker(alert_threshold_mb=",
        "results = tracker.stop_tracking()",
    ],
    "docs/tensorflow_testing_guide.md": [
        "from tfmemprof import MemoryTracker",
        "enable_alerts=True",
        "TFCPUMemoryTracker",
    ],
}

_PARAMS = [
    (relative_path, snippet)
    for relative_path, snippets in _BANNED_DOC_SNIPPETS.items()
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
