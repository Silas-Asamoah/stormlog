import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = REPO_ROOT / "docs"
DOC_SOURCE_FILES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "CONTRIBUTING.md",
    REPO_ROOT / "RELEASE_CHECKLIST.md",
    *sorted(
        path
        for path in DOC_ROOT.rglob("*.md")
        if "_build" not in path.parts
        and path.relative_to(DOC_ROOT).parts[:2] != ("reference", "generated")
    ),
]

_BANNED_DOC_SNIPPETS = {
    "docs/pytorch_testing_guide.md": [
        "tracker.get_tracking_results()",
        "results['memory_samples']",
    ],
    "docs/article.md": [
        "MemoryTracker(alert_threshold_mb=",
        "results = tracker.stop_tracking()",
        "tui-distributed-diagnostics-workflow.svg",
    ],
    "docs/tensorflow_testing_guide.md": [
        "from tfmemprof import MemoryTracker",
        "TFCPUMemoryTracker",
    ],
    "README.md": [
        "python -m gpumemprof.cli",
        "python -m tfmemprof.cli",
    ],
    "docs/cli.md": [
        "python -m gpumemprof.cli",
        "python -m tfmemprof.cli",
    ],
    "docs/usage.md": [
        "python -m gpumemprof.cli",
        "python -m tfmemprof.cli",
    ],
    "docs/tui.md": [
        "tui-distributed-diagnostics-workflow.svg",
        "tui-distributed-diagnostics-workflow.png",
    ],
    "docs/examples/test_guides/README.md": [
        'pip install "stormlog[tui]"',
    ],
    "CONTRIBUTING.md": [
        "gpu-memory-profiler.git",
    ],
}

_PARAMS = [
    (relative_path, snippet)
    for relative_path, snippets in _BANNED_DOC_SNIPPETS.items()
    for snippet in snippets
]


@pytest.mark.parametrize(
    ("relative_path", "snippet"), _PARAMS
)  # type: ignore[misc, unused-ignore]
def test_docs_do_not_reintroduce_known_stale_api_snippets(
    relative_path: str, snippet: str
) -> None:
    content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    assert (
        snippet not in content
    ), f"{relative_path} contains stale docs snippet: {snippet!r}"


_MARKDOWN_LINK_RE = re.compile(
    r"!\[[^\]]*\]\(([^)#]+)(?:#[^)]+)?\)|\[[^\]]+\]\(([^)#]+)(?:#[^)]+)?\)"
)
_HTML_IMAGE_RE = re.compile(r'<img\s+[^>]*src="([^"]+)"')


def _iter_local_doc_links(doc_path: Path) -> list[str]:
    content = doc_path.read_text(encoding="utf-8")
    links: list[str] = []
    for markdown_match in _MARKDOWN_LINK_RE.finditer(content):
        target = markdown_match.group(1) or markdown_match.group(2)
        if target:
            links.append(target)
    links.extend(_HTML_IMAGE_RE.findall(content))
    return [
        link
        for link in links
        if not link.startswith(("http://", "https://", "mailto:"))
    ]


@pytest.mark.parametrize("doc_path", DOC_SOURCE_FILES, ids=lambda path: path.name)
def test_doc_links_and_media_targets_exist(doc_path: Path) -> None:
    for target in _iter_local_doc_links(doc_path):
        if target.startswith("/"):
            resolved = (REPO_ROOT / target.lstrip("/")).resolve()
        else:
            resolved = (doc_path.parent / target).resolve()
        assert (
            resolved.exists()
        ), f"{doc_path.relative_to(REPO_ROOT)} references missing path: {target}"


def test_readme_uses_absolute_urls_for_pypi_rendering() -> None:
    readme_links = _iter_local_doc_links(REPO_ROOT / "README.md")
    assert not readme_links, (
        "README.md contains repo-relative links or media targets that break on PyPI: "
        f"{readme_links}"
    )
