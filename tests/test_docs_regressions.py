import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = REPO_ROOT / "docs"
DOC_SOURCE_FILES = [
    *[
        path
        for path in (
            REPO_ROOT / "README.md",
            REPO_ROOT / "CONTRIBUTING.md",
            REPO_ROOT / "RELEASE_CHECKLIST.md",
        )
        if path.exists()
    ],
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
        "python -m gpumemprof.cli analyze --input",
        "--detect-leaks --visualize",
    ],
    "docs/article.md": [
        "MemoryTracker(alert_threshold_mb=",
        "results = tracker.stop_tracking()",
        "tui-distributed-diagnostics-workflow.svg",
        'pip install "stormlog[tui]"',
    ],
    "docs/tensorflow_testing_guide.md": [
        "from stormlog.tensorflow import MemoryTracker",
        "TFCPUMemoryTracker",
        "from tfmemprof import MemoryTracker",
    ],
    "docs/index.md": [
        "The `gpumemprof` and `tfmemprof` CLIs, the `stormlog` TUI entrypoint, and the public Python APIs work with a pip install.",
        "Stormlog ships three surfaces",
    ],
    "docs/architecture.md": [
        "The shared terminal UI is the `stormlog` entrypoint implemented in `stormlog.tui`.",
        "The `stormlog` console script points to `stormlog.tui:run_app`.",
    ],
    "docs/cpu_compatibility.md": [
        "python -m examples.cli.quickstart",
    ],
    "docs/testing.md": [
        "export CUDA_VISIBLE_DEVICES=\npython3 -m examples.cli.quickstart",
    ],
    "docs/examples/test_guides/README.md": [
        "# Run the CLI quickstart (also part of CI)",
        "# Step 1: force CPU mode and run the CLI walkthrough",
        'pip install "stormlog[tui]"',
    ],
    "docs/usage.md": [
        "from gpumemprof import CPUMemoryTracker, MemoryTracker",
        "python -m stormlog.cli",
        "python -m stormlog.tensorflow.cli",
        "interactive WebAssembly graph viewer",
        "no Go, `protoc`, or Graphviz installation required",
    ],
    "README.md": [
        "gpu-profiler-overview.gif",
        "python -m stormlog.cli",
        "python -m stormlog.tensorflow.cli",
        "There is no top-level `stormlog` Python module today.",
    ],
    "docs/cli.md": [
        "python -m stormlog.cli",
        "python -m stormlog.tensorflow.cli",
        "JAX device allocation fields are only emitted",
    ],
    "docs/installation.md": [
        "There is no top-level `import stormlog` module in the current package layout.",
    ],
    "docs/api.md": [
        "There is no top-level `stormlog` module today.",
        "JAX snapshots and tracking results retain numeric fields",
    ],
    "docs/jax_testing_guide.md": [
        "it never treats unavailable",
        "Use the generated pprof/OOM artifacts",
    ],
    "docs/native_probe_feasibility.md": [
        "Stormlog measured native-probe overhead",
        "Stormlog achieved less than 4% overhead",
        "production-ready native collector",
    ],
    "docs/native_probe_qualification.md": [
        "all unsupported combinations passed",
        "missing values are zero",
    ],
    "docs/native_trace_integration.md": [
        "synthetic traces prove CUPTI support",
        "native tracing is enabled by default",
        "ship or invoke a native collector.",
    ],
    "docs/tui.md": [
        "tui-distributed-diagnostics-workflow.svg",
        "tui-distributed-diagnostics-workflow.png",
        'pip install "stormlog[tui]"',
    ],
    "CONTRIBUTING.md": [
        "gpu-memory-profiler.git",
    ],
}

_REQUIRED_DOC_SNIPPETS = {
    "CHANGELOG.md": [
        "Callers that need tensor counts must now pass `track_tensors=True`",
    ],
    "docs/index.md": [
        "## Important: Pip vs Source Checkout",
        "The `examples/` package is **not** included.",
        "Stormlog ships four surfaces",
        "`stormlog infer`",
        "Install `stormlog[tui,torch]` if you want `stormlog` or `stormlog tui` to launch the TUI from a pip install.",
    ],
    "README.md": [
        "source-checkout only",
        "gpumemprof info",
        "tfmemprof info",
        "stormlog infer profile",
    ],
    "docs/article.md": [
        'pip install "stormlog[tui,torch]"',
        "If you are working from a source checkout, you can optionally add:",
    ],
    "docs/cli.md": [
        "**Pip users** should use this CLI-only sequence instead:",
        "`--device /CPU:0`",
        "stormlog infer profile",
        "JAX memory sample events are emitted only",
    ],
    "docs/inference.md": [
        "stormlog infer profile",
        "server_usage",
        "Chunk inter-arrival timing is chunk-level timing.",
    ],
    "docs/cpu_compatibility.md": [
        "Do **not** use `examples.cli.quickstart` for pip installs",
        "non-CUDA smoke test rather than a strict CPU-only force.",
    ],
    "docs/examples.md": [
        "CLI-only validation for pip users",
        "not included in the PyPI distribution",
    ],
    "docs/examples/test_guides/README.md": [
        "Treat this as a non-CUDA",
        "strict CPU-only force.",
        'pip install "stormlog[tui,torch]"',
    ],
    "docs/installation.md": [
        "Source-only examples and guides",
        "A plain `pip install stormlog` does not include them.",
    ],
    "docs/pytorch_testing_guide.md": [
        "source-checkout only",
        "installed from PyPI",
    ],
    "docs/tensorflow_testing_guide.md": [
        "source-checkout only",
        "`--device /CPU:0`",
    ],
    "docs/testing.md": [
        "CLI-only validation (pip install)",
        "non-CUDA smoke test rather than a strict CPU-only force.",
    ],
    "docs/tui.md": [
        "`Capability Matrix` or `OOM Scenario` fails with `ModuleNotFoundError`",
        'pip install -e ".[tui,torch]"',
    ],
    "docs/usage.md": [
        "The `examples/` package is not included in the PyPI distribution.",
        "stormlog[torch]",
        'JAXMemoryProfiler(device_index="cpu")',
        "if results.device_memory_available:",
        "requires the Graphviz `dot` executable at generation time",
        "--oom-flight-recorder",
    ],
    "docs/api.md": [
        "`TrackingResult` additionally exposes",
    ],
    "docs/jax_testing_guide.md": [
        "if results.device_memory_available:",
        "`jaxmemprof diagnose --duration 0` skips timeline capture",
        "`jaxmemprof track --profile`",
        "`--oom-flight-recorder`",
    ],
    "docs/telemetry_schema.md": [
        "`stormlog.jax.memory_tracker`",
    ],
    "docs/native_probe_feasibility.md": [
        "**Prototype CUPTI Activity collection first",
        "no Stormlog native-probe performance",
        "opt-in, separately built native helper",
        "High-volume native activity must **not** be encoded as memory samples",
    ],
    "docs/native_probe_qualification.md": [
        "that the experiments have run.",
        "A lane that skips every applicable GPU scenario is not a successful run.",
        "Unknown and unsupported values remain explicit.",
        "is valid when the manifest and raw evidence",
    ],
    "docs/native_trace_integration.md": [
        "disabled by default",
        "CUDA_INJECTION64_PATH",
        "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
        "They are not converted into",
        "qualification remains in",
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


@pytest.mark.parametrize(
    ("relative_path", "snippet"), _REQUIRED_PARAMS
)  # type: ignore[misc, unused-ignore]
def test_docs_keep_current_pip_vs_source_and_runtime_caveats(
    relative_path: str, snippet: str
) -> None:
    content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    assert (
        snippet in content
    ), f"{relative_path} is missing required docs snippet: {snippet!r}"


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


def test_doc_links_and_media_targets_exist() -> None:
    for doc_path in DOC_SOURCE_FILES:
        for target in _iter_local_doc_links(doc_path):
            if target.startswith("/"):
                resolved = (REPO_ROOT / target.lstrip("/")).resolve()
            else:
                resolved = (doc_path.parent / target).resolve()
            assert (
                resolved.exists()
            ), f"{doc_path.relative_to(REPO_ROOT)} references missing path: {target}"


def test_docs_use_human_readable_markdown_link_labels() -> None:
    label_pattern = re.compile(r"\[[^\]]+\.md\]\(")

    for doc_path in DOC_SOURCE_FILES:
        content = doc_path.read_text(encoding="utf-8")
        assert not label_pattern.search(content), (
            f"{doc_path.relative_to(REPO_ROOT)} contains a markdown link label "
            "that exposes a .md filename"
        )


def test_readme_uses_absolute_urls_for_pypi_rendering() -> None:
    readme_links = _iter_local_doc_links(REPO_ROOT / "README.md")
    assert not readme_links, (
        "README.md contains repo-relative links or media targets that break on PyPI: "
        f"{readme_links}"
    )


def test_readme_uses_json_backed_pypi_badge() -> None:
    content = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://img.shields.io/badge/dynamic/json" in content
    assert "https%3A%2F%2Fpypi.org%2Fpypi%2Fstormlog%2Fjson" in content
    assert "https://img.shields.io/pypi/v/stormlog.svg" not in content


def test_readme_uses_live_github_actions_badge() -> None:
    content = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert (
        "https://github.com/Silas-Asamoah/stormlog/actions/workflows/ci.yml/badge.svg?branch=main"
        in content
    )
    assert "https://img.shields.io/badge/build-passing-brightgreen" not in content


def test_docs_conf_uses_stormlog_canonical_baseurl() -> None:
    content = (DOC_ROOT / "conf.py").read_text(encoding="utf-8")
    assert "https://stormlog.readthedocs.io/en/latest/" in content
    assert "https://gpu-memory-profiler.readthedocs.io/" not in content
