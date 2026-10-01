import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ci_workflow_content() -> str:
    return (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")


def test_ci_uses_built_wheel_for_cli_smoke() -> None:
    content = _ci_workflow_content()
    start = content.index("artifact-cli-smoke:")
    end = content.index("examples-smoke:", start)
    job_block = content[start:end]

    assert re.search(
        r"uses: actions/download-artifact@[0-9a-f]{40}\s+# v4\.",
        job_block,
    )
    assert "python3 -m venv .venv-wheel-smoke" in job_block
    assert 'pip install "$WHEEL_PATH"' in job_block
    assert "torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu" in job_block
    assert "gpumemprof info" in job_block
    assert "examples.cli.quickstart" not in job_block
    assert "pip install -e ." not in job_block


def test_ci_triggers_include_release_dev() -> None:
    content = _ci_workflow_content()

    assert "branches: [main, develop, release/dev]" in content
    assert "branches: [main, release/v0.2-readiness, release/dev]" in content


def test_ci_jax_leg_verifies_runtime() -> None:
    content = _ci_workflow_content()
    start = content.index("- name: Verify JAX runtime")
    end = content.index("- name: Run tests", start)
    runtime_step = content[start:end]

    assert "if: matrix.framework == 'jax'" in runtime_step
    assert "import jax" in runtime_step
    assert "import jaxlib" in runtime_step
    assert "jax.__version__" in runtime_step
    assert "jaxlib.__version__" in runtime_step
    assert "devices = jax.devices()" in runtime_step
    assert "if not devices:" in runtime_step
    assert "jax.numpy.arange(3) + 1" in runtime_step
    assert "result.block_until_ready()" in runtime_step

    jax_test_command = re.search(
        r'python3 -m pytest tests/ -o "python_files=test_jax\*\.py"[^\n]+',
        content,
    )
    assert jax_test_command is not None
    assert "-ra" in jax_test_command.group(0)


def test_ci_wires_memory_regression_gate_job() -> None:
    content = _ci_workflow_content()
    start = content.index("memory-regression-gate:")
    end = content.index("memory-operability-budget:", start)
    job_block = content[start:end]

    assert "runs-on: ubuntu-24.04" in job_block
    assert "tensorflow-cpu==2.15.0" in job_block
    assert "--profile pr" in job_block
    assert "--mode all" in job_block
    assert "--gate-mode regression" in job_block
    assert "docs/benchmarks/v0.4_baseline.json" in job_block
    assert "docs/benchmarks/v0.4_tolerances.json" in job_block
    assert "artifacts/benchmarks/ci_regression.json" in job_block
    assert "artifacts/benchmarks/ci_scenarios" in job_block
    assert "--iterations 5000" in job_block
    assert re.search(
        r"uses: actions/upload-artifact@[0-9a-f]{40}\s+# v4\.",
        job_block,
    )


def test_ci_wires_memory_operability_budget_job() -> None:
    content = _ci_workflow_content()
    start = content.index("memory-operability-budget:")
    end = content.index("artifact-cli-smoke:", start)
    job_block = content[start:end]

    assert "runs-on: ubuntu-24.04" in job_block
    assert "tensorflow-cpu==2.15.0" in job_block
    assert "--profile nightly" in job_block
    assert "--mode all" in job_block
    assert "--gate-mode budget" in job_block
    assert "docs/benchmarks/v0.4_operating_budget.json" in job_block
    assert "--iterations 5000" in job_block
    assert re.search(
        r"uses: actions/upload-artifact@[0-9a-f]{40}\s+# v4\.",
        job_block,
    )


def test_ci_uses_supported_python_and_actions_pins() -> None:
    content = _ci_workflow_content()

    assert not re.search(r"uses:\s+\S+@v\d", content)
    assert re.search(
        r"uses: actions/setup-python@[0-9a-f]{40}\s+# v6\.",
        content,
    )
    assert "actions/cache@" not in content


def test_ci_declares_minimal_permissions_and_hardens_checkout() -> None:
    content = _ci_workflow_content()
    checkout_uses = re.findall(r"uses: actions/checkout@[0-9a-f]{40}", content)

    assert "permissions: {}" in content
    assert checkout_uses
    assert content.count("persist-credentials: false") == len(checkout_uses)


def test_ci_runs_zizmor_workflow_audit() -> None:
    content = _ci_workflow_content()
    start = content.index("workflow-audit:")
    end = content.index("docs:", start)
    job_block = content[start:end]

    assert "contents: read" in job_block
    assert "pip install zizmor" in job_block
    assert "zizmor .github/workflows/*.yml" in job_block
    assert "persist-credentials: false" in job_block


def test_ci_enforces_radon_complexity_without_framework_dependencies() -> None:
    content = _ci_workflow_content()
    start = content.index("    complexity:")
    end = content.index("    workflow-audit:", start)
    job_block = content[start:end]

    assert "requirements-complexity.txt" in job_block
    assert 'python-version: ["3.10", "3.12", "3.13", "3.14"]' in job_block
    assert "python-version: ${{ matrix.python-version }}" in job_block
    assert "python3 -m pytest tests/test_complexity_gate.py" in job_block
    assert "python3 scripts/check_complexity.py --json" in job_block
    assert "continue-on-error" not in job_block
    assert "pip install -e" not in job_block
