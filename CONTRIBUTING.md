# Contributing to Stormlog

Thank you for helping improve Stormlog. This guide covers the contribution
workflow, local setup, validation commands, and project conventions used by the
current codebase.

## Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Project Layout](#project-layout)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Requests](#pull-requests)
- [Getting Help](#getting-help)
- [License](#license)

## Code of Conduct

This project and everyone participating in it is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.

## Ways to Contribute

### Report Bugs

- Search existing issues before opening a new one.
- Include a clear title and a small reproduction when possible.
- Include the operating system, Python version, framework version, and backend
  details such as CUDA, ROCm, MPS, TensorFlow Metal, or CPU-only mode.
- Include the exact command, traceback, and any generated Stormlog artifact paths
  that are safe to share.

### Suggest Enhancements

- Open an issue with the user problem, proposed behavior, and likely scope.
- Call out the affected surface: Python API, `gpumemprof`, `tfmemprof`, TUI,
  telemetry schema, docs, packaging, or examples.
- Keep optional integrations optional. Stormlog should remain local-first and
  usable without hosted services.

### Submit Pull Requests

- Use a focused branch and keep each PR scoped to one logical change.
- Add or update tests for behavior changes.
- Update docs when behavior, commands, schemas, examples, or install paths
  change.
- Prefer small, reviewable PRs over broad drive-by cleanup.

## Development Setup

### Prerequisites

- Python 3.10 or newer
- Git
- pip
- Optional runtime dependencies for the surface you are changing:
  - PyTorch: `stormlog[torch]`
  - TensorFlow: `stormlog[tf]`
  - TUI: `stormlog[tui,torch]`
  - Visualization exports: `stormlog[viz]`

### Clone and Install

```bash
git clone https://github.com/Silas-Asamoah/stormlog.git
cd stormlog
git checkout release/dev

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip

# Core development dependencies
python3 -m pip install -e ".[dev]"

# Add framework/UI extras when your change needs them
python3 -m pip install -e ".[dev,torch]"
python3 -m pip install -e ".[dev,tf]"
python3 -m pip install -e ".[dev,tui,torch]"
python3 -m pip install -e ".[dev,all]"
```

You can also install from the checked-in requirement files when reproducing CI
or release behavior:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pip install -r requirements-test.txt
python3 -m pip install -r requirements-ci-lint.txt
```

Install pre-commit hooks if you want local checks to run before each commit:

```bash
pre-commit install
```

## Project Layout

- `stormlog/`: PyTorch-facing APIs, CPU fallback utilities, telemetry,
  diagnostics, visualization, W&B integration, and the Textual TUI.
- `stormlog/tensorflow/`: TensorFlow-specific profiler, tracker, analyzer,
  diagnose, CLI, and runtime helpers.
- `stormlog/tui/`: Textual application, widgets, command helpers, and artifact
  diagnostics.
- `docs/`: user guides, architecture notes, schema documentation, and cookbook
  recipes.
- `docs/schemas/`: versioned JSON schemas for exported telemetry records.
- `examples/`: source-checkout-only examples and scenario runners.
- `tests/`: unit, integration, TUI, and end-to-end coverage.

The PyPI package and Python import root are both `stormlog`. The CLI entrypoints
are `gpumemprof`, `tfmemprof`, and `stormlog`.

## Code Style

The repository is configured for Black, isort, flake8, mypy, and pytest.

```bash
python3 -m isort stormlog/ tests/ examples/
python3 -m black stormlog/ tests/ examples/
python3 -m flake8 stormlog/ tests/ examples/ --show-source --statistics
python3 -m mypy stormlog/
```

Project conventions:

- Keep changes surgical and match the surrounding code style.
- Avoid importing optional framework dependencies at module import time unless
  the module already requires them.
- Keep GPU-specific behavior capability-gated so CPU-only environments continue
  to pass the relevant tests.
- Preserve backward compatibility for exported telemetry, manifests, CLI output,
  and public Python APIs unless a breaking change is explicitly planned.
- Add comments only when they clarify non-obvious behavior.

### Cyclomatic Complexity

All scoped Python functions and methods must have Radon complexity **10 or
less** (grade A or B). The CI complexity job checks `stormlog/` and the checker
itself, including nested functions and methods of local classes. Class aggregate
scores are excluded. Tests and examples are outside this gate.

```bash
python3 -m pip install -r requirements-complexity.txt
python3 scripts/check_complexity.py
python3 scripts/check_complexity.py --json
```

The legacy exception map in `.ci/complexity-baseline.json` is empty. Keep it
empty: do not add grace entries or raise the limit. A regression test enforces
this policy. Prefer cohesive helpers and focused tests that preserve behavior,
outputs, and error handling. CI checks the gate on Python 3.10, 3.12, 3.13,
and 3.14.

For compatibility with older branches, the checker still supports pruning
resolved or deleted legacy entries:

```bash
python3 scripts/check_complexity.py --update-baseline
```

This command runs the full check first and refuses to add new exceptions or
accept changed complex callables. It only prunes resolved or deleted entries.
Radon is pinned to 6.0.1 so measurements remain reproducible. Exit status is 0
for a pass, 1 for complexity violations, and 2 for invalid source, configuration,
or file access. The JSON report includes status, counts, and each violation's
source location and score.

Use conventional commit messages:

```text
docs: update contributor setup guide
fix(tracker): preserve collector health metadata
test(tui): cover diagnostics session selection
```

Common types are `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`,
and `build`.

## Testing

Run focused checks for the files you changed, then broaden when the change
touches shared contracts or user-facing behavior.

### Fast Local Checks

```bash
python3 -m pytest tests/test_docs_regressions.py -v
python3 -m pytest tests/test_telemetry_v2.py -v
python3 -m pytest tests/test_cli_info.py -v
```

### Main Test Suite

```bash
python3 -m pytest tests/ -v -m "not tui_pilot and not tui_snapshot and not tui_pty"
```

### TensorFlow-Focused Tests

```bash
python3 -m pytest tests/ -o "python_files=test_tf*.py" -v -m "not tui_pilot and not tui_snapshot and not tui_pty"
```

### TUI and E2E Tests

```bash
python3 -m pytest tests/tui/ -m "tui_pilot or tui_snapshot" -v
python3 -m pytest tests/e2e/test_tui_pty.py -m tui_pty -v
```

### Docs Build

```bash
python3 -m sphinx -W --keep-going -b html docs docs/_build/html
```

### Example Smoke Checks

Examples are available only from a source checkout, not from a plain PyPI
install.

```bash
python3 -m examples.cli.quickstart
python3 -m examples.cli.capability_matrix --mode smoke --target both --oom-mode simulated
python3 -m examples.scenarios.cpu_telemetry_scenario
python3 -m examples.scenarios.oom_flight_recorder_scenario --mode simulated
```

GPU, ROCm, MPS, TensorFlow, and TUI checks may require platform-specific
hardware or optional dependencies. If you cannot run a relevant check locally,
state that in the PR.

## Documentation

- Update `README.md` for top-level install, quickstart, or feature-positioning
  changes.
- Update `docs/` when CLI flags, Python APIs, telemetry artifacts, examples, or
  troubleshooting steps change.
- Update `docs/schemas/` and `docs/telemetry_schema.md` together for telemetry
  schema changes.
- Do not commit generated docs build output unless a maintainer explicitly asks
  for it.
- Keep command examples copy-pasteable and note when examples require a source
  checkout or optional extras.

## Pull Requests

1. Start from the active target branch, usually `release/dev` for in-flight
   development work unless a maintainer asks for another base.
2. Create a focused branch:

   ```bash
   git checkout release/dev
   git pull --ff-only
   git checkout -b docs/update-contributing
   ```

3. Make the change and run the relevant checks.
4. Inspect your diff before committing:

   ```bash
   git diff --check
   git diff
   ```

5. Commit with a conventional message and push your branch.
6. Open a PR against the intended base branch.

### PR Checklist

- The PR description explains what changed and why.
- Related issues are linked when applicable.
- Tests or docs checks are listed, including checks you could not run.
- Screenshots or terminal captures are included for visible TUI changes.
- New dependencies, optional extras, schema changes, and compatibility impacts
  are called out explicitly.

## Getting Help

- Use GitHub Issues for bug reports and scoped feature requests.
- Use GitHub Discussions, when available, for broader questions.
- Check the guides in `docs/` for current install, usage, testing, and
  troubleshooting details.

## License

By contributing to Stormlog, you agree that your contributions will be licensed
under the MIT License.
