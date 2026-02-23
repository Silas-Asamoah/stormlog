# Stormlog Rebrand Spec (Decision-Complete, Atomic Commits)

## Summary
This spec renames the project branding and package metadata to `stormlog` while keeping runtime Python module imports and CLI commands unchanged (`gpumemprof`, `tfmemprof`, `gpu-profiler`).  
It also includes GitHub repository rename steps (`Silas-Asamoah/gpu-memory-profiler` -> `Silas-Asamoah/stormlog`), PyPI package rename to `stormlog`, and Read the Docs slug updates.  
Execution must follow Karpathy principles: think first, smallest safe change, surgical edits, and explicit verification after each atomic commit.

## Public Interfaces And Compatibility
1. Packaging interface change (breaking):
`pip install gpu-memory-profiler[...]` becomes `pip install stormlog[...]`.
`[project].name` in `/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/pyproject.toml` becomes `stormlog`.

2. Repository/docs canonical URLs change:
GitHub URLs become `https://github.com/Silas-Asamoah/stormlog`.
Docs host becomes `https://stormlog.readthedocs.io/en/latest/`.

3. Runtime compatibility preserved:
No rename of `/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/gpumemprof`.
No rename of `/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/tfmemprof`.
No rename of `gpumemprof` or `tfmemprof` console scripts in `/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/pyproject.toml`.
No telemetry collector identifier rename (`gpumemprof.*`, `tfmemprof.*` strings remain unchanged where they represent runtime IDs).

4. No API/type surface refactor:
No behavioral feature changes, no new abstractions, no unrelated cleanup.

## Karpathy-Guidelines Execution Rules
1. Think before coding:
Run a baseline search and keep an explicit allowlist of old-name references that are intentionally retained (historical changelog context only).

2. Simplicity first:
Use direct string updates; do not introduce compatibility wrapper code because module/CLI names remain unchanged by decision.

3. Surgical changes:
Edit only files containing branding/package/URL surfaces. Do not refactor unrelated logic or formatting.

4. Goal-driven verification:
Each commit has a dedicated validation command set and must pass before the next commit.

## Branching And Atomic Commit Workflow
1. Create branch from latest `main`:
```bash
cd /Users/jojoasamoah/Documents/learnings/gpu-memory-profiler
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c chore/stormlog-rebrand
```

2. Preflight checks before first edit:
```bash
gh auth status
gh repo view --json nameWithOwner,url
python3 -m pip index versions stormlog
```
If `stormlog` on PyPI is not owned/available for maintainers, stop and report blocker; do not choose a fallback name.

3. Commit discipline:
One cohesive concern per commit, no amend, no squash-rewrite during implementation, no unrelated file churn.

## Atomic Commit Plan
1. Commit 1  
Commit message: `chore(rebrand): rename distribution metadata to stormlog`  
Files:
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/pyproject.toml`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/setup.py`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/requirements.txt`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/requirements-dev.txt`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/requirements-test.txt`  
Changes:
Set project name to `stormlog`; update package/install comments and metadata URLs to new canonical slug. Keep existing script entrypoints unchanged.  
Verify:
```bash
python3 -m pip install -e . --no-deps
python3 - <<'PY'
import importlib.metadata as m
print(m.metadata("stormlog")["Name"])
assert m.metadata("stormlog")["Name"] == "stormlog"
PY
```

2. Commit 2  
Commit message: `chore(rebrand): update runtime user-facing branding strings`  
Files:
All matched tracked files in code/examples/tests for branding/install guidance under:  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/gpumemprof/`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/tfmemprof/`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/examples/`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/tests/`  
Changes:
Replace project name text and install guidance (`gpu-memory-profiler[...]` -> `stormlog[...]`) in user-facing messages and corresponding assertions. Keep command names/import paths unchanged.  
Verify:
```bash
python3 -m pytest /Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/tests/test_import_hardening.py -q
python3 -m pytest /Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/tests/test_cli_info.py -q
```

3. Commit 3  
Commit message: `docs(rebrand): update core docs, install instructions, and canonical links`  
Files:
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/README.md`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/CONTRIBUTING.md`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/docs/index.md`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/docs/installation.md`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/docs/usage.md`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/docs/tui.md`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/docs/examples.md`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/docs/gpu_setup.md`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/docs/troubleshooting.md`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/docs/cpu_compatibility.md`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/docs/conf.py`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/docs/schemas/telemetry_event_v2.schema.json`  
Changes:
Rename project wording to Stormlog, update install commands to `stormlog[...]`, and move canonical URLs/asset links to `Silas-Asamoah/stormlog` and `stormlog.readthedocs.io`.  
Verify:
```bash
python3 -m pytest /Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/tests/test_docs_regressions.py -q
python3 -m sphinx -W --keep-going -b html /Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/docs /Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/docs/_build/html
```

4. Commit 4  
Commit message: `docs(rebrand): update long-form and governance docs with migration note`  
Files:
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/CHANGELOG.md`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/PROJECT_STATUS.md`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/STYLE_GUIDE.md`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/SECURITY.md`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/CODE_OF_CONDUCT.md`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/LICENSE`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/docs/article.md`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/docs/architecture.md`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/docs/pytorch_testing_guide.md`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/docs/tensorflow_testing_guide.md`  
`/Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/docs/testing.md`  
Changes:
Complete remaining branding updates, add explicit migration note for package rename (`gpu-memory-profiler` -> `stormlog`) and explicitly state runtime CLI/import compatibility.  
Verify:
```bash
rg -n --glob '!docs/_build/**' "gpu-memory-profiler|gpu_memory_profiler|GPU Memory Profiler|gpu-memory-profiler.readthedocs.io|github.com/Silas-Asamoah/gpu-memory-profiler|pypi.org/project/gpu-memory-profiler" /Users/jojoasamoah/Documents/learnings/gpu-memory-profiler
python3 -m sphinx -W --keep-going -b html /Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/docs /Users/jojoasamoah/Documents/learnings/gpu-memory-profiler/docs/_build/html
```
Expected grep result:
Only intentional historical/migration context in `CHANGELOG.md`; zero matches elsewhere.

## External Platform Operations (Post-PR Approval, Same Rollout Window)
1. GitHub repository rename:
```bash
cd /Users/jojoasamoah/Documents/learnings/gpu-memory-profiler
gh repo rename stormlog --yes
gh repo view --json nameWithOwner,url
```

2. Local remote normalization after rename:
```bash
git remote set-url origin https://github.com/Silas-Asamoah/stormlog.git
git remote -v
```

3. Read the Docs:
Rename/create project slug `stormlog`, ensure `latest` builds from main and canonical docs URL is `https://stormlog.readthedocs.io/en/latest/`.

4. PyPI:
Publish next release as package `stormlog`; keep release notes with explicit migration commands from old package name.

## Final Verification Gate (Before Merge)
```bash
cd /Users/jojoasamoah/Documents/learnings/gpu-memory-profiler
python3 -m pip install -e .[dev,test]
pre-commit run --all-files
python3 -m flake8 gpumemprof/ tfmemprof/ tests/ examples/ --show-source --statistics
python3 -m mypy gpumemprof/ tfmemprof/
python3 -m pytest tests/ -m "not tui_pilot and not tui_snapshot and not tui_pty" -q
python3 -m build
python3 -m twine check dist/*
gpumemprof --help
tfmemprof --help
```

## Test Cases And Scenarios
1. Install and metadata scenario:
Editable install reports package name `stormlog`.

2. Runtime compatibility scenario:
`gpumemprof` and `tfmemprof` commands still work and docs/examples still show these command names.

3. Missing dependency messaging scenario:
Import guidance points to `stormlog[torch]` and `stormlog[viz]` where applicable; updated tests pass.

4. Documentation integrity scenario:
Sphinx builds warning-clean; docs regression tests pass; all canonical links point to `stormlog` repo/docs.

5. Rename hygiene scenario:
No stale canonical old-name links outside explicit changelog migration/history references.

## Assumptions And Defaults
1. Canonical naming is fixed to `stormlog` everywhere externally (repo slug, package name, docs host).
2. Repository owner remains `Silas-Asamoah`.
3. Scope is brand + metadata rename only; import paths and CLI binary names remain unchanged.
4. Existing telemetry collector IDs remain unchanged for compatibility.
5. `docs/_build`, `dist`, `*.egg-info`, and other generated artifacts are not edited or committed.
6. Merge strategy should preserve atomic commits (no squash merge for this PR).
