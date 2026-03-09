from __future__ import annotations

import sys
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace

from examples.cli import capability_matrix


def test_run_stormlog_diagnose_uses_absolute_output_path(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    observed_cmd: list[str] = []

    def _fake_run_command(cmd, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal observed_cmd
        observed_cmd = list(cmd)
        assert kwargs["cwd"] == capability_matrix.REPO_ROOT
        return CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(capability_matrix, "run_command", _fake_run_command)

    result = capability_matrix._run_stormlog_diagnose(Path("relative-artifacts"))

    output_path = Path(observed_cmd[observed_cmd.index("--output") + 1])
    assert output_path.is_absolute()
    assert result["artifact_dir"] == str(output_path)


def test_run_benchmark_check_uses_absolute_artifact_paths(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    observed_cmd: list[str] = []

    def _fake_run_command(cmd, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal observed_cmd
        observed_cmd = list(cmd)
        assert kwargs["cwd"] == capability_matrix.REPO_ROOT
        return CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(capability_matrix, "run_command", _fake_run_command)

    result = capability_matrix._run_benchmark_check(Path("relative-artifacts"), "smoke")

    output_path = Path(observed_cmd[observed_cmd.index("--output") + 1])
    artifact_root = Path(observed_cmd[observed_cmd.index("--artifact-root") + 1])
    assert output_path.is_absolute()
    assert artifact_root.is_absolute()
    assert result["output"] == str(output_path)


def test_run_tui_smoke_skips_when_tui_extras_are_missing(monkeypatch) -> None:
    class FakeSpawnError(Exception):
        pass

    class FakeChild:
        def __init__(self) -> None:
            self.before = (
                "The Stormlog TUI requires optional dependencies. "
                "Install with `pip install 'stormlog[tui,torch]'`."
            )

        def expect(self, _pattern, timeout=None) -> None:  # type: ignore[no-untyped-def]
            raise FakeSpawnError("missing textual")

        def send(self, _chars: str) -> None:
            raise AssertionError("send should not be reached when the TUI exits early")

        def isalive(self) -> bool:
            return False

        def terminate(self, force: bool = False) -> None:
            return None

        def close(self) -> None:
            return None

    fake_pexpect = SimpleNamespace(
        spawn=lambda *args, **kwargs: FakeChild(),
        EOF=FakeSpawnError,
    )
    monkeypatch.setitem(sys.modules, "pexpect", fake_pexpect)

    result = capability_matrix._run_tui_smoke()

    assert result["status"] == "SKIP"
    assert result["reason"] == "missing TUI extras"
