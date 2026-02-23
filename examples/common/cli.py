"""Helpers for demonstrating CLI usage in examples."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional

from .formatting import print_kv, print_section


def run_cli_command(
    command: List[str],
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """Run a CLI command and return the completed process."""
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    return result


def print_cli_result(title: str, result: subprocess.CompletedProcess) -> None:
    """Pretty-print stdout/stderr from a CLI invocation."""
    print_section(title)
    print_kv("Command", " ".join(result.args))
    print_kv("Exit code", result.returncode)
    if result.stdout:
        print("\nstdout:")
        print(result.stdout.strip())
    if result.stderr:
        print("\nstderr:")
        print(result.stderr.strip())


def ensure_cli_available(executable: str) -> None:
    """Raise a helpful error if the CLI executable is missing."""
    from shutil import which

    if which(executable) is None:
        raise RuntimeError(
            f"Executable '{executable}' not found. "
            "Install the package locally with 'pip install -e .' and try again."
        )
