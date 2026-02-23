"""Quickstart script demonstrating the gpumemprof/tfmemprof CLI tools."""

from __future__ import annotations

import shutil

from examples.common import (
    ensure_cli_available,
    print_cli_result,
    print_header,
    print_section,
    run_cli_command,
)


def _torch_cuda_available() -> bool:
    """Return True if torch+CUDA are available without requiring a hard dependency."""
    try:
        import torch
    except (ImportError, ModuleNotFoundError):
        return False
    return bool(torch.cuda.is_available())


def run_gpumemprof_examples() -> None:
    ensure_cli_available("gpumemprof")
    commands = [
        ("gpumemprof --help", ["gpumemprof", "--help"]),
        ("gpumemprof info (summary)", ["gpumemprof", "info"]),
    ]

    if _torch_cuda_available():
        commands.append(
            (
                "gpumemprof monitor (5s)",
                ["gpumemprof", "monitor", "--duration", "5", "--interval", "0.5"],
            )
        )

    for title, cmd in commands:
        result = run_cli_command(cmd)
        print_cli_result(title, result)


def run_tfmemprof_examples() -> None:
    if shutil.which("tfmemprof") is None:
        print("tfmemprof CLI not installed; skipping TensorFlow demo.")
        return

    commands = [
        ("tfmemprof --help", ["tfmemprof", "--help"]),
        ("tfmemprof info", ["tfmemprof", "info"]),
    ]
    for title, cmd in commands:
        result = run_cli_command(cmd)
        print_cli_result(title, result)


def main() -> None:
    print_header("GPU Memory Profiler CLI Quickstart")
    print_section("gpumemprof commands")
    run_gpumemprof_examples()

    print_section("tfmemprof commands")
    run_tfmemprof_examples()


if __name__ == "__main__":
    main()
