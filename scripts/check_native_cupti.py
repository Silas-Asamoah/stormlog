"""Compile the optional CUPTI injection library against pinned NVIDIA headers."""

from __future__ import annotations

import argparse
import importlib.util
import shutil

# The compiler is invoked with a constructed argument vector and no shell.
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path
from typing import Sequence


def _package_root(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    locations = None if spec is None else spec.submodule_search_locations
    if not locations:
        raise RuntimeError(f"required native CI package is unavailable: {module_name}")
    return Path(next(iter(locations))).resolve()


def _installed_paths() -> tuple[tuple[Path, ...], Path]:
    cupti = _package_root("nvidia.cuda_cupti")
    runtime = _package_root("nvidia.cuda_runtime")
    nvcc = _package_root("nvidia.cuda_nvcc")
    return (cupti / "include", runtime / "include", nvcc / "include"), cupti / "lib"


def _root_paths(root: Path) -> tuple[tuple[Path, ...], Path]:
    return (
        (
            root / "cuda_cupti" / "include",
            root / "cuda_runtime" / "include",
            root / "cuda_nvcc" / "include",
        ),
        root / "cuda_cupti" / "lib",
    )


def _validate_paths(include_paths: Sequence[Path], library_path: Path) -> None:
    required = (
        include_paths[0] / "cupti.h",
        include_paths[1] / "cuda.h",
        include_paths[2] / "crt" / "host_defines.h",
        library_path / "libcupti.so.12",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("missing native build inputs: " + ", ".join(missing))


def _compile(
    compiler: str,
    source: Path,
    output: Path,
    include_paths: Sequence[Path],
    library_path: Path,
    *,
    syntax_only: bool,
) -> None:
    command = [
        compiler,
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
    ]
    for include_path in include_paths:
        command.extend(("-isystem", str(include_path)))
    if syntax_only:
        command.extend(("-fsyntax-only", str(source)))
    else:
        command.extend(
            (
                "-shared",
                "-fPIC",
                str(source),
                f"-L{library_path}",
                "-l:libcupti.so.12",
                "-pthread",
                "-o",
                str(output),
            )
        )
    subprocess.run(command, check=True)  # nosec B603


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Stormlog repository root.",
    )
    parser.add_argument(
        "--nvidia-package-root",
        type=Path,
        help="Directory containing extracted cuda_cupti/runtime/nvcc packages.",
    )
    parser.add_argument(
        "--syntax-only",
        action="store_true",
        help="Parse and type-check C++ without invoking the platform linker.",
    )
    args = parser.parse_args(argv)
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("a C++ compiler is required")
    if args.nvidia_package_root is None:
        include_paths, library_path = _installed_paths()
    else:
        include_paths, library_path = _root_paths(args.nvidia_package_root.resolve())
    _validate_paths(include_paths, library_path)
    source = args.root.resolve() / "native" / "cupti" / "stormlog_cupti_injection.cpp"
    with tempfile.TemporaryDirectory(prefix="stormlog-cupti-build-") as temporary:
        output = Path(temporary) / "libstormlog_cupti_injection.so"
        _compile(
            compiler,
            source,
            output,
            include_paths,
            library_path,
            syntax_only=args.syntax_only,
        )
        if args.syntax_only:
            print("CUPTI injection source passed syntax and type checking")
            return 0
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("native compiler did not produce the injection library")
        print(f"built {output.name} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
