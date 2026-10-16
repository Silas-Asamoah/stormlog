"""Explicit same-process entrypoints for version-qualified server candidates.

Version gates identify the inspected API, not proof that every model works.
Live qualifications and entrypoint fingerprints are recorded separately.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import signal
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..scrub import redact_url
from .http_transport import safe_error


@dataclass(frozen=True)
class Launcher:
    engine: str
    distribution: str
    module: str
    versions: tuple[str, ...]


LAUNCHERS = {
    "omlx": Launcher("omlx", "omlx", "omlx.cli", ("0.7.0.dev2",)),
    "mlx-lm": Launcher("mlx-lm", "mlx-lm", "mlx_lm.server", ("0.31.3",)),
    "mlx-audio": Launcher("mlx-audio", "mlx-audio", "mlx_audio.server", ("0.4.6",)),
}
UNSUPPORTED_FLAGS = frozenset(
    {
        "--workers",
        "--worker",
        "--reload",
        "--pipeline",
        "--distributed",
        "--tensor-parallel-size",
        "--start-ui",
        "--realtime",
    }
)
SAFE_VALUE_FLAGS = frozenset(
    {
        "--host",
        "--port",
        "--model",
        "--model-dir",
        "--base-path",
        "--log-level",
        "--max-tokens",
        "--seed",
        "--temperature",
        "--max-concurrent-requests",
        "--tts-max-batch-size",
        "--memory-guard",
        "--memory-guard-gb",
        "--log-dir",
    }
)


def prepare_launcher(
    engine: str, args: Sequence[str]
) -> tuple[Launcher, str, list[str]]:
    if engine not in LAUNCHERS:
        raise ValueError("Unknown MLX server engine")
    launcher = LAUNCHERS[engine]
    actual = list(args)
    if actual[:1] == ["--"]:
        actual.pop(0)
    if engine == "omlx" and actual[:1] == ["serve"]:
        actual.pop(0)
    _validate_process_options(actual)
    if engine == "omlx":
        _validate_omlx_config(actual)
        actual.insert(0, "serve")
    try:
        version = importlib.metadata.version(launcher.distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError(
            f"Install {launcher.distribution} in the dedicated server interpreter"
        ) from exc
    if version not in launcher.versions:
        raise ValueError(
            f'{engine} launcher supports {", ".join(launcher.versions)}; installed {version}. Use the explicit capture_mlx_server embedding context for other entrypoints.'
        )
    return launcher, version, actual


def _validate_process_options(args: Sequence[str]) -> None:
    for arg in args:
        if arg.split("=", 1)[0] in UNSUPPORTED_FLAGS:
            raise ValueError(
                "Workers, reload, distributed and auxiliary-process server modes are not qualified for same-process capture"
            )
    for name in ("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "WEB_CONCURRENCY"):
        if os.environ.get(name, "1") not in {"", "1"}:
            raise ValueError(
                "A multiprocess environment is incompatible with this launcher"
            )
    if any(
        os.environ.get(name) for name in ("MLX_HOSTFILE", "MLX_RANK", "UVICORN_RELOAD")
    ):
        raise ValueError(
            "Distributed/reload environment is incompatible with same-process capture"
        )


def _option(args: Sequence[str], flag: str) -> str | None:
    for index, arg in enumerate(args):
        if arg.startswith(flag + "="):
            return arg.partition("=")[2]
        if arg == flag and index + 1 < len(args):
            return args[index + 1]
    return None


def _validate_omlx_config(args: Sequence[str]) -> None:
    import json

    base = _option(args, "--base-path")
    if not base:
        raise ValueError(
            "Instrumented oMLX requires an explicit dedicated --base-path; existing service settings are not reused implicitly"
        )
    config = Path(base).expanduser() / "settings.json"
    if config.exists():
        from .audio_workload import bounded_file

        settings = json.loads(bounded_file(config, 1024**2))
        cluster = settings.get("cluster", {}) if isinstance(settings, dict) else {}
        server = settings.get("server", {}) if isinstance(settings, dict) else {}
        distributed = isinstance(server, dict) and server.get(
            "distributed_inference_enabled"
        )
        if (isinstance(cluster, dict) and cluster.get("enabled")) or distributed:
            raise ValueError(
                "Distributed oMLX settings are not qualified for process-local capture"
            )


@contextmanager
def server_arguments(module: str, args: Sequence[str]) -> Iterator[None]:
    previous = sys.argv
    sys.argv = [module, *args]
    try:
        yield
    finally:
        sys.argv = previous


def launch(launcher: Launcher, args: Sequence[str]) -> None:
    with server_arguments(launcher.module, args), server_signals():
        module = importlib.import_module(launcher.module)
        entrypoint: Callable[[], Any] = getattr(module, "main")
        entrypoint()


@contextmanager
def server_signals() -> Iterator[None]:
    previous = {}
    if threading.current_thread() is threading.main_thread():
        previous = {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def redacted_arguments(args: Sequence[str]) -> list[str]:
    """Unknown values are omitted; known credential flags work in both forms."""
    output: list[str] = []
    pending: str | None = None
    for argument in args:
        if pending is not None and not argument.startswith("--"):
            output.append(_argument_value(pending, argument))
            pending = None
        elif argument.startswith("--"):
            pending = None
            flag, separator, value = argument.partition("=")
            if separator:
                output.append(flag + "=" + _argument_value(flag, value))
            else:
                output.append(flag)
                pending = flag
        else:
            output.append("serve" if argument == "serve" else "[OMITTED]")
    return output


def _argument_value(flag: str, value: str) -> str:
    if (
        any(
            word in flag.lower()
            for word in ("key", "token", "password", "secret", "auth")
        )
        and flag not in SAFE_VALUE_FLAGS
    ):
        return "[REDACTED]"
    if flag not in SAFE_VALUE_FLAGS:
        return "[OMITTED]"
    if value.startswith(("http://", "https://")):
        return redact_url(value) or "[OMITTED]"
    return safe_error(value)
