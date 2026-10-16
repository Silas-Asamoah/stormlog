"""Foreground same-interpreter MLX server capture and public embedding context."""

from __future__ import annotations

import os
import platform
import re
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import psutil

from .. import __version__
from ..exit_codes import ExitCode
from ..mlx.artifacts import CaptureBundle, CaptureOptions, Outcome
from ..mlx.runner import workload_failure
from ..mlx.runtime import MLXRuntimeError, Runtime
from .host_clock import host_boot_id
from .mlx_evidence import build_manifest, collect_files, write_manifest
from .mlx_launchers import Launcher, launch, prepare_launcher, redacted_arguments


class ServerCapture:
    def __init__(
        self,
        *,
        engine: str,
        run_id: str,
        output: str | Path,
        options: CaptureOptions,
        server_version: str | None,
        mode: str,
        argv: Sequence[str],
        endpoint_host: str | None,
        endpoint_port: int | None,
        entrypoint: str | None,
        source_revision: str | None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", run_id):
            raise ValueError(
                "run_id must be a short identifier without spaces or separators"
            )
        if engine not in {"omlx", "mlx-lm", "mlx-audio"}:
            raise ValueError("Unsupported MLX server engine")
        if options.job_id is not None and options.job_id != run_id:
            raise ValueError("Capture job ID must equal the workload run ID")
        self.engine, self.run_id, self.options = (
            engine,
            run_id,
            replace(options, job_id=run_id),
        )
        self.version, self.mode, self.argv = (
            server_version,
            mode,
            redacted_arguments(argv),
        )
        self.endpoint_host, self.endpoint_port = endpoint_host, endpoint_port
        self.entrypoint, self.source_revision = entrypoint, source_revision
        process = psutil.Process(os.getpid())
        self.host, self.pid = platform.node(), process.pid
        self.start_time_ns = round(process.create_time() * 1e9)
        self.boot_id = host_boot_id()
        self.started_at_ns = time.time_ns()
        self.outcome = Outcome(
            workload={"engine": engine, "argv": self.argv, "status": "running"}
        )
        identity = {
            "run_id": run_id,
            "host": self.host,
            "boot_id": self.boot_id,
            "pid": self.pid,
            "process_start_ns": self.start_time_ns,
            "engine": engine,
            "version": server_version,
            "argv": self.argv,
        }
        self.bundle = CaptureBundle(
            output,
            "server",
            self.options,
            identity=identity,
            command_line=f"stormlog infer mlx-serve --engine {engine}",
        )
        self.directory = self.bundle.directory
        self._proof = ListenerProof(
            self.pid, self.start_time_ns, endpoint_host, endpoint_port
        )

    def start(self, runtime: Runtime | None) -> None:
        self.bundle.start(runtime)
        self._proof.start()

    def record_failure(self, exc: BaseException) -> None:
        self.outcome = workload_failure(exc)
        # Preserve classification and exit status without storing arbitrary
        # server exception text (which can contain arguments/credentials).
        self.outcome.error = None if self.outcome.exit_code == 0 else type(exc).__name__
        self.outcome.workload = {
            "engine": self.engine,
            "argv": self.argv,
            "status": "completed" if self.outcome.exit_code == 0 else "failed",
            "exception_type": type(exc).__name__,
            "oom_occurred": self.outcome.workload.get("oom_occurred", False),
        }
        if isinstance(exc, (MLXRuntimeError, ImportError)):
            self.outcome.exit_code = int(ExitCode.USAGE)

    def finish(self) -> None:
        self._proof.stop()
        self.bundle.finish(self.outcome)
        tracker = self.bundle.tracker
        session = tracker.session_summary if tracker is not None else None
        status = {0: "complete", 130: "interrupted"}.get(
            self.outcome.exit_code, "incomplete"
        )
        manifest = build_manifest(
            run_id=self.run_id,
            engine=self.engine,
            mode=self.mode,
            producer_version=__version__,
            server_version=self.version,
            version_provenance=self._version_provenance(),
            entrypoint=self.entrypoint,
            source_revision=self.source_revision,
            host=self.host,
            boot_id=self.boot_id,
            pid=self.pid,
            start_time_ns=self.start_time_ns,
            start_time_source="psutil_process_create_time",
            session_id=session.session_id if session else None,
            capture_status=status,
            started_at_ns=self.started_at_ns,
            ended_at_ns=time.time_ns(),
            interval_seconds=self.options.interval,
            exit_code=self.outcome.exit_code,
            argv=self.argv,
            config={},
            endpoint_host=self.endpoint_host,
            endpoint_port=self.endpoint_port,
            endpoint_declared_by=(
                "launch_arguments"
                if self.mode == "mlx-serve"
                else "operator" if self.endpoint_port else "unknown"
            ),
            files=collect_files(self.directory, stats=False),
            stats_sampling=False,
            endpoint_proof=self._proof.proof if self.mode == "mlx-serve" else None,
        )
        write_manifest(self.directory, manifest)

    def _version_provenance(self) -> str:
        if self.version is None:
            return "unknown"
        return (
            "installed_distribution_metadata"
            if self.mode == "mlx-serve"
            else "user_declared"
        )


class ListenerProof:
    """Observe a literal loopback listening socket owned by the sampled PID.

    This establishes the controlled launch's HTTP process; it is not per-request
    execution ownership. Embedded custom entrypoints keep declared association.
    """

    def __init__(
        self, pid: int, process_start_ns: int, host: str | None, port: int | None
    ) -> None:
        self.pid, self.process_start_ns, self.host, self.port = (
            pid,
            process_start_ns,
            host,
            port,
        )
        self.proof: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.host not in {"127.0.0.1", "::1"} or self.port is None:
            return
        self._thread = threading.Thread(
            target=self._observe, daemon=True, name="stormlog-server-identity"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _observe(self) -> None:
        process = psutil.Process(self.pid)
        while not self._stop.is_set():
            try:
                if round(process.create_time() * 1e9) != self.process_start_ns:
                    return
                for connection in process.net_connections(kind="inet"):
                    if (
                        connection.status == psutil.CONN_LISTEN
                        and connection.laddr.port == self.port
                        and connection.laddr.ip == self.host
                    ):
                        self.proof = {
                            "method": "psutil_process_listening_socket",
                            "pid": self.pid,
                            "start_time_ns": self.process_start_ns,
                            "address": connection.laddr.ip,
                            "port": self.port,
                            "observed_at_ns": time.time_ns(),
                        }
                        return
            except (psutil.Error, OSError):
                return
            self._stop.wait(0.1)


@contextmanager
def capture_mlx_server(
    *,
    engine: str,
    run_id: str,
    output: str | Path,
    options: CaptureOptions | None = None,
    runtime: Runtime | None = None,
    server_version: str | None = None,
    argv: Sequence[str] = (),
    endpoint_host: str | None = None,
    endpoint_port: int | None = None,
    _launcher: Launcher | None = None,
) -> Iterator[ServerCapture]:
    """Capture native memory in the serving interpreter and preserve exceptions.

    Custom entrypoints must do inference in this interpreter. Children are not
    sampled or attributed to their parent. Server arguments are passed unchanged;
    only the metadata copy is redacted. No allocator state is modified.
    """
    capture = ServerCapture(
        engine=engine,
        run_id=run_id,
        output=output,
        options=options or CaptureOptions(),
        server_version=server_version,
        mode="mlx-serve" if _launcher else "embedded",
        argv=argv,
        endpoint_host=endpoint_host,
        endpoint_port=endpoint_port,
        entrypoint=_launcher.module if _launcher else None,
        source_revision=None,
    )
    primary: BaseException | None = None
    try:
        capture.start(runtime)
        yield capture
        capture.outcome.workload["status"] = "completed"
    except BaseException as exc:
        primary = exc
        capture.record_failure(exc)
        raise
    finally:
        try:
            capture.finish()
        except BaseException as exc:
            if primary is None:
                capture.record_failure(exc)
                raise
            # Finalization must not replace the original workload exception.


def run_mlx_server(
    *,
    engine: str,
    run_id: str,
    output: str | Path,
    args: Sequence[str],
    options: CaptureOptions | None = None,
    runtime: Runtime | None = None,
) -> tuple[Path, Outcome]:
    launcher, version, actual = prepare_launcher(engine, args)
    from .mlx_launchers import _option

    host = _option(actual, "--host")
    port_value = _option(actual, "--port")
    port = int(port_value) if port_value else None
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("Server port must be in [1, 65535]")
    capture: ServerCapture | None = None
    try:
        with capture_mlx_server(
            engine=engine,
            run_id=run_id,
            output=output,
            options=options,
            runtime=runtime,
            server_version=version,
            argv=actual,
            endpoint_host=host,
            endpoint_port=port,
            _launcher=launcher,
        ) as capture:
            print(f"MLX server evidence: {capture.directory}", flush=True)
            launch(launcher, actual)
    except BaseException:
        if capture is None:
            raise
        return capture.directory, capture.outcome
    return capture.directory, capture.outcome
