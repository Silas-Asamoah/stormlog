"""A fake Linux /proc, laid out as a vLLM 0.30.0 server shows it."""

from __future__ import annotations

from pathlib import Path


def _stat(
    pid: int, comm: str, ppid: int, pgid: int, sid: int, start: int, state: str
) -> str:
    # Fields 3..22 of /proc/<pid>/stat; starttime is the twentieth after comm.
    after = [state, ppid, pgid, sid] + [0] * 15 + [start] + [0] * 30
    return f"{pid} ({comm}) " + " ".join(str(field) for field in after) + "\n"


def fake_process(
    proc: Path,
    pid: int,
    *,
    ppid: int = 1,
    pgid: int | None = None,
    sid: int | None = None,
    start: int = 1000,
    comm: str = "python3",
    cmdline: tuple[str, ...] = ("python3",),
    cpus: str | None = "0-7",
    environ: dict[str, str] | None = None,
    state: str = "S",
) -> None:
    base = proc / str(pid)
    base.mkdir(parents=True)
    pgid = pid if pgid is None else pgid
    sid = pid if sid is None else sid
    (base / "stat").write_text(_stat(pid, comm, ppid, pgid, sid, start, state))
    (base / "cmdline").write_bytes("\0".join(cmdline).encode() + b"\0")
    status = f"Name:\t{comm}\n" + (f"Cpus_allowed_list:\t{cpus}\n" if cpus else "")
    (base / "status").write_text(status)
    if environ is not None:
        raw = "\0".join(f"{k}={v}" for k, v in environ.items()) + "\0"
        (base / "environ").write_bytes(raw.encode())


def fake_server(tmp_path: Path) -> Path:
    """A vLLM 0.30.0 server tree as /proc shows it, plus an unrelated process."""
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "stat").write_text("cpu 1 2 3\nbtime 1700000000\nprocesses 9\n")
    serve = ("/usr/bin/python3", "/usr/local/bin/vllm", "serve", "Qwen/Qwen2.5-0.5B")
    fake_process(proc, 100, start=500, cmdline=serve, environ={"VLLM_PORT": "8000"})
    fake_process(
        proc,
        101,
        ppid=100,
        pgid=100,
        sid=100,
        start=510,
        comm="VLLM::EngineCor",
        cmdline=("VLLM::EngineCore",),
    )
    fake_process(
        proc,
        102,
        ppid=101,
        pgid=100,
        sid=100,
        start=520,
        comm="VLLM::Worker_TP",
        cmdline=("VLLM::Worker_TP0",),
        cpus="0-3",
    )
    fake_process(
        proc,
        103,
        ppid=100,
        pgid=100,
        sid=100,
        start=505,
        cmdline=(
            "python3",
            "-c",
            "from multiprocessing.resource_tracker import main;main(9)",
        ),
    )
    fake_process(
        proc,
        104,
        ppid=100,
        pgid=100,
        sid=100,
        start=600,
        comm="pip",
        cmdline=("/usr/bin/python3", "-m", "pip", "list", "--format=freeze"),
    )
    fake_process(proc, 200, start=50, comm="bash", cmdline=("bash",))
    return proc
