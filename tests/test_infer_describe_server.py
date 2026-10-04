"""`stormlog infer describe-server`: one description of a running server."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from stormlog.exit_codes import ExitCode
from stormlog.infer.cli import main as infer_main
from stormlog.infer.describe_server import (
    DESCRIPTION_FORMAT,
    DescribeOptions,
    describe_server,
    load_description,
    write_description,
)
from stormlog.infer.errors import InferInputError, InferUsageError
from stormlog.infer.server_collector import NvmlUnavailableError
from tests.infer_proc_helpers import fake_process, fake_server

SECRET = "hf_plantedSecret0123456789"
SMI = "<nvidia_smi_log>...</nvidia_smi_log>"


class _Gpus:
    def __init__(self) -> None:
        self.closed = False

    def driver_version(self) -> str:
        return "580.82.07"

    def cuda_driver_version(self) -> int:
        return 13000

    def device_count(self) -> int:
        return 1

    def uuid(self, index: int) -> str:
        return "GPU-aaaa"

    def compute_pids(self, index: int) -> set[int]:
        return {102}

    def read(self, index: int, name: str) -> Any:
        return {"name": "NVIDIA A30"}.get(name, 1)

    def close(self) -> None:
        self.closed = True


def _run(arguments: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
    if arguments[0] == "nvidia-smi":
        return subprocess.CompletedProcess(arguments, 0, SMI, "")
    report = {"python": "3.12.3", "packages": {"torch": "2.9.0", "vllm": "0.30.0"}}
    return subprocess.CompletedProcess(arguments, 0, json.dumps(report), "")


@pytest.fixture
def proc(tmp_path: Path) -> Path:
    proc = fake_server(tmp_path)
    environ = (proc / "100" / "environ").read_bytes()
    extra = f"HF_TOKEN={SECRET}\0VLLM_WORKER_MULTIPROC_METHOD=spawn\0PATH=/bin\0"
    (proc / "100" / "environ").write_bytes(environ + extra.encode())
    return proc


def _describe(proc: Path, **options: Any) -> dict[str, Any]:
    gpus = _Gpus()
    return describe_server(
        DescribeOptions(pid=100, proc=proc, **options),
        gpu_reader=lambda: gpus,
        run=_run,
    )


def test_a_description_covers_the_server_tree_and_its_gpus(proc: Path) -> None:
    document = _describe(proc)

    assert (document["format"], document["version"]) == (DESCRIPTION_FORMAT, 1)
    server = document["server"]
    assert server["pid"] == 100 and server["start_ticks"] == 500
    roles = {item["pid"]: item["role"] for item in server["processes"]}
    assert roles[101] == "engine_core" and roles[102] == "worker"
    assert server["launch"]["model"] == "Qwen/Qwen2.5-0.5B"
    assert server["start_method"] == {"configured": "spawn", "spawn_marker": False}
    assert "PATH" not in server["environ"]
    assert document["gpus"]["server_uuids"] == ["GPU-aaaa"]
    assert document["runtime"] == {
        "interpreter": "/usr/bin/python3",
        "python": "3.12.3",
        "packages": {"torch": "2.9.0", "vllm": "0.30.0"},
    }
    assert document["nvidia_smi"] == {
        "sha256": hashlib.sha256(SMI.encode()).hexdigest(),
        "bytes": len(SMI),
    }
    assert document["model"]["identity_evidence"] == "unresolved"
    assert document["issues"] == []
    assert SECRET not in json.dumps(document)


def test_a_written_description_loads_back_and_a_changed_one_does_not(
    proc: Path, tmp_path: Path
) -> None:
    path = tmp_path / "server.json"
    write_description(_describe(proc), path)
    assert load_description(path)["server"]["pid"] == 100

    tampered = json.loads(path.read_text())
    tampered["server"]["pid"] = 999
    path.write_text(json.dumps(tampered))
    with pytest.raises(InferInputError, match="sha256 does not match"):
        load_description(path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "server description"),
        ('{"format": "other"}', "not stormlog.infer.server_description"),
        ('{"format": "stormlog.infer.server_description", "version": 2}', "version 2"),
    ],
)
def test_a_file_that_is_not_a_description_is_invalid_input(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "server.json"
    path.write_text(content)
    with pytest.raises(InferInputError, match=message):
        load_description(path)


def test_a_process_that_is_not_the_api_server_is_described_with_an_issue(
    proc: Path,
) -> None:
    document = describe_server(
        DescribeOptions(pid=101, proc=proc, no_gpu=True, python="none"), run=_run
    )
    assert [item["pid"] for item in document["server"]["processes"]] == [101, 102]
    assert any("engine_core process" in issue for issue in document["issues"])
    assert any("environment could not be read" in issue for issue in document["issues"])


def test_without_gpus_or_python_those_parts_are_null(proc: Path) -> None:
    document = _describe(proc, no_gpu=True, python="none")
    assert document["gpus"] is None
    assert document["nvidia_smi"] is None
    assert document["runtime"] is None


def test_a_retitled_root_shows_no_interpreter(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    fake_process(proc, 5, comm="VLLM::EngineCor", cmdline=("VLLM::EngineCore",))
    document = describe_server(DescribeOptions(pid=5, proc=proc, no_gpu=True), run=_run)
    assert document["runtime"] is None


def test_a_missing_process_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(InferUsageError, match="no such process"):
        describe_server(DescribeOptions(pid=4242, proc=tmp_path))


def test_a_server_log_that_cannot_be_read_is_invalid_input(
    proc: Path, tmp_path: Path
) -> None:
    with pytest.raises(InferInputError, match="--server-log"):
        _describe(proc, server_log=tmp_path / "missing.log")


def test_a_server_log_is_read_into_the_description(proc: Path, tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text(
        "INFO [core.py:124] Initializing a V1 LLM engine (v0.30.0) with config: x\n"
        "INFO [cuda.py:539] Using FLASH_ATTN attention backend out of potential "
        "backends: ['FLASH_ATTN'].\n"
    )
    document = _describe(proc, server_log=log)
    assert document["log"]["attention_backend"] == "FLASH_ATTN"
    assert document["log"]["path"] == str(log)


def _cli(*argv: str) -> tuple[int, str]:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
        code = infer_main(["describe-server", *argv])
    return code, stderr.getvalue()


def test_the_cli_writes_the_description(
    monkeypatch: pytest.MonkeyPatch, proc: Path, tmp_path: Path
) -> None:
    from stormlog.infer import cli

    def fake(options: DescribeOptions) -> dict[str, Any]:
        assert (options.pid, options.hash_weights, options.python) == (
            100,
            True,
            "auto",
        )
        return _describe(proc)

    monkeypatch.setattr(cli, "describe_server", fake)
    output = tmp_path / "out" / "server.json"
    code, _err = _cli("--pid", "100", "--hash-weights", "--output", str(output))
    assert code == ExitCode.OK
    assert load_description(output)["server"]["pid"] == 100


def test_the_cli_without_nvml_says_to_pass_no_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from stormlog.infer import cli

    def fake(options: DescribeOptions) -> dict[str, Any]:
        raise NvmlUnavailableError("NVML is unavailable on this host")

    monkeypatch.setattr(cli, "describe_server", fake)
    code, err = _cli("--pid", "1", "--output", str(tmp_path / "s.json"))
    assert code == ExitCode.USAGE
    assert "--no-gpu" in err


def test_the_cli_refuses_a_process_it_cannot_find(tmp_path: Path) -> None:
    code, err = _cli("--pid", "999999999", "--no-gpu", "--output", str(tmp_path / "s"))
    assert code == ExitCode.USAGE
    assert "no such process" in err
