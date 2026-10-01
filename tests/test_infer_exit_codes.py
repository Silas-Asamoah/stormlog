"""`stormlog infer` returns codes from the shared exit-code contract."""

import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

from stormlog.exit_codes import ExitCode
from stormlog.infer.cli import main as infer_main


def _infer(*argv: str) -> tuple[int, str]:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
        code = infer_main(list(argv))
    return code, stderr.getvalue()


def _profile(tmp_path: Path, *flags: str) -> tuple[int, str]:
    return _infer(
        "profile",
        "--endpoint",
        "http://127.0.0.1:1/v1/chat/completions",
        "--model",
        "fake-model",
        "--system-sampler",
        "none",
        "--output",
        str(tmp_path / "infer.jsonl"),
        *flags,
    )


def test_a_requested_tokenizer_that_is_not_installed_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    code, stderr = _profile(tmp_path, "--tokenizer", "tiktoken")
    assert code == ExitCode.USAGE
    assert "install it or choose another --tokenizer" in stderr
    assert not (tmp_path / "infer.jsonl").exists()


def test_an_argparse_error_keeps_its_usage_code(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as stopped:
        _profile(tmp_path, "--requests", "two")
    assert stopped.value.code == ExitCode.USAGE


def _artifact(path: Path, *records: dict[str, object]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


_FAILED_REQUEST: dict[str, object] = {
    "event_type": "infer.request",
    "phase": "measured",
    "case_id": "c1_in8_out4",
    "status": "error",
}


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "not found"),
        ("{not json\n", "Expecting property name"),
        ("[1]\n", "Line 1 is not a JSON object"),
        (b"\xff\xfe\n", "can't decode byte"),
        # Readable, but nothing in it came from a profile.
        ("", "not an inference artifact"),
        ("{}\n{}\n", "not an inference artifact"),
    ],
)
def test_analyze_exits_invalid_input_for_an_artifact_it_cannot_read(
    tmp_path: Path, content: str | bytes | None, message: str
) -> None:
    artifact = tmp_path / "infer.jsonl"
    if isinstance(content, bytes):
        artifact.write_bytes(content)
    elif content is not None:
        artifact.write_text(content)
    code, stderr = _infer("analyze", str(artifact))
    assert code == ExitCode.INVALID_INPUT
    assert message in stderr


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "No such file or directory"),
        ("", "telemetry artifact has no samples"),
        ("[]\n", "invalid telemetry line 1"),
    ],
)
def test_analyze_exits_invalid_input_for_server_telemetry_it_cannot_read(
    tmp_path: Path, content: str | None, message: str
) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl", _FAILED_REQUEST)
    telemetry = tmp_path / "server.jsonl"
    if content is not None:
        telemetry.write_text(content)
    code, stderr = _infer(
        "analyze", str(artifact), "--server-telemetry", str(telemetry)
    )
    assert code == ExitCode.INVALID_INPUT
    assert f"--server-telemetry {telemetry}: {message}" in stderr


def test_analyze_reports_findings_without_failing(tmp_path: Path) -> None:
    # analyze is read-only: a run where every request failed is still analysed.
    artifact = _artifact(tmp_path / "infer.jsonl", _FAILED_REQUEST)
    code, _stderr = _infer("analyze", str(artifact))
    assert code == ExitCode.OK


def test_analyze_exits_error_when_it_cannot_write_its_output(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path / "infer.jsonl", _FAILED_REQUEST)
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    code, stderr = _infer(
        "analyze", str(artifact), "--output", str(blocker / "report.txt")
    )
    assert code == ExitCode.ERROR
    assert stderr.startswith("Error: ")
