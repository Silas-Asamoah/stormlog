"""`stormlog infer` returns codes from the shared exit-code contract."""

import contextlib
import io
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
