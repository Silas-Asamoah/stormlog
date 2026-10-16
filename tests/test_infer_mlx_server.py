"""Same-process capture contracts with an injectable native runtime."""

import json
import os
import signal
import sys
import time
from types import SimpleNamespace

import pytest

from stormlog.infer.mlx_evidence import load_evidence
from stormlog.infer.mlx_launchers import prepare_launcher, redacted_arguments
from stormlog.infer.mlx_server import capture_mlx_server, run_mlx_server
from tests.mlx_fakes import FakeCore, make_runtime


def test_embedding_same_pid_passive_sampling_and_primary_exception(tmp_path):
    core = FakeCore()
    with capture_mlx_server(
        engine="mlx-lm",
        run_id="qualified-run",
        output=tmp_path / "evidence",
        runtime=make_runtime(core),
    ) as capture:
        assert capture.pid == os.getpid()
        time.sleep(0.03)
    loaded = load_evidence(capture.directory)
    assert loaded.events
    assert all(event.pid == os.getpid() for event in loaded.events)
    assert all(event.job_id == "qualified-run" for event in loaded.events)
    assert loaded.manifest["capture"]["status"] == "complete"
    assert loaded.manifest["capture"]["sampling"] == "passive"
    assert not {"eval", "reset"} & set(
        call for call in core.calls if isinstance(call, str)
    )
    assert not any(isinstance(call, tuple) and call[0] == "sync" for call in core.calls)
    primary = RuntimeError("secret server exception with private arguments")
    with pytest.raises(RuntimeError) as raised:
        with capture_mlx_server(
            engine="mlx-lm",
            run_id="failed-run",
            output=tmp_path / "failure",
            runtime=make_runtime(core),
        ) as failed:
            raise primary
    assert raised.value is primary
    assert load_evidence(failed.directory).capture_status == "incomplete"
    for path in failed.directory.rglob("*.json"):
        assert "private arguments" not in path.read_text()


def test_launcher_restores_arguments_and_records_clean_metadata(tmp_path, monkeypatch):
    import importlib.metadata

    import stormlog.infer.mlx_launchers as launchers

    original_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.31.3" if name == "mlx-lm" else original_version(name),
    )
    original_import = launchers.importlib.import_module
    seen = []

    def main():
        seen.append((os.getpid(), list(sys.argv)))
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    monkeypatch.setattr(
        launchers.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(main=main)
            if name == "mlx_lm.server"
            else original_import(name)
        ),
    )
    argv = sys.argv
    handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    secret = "my-private-server-api-key"
    directory, outcome = run_mlx_server(
        engine="mlx-lm",
        run_id="test-run",
        output=tmp_path / "evidence",
        args=["--model", "/opaque/model", "--api-key", secret],
        runtime=make_runtime(),
    )
    assert outcome.exit_code == 0
    assert seen == [
        (
            os.getpid(),
            ["mlx_lm.server", "--model", "/opaque/model", "--api-key", secret],
        )
    ]
    assert sys.argv is argv
    assert all(
        signal.getsignal(signum) == handler for signum, handler in handlers.items()
    )
    manifest = load_evidence(directory).manifest
    assert manifest["engine"]["version_provenance"] == "installed_distribution_metadata"
    for path in directory.rglob("*.json"):
        assert secret not in path.read_text()


@pytest.mark.parametrize(
    "args", [["--reload"], ["--workers=2"], ["--pipeline"], ["--start-ui"]]
)
def test_multiprocess_flags_fail_before_capture(tmp_path, args):
    with pytest.raises(ValueError, match="same-process"):
        run_mlx_server(
            engine="mlx-lm",
            run_id="run",
            output=tmp_path / "evidence",
            args=args,
            runtime=make_runtime(),
        )
    assert not (tmp_path / "evidence").exists()


def test_unknown_versions_and_existing_omlx_settings_are_not_implicit(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "stormlog.infer.mlx_launchers.importlib.metadata.version", lambda _: "999"
    )
    with pytest.raises(ValueError, match="installed 999"):
        prepare_launcher("mlx-lm", [])
    with pytest.raises(ValueError, match="dedicated"):
        prepare_launcher("omlx", [])
    base = tmp_path / "base"
    base.mkdir()
    for settings in (
        {"cluster": {"enabled": True}},
        {"server": {"distributed_inference_enabled": True}},
    ):
        (base / "settings.json").write_text(json.dumps(settings))
        with pytest.raises(ValueError, match="Distributed"):
            prepare_launcher("omlx", ["--base-path", str(base)])


def test_arguments_redact_both_credential_forms_and_urls():
    clean = redacted_arguments(
        [
            "--api-key",
            "key-value",
            "--token=token-value",
            "--model",
            "http://user:pass@example/model?token=value",
            "--no-cache",
            "--port",
            "8100",
        ]
    )
    assert "key-value" not in " ".join(clean)
    assert "token-value" not in " ".join(clean)
    assert "pass" not in " ".join(clean) and "token=value" not in " ".join(clean)
    assert clean[-1] == "8100"


def test_artifact_finalization_cannot_replace_workload_failure(tmp_path, monkeypatch):
    from stormlog.infer import mlx_server

    primary = RuntimeError("primary")
    monkeypatch.setattr(
        mlx_server,
        "write_manifest",
        lambda *_: (_ for _ in ()).throw(OSError("finalization")),
    )
    with pytest.raises(RuntimeError) as raised:
        with capture_mlx_server(
            engine="mlx-lm",
            run_id="run",
            output=tmp_path / "evidence",
            runtime=make_runtime(),
        ):
            raise primary
    assert raised.value is primary
