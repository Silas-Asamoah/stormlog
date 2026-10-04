"""Model weights fixed and verified before the runner launches a server."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest

from stormlog.infer.describe_server import DescribeOptions, describe_server
from stormlog.infer.errors import InferInputError
from stormlog.infer.model_identity import changed_files, prepare_model
from tests.infer_proc_helpers import fake_server

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
COMMIT = "c" * 40
WEIGHTS = b"\x00weights" * 200
CONFIG = b'{"architectures": ["Qwen2ForCausalLM"]}'


def _git_sha1(content: bytes) -> str:
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def _hub(root: Path, weights: bytes = WEIGHTS) -> Path:
    cache = root / "hub"
    repo = cache / ("models--" + REPO.replace("/", "--"))
    (repo / "blobs").mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(COMMIT)
    snapshot = repo / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    blobs = {
        "model.safetensors": (WEIGHTS, hashlib.sha256(WEIGHTS).hexdigest()),
        "config.json": (CONFIG, _git_sha1(CONFIG)),
    }
    for name, (content, blob) in blobs.items():
        # The blob is named by the true content's digest; `weights` may differ.
        stored = weights if name == "model.safetensors" else content
        (repo / "blobs" / blob).write_bytes(stored)
        (snapshot / name).symlink_to(Path("../../blobs") / blob)
    return cache


def test_a_pinned_snapshot_is_checked_and_the_server_pointed_at_its_commit(
    tmp_path: Path,
) -> None:
    model = prepare_model(
        {"route": "pinned_hub", "repo": REPO, "hub_cache": str(_hub(tmp_path))}
    )

    assert model.commit == COMMIT
    assert model.server_args == ("--revision", COMMIT, "--tokenizer-revision", COMMIT)
    assert model.env["HF_HUB_OFFLINE"] == "1"
    record = model.record()
    assert record["identity_evidence"] == "pinned_commit_verified"
    assert record["files"]["model.safetensors"]["algorithm"] == "sha256"
    assert record["files"]["config.json"]["algorithm"] == "git-sha1"
    assert all(item["checked"] for item in record["files"].values())
    assert record["weights_digest"] is not None
    assert changed_files(model) == []


def test_a_blob_whose_content_is_not_its_name_is_refused(tmp_path: Path) -> None:
    cache = _hub(tmp_path, weights=WEIGHTS[::-1])
    with pytest.raises(InferInputError, match="does not match its blob"):
        prepare_model({"route": "pinned_hub", "repo": REPO, "hub_cache": str(cache)})


def test_a_revision_not_in_the_cache_is_refused(tmp_path: Path) -> None:
    with pytest.raises(InferInputError, match="no snapshot"):
        prepare_model(
            {
                "route": "pinned_hub",
                "repo": REPO,
                "revision": "d" * 40,
                "hub_cache": str(_hub(tmp_path)),
            }
        )


def test_a_staged_snapshot_is_named_by_its_content_and_read_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model"
    source.mkdir()
    (source / "model.safetensors").write_bytes(WEIGHTS)
    (source / "config.json").write_bytes(CONFIG)
    store = tmp_path / "store"
    model = prepare_model(
        {"route": "staged", "source": str(source), "store": str(store)}
    )

    digest = model.record()["weights_digest"]
    assert model.directory == store / digest
    assert model.model == str(store / digest)
    assert model.record()["identity_evidence"] == "staged_snapshot_verified"
    mode = (model.directory / "config.json").stat().st_mode
    assert not mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    # Staging again reuses the directory, after checking it.
    assert (
        prepare_model(
            {"route": "staged", "source": str(source), "store": str(store)}
        ).directory
        == model.directory
    )
    # A file that moves after verification is caught.
    target = model.directory / "config.json"
    os.chmod(model.directory, 0o755)
    os.chmod(target, 0o644)
    target.write_bytes(b"{}")
    assert changed_files(model) == ["config.json"]


def test_an_unknown_route_is_refused() -> None:
    with pytest.raises(InferInputError, match="route"):
        prepare_model({"route": "trust_me"})


def test_a_runner_description_carries_the_verified_identity_bound_to_its_server(
    tmp_path: Path,
) -> None:
    proc = fake_server(tmp_path)
    identity = {
        "identity_evidence": "pinned_commit_verified",
        "weights_digest": "w" * 64,
    }
    document = describe_server(
        DescribeOptions(
            pid=100, proc=proc, no_gpu=True, python="none", model_identity=identity
        )
    )
    assert document["model"]["identity_evidence"] == "pinned_commit_verified"
    assert document["model"]["bound_to"] == {"pid": 100, "start_ticks": 500}
