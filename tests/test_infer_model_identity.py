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
TOKENIZER = b'{"model": {"type": "BPE"}}'


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
        "tokenizer.json": (TOKENIZER, _git_sha1(TOKENIZER)),
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
    # rev-213-a's E3, as the lead ruled: without the commit's file list,
    # recorded when it was pinned, the evidence does not claim the commit.
    assert record["identity_evidence"] == "pinned_snapshot_verified"
    assert record["files"]["model.safetensors"]["algorithm"] == "sha256"
    assert record["files"]["config.json"]["algorithm"] == "git-sha1"
    assert all(item["checked"] for item in record["files"].values())
    assert record["weights_digest"] is not None
    assert changed_files(model) == []


def test_a_blob_whose_content_is_not_its_name_is_refused(tmp_path: Path) -> None:
    cache = _hub(tmp_path, weights=WEIGHTS[::-1])
    with pytest.raises(InferInputError, match="does not match its blob"):
        prepare_model({"route": "pinned_hub", "repo": REPO, "hub_cache": str(cache)})


def _deduplicated(cache: Path) -> Path:
    """The weights' blob made a link into a shared store under another
    name, as the A30 box's hub cache has it."""
    blob = (
        cache
        / ("models--" + REPO.replace("/", "--"))
        / "blobs"
        / hashlib.sha256(WEIGHTS).hexdigest()
    )
    store = cache / "blobs" / "bb"
    store.mkdir(parents=True)
    other = store / ("bb" + "5" * 62)
    other.write_bytes(blob.read_bytes())
    blob.unlink()
    blob.symlink_to(Path("../../blobs/bb") / other.name)
    return cache


def test_a_blob_that_links_on_to_a_shared_store_is_checked_by_its_own_name(
    tmp_path: Path,
) -> None:
    cache = _deduplicated(_hub(tmp_path))
    model = prepare_model(
        {"route": "pinned_hub", "repo": REPO, "hub_cache": str(cache)}
    )
    weights = model.record()["files"]["model.safetensors"]
    assert weights["digest"] == hashlib.sha256(WEIGHTS).hexdigest()
    assert weights["checked"] is True


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
    (source / "tokenizer.json").write_bytes(TOKENIZER)
    store = tmp_path / "store"
    model = prepare_model(
        {"route": "staged", "source": str(source), "store": str(store)}
    )

    digest = model.record()["weights_digest"]
    assert model.directory == store / digest
    # rev-213-a's E5: a hard link shared the source's inode, so locking the
    # store locked the user's source, and an edit to one changed the other.
    stored = model.directory / "model.safetensors"
    assert stored.stat().st_ino != (source / "model.safetensors").stat().st_ino
    assert (source / "config.json").stat().st_mode & stat.S_IWUSR
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


def test_a_staged_hub_snapshot_stores_its_files_not_its_links(
    tmp_path: Path,
) -> None:
    # A snapshot's relative links would dangle in the store.
    cache = _deduplicated(_hub(tmp_path))
    snapshot = cache / ("models--" + REPO.replace("/", "--")) / "snapshots" / COMMIT
    model = prepare_model(
        {"route": "staged", "source": str(snapshot), "store": str(tmp_path / "s")}
    )
    stored = model.directory / "model.safetensors"
    assert not stored.is_symlink()
    assert stored.read_bytes() == WEIGHTS


def _repo(cache: Path) -> Path:
    return cache / ("models--" + REPO.replace("/", "--"))


def test_a_dangling_snapshot_link_is_refused_not_a_crash(tmp_path: Path) -> None:
    # rev-213-a's E4: FileNotFoundError, exit 1, before any server started.
    cache = _hub(tmp_path)
    (_repo(cache) / "blobs" / hashlib.sha256(WEIGHTS).hexdigest()).unlink()
    with pytest.raises(InferInputError, match="model.safetensors: its blob is missing"):
        prepare_model({"route": "pinned_hub", "repo": REPO, "hub_cache": str(cache)})
    snapshot = _repo(cache) / "snapshots" / COMMIT
    with pytest.raises(InferInputError, match="model.safetensors: cannot be read"):
        prepare_model(
            {"route": "staged", "source": str(snapshot), "store": str(tmp_path / "s")}
        )


def _add(cache: Path, name: str, content: bytes) -> None:
    """Another file of the commit, stored as the hub stores a small file."""
    blob = _git_sha1(content)
    (_repo(cache) / "blobs" / blob).write_bytes(content)
    (_repo(cache) / "snapshots" / COMMIT / name).symlink_to(Path("../../blobs") / blob)


def test_a_snapshot_link_into_another_directorys_blobs_is_refused(
    tmp_path: Path,
) -> None:
    # rev-213-a's E3: any directory named blobs passed.
    cache = _hub(tmp_path)
    elsewhere = tmp_path / "elsewhere" / "blobs"
    elsewhere.mkdir(parents=True)
    blob = hashlib.sha256(WEIGHTS).hexdigest()
    (elsewhere / blob).write_bytes(WEIGHTS)
    link = _repo(cache) / "snapshots" / COMMIT / "model.safetensors"
    link.unlink()
    link.symlink_to(Path(os.path.relpath(elsewhere / blob, link.parent)))
    with pytest.raises(InferInputError, match="not a link into its repository's"):
        prepare_model({"route": "pinned_hub", "repo": REPO, "hub_cache": str(cache)})


def test_a_snapshot_missing_what_a_load_reads_is_refused(tmp_path: Path) -> None:
    # A file of the commit gone from the snapshot verified as the commit.
    cache = _hub(tmp_path)
    index = (
        b'{"weight_map": {"a": "model.safetensors", '
        b'"b": "model-00002-of-00002.safetensors"}}'
    )
    _add(cache, "model.safetensors.index.json", index)
    with pytest.raises(
        InferInputError, match="model-00002-of-00002.safetensors, which its index"
    ):
        prepare_model({"route": "pinned_hub", "repo": REPO, "hub_cache": str(cache)})

    cache = _hub(tmp_path / "again")
    (_repo(cache) / "snapshots" / COMMIT / "config.json").unlink()
    with pytest.raises(InferInputError, match="no config.json"):
        prepare_model({"route": "pinned_hub", "repo": REPO, "hub_cache": str(cache)})


def _rewrite_in_place(path: Path, content: bytes) -> None:
    """New bytes in the same inode, with the size and times put back."""
    before = path.stat()
    os.chmod(path, 0o644)
    with path.open("r+b") as handle:
        handle.write(content)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def test_a_file_rewritten_in_place_with_its_size_and_time_kept_is_caught(
    tmp_path: Path,
) -> None:
    # fable-213's P3: the post-run check was by size, time and inode only.
    cache = _hub(tmp_path)
    model = prepare_model(
        {"route": "pinned_hub", "repo": REPO, "hub_cache": str(cache)}
    )
    blob = _repo(cache) / "blobs" / hashlib.sha256(WEIGHTS).hexdigest()
    _rewrite_in_place(blob, WEIGHTS[::-1])
    assert changed_files(model) == ["model.safetensors"]

    source = tmp_path / "model"
    source.mkdir()
    (source / "model.safetensors").write_bytes(WEIGHTS)
    (source / "config.json").write_bytes(CONFIG)
    (source / "tokenizer.json").write_bytes(TOKENIZER)
    staged = prepare_model(
        {"route": "staged", "source": str(source), "store": str(tmp_path / "s")}
    )
    _rewrite_in_place(staged.directory / "config.json", CONFIG[::-1])
    assert changed_files(staged) == ["config.json"]
    # A file added after verification is a change too.
    os.chmod(staged.directory, 0o755)
    (staged.directory / "extra.safetensors").write_bytes(b"x")
    assert changed_files(staged) == ["config.json", "extra.safetensors"]


def test_a_staged_store_whose_content_changed_is_refused_when_reused(
    tmp_path: Path,
) -> None:
    # rev-213-a's mutant r9: nothing failed when a reused store went unhashed.
    source = tmp_path / "model"
    source.mkdir()
    (source / "model.safetensors").write_bytes(WEIGHTS)
    (source / "config.json").write_bytes(CONFIG)
    (source / "tokenizer.json").write_bytes(TOKENIZER)
    spec = {"route": "staged", "source": str(source), "store": str(tmp_path / "s")}
    staged = prepare_model(spec)
    _rewrite_in_place(staged.directory / "model.safetensors", WEIGHTS[::-1])
    with pytest.raises(InferInputError, match="model.safetensors changed"):
        prepare_model(spec)


def test_a_pin_with_the_commits_file_list_verifies_the_commit(
    tmp_path: Path,
) -> None:
    cache = _hub(tmp_path)
    spec = {"route": "pinned_hub", "repo": REPO, "hub_cache": str(cache)}
    listed = ["config.json", "model.safetensors", "tokenizer.json"]
    model = prepare_model({**spec, "files": listed})
    assert model.record()["identity_evidence"] == "pinned_commit_verified"
    with pytest.raises(InferInputError, match="README.md, which the pin lists"):
        prepare_model({**spec, "files": [*listed, "README.md"]})


def test_a_snapshot_without_a_tokenizer_is_refused(tmp_path: Path) -> None:
    cache = _hub(tmp_path)
    (_repo(cache) / "snapshots" / COMMIT / "tokenizer.json").unlink()
    with pytest.raises(InferInputError, match="no tokenizer"):
        prepare_model({"route": "pinned_hub", "repo": REPO, "hub_cache": str(cache)})


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
