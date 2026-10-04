"""Which model weights a vLLM server was started with."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from stormlog.infer.server_model import (
    GIT_SHA1,
    INFERRED,
    PINNED_COMMIT,
    POST_LAUNCH_DIGEST,
    SHA256,
    SIZE_ONLY,
    UNRESOLVED,
    LaunchArguments,
    describe_model,
    hub_cache_dir,
    launch_arguments,
    local_files,
    weights_digest,
)

COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40
REPO = "Qwen/Qwen2.5-0.5B-Instruct"
WEIGHTS = b"\x00\x01weights" * 100
CONFIG = b'{"architectures": ["Qwen2ForCausalLM"]}'
TOKENIZER_CONFIG = json.dumps({"chat_template": "{{ messages }}"}).encode()
GENERATION = json.dumps({"temperature": 0.7, "top_p": 0.8}).encode()


def _git_sha1(content: bytes) -> str:
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def _hub(tmp_path: Path, *, weights: bytes = WEIGHTS) -> Path:
    """A hub cache as huggingface_hub lays it out: links into named blobs."""
    cache = tmp_path / "hub"
    repo = cache / ("models--" + REPO.replace("/", "--"))
    (repo / "blobs").mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(COMMIT)
    snapshot = repo / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    files = {
        "model.safetensors": (weights, hashlib.sha256(weights).hexdigest()),
        "config.json": (CONFIG, _git_sha1(CONFIG)),
        "tokenizer_config.json": (TOKENIZER_CONFIG, _git_sha1(TOKENIZER_CONFIG)),
        "generation_config.json": (GENERATION, _git_sha1(GENERATION)),
    }
    for name, (content, blob) in files.items():
        (repo / "blobs" / blob).write_bytes(content)
        (snapshot / name).symlink_to(Path("../../blobs") / blob)
    return cache


def test_the_command_line_names_the_model_and_its_revision() -> None:
    launch = launch_arguments(
        [
            "/usr/bin/python3",
            "/usr/local/bin/vllm",
            "serve",
            REPO,
            "--revision",
            COMMIT,
            "--tokenizer-revision=" + COMMIT,
            "--max-num-seqs",
            "64",
        ]
    )
    assert launch.model == REPO
    assert launch.revision == COMMIT
    assert launch.tokenizer_revision == COMMIT
    assert launch.tokenizer is None
    module = launch_arguments(
        ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "/m"]
    )
    assert module.model == "/m"


def test_launch_values_lose_credentials_and_queries() -> None:
    secret = "hf_plantedSecret0123456789"
    launch = launch_arguments(
        [
            "vllm",
            "serve",
            f"https://user:{secret}@models.example/m",
            f"--revision=main?token={secret}",
            "--tokenizer",
            f" https://u:{secret}@h/t",
        ]
    )
    record = launch.to_record()
    assert secret not in str(record)
    # The path is replaced by its digest: a token can sit in it.
    assert record["model"] == "https://models.example/<sha256:4da12da337c2>"
    assert record["revision"] == "main?<redacted>"
    assert record["tokenizer"] == "https://h/<sha256:599c83a04aed>"


def test_inline_templates_that_differ_after_a_question_mark_digest_apart(
    tmp_path: Path,
) -> None:
    # The query rule for URLs cut an inline Jinja template at its first "?",
    # so two templates agreeing up to it shared a chat_template_digest.
    prefix = "{% for m in messages %}{{ m.content }}{% endfor %} Ready?"
    templates = [prefix + " Go.", prefix + " Stop."]
    digests = []
    for index, template in enumerate(templates):
        launch = launch_arguments(["vllm", "serve", REPO, "--chat-template", template])
        model = describe_model(launch, hub_cache=_hub(tmp_path / str(index)))
        digests.append(model["chat_template_digest"])
        # A template is no URL: its recorded copy is kept whole.
        assert launch.to_record()["chat_template"] == template
    assert digests[0] != digests[1]


def test_the_hub_cache_follows_the_servers_own_settings() -> None:
    assert hub_cache_dir({"HF_HUB_CACHE": "/c"}, None) == Path("/c")
    assert hub_cache_dir({"HF_HOME": "/h", "HOME": "/root"}, None) == Path("/h/hub")
    assert hub_cache_dir({"HOME": "/root"}, None) == Path(
        "/root/.cache/huggingface/hub"
    )
    assert hub_cache_dir({"HOME": "/root"}, "/models") == Path("/models")
    assert hub_cache_dir({}, None) is None
    # huggingface_hub honours its older name and the XDG cache home too.
    assert hub_cache_dir({"HUGGINGFACE_HUB_CACHE": "/old"}, None) == Path("/old")
    assert hub_cache_dir({"XDG_CACHE_HOME": "/x", "HOME": "/root"}, None) == Path(
        "/x/huggingface/hub"
    )


def test_a_mutable_revision_resolved_from_the_cache_is_inferred(tmp_path: Path) -> None:
    model = describe_model(LaunchArguments(model=REPO), hub_cache=_hub(tmp_path))

    assert model["identity_evidence"] == INFERRED
    assert model["revision_immutable"] is False
    assert model["resolved_snapshot"] == COMMIT
    weights = model["files"]["model.safetensors"]
    assert weights == {
        "algorithm": SHA256,
        "digest": hashlib.sha256(WEIGHTS).hexdigest(),
        "size": len(WEIGHTS),
        "checked": False,
    }
    assert model["files"]["config.json"]["algorithm"] == GIT_SHA1
    assert model["weights_digest"] is not None
    assert model["generation_config"] == {"temperature": 0.7, "top_p": 0.8}
    template = hashlib.sha256(b"{{ messages }}").hexdigest()
    assert model["chat_template_digest"] == template


def test_a_pinned_commit_is_named_as_such(tmp_path: Path) -> None:
    launch = LaunchArguments(model=REPO, revision=COMMIT)
    model = describe_model(launch, hub_cache=_hub(tmp_path))
    assert (model["identity_evidence"], model["revision_immutable"]) == (
        PINNED_COMMIT,
        True,
    )


def test_a_revision_missing_from_the_cache_is_unresolved(tmp_path: Path) -> None:
    launch = LaunchArguments(model=REPO, revision=OTHER_COMMIT)
    model = describe_model(launch, hub_cache=_hub(tmp_path))
    assert model["identity_evidence"] == UNRESOLVED
    assert model["files"] == {} and model["weights_digest"] is None


def _nested_hub(tmp_path: Path) -> Path:
    """A hub cache whose weights blob is itself a link into a shared store,
    as a deduplicated cache lays it out: blobs/<sha256> -> store/bb/<other>."""
    cache = _hub(tmp_path)
    blobs = cache / ("models--" + REPO.replace("/", "--")) / "blobs"
    blob = blobs / hashlib.sha256(WEIGHTS).hexdigest()
    store = cache / "blobs" / "bb"
    store.mkdir(parents=True)
    other = store / ("bb" + "5" * 62)
    other.write_bytes(blob.read_bytes())
    blob.unlink()
    blob.symlink_to(Path("../../blobs/bb") / other.name)
    return cache


@pytest.mark.parametrize("verify", [False, True])
def test_a_file_is_named_by_the_blob_its_snapshot_links_to(
    tmp_path: Path, verify: bool
) -> None:
    # Not by whatever file the blob itself leads to: on the A30 box the
    # weights' blob is a link into a shared store with another name.
    model = describe_model(
        LaunchArguments(model=REPO),
        hub_cache=_nested_hub(tmp_path),
        verify_blobs=verify,
    )
    weights = model["files"]["model.safetensors"]
    assert (weights["algorithm"], weights["digest"], weights["size"]) == (
        SHA256,
        hashlib.sha256(WEIGHTS).hexdigest(),
        len(WEIGHTS),
    )
    assert weights["checked"] is verify
    assert model["weights_digest"] is not None


def test_a_name_that_looks_like_a_digest_is_none_outside_the_blobs(
    tmp_path: Path,
) -> None:
    # Only a link into the repository's blobs vouches for a digest: a copied
    # file, or a link elsewhere, named like one has none.
    cache = _hub(tmp_path)
    snapshot = cache / ("models--" + REPO.replace("/", "--")) / "snapshots" / COMMIT
    digest_like = "d" * 64
    (snapshot / digest_like).write_bytes(b"copied")
    elsewhere = cache / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / ("e" * 64)).write_bytes(b"linked")
    (snapshot / "linked.bin").symlink_to(Path("../../../elsewhere") / ("e" * 64))
    files = describe_model(LaunchArguments(model=REPO), hub_cache=cache)["files"]
    assert (files[digest_like]["algorithm"], files[digest_like]["digest"]) == (
        "none",
        "",
    )
    assert (files["linked.bin"]["algorithm"], files["linked.bin"]["digest"]) == (
        "none",
        "",
    )


def test_verifying_blobs_hashes_their_content(tmp_path: Path) -> None:
    cache = _hub(tmp_path)
    model = describe_model(
        LaunchArguments(model=REPO), hub_cache=cache, verify_blobs=True
    )
    files = model["files"]
    assert files["model.safetensors"]["checked"] is True
    assert files["model.safetensors"]["digest"] == hashlib.sha256(WEIGHTS).hexdigest()
    assert files["config.json"]["digest"] == _git_sha1(CONFIG)


def _copied_hub(tmp_path: Path, weights: bytes) -> Path:
    """A hub cache where the snapshot holds copies, not links into blobs."""
    cache = _hub(tmp_path, weights=weights)
    snapshot = cache / ("models--" + REPO.replace("/", "--")) / "snapshots" / COMMIT
    for link in list(snapshot.iterdir()):
        content = link.read_bytes()
        link.unlink()
        link.write_bytes(content)
    return cache


def test_a_snapshot_of_copies_is_not_named_by_its_file_names(tmp_path: Path) -> None:
    # huggingface_hub copies files where links are unsupported; a file's name
    # is then no digest, and two weights of one size must not look alike.
    first = describe_model(
        LaunchArguments(model=REPO), hub_cache=_copied_hub(tmp_path / "1", WEIGHTS)
    )
    weights = first["files"]["model.safetensors"]
    assert (weights["algorithm"], weights["digest"]) == ("none", "")
    assert first["weights_digest"] is None
    hashed = [
        describe_model(
            LaunchArguments(model=REPO),
            hub_cache=_copied_hub(tmp_path / name, content),
            verify_blobs=True,
        )
        for name, content in (("3", WEIGHTS), ("4", WEIGHTS[::-1]))
    ]
    assert hashed[0]["files"]["model.safetensors"]["digest"] == (
        hashlib.sha256(WEIGHTS).hexdigest()
    )
    assert hashed[0]["weights_digest"] != hashed[1]["weights_digest"]


def test_verifying_records_what_a_damaged_blob_holds(tmp_path: Path) -> None:
    cache = _hub(tmp_path)
    repo = cache / ("models--" + REPO.replace("/", "--"))
    named = hashlib.sha256(WEIGHTS).hexdigest()
    (repo / "blobs" / named).write_bytes(b"corrupt")
    plain = describe_model(LaunchArguments(model=REPO), hub_cache=cache)
    checked = describe_model(
        LaunchArguments(model=REPO), hub_cache=cache, verify_blobs=True
    )
    assert plain["files"]["model.safetensors"]["digest"] == named
    assert checked["files"]["model.safetensors"]["digest"] == (
        hashlib.sha256(b"corrupt").hexdigest()
    )
    assert plain["weights_digest"] != checked["weights_digest"]


@pytest.mark.parametrize("damage", ["dangling", "loop"])
def test_a_damaged_snapshot_is_described_not_raised(
    tmp_path: Path, damage: str
) -> None:
    cache = _hub(tmp_path)
    snapshot = cache / ("models--" + REPO.replace("/", "--")) / "snapshots" / COMMIT
    broken = snapshot / "extra.bin"
    if damage == "dangling":
        broken.symlink_to(Path("../../blobs") / ("f" * 64))
    else:
        broken.symlink_to(broken)
    model = describe_model(LaunchArguments(model=REPO), hub_cache=cache)
    assert model["files"]["extra.bin"]["algorithm"] == "none"
    assert model["weights_digest"] is None


def test_different_weights_give_different_digests(tmp_path: Path) -> None:
    first = describe_model(LaunchArguments(model=REPO), hub_cache=_hub(tmp_path / "1"))
    second = describe_model(
        LaunchArguments(model=REPO),
        hub_cache=_hub(tmp_path / "2", weights=WEIGHTS[::-1]),
    )
    assert first["weights_digest"] != second["weights_digest"]


def _local(tmp_path: Path, weights: bytes) -> Path:
    directory = tmp_path / "model"
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(weights)
    (directory / "config.json").write_bytes(CONFIG)
    return directory


def test_same_size_local_weights_differ_only_when_hashed(tmp_path: Path) -> None:
    first = _local(tmp_path / "1", WEIGHTS)
    second = _local(tmp_path / "2", WEIGHTS[::-1])
    sized = [
        describe_model(LaunchArguments(model=str(path)), hub_cache=None)
        for path in (first, second)
    ]
    hashed = [
        describe_model(
            LaunchArguments(model=str(path)), hub_cache=None, hash_weights=True
        )
        for path in (first, second)
    ]
    assert [model["identity_evidence"] for model in sized] == [SIZE_ONLY, SIZE_ONLY]
    assert sized[0]["weights_digest"] is None
    assert [model["identity_evidence"] for model in hashed] == [
        POST_LAUNCH_DIGEST,
        POST_LAUNCH_DIGEST,
    ]
    assert hashed[0]["weights_digest"] != hashed[1]["weights_digest"]


def test_a_relative_model_path_resolves_from_the_servers_directory(
    tmp_path: Path,
) -> None:
    _local(tmp_path, WEIGHTS)
    model = describe_model(LaunchArguments(model="model"), hub_cache=None, cwd=tmp_path)
    assert set(model["files"]) == {"config.json", "model.safetensors"}


def test_local_digests_are_cached_by_path_size_mtime_and_inode(tmp_path: Path) -> None:
    directory = _local(tmp_path, WEIGHTS)
    cache = tmp_path / "digests.json"
    first = local_files(directory, hash_contents=True, cache=cache)
    stored = json.loads(cache.read_text())
    assert len(stored) == 2
    # A cached entry is used as is: plant a marker to see it come back.
    key = next(k for k in stored if k.split("|")[0].endswith("model.safetensors"))
    stored[key] = "cached-marker"
    cache.write_text(json.dumps(stored))
    second = local_files(directory, hash_contents=True, cache=cache)
    digests = {item.path: item.digest for item in second}
    assert digests["model.safetensors"] == "cached-marker"
    assert {item.path for item in first} == set(digests)


def test_a_cached_digest_is_not_used_after_an_in_place_rewrite(tmp_path: Path) -> None:
    # rsync --inplace -t or cp --preserve keep the size and the mtime; the
    # change time still moves.
    directory = _local(tmp_path, WEIGHTS)
    cache = tmp_path / "digests.json"
    local_files(directory, hash_contents=True, cache=cache)
    weights = directory / "model.safetensors"
    stat = weights.stat()
    time.sleep(0.01)
    weights.write_bytes(WEIGHTS[::-1])
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    digests = {
        item.path: item.digest
        for item in local_files(directory, hash_contents=True, cache=cache)
    }
    assert digests["model.safetensors"] == hashlib.sha256(WEIGHTS[::-1]).hexdigest()


def test_the_weights_digest_needs_every_file_named() -> None:
    assert weights_digest([]) is None


@pytest.mark.parametrize(
    ("template", "expected"),
    [("{{ inline }}", "{{ inline }}"), ("chat.jinja", "{{ from file }}")],
)
def test_a_chat_template_flag_overrides_the_snapshots(
    tmp_path: Path, template: str, expected: str
) -> None:
    (tmp_path / "chat.jinja").write_text("{{ from file }}")
    model = describe_model(
        LaunchArguments(model=REPO, chat_template=template),
        hub_cache=_hub(tmp_path),
        cwd=tmp_path,
    )
    assert (
        model["chat_template_digest"] == hashlib.sha256(expected.encode()).hexdigest()
    )
