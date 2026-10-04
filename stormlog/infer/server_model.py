"""Which model weights a vLLM server was started with, as far as can be shown.

The server's command line names a model and a revision. For a Hugging Face
repository, the hub cache (``models--org--name/snapshots/<commit>/``) maps
each file to a blob named by its own digest: SHA-256 for a file stored in
LFS, git's SHA-1 for a small one. For a local directory there is no such
name, so each file's SHA-256 is computed (``hash_weights``) and cached by
path, size, ``mtime_ns`` and inode.

None of this can show what a server loaded earlier: the cache or the
directory may have changed since. So a description only names its
evidence (``pinned_commit``, ``inferred``, ``post_launch_digest`` or
``size_only``); identity is verified only for a launch that the experiment
runner controlled.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .server_privacy import scrub_argument

SHA256 = "sha256"
GIT_SHA1 = "git-sha1"
PINNED_COMMIT = "pinned_commit"
INFERRED = "inferred"
POST_LAUNCH_DIGEST = "post_launch_digest"
SIZE_ONLY = "size_only"
UNRESOLVED = "unresolved"

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK = 8 * 1024 * 1024
# vLLM 0.30.0 options that name what is loaded, as --flag VALUE or --flag=VALUE.
_VALUE_OPTIONS = {
    "--model": "model",
    "--revision": "revision",
    "--tokenizer": "tokenizer",
    "--tokenizer-revision": "tokenizer_revision",
    "--chat-template": "chat_template",
    "--served-model-name": "served_model_name",
    "--download-dir": "download_dir",
}


@dataclass(frozen=True)
class LaunchArguments:
    """What the server's command line says it loads."""

    model: str | None = None
    revision: str | None = None
    tokenizer: str | None = None
    tokenizer_revision: str | None = None
    chat_template: str | None = None
    served_model_name: str | None = None
    download_dir: str | None = None

    def to_record(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ModelFile:
    """One file of the model, named by a digest of its content."""

    path: str
    algorithm: str
    digest: str
    size: int
    checked: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "digest": self.digest,
            "size": self.size,
            "checked": self.checked,
        }


def launch_arguments(cmdline: Sequence[str]) -> LaunchArguments:
    """``vllm serve MODEL --revision R ...``, or the server module's flags."""
    values: dict[str, str] = {}
    arguments = list(cmdline)
    if "serve" in arguments:
        rest = arguments[arguments.index("serve") + 1 :]
        if rest and not rest[0].startswith("-"):
            values["model"] = rest[0]
    for index, argument in enumerate(arguments):
        name, separator, value = argument.partition("=")
        key = _VALUE_OPTIONS.get(name)
        if key is None:
            continue
        if not separator and index + 1 < len(arguments):
            value = arguments[index + 1]
        if value:
            values.setdefault(key, value)
    return LaunchArguments(**{key: scrub_argument(v) for key, v in values.items()})


def hub_cache_dir(environ: Mapping[str, str], download_dir: str | None) -> Path | None:
    """Where the server's Hugging Face cache is, by the server's own settings."""
    if download_dir:
        return Path(download_dir)
    # huggingface_hub's order: its cache, its older name, its home, XDG's.
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if environ.get(name):
            return Path(environ[name])
    if environ.get("HF_HOME"):
        return Path(environ["HF_HOME"]) / "hub"
    if environ.get("XDG_CACHE_HOME"):
        return Path(environ["XDG_CACHE_HOME"]) / "huggingface" / "hub"
    if environ.get("HOME"):
        return Path(environ["HOME"]) / ".cache" / "huggingface" / "hub"
    return None


def describe_model(
    launch: LaunchArguments,
    *,
    hub_cache: Path | None,
    cwd: Path | None = None,
    hash_weights: bool = False,
    verify_blobs: bool = False,
    digest_cache: Path | None = None,
) -> dict[str, Any]:
    """The model's files and digests, and what evidence they are.

    ``cwd`` is the server's working directory, against which a relative
    model path on its command line resolves.
    """
    model = launch.model
    launch = _resolved_template(launch, cwd)
    record: dict[str, Any] = {
        "configured": model,
        "configured_revision": launch.revision,
        "revision_immutable": bool(launch.revision and _COMMIT.match(launch.revision)),
    }
    local = _local_directory(model, cwd)
    if local is not None:
        files = local_files(local, hash_contents=hash_weights, cache=digest_cache)
        evidence = POST_LAUNCH_DIGEST if hash_weights else SIZE_ONLY
        return {**record, **_files_record(None, files, evidence, local, launch)}
    snapshot = (
        hub_snapshot(hub_cache, model, launch.revision)
        if model and hub_cache is not None
        else None
    )
    if snapshot is None:
        return {**record, **_files_record(None, [], UNRESOLVED, None, launch)}
    directory, commit = snapshot
    files = snapshot_files(directory, verify=verify_blobs)
    evidence = PINNED_COMMIT if record["revision_immutable"] else INFERRED
    return {**record, **_files_record(commit, files, evidence, directory, launch)}


def hub_snapshot(
    cache: Path, repo_id: str, revision: str | None
) -> tuple[Path, str] | None:
    """The snapshot directory and commit a revision resolves to in the cache."""
    repo = cache / ("models--" + repo_id.replace("/", "--"))
    wanted = revision or "main"
    commit = wanted if _COMMIT.match(wanted) else _read_ref(repo, wanted)
    if commit is None:
        return None
    directory = repo / "snapshots" / commit
    return (directory, commit) if directory.is_dir() else None


def snapshot_files(directory: Path, *, verify: bool = False) -> list[ModelFile]:
    """Each file of a hub snapshot, named by the blob its link points to.

    Only a link into the repository's ``blobs`` whose name is a digest is
    named by it. huggingface_hub copies files where links are unsupported,
    and a copy's name is no digest: it has none unless ``verify`` hashes it.
    A link that leads nowhere, or in a loop, has none either.
    """
    files = []
    for path in _walk(directory):
        relative = path.relative_to(directory).as_posix()
        try:
            blob = path.resolve(strict=True)
            size = blob.stat().st_size
        except (OSError, RuntimeError):
            files.append(ModelFile(relative, "none", "", 0, checked=False))
            continue
        algorithm, digest = _blob_name(path, blob)
        if verify:
            algorithm = algorithm if algorithm != "none" else SHA256
            digest = _sha256(blob) if algorithm == SHA256 else _git_sha1(blob)
        files.append(ModelFile(relative, algorithm, digest, size, checked=verify))
    return files


def _blob_name(path: Path, blob: Path) -> tuple[str, str]:
    """The digest a blob's name gives, when the file is a link to a blob."""
    if not path.is_symlink() or blob.parent.name != "blobs":
        return "none", ""
    if _SHA256.match(blob.name):
        return SHA256, blob.name
    if _COMMIT.match(blob.name):
        return GIT_SHA1, blob.name
    return "none", ""


def local_files(
    directory: Path, *, hash_contents: bool, cache: Path | None = None
) -> list[ModelFile]:
    """Each file of a local model directory, by size or by SHA-256."""
    known = _load_cache(cache) if hash_contents else {}
    files = []
    for path in _walk(directory):
        relative = path.relative_to(directory).as_posix()
        try:
            stat = path.stat()
            digest = _cached_sha256(path, stat, known) if hash_contents else ""
        except (OSError, RuntimeError):
            files.append(ModelFile(relative, "none", "", 0, checked=False))
            continue
        algorithm = SHA256 if hash_contents else "none"
        files.append(
            ModelFile(relative, algorithm, digest, stat.st_size, checked=hash_contents)
        )
    if hash_contents and cache is not None:
        _save_cache(cache, known)
    return files


def _cached_sha256(path: Path, stat: os.stat_result, known: dict[str, str]) -> str:
    """A file's SHA-256, reused while its path, size, times and inode hold.

    The change time is part of the key: an in-place rewrite that restores
    the size and modification time (``rsync --inplace -t``) still moves it.
    """
    key = (
        f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{stat.st_ctime_ns}"
        f"|{stat.st_ino}"
    )
    digest = known.get(key) or _sha256(path)
    known[key] = digest
    return digest


def weights_digest(files: Sequence[ModelFile]) -> str | None:
    """SHA-256 over the sorted file list: path, algorithm, digest and size."""
    if not files or any(not item.digest for item in files):
        return None
    lines = [
        f"{item.path}\0{item.algorithm}\0{item.digest}\0{item.size}\n"
        for item in sorted(files, key=lambda item: item.path)
    ]
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def chat_template_digest(
    directory: Path | None, chat_template: str | None
) -> str | None:
    """SHA-256 of the chat template the server uses, when it can be found.

    ``--chat-template`` names a file or gives the template inline; otherwise
    it is the snapshot's ``chat_template.jinja`` or the ``chat_template`` of
    its ``tokenizer_config.json``.
    """
    template = _template_text(directory, chat_template)
    return None if template is None else hashlib.sha256(template.encode()).hexdigest()


def generation_config(directory: Path | None) -> dict[str, Any] | None:
    """The snapshot's ``generation_config.json``: the sampling defaults."""
    if directory is None:
        return None
    loaded = _read_json(directory / "generation_config.json")
    return loaded if isinstance(loaded, dict) else None


def _files_record(
    commit: str | None,
    files: list[ModelFile],
    evidence: str,
    directory: Path | None,
    launch: LaunchArguments,
) -> dict[str, Any]:
    return {
        "resolved_snapshot": commit,
        "files": {item.path: item.to_record() for item in files},
        "weights_digest": weights_digest(files),
        "identity_evidence": evidence,
        "chat_template_digest": chat_template_digest(directory, launch.chat_template),
        "generation_config": generation_config(directory),
    }


def _resolved_template(launch: LaunchArguments, cwd: Path | None) -> LaunchArguments:
    """A relative ``--chat-template`` file, found from the server's cwd."""
    template = launch.chat_template
    if not template or cwd is None or Path(template).is_absolute():
        return launch
    candidate = cwd / template
    return (
        replace(launch, chat_template=str(candidate)) if candidate.is_file() else launch
    )


def _local_directory(model: str | None, cwd: Path | None) -> Path | None:
    if not model:
        return None
    path = Path(model)
    if not path.is_absolute() and cwd is not None:
        path = cwd / path
    return path if path.is_dir() else None


def _template_text(directory: Path | None, chat_template: str | None) -> str | None:
    if chat_template:
        path = Path(chat_template)
        try:
            return path.read_text() if path.is_file() else chat_template
        except OSError:
            return chat_template
    if directory is None:
        return None
    try:
        return (directory / "chat_template.jinja").read_text()
    except OSError:
        pass
    config = _read_json(directory / "tokenizer_config.json")
    template = config.get("chat_template") if isinstance(config, dict) else None
    if isinstance(template, str):
        return template
    return json.dumps(template, sort_keys=True) if template is not None else None


def _read_ref(repo: Path, name: str) -> str | None:
    try:
        commit = (repo / "refs" / name).read_text().strip()
    except OSError:
        return None
    return commit if _COMMIT.match(commit) else None


def _walk(directory: Path) -> Iterator[Path]:
    for root, _dirs, names in sorted(os.walk(directory)):
        for name in sorted(names):
            yield Path(root) / name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha1(path: Path) -> str:
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())  # noqa: S324
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _load_cache(path: Path | None) -> dict[str, str]:
    loaded = _read_json(path) if path is not None else None
    if not isinstance(loaded, dict):
        return {}
    return {str(key): str(value) for key, value in loaded.items()}


def _save_cache(path: Path, known: Mapping[str, str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(sorted(known.items())), indent=0) + "\n")
    except OSError:
        return


__all__ = [
    "GIT_SHA1",
    "INFERRED",
    "PINNED_COMMIT",
    "POST_LAUNCH_DIGEST",
    "SHA256",
    "SIZE_ONLY",
    "UNRESOLVED",
    "LaunchArguments",
    "ModelFile",
    "chat_template_digest",
    "describe_model",
    "generation_config",
    "hub_cache_dir",
    "hub_snapshot",
    "launch_arguments",
    "local_files",
    "snapshot_files",
    "weights_digest",
]
