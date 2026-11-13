"""The frontend's static files, served from package data (design §2.3).

Only the files ``static/manifest.json`` lists are served, by exact name:
the server lists no directory and joins no path from a request. ``/``
serves ``index.html``; ``/static/<name>`` serves ``<name>``. Each file is
read once, at start, through :mod:`importlib.resources`, so a wheel install
and a source checkout behave the same.

Caching: API responses carry ``Cache-Control: no-store``. Static files
carry an ETag (the SHA-256 of their bytes) and ``Cache-Control: no-cache``,
so the browser may keep a copy but revalidates it on every load and never
runs a page from an older install against a newer API. A matching
``If-None-Match`` gets 304.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from typing import Any, Protocol, cast

from stormlog.studio.contracts import HttpResponse


class Traversable(Protocol):
    """What this module reads of ``importlib.resources``' Traversable.

    Its home moved from ``importlib.abc`` (3.10) to
    ``importlib.resources.abc`` (3.11), so the methods used are named here.
    """

    def joinpath(self, *descendants: str) -> Traversable: ...

    def read_bytes(self) -> bytes: ...

    def read_text(self, encoding: str | None = None) -> str: ...


__all__ = [
    "MANIFEST",
    "MANIFEST_FORMAT",
    "STATIC_PREFIX",
    "CONTENT_SECURITY_POLICY",
    "StaticFile",
    "StaticFiles",
]

MANIFEST = "manifest.json"
MANIFEST_FORMAT = "stormlog.studio.static_manifest"
STATIC_PREFIX = "/static/"
# §7.5: no inline script or style, no framing, no form posts.
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'self'; style-src 'self'; "
    "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
_INDEX = "index.html"
# A flat directory plus vendor/, as pyproject's package-data patterns ship.
_NAME = re.compile(r"(vendor/)?[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class StaticFile:
    """One manifest entry, with its bytes and ETag."""

    name: str
    content_type: str
    body: bytes
    etag: str


class StaticFiles:
    """The manifest's files, loaded once from ``stormlog.studio/static``."""

    def __init__(self, directory: Traversable | None = None) -> None:
        base = directory if directory is not None else _package_directory()
        manifest = json.loads(base.joinpath(MANIFEST).read_text(encoding="utf-8"))
        self.files: Mapping[str, StaticFile] = _load(base, manifest)
        if _INDEX not in self.files:
            raise ValueError(f"the static manifest does not list {_INDEX}")

    def lookup(self, path: str) -> StaticFile | None:
        """The file a request path names, or None."""
        if path == "/":
            return self.files[_INDEX]
        if path.startswith(STATIC_PREFIX):
            return self.files.get(path[len(STATIC_PREFIX) :])
        return None

    def respond(self, file: StaticFile, if_none_match: str | None) -> HttpResponse:
        """200 with the file, or 304 when ``If-None-Match`` names its ETag."""
        headers = (
            ("ETag", file.etag),
            ("Cache-Control", "no-cache"),
            ("Content-Security-Policy", CONTENT_SECURITY_POLICY),
        )
        if if_none_match is not None and _matches(if_none_match, file.etag):
            return HttpResponse(304, file.content_type, b"", headers)
        return HttpResponse(200, file.content_type, file.body, headers)


def _package_directory() -> Traversable:
    return cast(Traversable, resources.files("stormlog.studio").joinpath("static"))


def _load(base: Traversable, manifest: Any) -> dict[str, StaticFile]:
    if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"the static manifest is not a {MANIFEST_FORMAT}")
    loaded: dict[str, StaticFile] = {}
    for name, entry in dict(manifest.get("files") or {}).items():
        if not _NAME.fullmatch(name) or name == MANIFEST:
            raise ValueError(f"the static manifest lists a bad name: {name!r}")
        body = base.joinpath(*name.split("/")).read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        loaded[name] = StaticFile(
            name, str(entry["content_type"]), body, f'"sha256-{digest}"'
        )
    return loaded


def _matches(if_none_match: str, etag: str) -> bool:
    candidates = [item.strip() for item in if_none_match.split(",")]
    weak = "W/" + etag
    return any(item in ("*", etag, weak) for item in candidates)
