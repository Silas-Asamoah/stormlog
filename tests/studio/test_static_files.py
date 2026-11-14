"""static_files.py: only manifest files, ETags and no-cache (§2.3, §7.5)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from stormlog.studio.static_files import (
    CONTENT_SECURITY_POLICY,
    MANIFEST_FORMAT,
    StaticFiles,
)

PACKAGE_STATIC = Path(__file__).resolve().parents[2] / "stormlog" / "studio" / "static"


def write_tree(directory: Path, files: dict[str, Any], **contents: bytes) -> Path:
    manifest = {"format": MANIFEST_FORMAT, "schema_version": 1, "files": files}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    for name, body in contents.items():
        path = directory / name.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return directory


HTML = {"content_type": "text/html; charset=utf-8"}
JS = {"content_type": "text/javascript; charset=utf-8"}


def test_the_packaged_manifest_lists_exactly_the_shipped_files() -> None:
    manifest = json.loads((PACKAGE_STATIC / "manifest.json").read_text())
    assert manifest["format"] == MANIFEST_FORMAT
    shipped = {
        path.relative_to(PACKAGE_STATIC).as_posix()
        for path in PACKAGE_STATIC.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert set(manifest["files"]) == shipped
    static = StaticFiles()
    assert set(static.files) == shipped


def test_index_has_no_inline_script_or_style() -> None:
    page = (PACKAGE_STATIC / "index.html").read_text()
    assert "<script" not in page or 'src="' in page
    assert "<style" not in page
    assert " style=" not in page


def test_root_serves_index_and_static_serves_listed_names(tmp_path: Path) -> None:
    tree = write_tree(
        tmp_path,
        {"index.html": HTML, "app.js": JS, "vendor/preact.js": JS},
        **{"index.html": b"<p>", "app.js": b"1", "vendor__preact.js": b"2"},
    )
    static = StaticFiles(tree)
    root = static.lookup("/")
    assert root is not None and root.body == b"<p>"
    app = static.lookup("/static/app.js")
    assert app is not None and app.content_type == JS["content_type"]
    vendor = static.lookup("/static/vendor/preact.js")
    assert vendor is not None and vendor.body == b"2"
    for path in (
        "/static/manifest.json",
        "/static/../cli.py",
        "/static/vendor/../app.js",
        "/static/",
        "/index.html",
        "/static/missing.js",
        "/app.js",
    ):
        assert static.lookup(path) is None, path


def test_files_not_in_the_manifest_are_never_served(tmp_path: Path) -> None:
    tree = write_tree(
        tmp_path, {"index.html": HTML}, **{"index.html": b"x", "secret.js": b"s"}
    )
    assert StaticFiles(tree).lookup("/static/secret.js") is None


@pytest.mark.parametrize(
    "name", ["../x.js", "/abs.js", "a/b/c.js", ".hidden", "manifest.json", "x y.js"]
)
def test_bad_manifest_names_are_refused(tmp_path: Path, name: str) -> None:
    tree = write_tree(tmp_path, {"index.html": HTML, name: JS}, **{"index.html": b""})
    with pytest.raises(ValueError):
        StaticFiles(tree)


def test_a_manifest_without_index_or_format_is_refused(tmp_path: Path) -> None:
    tree = write_tree(tmp_path, {"app.js": JS}, **{"app.js": b""})
    with pytest.raises(ValueError):
        StaticFiles(tree)
    (tmp_path / "manifest.json").write_text('{"files": {}}')
    with pytest.raises(ValueError):
        StaticFiles(tmp_path)


def test_etag_is_the_sha256_and_revalidation_gives_304(tmp_path: Path) -> None:
    tree = write_tree(tmp_path, {"index.html": HTML}, **{"index.html": b"hello"})
    static = StaticFiles(tree)
    page = static.lookup("/")
    assert page is not None
    expected = f'"sha256-{hashlib.sha256(b"hello").hexdigest()}"'
    assert page.etag == expected
    fresh = static.respond(page, None)
    assert fresh.status == 200 and fresh.body == b"hello"
    headers = dict(fresh.headers)
    assert headers["ETag"] == expected
    assert headers["Cache-Control"] == "no-cache"
    assert headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
    for match in (expected, f'"x", {expected}', "W/" + expected, "*"):
        not_modified = static.respond(page, match)
        assert not_modified.status == 304
        assert not_modified.body == b""
    assert static.respond(page, '"other"').status == 200


def test_csp_is_the_section_7_5_policy() -> None:
    directives = dict(
        part.strip().split(" ", 1) for part in CONTENT_SECURITY_POLICY.split(";")
    )
    assert directives["default-src"] == "'none'"
    assert directives["script-src"] == "'self'"
    assert directives["frame-ancestors"] == "'none'"
    assert directives["form-action"] == "'none'"
    assert "unsafe" not in CONTENT_SECURITY_POLICY
