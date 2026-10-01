"""Behavioral contracts for the standalone Radon gate."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

pytest.importorskip("radon")

from scripts.check_complexity import (
    _source_hash,
    baseline_payload,
    evaluate,
    main,
    read_baseline,
    scan,
    score_source,
)


def test_repository_has_no_complexity_grace_entries() -> None:
    root = Path(__file__).resolve().parents[1]
    baseline = read_baseline(root / ".ci/complexity-baseline.json")
    assert baseline["blocks"] == {}


def _complex_source(branches: int = 10, name: str = "legacy") -> str:
    body = "\n".join(
        f"    if value == {index}: return {index}" for index in range(branches)
    )
    return f"def {name}(value):\n{body}\n    return -1\n"


def _project(tmp_path: Path, source: str) -> Path:
    (tmp_path / "stormlog").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "stormlog/example.py").write_text(source, encoding="utf-8")
    (tmp_path / "scripts/check_complexity.py").write_text("", encoding="utf-8")
    (tmp_path / ".ci").mkdir()
    path = tmp_path / ".ci/complexity-baseline.json"
    scores = scan(tmp_path, ["stormlog", "scripts/check_complexity.py"])
    path.write_text(json.dumps(baseline_payload(scores)), encoding="utf-8")
    return path


def test_radon_scores_methods_closures_async_and_local_classes() -> None:
    source = """
class Outer:
    async def run(self):
        def nested(value):
            return 1 if value else 0
        class Local:
            def method(self):
                assert self
        return nested(True)
"""
    scores = score_source(source, "sample.py")

    assert {score.name: score.complexity for score in scores} == {
        "Outer.run": 1,
        "Outer.run.nested": 2,
        "Outer.run.Local.method": 2,
    }
    assert {score.name: score.source_hash for score in scores} == {
        "Outer.run": "2668f7cf7228671b1aa66266c4747c9a2355858aa031b3af6e7e188a346121bf",
        "Outer.run.nested": "71500992289c532ad31d3070f5d69740b5f6fc8bd7a524e5c3b7381aa059ee88",
        "Outer.run.Local.method": "9612fa96a834b9cf1ce7044ad75d8dbe241b6453ecb34362c1fa26799f6b5a4f",
    }


@pytest.mark.parametrize(
    "source", ["def f(): pass", "async def f(): pass", "class C: pass"]
)
def test_hash_normalizes_only_empty_type_parameters(source: str) -> None:
    node = ast.parse(source).body[0]
    expected = _source_hash(node)
    if "type_params" not in node._fields:
        setattr(node, "_fields", (*node._fields, "type_params"))
    setattr(node, "type_params", [])
    before = ast.dump(node)

    assert _source_hash(node) == expected
    assert ast.dump(node) == before

    setattr(node, "type_params", [ast.Name(id="T", ctx=ast.Load())])
    assert _source_hash(node) != expected


def test_duplicate_names_do_not_overwrite_scores() -> None:
    source = "def conditional(): pass\ndef conditional(): pass\n"
    assert [score.key for score in score_source(source, "test.py")] == [
        "test.py:conditional",
        "test.py:conditional#2",
    ]


def test_unchanged_legacy_code_tolerates_line_shifts_and_formatting() -> None:
    source = _complex_source()
    baseline = baseline_payload(score_source(source, "test.py"))
    shifted = "# added comment\n\n" + source.replace("return -1", "return (-1)")

    assert evaluate(score_source(shifted, "test.py"), baseline)["status"] == "pass"


def test_python_310_legacy_baseline_remains_valid() -> None:
    baseline = {
        "blocks": {
            "test.py:legacy": {
                "complexity": 11,
                "source_hash": (
                    "c48792a148fb4b2a2df00ae7509ef59e58199061a14dd37392eb64070c89f613"
                ),
            }
        }
    }

    report = evaluate(score_source(_complex_source(), "test.py"), baseline)

    assert report["status"] == "pass"
    assert report["baselined"] == 1
    assert report["stale_baseline_entries"] == []


@pytest.mark.parametrize(
    "replacement",
    [_complex_source(11), _complex_source().replace("return -1", "return -2")],
)
def test_modified_legacy_callable_must_meet_limit(replacement: str) -> None:
    baseline = baseline_payload(score_source(_complex_source(), "test.py"))
    report = evaluate(score_source(replacement, "test.py"), baseline)

    assert report["status"] == "fail"
    assert report["violations"][0]["name"] == "legacy"


def test_new_callable_limit_and_deleted_exception() -> None:
    baseline = baseline_payload(score_source(_complex_source(), "test.py"))
    scores = score_source(
        _complex_source(9, "acceptable") + _complex_source(10, "new"), "test.py"
    )
    report = evaluate(scores, baseline)

    assert [item["name"] for item in report["violations"]] == ["new"]
    assert report["stale_baseline_entries"] == ["test.py:legacy"]


def test_cli_prunes_resolved_exceptions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _project(tmp_path, _complex_source())
    (tmp_path / "stormlog/example.py").write_text(_complex_source(9), encoding="utf-8")

    assert main(["--root", str(tmp_path), "--update-baseline", "--json"]) == 0
    assert read_baseline(path)["blocks"] == {}
    assert json.loads(capsys.readouterr().out)["status"] == "pass"


def test_cli_update_cannot_baseline_new_debt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _project(tmp_path, "")
    before = path.read_bytes()
    (tmp_path / "stormlog/example.py").write_text(_complex_source(), encoding="utf-8")

    assert main(["--root", str(tmp_path), "--update-baseline", "--json"]) == 1
    assert path.read_bytes() == before
    assert json.loads(capsys.readouterr().out)["violations"][0]["complexity"] == 11


@pytest.mark.parametrize(
    "damage", ["syntax", "missing_source", "bad_baseline", "missing_baseline"]
)
def test_cli_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], damage: str
) -> None:
    path = _project(tmp_path, "")
    if damage == "syntax":
        (tmp_path / "stormlog/example.py").write_text("def broken(", encoding="utf-8")
    elif damage == "missing_source":
        (tmp_path / "scripts/check_complexity.py").unlink()
    elif damage == "bad_baseline":
        path.write_text('{"blocks": []}', encoding="utf-8")
    else:
        path.unlink()

    assert main(["--root", str(tmp_path), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"
