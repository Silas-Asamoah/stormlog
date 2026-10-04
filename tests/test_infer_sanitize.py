"""A bundle is publishable only with no secret left in it."""

from __future__ import annotations

from pathlib import Path

from stormlog.infer.sanitize import sanitize_bundle

SECRET = "plantedSecretValue-0123"


def test_a_clean_bundle_is_publishable(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "commands.sh").write_text("TOKEN=${TOKEN} vllm serve m\n")
    report = sanitize_bundle(tmp_path, [SECRET])
    assert report == {
        "publishable": True,
        "files_scanned": 1,
        "secrets_checked": 1,
        "hits": [],
    }


def test_planted_secrets_and_token_shapes_are_found_by_file_and_line(
    tmp_path: Path,
) -> None:
    (tmp_path / "server.log").write_text(f"starting\nkey is {SECRET}\n")
    (tmp_path / "c1.jsonl").write_text(
        '{"a": 1}\n{"headers": "Authorization: Bearer abcdefghijk12345"}\n'
    )
    (tmp_path / "env.txt").write_text(
        "HF=hf_abcdefghijklmnopqrstuvwxyz\nKEY=sk-abcdefghijklmnopqrstuvwx\n"
    )
    report = sanitize_bundle(tmp_path, [SECRET])
    found = {(hit["file"], hit["line"], hit["kind"]) for hit in report["hits"]}
    assert report["publishable"] is False
    assert found == {
        ("server.log", 2, "secret_value"),
        ("c1.jsonl", 2, "bearer"),
        ("env.txt", 1, "huggingface_token"),
        ("env.txt", 2, "sk_key"),
    }
    # A hit never repeats the value it found.
    assert SECRET not in str(report)
