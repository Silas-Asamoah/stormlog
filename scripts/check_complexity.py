"""Check Radon complexity, allowing only unchanged, baselined legacy callables."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import sys
import tokenize
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import radon  # type: ignore[import-untyped, unused-ignore]
from radon.complexity import cc_rank  # type: ignore[import-untyped, unused-ignore]
from radon.visitors import (
    ComplexityVisitor,  # type: ignore[import-untyped, unused-ignore]
)

RADON_VERSION = "6.0.1"
MAX_COMPLEXITY = 10
DEFAULT_PATHS = ("stormlog", "scripts/check_complexity.py")
BASELINE_PATH = Path(".ci/complexity-baseline.json")


@dataclass(frozen=True)
class CallableScore:
    path: str
    name: str
    line: int
    complexity: int
    source_hash: str

    @property
    def key(self) -> str:
        return f"{self.path}:{self.name}"


def _callable_nodes(
    node: ast.AST, prefix: str = ""
) -> Iterator[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Include methods, closures and local classes without class aggregate scores."""
    for child in ast.iter_child_nodes(node):
        child_prefix = prefix
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            child_prefix = f"{prefix}.{child.name}" if prefix else child.name
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield child_prefix, child
        yield from _callable_nodes(child, child_prefix)


def _source_hash(node: ast.AST) -> str:
    normalized = copy.deepcopy(node)
    for child in ast.walk(normalized):
        # Python 3.12 adds this empty field even to pre-3.12 syntax.
        if getattr(child, "type_params", None) == []:
            delattr(child, "type_params")
    # Python 3.13 omits empty lists by default; retain the pre-3.13 representation.
    dump_options = {"show_empty": True} if sys.version_info >= (3, 13) else {}
    return hashlib.sha256(
        ast.dump(normalized, include_attributes=False, **dump_options).encode("utf-8")
    ).hexdigest()


def score_source(source: str, path: str) -> list[CallableScore]:
    tree = ast.parse(source, filename=path)
    occurrences: Counter[str] = Counter()
    scores = []
    for name, node in _callable_nodes(tree):
        occurrences[name] += 1
        # Conditional definitions and property accessors can share a lexical name.
        suffix = f"#{occurrences[name]}" if occurrences[name] > 1 else ""
        complexity = ComplexityVisitor.from_ast(node).functions[0].complexity
        source_hash = _source_hash(node)
        scores.append(
            CallableScore(path, name + suffix, node.lineno, complexity, source_hash)
        )
    return scores


def _source_paths(root: Path, paths: list[str]) -> list[Path]:
    sources: set[Path] = set()
    for name in paths:
        path = root / name
        if path.is_dir():
            sources.update(path.rglob("*.py"))
        elif path.is_file() and path.suffix == ".py":
            sources.add(path)
        else:
            raise ValueError(f"Python source path does not exist: {path}")
    if not sources:
        raise ValueError("No Python source files found")
    return sorted(sources)


def scan(root: Path, paths: list[str]) -> list[CallableScore]:
    scores = []
    for path in _source_paths(root, paths):
        with tokenize.open(path) as handle:
            source = handle.read()
        scores.extend(score_source(source, path.relative_to(root).as_posix()))
    return scores


def baseline_payload(scores: list[CallableScore]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "radon_version": RADON_VERSION,
        "max_complexity": MAX_COMPLEXITY,
        "blocks": {
            score.key: {
                "complexity": score.complexity,
                "source_hash": score.source_hash,
            }
            for score in sorted(scores, key=lambda item: item.key)
            if score.complexity > MAX_COMPLEXITY
        },
    }


def _validate_baseline_block(key: str, block: Any) -> None:
    if not isinstance(block, dict) or set(block) != {"complexity", "source_hash"}:
        raise ValueError(f"Invalid baseline entry: {key}")
    score = block["complexity"]
    digest = block["source_hash"]
    if type(score) is not int or score <= MAX_COMPLEXITY:
        raise ValueError(f"Invalid baseline complexity: {key}")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"Invalid baseline source hash: {key}")
    if any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"Invalid baseline source hash: {key}")


def read_baseline(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Complexity baseline must be an object")
    expected = {
        "schema_version": 1,
        "radon_version": RADON_VERSION,
        "max_complexity": MAX_COMPLEXITY,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Complexity baseline version or threshold does not match the checker"
        )
    blocks = payload.get("blocks")
    if not isinstance(blocks, dict):
        raise ValueError("Complexity baseline blocks must be an object")
    for key, block in blocks.items():
        _validate_baseline_block(key, block)
    return payload


def evaluate(scores: list[CallableScore], baseline: dict[str, Any]) -> dict[str, Any]:
    violations = []
    retained = set()
    for score in scores:
        if score.complexity <= MAX_COMPLEXITY:
            continue
        previous = baseline["blocks"].get(score.key)
        current = {"complexity": score.complexity, "source_hash": score.source_hash}
        if previous == current:
            retained.add(score.key)
            continue
        violations.append({**asdict(score), "rank": cc_rank(score.complexity)})
    return {
        "status": "fail" if violations else "pass",
        "radon_version": RADON_VERSION,
        "max_complexity": MAX_COMPLEXITY,
        "callables": len(scores),
        "rank_counts": dict(
            sorted(Counter(cc_rank(score.complexity) for score in scores).items())
        ),
        "baselined": len(retained),
        "violations": violations,
        "stale_baseline_entries": sorted(set(baseline["blocks"]) - retained),
    }


def _print_report(report: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if report["status"] == "error":
        print(f"Complexity check error: {report['error']}", file=sys.stderr)
        return
    print(
        f"Radon {RADON_VERSION}: {report['callables']} callables, "
        f"{report['baselined']} unchanged legacy exceptions, "
        f"{len(report['violations'])} violations (limit {MAX_COMPLEXITY})."
    )
    for violation in report["violations"]:
        print(
            f"{violation['path']}:{violation['line']}: {violation['name']} "
            f"has CC {violation['complexity']} ({violation['rank']}); "
            "new or changed callables must have CC <= 10."
        )
    if report["stale_baseline_entries"]:
        print(
            "Resolved/changed baseline entries: "
            + ", ".join(report["stale_baseline_entries"])
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument(
        "--json", action="store_true", help="Print a machine-readable report"
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Remove resolved exceptions after a passing full scan",
    )
    args = parser.parse_args(argv)
    try:
        if radon.__version__ != RADON_VERSION:
            raise ValueError(
                f"Install radon=={RADON_VERSION}; found {radon.__version__}"
            )
        root = args.root.resolve()
        baseline_path = root / args.baseline
        baseline = read_baseline(baseline_path)
        scores = scan(root, list(DEFAULT_PATHS))
        report = evaluate(scores, baseline)
        if args.update_baseline and report["status"] == "pass":
            baseline_path.write_text(
                json.dumps(baseline_payload(scores), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except (OSError, ValueError, SyntaxError) as exc:
        _print_report({"status": "error", "error": str(exc)}, args.json)
        return 2
    _print_report(report, args.json)
    return 1 if report["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
