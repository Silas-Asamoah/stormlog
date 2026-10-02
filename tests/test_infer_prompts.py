"""Deterministic prompts with controlled prefix sharing."""

import contextlib
import io
import json
import os
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.analysis import format_analysis_text
from stormlog.infer.cli import main as infer_main
from stormlog.infer.prompts import Prompt, PromptSource, PromptSpec
from stormlog.infer.tokens import EstimatedTokenCounter, TokenCount, generate_prompt
from tests.infer_workload_helpers import run_profile_with_fake_client


class _CharacterCounter:
    """Counts one token per four characters, unlike the word estimate."""

    source = "characters"
    exact = True

    def count_text(self, text: str) -> TokenCount:
        return TokenCount(value=len(text) // 4, source=self.source, exact=True)


def _source(spec: PromptSpec, **changes: Any) -> PromptSource:
    values: dict[str, Any] = {
        "counter": EstimatedTokenCounter(),
        "seed": 0,
        "case_id": "c1_in64_out16",
        "phase": "measured",
        "input_tokens": 64,
    }
    values.update(changes)
    return PromptSource(spec, **values)


def _common_prefix(left: str, right: str) -> str:
    return os.path.commonprefix([left, right])


# "[" and a 12-character nonce: two prompts whose nonces differ share less.
NONCE_HEAD = 13


def test_repeat_mode_keeps_the_prompt_stormlog_always_sent() -> None:
    counter = EstimatedTokenCounter()
    source = _source(PromptSpec(), seed=3)
    expected = generate_prompt(64, counter, seed=3 + 64)
    assert {source.prompt(i).text for i in range(5)} == {expected}
    warmup = _source(PromptSpec(), seed=3, phase="warmup")
    assert warmup.prompt(0).text == expected
    assert source.prompt(0).prompt_id == "repeat"


def test_phases_given_one_cache_build_the_repeated_prompt_once() -> None:
    built: list[int] = []

    class Counting(EstimatedTokenCounter):
        def count_text(self, text: str) -> TokenCount:
            built.append(len(text))
            return super().count_text(text)

    repeated: dict[tuple[int, int], Prompt] = {}
    phases = [
        _source(PromptSpec(), counter=Counting(), phase=phase, repeated=repeated)
        for phase in ("warmup", "measured")
    ]
    first = phases[0].prompt(0)
    calls = len(built)
    assert phases[1].prompt(0) is first and len(built) == calls
    # Another length, or another seed, is a different prompt.
    other = _source(PromptSpec(), input_tokens=32, repeated=repeated).prompt(0)
    reseeded = _source(PromptSpec(), seed=9, repeated=repeated).prompt(0)
    assert other.text != first.text and reseeded.text != first.text


@pytest.mark.parametrize("counter", [EstimatedTokenCounter(), _CharacterCounter()])
def test_unique_prompts_differ_from_the_first_token(counter: Any) -> None:
    source = _source(PromptSpec(mode="unique"), counter=counter)
    prompts = [source.prompt(i) for i in range(50)]
    nonces = [p.text[1:13] for p in prompts]
    assert len(set(nonces)) == 50
    assert all(p.text.startswith(f"[{n}] ") for p, n in zip(prompts, nonces))
    for prompt in prompts:
        assert prompt.count == counter.count_text(prompt.text)
        assert prompt.prefix_group is None
        # The filler is checked against the counter, so each prompt reaches
        # the target, overshooting only where the nonce and filler meet.
        assert 64 <= prompt.count.value <= 66


def test_prompts_are_reproducible_and_scoped_to_case_and_phase() -> None:
    spec = PromptSpec(mode="unique")
    first = [_source(spec).prompt(i).text for i in range(3)]
    assert first == [_source(spec).prompt(i).text for i in range(3)]
    variants: list[dict[str, Any]] = [
        {"seed": 1},
        {"phase": "warmup"},
        {"case_id": "c4_in64_out16"},
    ]
    for changes in variants:
        other = [_source(spec, **changes).prompt(i).text for i in range(3)]
        assert all(len(_common_prefix(a, b)) < NONCE_HEAD for a in first for b in other)


def test_shared_prefix_groups_share_exactly_their_prefix() -> None:
    spec = PromptSpec(mode="shared-prefix", shared_prefix_ratio=0.5, prefix_groups=3)
    counter = EstimatedTokenCounter()
    source = _source(spec, input_tokens=200)
    prompts = [source.prompt(i) for i in range(60)]
    groups: dict[int, list[str]] = {}
    for prompt in prompts:
        assert prompt.prefix_group is not None
        groups.setdefault(prompt.prefix_group, []).append(prompt.text)
        assert counter.count_text(prompt.text).value >= 200
    assert sorted(groups) == [0, 1, 2]
    for texts in groups.values():
        shared = _common_prefix(texts[0], texts[1])
        # The shared part is about half the prompt, then each request differs.
        assert 95 <= counter.count_text(shared).value <= 110
        assert len({text[len(shared) :] for text in texts}) == len(texts)
    heads = [texts[0] for texts in groups.values()]
    assert len(_common_prefix(heads[0], heads[1])) < NONCE_HEAD


def test_shared_prefixes_do_not_carry_over_from_warmup() -> None:
    spec = PromptSpec(mode="shared-prefix", shared_prefix_ratio=0.8)
    measured = _source(spec).prompt(0).text
    warmup = _source(spec, phase="warmup").prompt(0).text
    assert len(_common_prefix(measured, warmup)) < NONCE_HEAD


def test_the_phase_digest_covers_every_prompt_in_order() -> None:
    source = _source(PromptSpec(mode="unique"))
    assert source.digest() is None
    source.prepare(range(3))
    digest = source.digest()
    again = _source(PromptSpec(mode="unique"))
    again.prepare([2, 0, 1])
    assert again.digest() == digest
    again.prompt(3)
    assert again.digest() != digest


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"mode": "random"}, "prompt mode must be one of"),
        ({"mode": "shared-prefix"}, "shared-prefix ratio goes with"),
        ({"mode": "unique", "shared_prefix_ratio": 0.5}, "shared-prefix ratio goes"),
        ({"mode": "unique", "prefix_groups": 2}, "prefix groups go with"),
        ({"mode": "shared-prefix", "shared_prefix_ratio": 1.0}, "between 0 and 1"),
        (
            {"mode": "shared-prefix", "shared_prefix_ratio": 0.5, "prefix_groups": 0},
            "prefix groups must be >= 1",
        ),
    ],
)
def test_prompt_specs_reject_settings_their_mode_cannot_use(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PromptSpec(**changes)


def test_spec_records_name_the_sharing() -> None:
    spec = PromptSpec(mode="shared-prefix", shared_prefix_ratio=0.25)
    assert spec.to_record() == {
        "mode": "shared-prefix",
        "shared_prefix_ratio": 0.25,
        "prefix_groups": 1,
    }
    assert PromptSpec(mode="unique").to_record() == {"mode": "unique"}


def test_profiled_requests_record_the_prompt_they_sent(tmp_path: Path) -> None:
    requests, report, _client = run_profile_with_fake_client(
        tmp_path, latency_seconds=0.0, request_count=4, prompt_mode="unique"
    )
    assert {r["prompt_mode"] for r in requests} == {"unique"}
    assert len({r["prompt_digest"] for r in requests}) == 4
    assert all(r["prompt_id"].startswith("r-") for r in requests)
    prompts = report["cases"]["c1_in8_out4"]["prompts"]
    assert (prompts["mode"], prompts["distinct_prompts"]) == ("unique", 4)
    windows = [
        json.loads(line)
        for line in (tmp_path / "infer.jsonl").read_text().splitlines()
        if '"infer.phase_window"' in line
    ]
    assert prompts["prompts_digest"] == windows[-1]["prompts_digest"]
    assert "prompts: unique, 4 distinct" in format_analysis_text(report)
    lengths = report["cases"]["c1_in8_out4"]["lengths"]
    # The fake server reports 8 prompt and 4 output tokens for every request.
    assert lengths["prompt_tokens"] == {"min": 8, "p50": 8.0, "p95": 8.0, "max": 8}
    assert lengths["output_tokens"]["max"] == 4


def test_shared_prefix_requests_record_their_group(tmp_path: Path) -> None:
    requests, report, _client = run_profile_with_fake_client(
        tmp_path,
        latency_seconds=0.0,
        request_count=12,
        input_tokens=(64,),
        prompt_mode="shared-prefix",
        shared_prefix_ratio=0.5,
        prefix_groups=2,
    )
    assert {r["prefix_group"] for r in requests} == {0, 1}
    # Half of a 64-token prompt is the group prefix, give or take a token.
    assert all(31 <= r["shared_prefix_tokens"] <= 33 for r in requests)
    prompts = report["cases"]["c1_in64_out4"]["prompts"]
    assert prompts["prefix_groups_used"] == 2
    assert 31 <= prompts["shared_prefix_tokens"]["min"] <= 33
    text = format_analysis_text(report)
    assert "over 2 prefix groups, shared prefixes up to" in text


def test_repeat_mode_reports_one_distinct_prompt(tmp_path: Path) -> None:
    _requests, report, _client = run_profile_with_fake_client(
        tmp_path, latency_seconds=0.0, request_count=3
    )
    prompts = report["cases"]["c1_in8_out4"]["prompts"]
    assert (prompts["mode"], prompts["distinct_prompts"]) == ("repeat", 1)
    assert "prompts:" not in format_analysis_text(report)


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--shared-prefix-ratio", "0.5"], "does not apply to --prompt-mode repeat"),
        (["--prompt-mode", "shared-prefix"], "needs --shared-prefix-ratio"),
        (
            ["--prompt-mode", "unique", "--prefix-groups", "2"],
            "--prefix-groups only applies",
        ),
        (
            ["--prompt-mode", "shared-prefix", "--shared-prefix-ratio", "1.5"],
            "between 0 and 1",
        ),
    ],
)
def test_prompt_flags_are_checked_before_any_request(
    tmp_path: Path, flags: list[str], message: str
) -> None:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        code = infer_main(
            [
                "profile",
                "--endpoint",
                "http://127.0.0.1:1/v1/chat/completions",
                "--model",
                "fake-model",
                "--output",
                str(tmp_path / "never.jsonl"),
                *flags,
            ]
        )
    assert code == 1
    assert message in stderr.getvalue()
    assert not (tmp_path / "never.jsonl").exists()


class _CountingCounter:
    """Counts how many texts are tokenized, and how many tokens in all."""

    source = "counting"
    exact = False

    def __init__(self) -> None:
        self.calls = 0
        self.tokens = 0

    def count_text(self, text: str) -> TokenCount:
        value = EstimatedTokenCounter().count_text(text).value
        self.calls += 1
        self.tokens += value
        return TokenCount(value=value, source=self.source, exact=False)


@pytest.mark.parametrize(
    "spec",
    [
        PromptSpec(mode="unique"),
        PromptSpec(mode="shared-prefix", shared_prefix_ratio=0.5, prefix_groups=4),
    ],
)
def test_building_a_prompt_tokenizes_only_its_nonce(spec: PromptSpec) -> None:
    counter = _CountingCounter()
    source = _source(spec, counter=counter, input_tokens=2048)
    source.warm()
    warm_tokens = counter.tokens
    source.prepare(range(1000))
    # Once warm, a thousand 2048-token prompts cost a few tokens each.
    assert (counter.tokens - warm_tokens) / 1000 < 10
    # Warming hands out no prompts, so the phase digest is untouched.
    fresh = _source(spec, input_tokens=2048)
    fresh.prepare(range(1000))
    assert source.digest() == fresh.digest()


def test_used_prompts_keep_only_their_digest() -> None:
    source = _source(PromptSpec(mode="unique"))
    for index in range(200):
        assert source.take(index).text.startswith("[")
        source.forget(index)
    assert source._prompts == {}
    again = _source(PromptSpec(mode="unique"))
    again.prepare(range(200))
    assert source.digest() == again.digest()


def test_short_prompts_with_nonces_get_a_warning(tmp_path: Path) -> None:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
        infer_main(
            [
                "profile",
                "--endpoint",
                "http://127.0.0.1:1/v1/chat/completions",
                "--model",
                "fake-model",
                "--input-tokens",
                "16",
                "--prompt-mode",
                "unique",
                "--timeout",
                "0.5",
                "--system-sampler",
                "none",
                "--tokenizer",
                "none",
                "--output",
                str(tmp_path / "infer.jsonl"),
            ]
        )
    assert "--input-tokens below 32" in stderr.getvalue()
