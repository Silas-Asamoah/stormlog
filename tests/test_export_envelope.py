"""Export envelopes: fixed fields, capped whatever the record held."""

import pytest

from stormlog._export import envelope as envelope_module
from stormlog._export.envelope import EnvelopeLimits, make_envelope


def test_an_envelope_keeps_fields_in_order_and_reads_them_back() -> None:
    envelope = make_envelope(
        "infer.request",
        [("status", "ok"), ("e2e_ms", 12.5), ("held", False), ("gaps", (1, 2.5))],
        EnvelopeLimits(),
    )
    assert envelope.kind == "infer.request"
    assert envelope.as_dict() == {
        "status": "ok",
        "e2e_ms": 12.5,
        "held": False,
        "gaps": (1, 2.5),
    }
    assert envelope.get("status") == "ok" and envelope.get("missing") is None
    assert envelope.truncated == 0


def test_fields_past_the_limit_are_left_out_and_counted() -> None:
    envelope = make_envelope(
        "k", [(f"f{i}", i) for i in range(40)], EnvelopeLimits(max_fields=32)
    )
    assert len(envelope.fields) == 32 and envelope.truncated == 8


def test_strings_and_tuples_are_cut_and_counted() -> None:
    limits = EnvelopeLimits(max_string=4, max_tuple=3)
    envelope = make_envelope(
        "k", [("name", "abcdefgh"), ("ids", ("abcdef", "x", "y", "z"))], limits
    )
    assert envelope.get("name") == "abcd"
    assert envelope.get("ids") == ("abcd", "x", "y")
    assert envelope.truncated == 2


def test_content_fields_are_cut_by_utf8_bytes_not_characters() -> None:
    limits = EnvelopeLimits(
        max_string=4, content_fields=frozenset({"error"}), max_content_bytes=5
    )
    envelope = make_envelope("k", [("error", "日本語テキスト")], limits)
    assert envelope.get("error") == "日"  # 3 bytes; a second character needs 6
    assert envelope.truncated == 1


def test_a_megabyte_content_field_is_cut_without_encoding_it_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int] = []
    real = envelope_module.truncate_utf8

    def counting(text: str, max_bytes: int) -> str:
        seen.append(len(text))
        return real(text, max_bytes)

    monkeypatch.setattr(envelope_module, "truncate_utf8", counting)
    limits = EnvelopeLimits(content_fields=frozenset({"error"}))
    body = "é" * 1_000_000
    envelope = make_envelope("k", [("error", body)], limits)
    assert envelope.get("error") == "é" * 512  # 1,024 bytes
    assert envelope.truncated == 1
    # Only a cap's worth of characters is ever encoded, on the producer.
    assert seen == [limits.max_content_bytes]


def test_a_megabyte_field_cannot_make_a_big_envelope() -> None:
    body = "x" * 1_000_000
    envelope = make_envelope("k", [("error_message", body)], EnvelopeLimits())
    assert envelope.size < EnvelopeLimits().max_bytes
    kept = envelope.get("error_message")
    assert isinstance(kept, str) and len(kept) == 256


def test_trailing_fields_past_the_byte_budget_are_left_out() -> None:
    # A field of 100 characters holds about 220 bytes: its string object,
    # its pair and its pointer; the envelope itself about 100.
    limits = EnvelopeLimits(max_string=100, max_bytes=600)
    envelope = make_envelope("k", [(f"f{i}", "y" * 100) for i in range(5)], limits)
    assert envelope.size <= 600
    assert len(envelope.fields) == 2 and envelope.truncated == 3


@pytest.mark.parametrize("value", [[1, 2], {"a": 1}, ((1, 2),), object()])
def test_only_flat_scalar_values_are_accepted(value: object) -> None:
    with pytest.raises(TypeError):
        make_envelope("k", [("v", value)], EnvelopeLimits())  # type: ignore[list-item]
