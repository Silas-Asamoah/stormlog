"""The exporters' metric registry: declared up front, budgeted, never unbounded."""

import threading
from typing import Any

import pytest

# An independent reader of the text format; test-only.
from prometheus_client.parser import (  # type: ignore[import-not-found, unused-ignore]
    text_string_to_metric_families,
)

from stormlog._export.registry import (
    MAX_LABEL_VALUE,
    OVERFLOW,
    BudgetExceeded,
    FamilySpec,
    Registry,
    bounded_value,
    format_value,
    render,
)
from tests.export_conformance import check_exposition

STATUSES = ("ok", "error", "timeout")


def _requests(registry: Registry, cases: list[str]) -> Any:
    return registry.add(
        FamilySpec(
            "stormlog_infer_requests_total",
            "counter",
            "Requests by outcome.",
            labels=("case", "status"),
            enums={"status": STATUSES},
        ),
        known=[{"case": case} for case in cases],
    )


def _latency(registry: Registry, cases: list[str]) -> Any:
    return registry.add(
        FamilySpec(
            "stormlog_infer_request_duration_seconds",
            "histogram",
            "Latency of completed requests.",
            unit="seconds",
            labels=("case",),
            buckets=(0.1, 1.0, 10.0),
        ),
        known=[{"case": case} for case in cases],
    )


def _text(registry: Registry) -> str:
    return render(registry.snapshot()).decode()


# ------------------------------------------------------------------ declaring
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "infer_requests_total"}, "stormlog_"),
        ({"name": "stormlog_requests"}, "_total"),
        ({"name": "stormlog_bad-name_total"}, "valid"),
        ({"name": "stormlog_wait_total", "unit": "seconds"}, "_seconds"),
        ({"labels": ("a", "a")}, "unique"),
        ({"labels": ("__a",)}, "invalid label"),
        ({"labels": ("a",), "enums": {"b": ("x",)}}, "not a label"),
        ({"labels": ("a",), "enums": {"a": ()}}, "distinct"),
        ({"buckets": (1.0,)}, "only a histogram"),
        # The textfile writer adds these two itself.
        ({"name": "stormlog_run_active", "kind": "gauge"}, "textfile"),
        (
            {"name": "stormlog_textfile_updated_timestamp_seconds", "kind": "gauge"},
            "textfile",
        ),
    ],
)
def test_family_declarations_are_checked(kwargs: dict, message: str) -> None:
    spec: dict[str, Any] = {
        "name": "stormlog_requests_total",
        "kind": "counter",
        "help": "h",
    }
    spec.update(kwargs)
    with pytest.raises(ValueError, match=message):
        FamilySpec(**spec)


@pytest.mark.parametrize(
    ("buckets", "labels"),
    [((), ()), ((1.0, 1.0), ()), ((float("inf"),), ()), ((1.0,), ("le",))],
)
def test_histogram_declarations_are_checked(buckets: tuple, labels: tuple) -> None:
    with pytest.raises(ValueError):
        FamilySpec(
            "stormlog_x_seconds",
            "histogram",
            "h",
            unit="seconds",
            labels=labels,
            buckets=buckets,
        )


def test_a_name_cannot_be_declared_twice_even_through_histogram_suffixes() -> None:
    registry = Registry()
    _latency(registry, ["c1"])
    with pytest.raises(ValueError, match="already declared"):
        registry.add(
            FamilySpec("stormlog_infer_request_duration_seconds_sum", "gauge", "h")
        )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        # The review's case: a gauge on a histogram's base name gave the
        # exposition two HELP and TYPE lines for one name.
        (("stormlog_lat_seconds", "histogram"), ("stormlog_lat_seconds", "gauge")),
        (("stormlog_lat_seconds", "gauge"), ("stormlog_lat_seconds", "histogram")),
        # A counter and a gauge of its stem read as one family name.
        (("stormlog_a_total", "counter"), ("stormlog_a", "gauge")),
        (("stormlog_a", "gauge"), ("stormlog_a_total", "counter")),
    ],
)
def test_a_family_name_is_reserved_whatever_its_kind(
    first: tuple[str, str], second: tuple[str, str]
) -> None:
    def spec(name: str, kind: str) -> FamilySpec:
        buckets = (1.0,) if kind == "histogram" else ()
        return FamilySpec(name, kind, "h", buckets=buckets)  # type: ignore[arg-type]

    registry = Registry()
    registry.add(spec(*first))
    with pytest.raises(ValueError, match="already declared"):
        registry.add(spec(*second))


def test_a_value_named_like_the_overflow_series_is_rejected() -> None:
    # It took a headroom series of its own, which the redirects past the
    # cap then shared, so the overflow series was not the overflow.
    registry = Registry(headroom=1)
    requests = _requests(registry, ["c1"])
    requests.inc((OVERFLOW, "ok"), 5)
    assert requests.stats.rejected == 1
    requests.inc(("c2", "ok"), 1)  # the headroom series
    requests.inc(("c3", "ok"), 2)  # past the cap
    exposition = check_exposition(_text(registry))
    assert (
        exposition.value("stormlog_infer_requests_total", case=OVERFLOW, status="ok")
        == 2
    )
    assert requests.stats.overflow_redirects == 1


def test_a_family_label_cannot_repeat_a_constant_label() -> None:
    # It would render {model="m",model="a"}, which a scrape rejects whole.
    registry = Registry(const_labels={"model": "m"})
    with pytest.raises(ValueError, match="model"):
        registry.add(
            FamilySpec("stormlog_x_total", "counter", "h", labels=("model",)),
            known=[{"model": "a"}],
        )
    assert registry.families == []
    with pytest.raises(ValueError, match="le"):  # a histogram's bucket label
        Registry(const_labels={"le": "x"}).add(
            FamilySpec("stormlog_x", "histogram", "h", buckets=(1.0,))
        )


def test_constant_label_names_are_checked() -> None:
    with pytest.raises(ValueError):
        Registry(const_labels={"bad-name": "x"})


# ------------------------------------------------------------------ series
def test_every_known_series_exists_at_zero_before_any_update() -> None:
    registry = Registry(const_labels={"stormlog_producer": "p"})
    _requests(registry, ["c1", "c2"])
    exposition = check_exposition(_text(registry))
    values = exposition.matching("stormlog_infer_requests_total")
    assert len(values) == 6 and set(values) == {0.0}


def test_updates_render_as_counters_and_cumulative_histograms() -> None:
    registry = Registry()
    requests = _requests(registry, ["c1"])
    latency = _latency(registry, ["c1"])

    def update() -> None:
        requests.inc(("c1", "ok"))
        requests.inc(("c1", "ok"))
        for value in (0.05, 0.5, 0.5, 20.0):
            latency.observe(("c1",), value)

    assert registry.apply(update)
    exposition = check_exposition(_text(registry))
    assert exposition.value("stormlog_infer_requests_total", status="ok") == 2
    bucket = "stormlog_infer_request_duration_seconds_bucket"
    assert exposition.value(bucket, le="0.1") == 1
    assert exposition.value(bucket, le="1.0") == 3
    assert exposition.value(bucket, le="10.0") == 3
    assert exposition.value(bucket, le="+Inf") == 4
    assert exposition.value("stormlog_infer_request_duration_seconds_sum") == 21.05
    assert exposition.value("stormlog_infer_request_duration_seconds_count") == 4


def test_pre_counted_observations_add_up_like_single_ones() -> None:
    registry = Registry()
    latency = _latency(registry, ["c1"])
    latency.observe_counts(("c1",), [1, 2, 0, 1], total=21.05)
    exposition = check_exposition(_text(registry))
    assert exposition.value("stormlog_infer_request_duration_seconds_count") == 4
    with pytest.raises(ValueError):
        latency.observe_counts(("c1",), [1, 2], total=1.0)


def test_non_finite_observations_are_rejected_and_counted() -> None:
    registry = Registry()
    latency = _latency(registry, ["c1"])
    latency.observe(("c1",), float("nan"))
    latency.observe_counts(("c1",), [0, 0, 0, 1], total=float("inf"))
    assert latency.stats.rejected == 2
    exposition = check_exposition(_text(registry))
    assert exposition.value("stormlog_infer_request_duration_seconds_count") == 0


def test_an_enum_value_outside_its_set_is_rejected_and_counted() -> None:
    registry = Registry()
    requests = _requests(registry, ["c1"])
    requests.inc(("c1", "unreachable"))
    requests.inc(("c1",))
    assert requests.stats.rejected == 2
    assert sum(check_exposition(_text(registry)).samples.values()) == 0


# ------------------------------------------------------------------ overflow
def test_counter_overflow_keeps_enums_so_status_totals_stay_exact() -> None:
    registry = Registry(headroom=1)
    requests = _requests(registry, ["c1"])
    observations = [("c1", "ok"), ("c2", "ok"), ("c3", "ok"), ("c4", "error")]
    for labels in observations:
        requests.inc(labels)
    exposition = check_exposition(_text(registry))
    name = "stormlog_infer_requests_total"
    assert sum(exposition.matching(name, status="ok")) == 3
    assert sum(exposition.matching(name, status="error")) == 1
    assert exposition.value(name, case=OVERFLOW, status="ok") == 1
    assert exposition.value(name, case=OVERFLOW, status="error") == 1
    assert requests.stats.overflow_redirects == 2


def test_histogram_overflow_keeps_every_observation_in_one_accumulator() -> None:
    registry = Registry(headroom=0)
    latency = _latency(registry, ["c1"])
    for case in ("c1", "c2", "c3"):
        latency.observe((case,), 0.5)
    exposition = check_exposition(_text(registry))
    counts = exposition.matching("stormlog_infer_request_duration_seconds_count")
    assert sum(counts) == 3
    assert (
        exposition.value("stormlog_infer_request_duration_seconds_count", case=OVERFLOW)
        == 2
    )


def test_gauge_label_sets_past_the_cap_are_rejected_not_merged() -> None:
    registry = Registry(headroom=1)
    gauge = registry.add(
        FamilySpec(
            "stormlog_watch_cooldown_seconds",
            "gauge",
            "h",
            unit="seconds",
            labels=("trigger_id",),
        ),
    )
    gauge.set(("t1",), 3.0)
    gauge.set(("t2",), 4.0)
    assert gauge.stats.rejected == 1
    exposition = check_exposition(_text(registry))
    assert exposition.matching("stormlog_watch_cooldown_seconds") == [3.0]


def test_the_overflow_sentinel_cannot_be_configured() -> None:
    with pytest.raises(ValueError, match="reserved"):
        _requests(Registry(), [OVERFLOW])


def test_long_configured_values_keep_a_digest_and_never_collide() -> None:
    first = "m" * 60 + "-first-variant-of-a-long-model-name"
    second = "m" * 60 + "-second-variant-of-a-long-model-name"
    assert bounded_value("short") == "short"
    assert len(bounded_value(first)) == MAX_LABEL_VALUE
    assert bounded_value(first) != bounded_value(second)
    registry = Registry(headroom=0)
    requests = _requests(registry, [first, second])
    requests.inc((first, "ok"))
    exposition = check_exposition(_text(registry))
    assert (
        exposition.value(
            "stormlog_infer_requests_total", case=bounded_value(first), status="ok"
        )
        == 1
    )
    assert requests.stats.overflow_redirects == 0


# ------------------------------------------------------------------ budget
def test_the_budget_counts_headroom_overflow_and_histogram_samples() -> None:
    registry = Registry(headroom=2)
    _requests(registry, ["c1"])  # 3 known + 2 headroom + 3 overflow series
    _latency(registry, ["c1"])  # 1 known + 2 headroom + 1 overflow, 3 + 3 each
    assert registry.budget().samples == 8 + 4 * 6


def test_the_byte_budget_covers_the_largest_possible_exposition() -> None:
    registry = Registry(headroom=3)
    requests = _requests(registry, ["c1"])
    latency = _latency(registry, ["c1"])
    budget = registry.budget()  # what start-up checks, before any update
    worst = '\\"' * MAX_LABEL_VALUE
    for index in range(10):  # fills the headroom, then the overflow series
        value = f"{index}{worst}"[:MAX_LABEL_VALUE]
        for status in STATUSES:
            requests.inc((value, status), 1e300)
        latency.observe((value,), 1e-300)
    assert len(_text(registry).encode()) <= budget.size


def test_the_byte_budget_covers_four_byte_characters_too() -> None:
    # Without enums there is one overflow series, so the headroom series
    # alone must be budgeted at four bytes a character.
    registry = Registry(headroom=8)
    family = registry.add(
        FamilySpec("stormlog_x_total", "counter", "h", labels=("case",)),
        known=[{"case": "c1"}],
    )
    budget = registry.budget()
    for index in range(9):
        family.inc((f"{index}" + "\U0001f600" * 63,), 1e300)
    assert family.stats.overflow_redirects == 1
    assert len(_text(registry).encode()) <= budget.size


def test_the_byte_budget_covers_the_widest_values() -> None:
    # The byte budget tests above use values of a few characters; the widest
    # a float renders is a negative one near the smallest normal, in full.
    widest = -1.2345678901234567e-308
    assert len(format_value(widest)) == 24
    registry = Registry(headroom=2)
    level = registry.add(
        FamilySpec("stormlog_level", "gauge", "h", labels=("case",)),
        known=[{"case": "c1"}],
    )
    budget = registry.budget()
    level.set(("c1",), widest)
    for index in range(2):  # the headroom, at the widest label values too
        level.set((chr(0x1F600 + index) * MAX_LABEL_VALUE,), widest)
    assert len(_text(registry).encode()) <= budget.size


def test_the_byte_budget_takes_the_enum_value_widest_in_bytes() -> None:
    # "\U0001f600" is one character but four bytes of UTF-8, more than "aaa";
    # a quote is one character but two bytes once escaped.
    widest = -1.2345678901234567e-308
    for wide in ("\U0001f600", '"' * 2):
        registry = Registry(headroom=2)
        level = registry.add(
            FamilySpec(
                "stormlog_level",
                "gauge",
                "h",
                labels=("case", "kind"),
                enums={"kind": ("aaa", wide)},
            ),
            known=[{"case": "c1", "kind": "aaa"}],
        )
        budget = registry.budget()
        cases = ["c1"] + [chr(0x1F600 + i) * MAX_LABEL_VALUE for i in range(2)]
        for case in cases:  # the known case, then the headroom, all at full width
            for kind in ("aaa", wide):
                level.set((case, kind), widest)
        assert len(_text(registry).encode()) <= budget.size, wide


def test_a_value_on_a_bucket_bound_falls_in_that_bucket() -> None:
    # Buckets count values less than or equal to their bound.
    registry = Registry()
    latency = _latency(registry, ["c1"])  # buckets 0.1, 1.0, 10.0
    latency.observe(("c1",), 1.0)
    exposition = check_exposition(_text(registry))
    name = "stormlog_infer_request_duration_seconds_bucket"
    assert exposition.value(name, case="c1", le="0.1") == 0
    assert exposition.value(name, case="c1", le="1.0") == 1


def test_a_registry_over_budget_is_refused_with_its_counts() -> None:
    registry = Registry(max_samples=10)
    _requests(registry, ["c1", "c2", "c3", "c4"])
    with pytest.raises(BudgetExceeded) as caught:
        registry.check_budget()
    assert caught.value.samples > 10 and caught.value.max_samples == 10
    small = Registry(max_bytes=100)
    _requests(small, ["c1"])
    with pytest.raises(BudgetExceeded):
        small.check_budget()


# ------------------------------------------------------------------ freezing
def test_nothing_changes_after_freeze() -> None:
    registry = Registry()
    requests = _requests(registry, ["c1"])
    requests.inc(("c1", "ok"))
    registry.freeze()
    before = _text(registry)
    assert not registry.apply(lambda: requests.inc(("c1", "ok")))
    requests.inc(("c1", "ok"))
    requests.inc(("c9", "ok"))
    assert _text(registry) == before
    assert registry.late_updates == 3 and registry.frozen


def test_a_snapshot_is_a_copy() -> None:
    registry = Registry()
    requests = _requests(registry, ["c1"])
    snapshot = registry.snapshot()
    requests.inc(("c1", "ok"))
    assert 'status="ok"} 0' in render(snapshot).decode()


def test_a_snapshot_waits_for_a_record_applied_whole() -> None:
    registry = Registry()
    requests = _requests(registry, ["c1"])
    latency = _latency(registry, ["c1"])
    halfway = threading.Event()
    snapshots: list[str] = []

    def take_snapshot() -> None:
        halfway.wait(5)
        snapshots.append(render(registry.snapshot()).decode())

    reader = threading.Thread(target=take_snapshot)
    reader.start()

    def update() -> None:
        requests.inc(("c1", "ok"))
        halfway.set()
        reader.join(0.2)  # the reader is blocked on the lock meanwhile
        latency.observe(("c1",), 0.5)

    registry.apply(update)
    reader.join(5)
    exposition = check_exposition(snapshots[0])
    assert exposition.value("stormlog_infer_requests_total", status="ok") == 1
    assert exposition.value("stormlog_infer_request_duration_seconds_count") == 1


# ------------------------------------------------------------------ rendering
def test_label_values_are_escaped_and_read_back_unchanged() -> None:
    awkward = 'back\\slash "quoted"\nnew line'
    registry = Registry(const_labels={"stormlog_producer": awkward}, headroom=0)
    requests = _requests(registry, [awkward])
    requests.inc((awkward, "ok"))
    text = _text(registry)
    check_exposition(text)
    families = {f.name: f for f in text_string_to_metric_families(text)}
    sample = next(
        s for s in families["stormlog_infer_requests"].samples if s.value == 1
    )
    assert sample.labels["case"] == awkward
    assert sample.labels["stormlog_producer"] == awkward


def test_help_text_is_escaped() -> None:
    registry = Registry()
    registry.add(FamilySpec("stormlog_up", "gauge", "first line\nback\\slash"))
    text = _text(registry)
    assert "# HELP stormlog_up first line\\nback\\\\slash\n" in text
    check_exposition(text)


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (0.0, "0"),
        (3.0, "3"),
        (0.25, "0.25"),
        (1e16, "1e+16"),
        (float("nan"), "NaN"),
        (float("inf"), "+Inf"),
        (float("-inf"), "-Inf"),
    ],
)
def test_values_are_formatted_for_the_text_format(value: float, text: str) -> None:
    assert format_value(value) == text


def test_a_final_update_that_fails_is_undone_and_the_registry_freezes() -> None:
    registry = Registry()
    done = registry.add(FamilySpec("stormlog_done_total", "counter", "h"))

    def final() -> None:
        done.inc((), 3)
        raise RuntimeError("final update failed")

    with pytest.raises(RuntimeError):
        registry.freeze(final)
    assert registry.frozen and registry.rolled_back == 1
    assert registry.apply(lambda: done.inc((), 1)) is False
    assert check_exposition(_text(registry)).value("stormlog_done_total") == 0


def test_a_final_update_lands_under_the_freeze() -> None:
    registry = Registry()
    requests = _requests(registry, ["c1"])
    registry.freeze(final=lambda: requests.inc(("c1", "error"), 5))
    registry.freeze(final=lambda: requests.inc(("c1", "error"), 5))  # no-op
    exposition = check_exposition(_text(registry))
    assert exposition.value("stormlog_infer_requests_total", status="error") == 5


def test_a_family_can_wait_for_values_instead_of_reading_zero() -> None:
    registry = Registry(headroom=0)
    family = registry.add(
        FamilySpec(
            "stormlog_up", "gauge", "h", labels=("state",), enums={"state": ("a", "b")}
        ),
        precreate=False,
    )
    budget = registry.budget()
    assert _text(registry).count("stormlog_up{") == 0
    family.set(("a",), 1.0)
    family.set(("b",), 0.0)
    exposition = check_exposition(_text(registry))
    assert exposition.value("stormlog_up", state="a") == 1
    assert len(_text(registry).encode()) <= budget.size
    assert budget.samples == 2


def _waiting(registry: Registry, kind: str, cases: list[str]) -> Any:
    return registry.add(
        FamilySpec(
            "stormlog_a_total" if kind == "counter" else "stormlog_a",
            kind,  # type: ignore[arg-type]
            "h",
            labels=("case",),
        ),
        known=[{"case": case} for case in cases],
        precreate=False,
    )


def test_a_waiting_counter_keeps_its_configured_series_after_strays() -> None:
    # Fable's case: label sets the configuration never named used up the
    # cap, so a configured one arriving later overflowed.
    registry = Registry(headroom=1)
    family = _waiting(registry, "counter", ["c0", "c1", "c2"])
    budget = registry.budget()
    for stray in ("u1", "u2", "u3", "u4"):
        family.inc((stray,), 1)
    family.inc(("c1",), 5)
    text = _text(registry)
    exposition = check_exposition(text)
    assert exposition.value("stormlog_a_total", case="c1") == 5
    assert exposition.value("stormlog_a_total", case="u1") == 1
    assert exposition.value("stormlog_a_total", case=OVERFLOW) == 3
    assert family.stats.overflow_redirects == 3
    # Every configured series too, and the budget still holds.
    for case in ("c0", "c2"):
        family.inc((case,), 1)
    assert len(_text(registry).encode()) <= budget.size
    assert _text(registry).count("stormlog_a_total{") <= budget.samples


def test_a_waiting_gauge_keeps_its_configured_series_after_strays() -> None:
    registry = Registry(headroom=1)
    family = _waiting(registry, "gauge", ["t1"])
    budget = registry.budget()
    family.set(("x1",), 1.0)
    family.set(("x2",), 2.0)
    family.set(("t1",), 3.0)
    exposition = check_exposition(_text(registry))
    assert exposition.value("stormlog_a", case="t1") == 3
    assert exposition.value("stormlog_a", case="x1") == 1
    assert family.stats.rejected == 1
    assert _text(registry).count("stormlog_a{") <= budget.samples


# ------------------------------------------------------------------ numbers
def _gauge(registry: Registry) -> Any:
    return registry.add(FamilySpec("stormlog_level", "gauge", "A level."))


def test_an_int_value_renders_on_every_supported_python() -> None:
    # int has no is_integer() before 3.12; a len() passed to set() used to
    # break every later render there.
    registry = Registry()
    gauge = _gauge(registry)
    requests = _requests(registry, ["c1"])
    gauge.set((), 3)
    requests.inc(("c1", "ok"), 2)
    exposition = check_exposition(_text(registry))
    assert exposition.value("stormlog_level") == 3
    assert exposition.value("stormlog_infer_requests_total", status="ok") == 2


def test_a_huge_int_stays_within_the_byte_budget() -> None:
    registry = Registry()
    gauge = _gauge(registry)
    gauge.set((), 10**300)
    assert len(render(registry.snapshot())) <= registry.budget().size
    gauge.set((), 10**400)  # beyond any float: rejected, the old value kept
    assert gauge.stats.rejected == 1
    assert check_exposition(_text(registry)).value("stormlog_level") == 1e300


@pytest.mark.parametrize(
    "amount",
    [10**400, float("nan"), float("inf"), -1, "3", True],
    ids=["huge", "nan", "inf", "negative", "str", "bool"],
)
def test_unusable_counter_increments_are_rejected_and_counted(amount: Any) -> None:
    registry = Registry()
    requests = _requests(registry, ["c1"])
    requests.inc(("c1", "ok"), 5)
    assert registry.apply(lambda: requests.inc(("c1", "ok"), amount))
    assert requests.stats.rejected == 1
    exposition = check_exposition(_text(registry))
    assert exposition.value("stormlog_infer_requests_total", status="ok") == 5


def test_a_record_that_fails_part_way_is_rolled_back() -> None:
    registry = Registry()
    requests = _requests(registry, ["c1"])
    latency = _latency(registry, ["c1"])
    before = _text(registry)

    def update() -> None:
        requests.inc(("c1", "ok"))
        latency.observe(("c1",), 0.5)
        latency.observe_counts(("c1",), [1, 0, 0, 0], 0.05)
        raise RuntimeError("the mapping broke")

    with pytest.raises(RuntimeError):
        registry.apply(update)
    assert _text(registry) == before
    assert registry.rolled_back == 1
    # The registry still takes the next record.
    assert registry.apply(lambda: requests.inc(("c1", "ok")))
    assert (
        check_exposition(_text(registry)).value(
            "stormlog_infer_requests_total", status="ok"
        )
        == 1
    )
