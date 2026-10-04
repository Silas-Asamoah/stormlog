"""``WatchStats``: a pure, cheap read of the watcher's own health (P5)."""

from __future__ import annotations

import time

import pytest

from stormlog.infer.watch.stats import (
    COUNTER,
    DESCRIPTORS,
    GAUGE,
    STATE,
    WatchStats,
    counter_value,
)


def test_health_is_a_copy_that_resets_nothing() -> None:
    stats = WatchStats()
    stats.add("scrapes_total", 3, ("ok",))
    stats.add("ticks_missed_total")
    stats.set("history_bytes", 1024)
    stats.set_trigger_state("queue", "pending")
    first = stats.health()
    second = stats.health()
    assert first == second
    first["scrapes_total"][("ok",)] = 99
    first["trigger_state"][("queue",)] = "firing"
    first["history_bytes"] = 0
    assert stats.health() == second
    assert counter_value(second, "scrapes_total", ("ok",)) == 3
    assert counter_value(second, "ticks_missed_total") == 1
    assert counter_value(second, "pruned_total") == 0
    assert second["trigger_state"] == {("queue",): "pending"}


def test_health_costs_the_number_of_series() -> None:
    small, large = WatchStats(), WatchStats()
    for index in range(10):
        small.set_trigger_state(f"t{index}", "inactive")
    for index in range(1000):
        large.set_trigger_state(f"t{index}", "inactive")

    def cost(stats: WatchStats) -> float:
        started = time.perf_counter()
        for _ in range(200):
            stats.health()
        return time.perf_counter() - started

    assert cost(small) < 0.05
    assert cost(large) < 2.0  # linear in series, no I/O


def test_labels_must_match_the_family() -> None:
    stats = WatchStats()
    with pytest.raises(ValueError, match="label values"):
        stats.add("incidents_total", labels=("signal",))
    with pytest.raises(ValueError, match="label values"):
        stats.add("suppressed_total", labels=("",))
    with pytest.raises(KeyError):
        stats.add("history_bytes")
    with pytest.raises(KeyError):
        stats.set("pruned_total", 1)
    with pytest.raises(ValueError, match="trigger state"):
        stats.set_trigger_state("queue", "exploded")


def test_closed_labels_take_only_their_values() -> None:
    """#220 budgets these series exactly, so an unlisted value is a bug."""
    stats = WatchStats()
    stats.add("incidents_total", labels=("signal", "disabled"))
    with pytest.raises(ValueError, match="is not one of"):
        stats.add("incidents_total", labels=("signal", "sort_of"))
    with pytest.raises(ValueError, match="is not one of"):
        stats.add("scrapes_total", labels=("meh",))
    stats.add("incident_windows_total", labels=("post", "missing", "none"))
    # trigger_id comes from the config: open, but bounded by it.
    stats.set_trigger_state("any_configured_id", "pending")


def test_a_gauge_set_to_none_leaves_the_snapshot() -> None:
    stats = WatchStats()
    stats.set("cooldown_remaining_seconds", 12.5)
    assert stats.health()["cooldown_remaining_seconds"] == 12.5
    stats.set("cooldown_remaining_seconds", None)
    assert "cooldown_remaining_seconds" not in stats.health()


def test_descriptors_are_well_formed() -> None:
    keys = [d.key for d in DESCRIPTORS]
    assert len(keys) == len(set(keys))
    for descriptor in DESCRIPTORS:
        assert descriptor.name == "stormlog_watch_" + descriptor.key
        assert descriptor.kind in (GAUGE, COUNTER, STATE)
        assert descriptor.help
        if descriptor.kind == COUNTER:
            assert descriptor.key.endswith("_total")
        if descriptor.kind == STATE:
            assert descriptor.states and descriptor.labels == ("trigger_id",)
        assert set(descriptor.enums) <= set(descriptor.labels)
        assert "trigger_id" not in descriptor.enums
        if descriptor.kind == COUNTER and descriptor.labels:
            assert set(descriptor.enums) == set(descriptor.labels), descriptor.key
    assert WatchStats().health_metrics() == DESCRIPTORS
