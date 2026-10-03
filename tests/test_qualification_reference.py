"""The reference channel's reader, against the fake vLLM engine."""

from __future__ import annotations

import json
from pathlib import Path

from examples.qualification.fake_engine import FakeEngine, FakeEngineConfig
from examples.qualification.reference import (
    HookTailer,
    ReferenceChannel,
    VictimView,
    chunk_gaps,
    scrape_metrics,
)
from tests.qualification_fake_engine_helpers import (
    chat,
    chats_in_background,
    join_all,
    run_profile,
    wait_until,
    words,
)

VICTIM = "chatcmpl-stormlog-victim-"


def _engine(hook: Path, **changes: object) -> FakeEngine:
    values: dict[str, object] = {
        "step_seconds": 0.001,
        "hook_dir": hook,
        "hook_seal_seconds": 0.2,
    }
    values.update(changes)
    return FakeEngine(FakeEngineConfig(**values))  # type: ignore[arg-type]


def test_the_tailer_reads_each_record_once_and_notes_when(tmp_path: Path) -> None:
    hook = tmp_path / "hook"
    firstseen, seals = tmp_path / "firstseen.jsonl", tmp_path / "seals.jsonl"
    tailer = HookTailer(hook, firstseen=firstseen, seals=seals)
    seen: list[dict[str, object]] = []
    with _engine(hook) as engine:
        for index in range(3):
            chat(engine, words(6, f"a{index}"), max_tokens=2)
            seen += tailer.poll()
        assert wait_until(lambda: bool(list(hook.rglob("*.jsonl"))), timeout=5)
        seen += tailer.poll()
    seen += tailer.poll()  # the closing goodbye
    keys = [(record["epoch"], record["seq"]) for record in seen]
    assert len(keys) == len(set(keys))
    by_epoch: dict[str, list[int]] = {}
    for epoch, seq in keys:
        by_epoch.setdefault(str(epoch), []).append(int(seq))  # type: ignore[call-overload]
    for sequence in by_epoch.values():
        assert sequence == list(range(len(sequence)))
    noted = [json.loads(line) for line in firstseen.read_text().splitlines()]
    assert len(noted) == len(seen)
    assert all(note["first_seen_ns"] > 0 for note in noted)
    assert seals.read_text().strip(), "no seal was observed"
    assert any(record["kind"] == "goodbye" for record in seen)


def test_a_line_still_being_written_waits(tmp_path: Path) -> None:
    epoch = tmp_path / "hook" / "host-boot" / "engine-1"
    epoch.mkdir(parents=True)
    part = epoch / "000000.jsonl.part"
    part.write_text('{"epoch": "engine-1", "seq": 0, "kind": "hello"}\n{"epoch": "eng')
    tailer = HookTailer(tmp_path / "hook")
    assert [record["seq"] for record in tailer.poll()] == [0]
    with part.open("a") as handle:
        handle.write('ine-1", "seq": 1, "kind": "heartbeat"}\n')
    part.rename(epoch / "000000.jsonl")
    assert [record["seq"] for record in tailer.poll()] == [1]
    assert tailer.poll() == []


def test_the_victim_view_follows_only_victim_requests(tmp_path: Path) -> None:
    hook = tmp_path / "hook"
    tailer = HookTailer(hook)
    view = VictimView(VICTIM, shared_prefix_tokens=8)
    shared = words(8, "shared")
    with _engine(hook, block_size=4) as engine:
        chat(engine, shared + " one", max_tokens=2, request_id="stormlog-victim-0")
        chat(engine, shared + " two", max_tokens=2, request_id="stormlog-victim-1")
        chat(engine, words(6, "n"), max_tokens=2, request_id="stormlog-neighbor-0")
        assert wait_until(lambda: len(engine.engine.finished) == 3)
        for record in tailer.poll():
            view.add(record)
    assert len(view.waits) == 2
    assert all(wait >= 0 for _at, wait in view.waits)
    # The second victim request found the shared prefix cached.
    assert [share for _at, share in view.cached_fraction][1] == 1.0
    assert view.preemptions == []
    assert len(view.step_starts) >= 4


def test_victim_preemptions_come_from_the_steps_preempted_lists(tmp_path: Path) -> None:
    hook = tmp_path / "hook"
    tailer = HookTailer(hook)
    view = VictimView(VICTIM, shared_prefix_tokens=4)
    config = {"num_gpu_blocks": 6, "block_size": 4, "max_num_seqs": 4}
    with _engine(hook, **config) as engine:
        engine.pause_engine()
        threads = chats_in_background(
            engine, [words(4, tag) for tag in "abc"], max_tokens=8
        )
        assert wait_until(lambda: len(engine.engine.waiting) == 3)
        engine.resume_engine()
        join_all(threads)
        log = list(engine.engine.preemption_log)
    # chats_in_background sends no X-Request-Id, so nothing is a victim here...
    for record in tailer.poll():
        view.add(record)
    assert log and view.preemptions == []
    # ...and naming every request a victim finds each logged preemption.
    every = VictimView("chatcmpl-", shared_prefix_tokens=4)
    for record in HookTailer(hook).poll():
        every.add(record)
    assert len(every.preemptions) == len(log)


def test_scrapes_read_the_waiting_count_and_kv_usage(tmp_path: Path) -> None:
    with FakeEngine(FakeEngineConfig(step_seconds=0.001)) as engine:
        chat(engine, words(6, "a"), max_tokens=2)
        taken = scrape_metrics(engine.metrics_url)
    assert taken.error is None
    assert (taken.waiting, taken.kv_usage) == (0.0, 0.0)
    gone = scrape_metrics("http://127.0.0.1:9/metrics", timeout_seconds=1)
    assert gone.error is not None and gone.waiting is None


def test_chunk_gaps_are_rebuilt_from_the_victims_client_records(tmp_path: Path) -> None:
    output = tmp_path / "victim.jsonl"
    with FakeEngine(FakeEngineConfig(step_seconds=0.001)) as engine:
        run_profile(engine, output, run_id="victim", request_count=2)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    gaps = chunk_gaps(records, VICTIM)
    requests = [r for r in records if r.get("event_type") == "infer.request"]
    expected = sum(len(r["chunk_interarrival_ms"]) for r in requests)
    assert len(gaps) == expected > 0
    assert gaps == sorted(gaps)
    assert chunk_gaps(records, "chatcmpl-stormlog-other-") == []


def test_the_channel_polls_into_signals(tmp_path: Path) -> None:
    hook = tmp_path / "hook"
    with _engine(hook) as engine:
        channel = ReferenceChannel(
            hook_root=hook,
            metrics_url=engine.metrics_url,
            victim_prefix=VICTIM,
            shared_prefix_tokens=4,
            reference_dir=tmp_path / "truth" / "reference",
            probes_dir=tmp_path / "probes",
        )
        chat(engine, words(6, "a"), max_tokens=2, request_id="stormlog-victim-0")
        channel.poll()
        channel.poll()
    signals = channel.signals()
    assert len(signals.waits) == 1
    assert len(signals.waiting) == 2
    assert (tmp_path / "truth" / "reference" / "scrapes.jsonl").exists()
    assert (tmp_path / "probes" / "hook-firstseen.jsonl").exists()
