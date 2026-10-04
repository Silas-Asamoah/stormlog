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
    merge_spans,
    request_spans,
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


def _hook_line(seq: int) -> bytes:
    record = {
        "epoch": "engine-1",
        "seq": seq,
        "kind": "scheduled",
        "start_wall_ns": seq,
    }
    return (json.dumps(record) + "\n").encode()


def _epoch_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "hook" / "host-a" / "engine-1"
    directory.mkdir(parents=True)
    return directory


def test_a_damaged_line_is_skipped_and_counted(tmp_path: Path) -> None:
    # A torn write after a crash: NULs between good records. The tailer
    # reads on, as vLLM's own log reader does, and notes the bad line.
    part = _epoch_dir(tmp_path) / "000001.jsonl.part"
    problems = tmp_path / "hook-problems.jsonl"
    tailer = HookTailer(tmp_path / "hook", problems=problems)
    part.write_bytes(_hook_line(1) + b"\x00\x00\x00\x00\n" + _hook_line(2) + b"[1]\n")
    assert [record["seq"] for record in tailer.poll()] == [1, 2]
    noted = [json.loads(line) for line in problems.read_text().splitlines()]
    assert [note["kind"] for note in noted] == ["bad_line", "bad_line"]
    assert tailer.bad_lines == 2


def test_a_segment_rewritten_shorter_is_read_again(tmp_path: Path) -> None:
    part = _epoch_dir(tmp_path) / "000001.jsonl.part"
    problems = tmp_path / "hook-problems.jsonl"
    tailer = HookTailer(tmp_path / "hook", problems=problems)
    part.write_bytes(_hook_line(1) + _hook_line(2) + _hook_line(3))
    assert len(tailer.poll()) == 3
    part.write_bytes(_hook_line(4))  # a writer that reopened it truncated
    assert [record["seq"] for record in tailer.poll()] == [4]
    assert (
        json.loads(problems.read_text().splitlines()[-1])["kind"] == "rewritten_shorter"
    )


def test_the_victims_chunk_gaps_are_read_incrementally(tmp_path: Path) -> None:
    # The signals are read every quarter second during recovery, on the GPU
    # host: each read takes only the lines appended since the last, so lines
    # already read are never parsed again.
    output = tmp_path / "victim.jsonl"
    with FakeEngine(FakeEngineConfig(step_seconds=0.001)) as engine:
        run_profile(engine, output, run_id="victim", request_count=4)
    lines = output.read_bytes().splitlines(keepends=True)
    first, rest = b"".join(lines[: len(lines) // 2]), b"".join(lines[len(lines) // 2 :])
    artifact = tmp_path / "growing.jsonl"
    artifact.write_bytes(first + rest[:5])  # a line still being written
    channel = ReferenceChannel(
        hook_root=tmp_path / "hook",
        metrics_url="http://127.0.0.1:9/metrics",
        victim_prefix=VICTIM,
        shared_prefix_tokens=4,
        reference_dir=tmp_path / "reference",
        probes_dir=tmp_path / "probes",
        victim_artifact=artifact,
    )
    early = channel.signals().chunk_gaps
    # Blank out what was read: a reader that starts over would lose it.
    artifact.write_bytes(b" " * (len(first) - 1) + b"\n" + rest)
    late = channel.signals()
    records = [json.loads(line) for line in (first + rest).splitlines()]
    assert early and late.chunk_gaps == chunk_gaps(records, VICTIM)
    # And the victim's requests become the in-flight intervals cadence needs.
    spans = request_spans(records, VICTIM)
    assert len(spans) == 4
    assert late.in_flight == merge_spans(spans)


def test_in_flight_intervals_merge_overlapping_requests() -> None:
    victim = [
        {"event_type": "infer.request", "x_request_id": f"{VICTIM[9:]}-{i}",
         "started_at_ns": start, "ended_at_ns": end}
        for i, (start, end) in enumerate([(10, 20), (15, 30), (40, 50)])
    ]  # fmt: skip
    other = {"event_type": "infer.request", "x_request_id": "neighbor-1",
             "started_at_ns": 0, "ended_at_ns": 100}  # fmt: skip
    spans = request_spans(victim + [other], VICTIM)
    assert spans == [(10, 20), (15, 30), (40, 50)]
    assert merge_spans(spans) == [(10, 30), (40, 50)]


def test_the_tailer_copies_the_epochs_it_read(tmp_path: Path) -> None:
    epoch = _epoch_dir(tmp_path)
    (epoch / "000001.jsonl").write_bytes(_hook_line(1))
    (epoch / "000002.jsonl.part").write_bytes(_hook_line(2))
    (epoch / "key").write_text("k")
    other = tmp_path / "hook" / "host-a" / "engine-0"
    other.mkdir()
    (other / "000001.jsonl").write_bytes(_hook_line(9))
    tailer = HookTailer(tmp_path / "hook")
    tailer.poll()
    # Both epochs were read: every file of each, the key and a segment still
    # being written included, is copied with its layout.
    copied = tailer.copy_to(tmp_path / "truth" / "hook")
    assert copied == 4
    assert (
        tmp_path / "truth" / "hook" / "host-a" / "engine-1" / "000002.jsonl.part"
    ).read_bytes() == _hook_line(2)


def test_the_view_keeps_each_engine_epochs_producer() -> None:
    view = VictimView(VICTIM, 4)
    view.add({"epoch": "engine-1", "kind": "hello", "producer": "vllm:a",
              "clock": {"wall_ns": 100}})  # fmt: skip
    view.add({"epoch": "worker-1", "kind": "hello", "producer": None})
    view.add({"epoch": "engine-2", "kind": "hello", "producer": "vllm:b",
              "clock": {"wall_ns": 900}})  # fmt: skip
    assert view.producers == [(100, "vllm:a"), (900, "vllm:b")]


def test_a_label_names_the_engine_serving_at_its_action() -> None:
    from examples.qualification.inject import name_engine
    from stormlog.infer.qualify.ground_truth import Expectation

    expects = (
        Expectation("kv_preemption_pressure", "kv_cache"),
        Expectation("load_increase", "workload", cause="workload_change"),
    )
    named = name_engine(expects, "vllm:a")
    assert [e.engine for e in named] == ["vllm:a", None]
    assert name_engine(expects, None) == expects


def test_the_engine_serving_is_the_last_to_say_hello_by_then() -> None:
    from examples.qualification.inject import _Truth
    from examples.qualification.outcomes import Slo
    from stormlog.infer.qualify.ground_truth import PhaseWindow

    truth = _Truth("q221-r", PhaseWindow(0, 1), (True, 1.0), Slo(), [], (0, 0),
                   None, False, producers=((900, "vllm:b"), (100, "vllm:a")))  # fmt: skip
    assert truth.producer_at(500) == "vllm:a"
    assert truth.producer_at(1000) == "vllm:b"
    # Before any hello, or with no action time: the first engine.
    assert truth.producer_at(50) == "vllm:a"
    assert truth.producer_at(None) == "vllm:a"


def test_the_view_says_where_a_moment_fell_in_the_step_loop() -> None:
    # A.4 (#218 R12): where each F4a pulse landed decides what the engine
    # was doing when it stopped: inside schedule(), in a step's execution
    # (a GPU wait), or between steps.
    view = VictimView(VICTIM, 4)
    for iteration, start in ((1, 100), (2, 300)):
        view.add(
            {
                "epoch": "engine-1",
                "kind": "scheduled",
                "iteration": str(iteration),
                "start_wall_ns": start,
                "end_wall_ns": start + 10,
                "members": [],
            }
        )
        view.add({"epoch": "engine-1", "kind": "completed", "iteration": str(iteration),
                  "wall_ns": start + 80, "members": []})  # fmt: skip
    assert view.landing(105) == "in_schedule"
    assert view.landing(150) == "in_step"
    assert view.landing(200) == "between_steps"
    assert view.landing(50) == "between_steps"
    assert view.landing(390) == "between_steps"
