"""Incident bundles: byte budgets, generations, readers, recovery and retention."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from stormlog.infer.watch import store as store_module
from stormlog.infer.watch.disk import BudgetExceeded, StoreLimits, bytes_on_disk
from stormlog.infer.watch.store import (
    BUNDLE_NAME,
    STATUS_INTERRUPTED,
    BundleManifest,
    GenerationWriter,
    IncidentStore,
    StoreInUse,
    open_incident_bundle,
    read_manifest_snapshot,
)

KIB = 1024


def _limits(**overrides: Any) -> StoreLimits:
    values: dict[str, Any] = {
        "max_total_bytes": 64 * KIB,
        "max_incident_bytes": 32 * KIB,
        "max_incidents": 10,
        "max_age_hours": 72.0,
    }
    values.update(overrides)
    return StoreLimits(**values)


def _gen0(store: IncidentStore, payload: bytes, *, now_ns: int | None = None) -> str:
    incident_id = store.new_incident_id(now_ns)
    writer = store.new_bundle(incident_id, len(payload) + KIB)
    assert writer is not None
    with writer.file("incident.jsonl") as out:
        out.write(payload)
    writer.publish(sealed_at_ns=now_ns)
    return incident_id


# ------------------------------------------------------------------ bundles


def test_a_sealed_bundle_names_gen0_with_digests(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path, _limits())
    trace = tmp_path / "rank0.1.pt.trace.json.gz"
    trace.write_bytes(b"t" * 300)
    incident_id = store.new_incident_id()
    assert BUNDLE_NAME.match(incident_id)
    writer = store.new_bundle(incident_id, 4 * KIB)
    assert writer is not None
    with writer.file("incident.jsonl") as out:
        out.write(b'{"event_type": "infer.incident"}\n')
    assert writer.adopt(trace, "traces/rank0.1.pt.trace.json.gz") == 300
    manifest = writer.publish()

    assert not trace.exists()  # linked in, then let go once published
    assert manifest.current == "gen-0" and manifest.complete
    assert [f.path for f in manifest.files] == [
        "gen-0/incident.jsonl",
        "gen-0/traces/rank0.1.pt.trace.json.gz",
    ]
    assert all(f.sha256 and len(f.sha256) == 64 for f in manifest.files)
    on_disk = json.loads(
        (tmp_path / "incidents" / incident_id / "manifest.json").read_text()
    )
    assert on_disk["membership_frozen"] is True
    assert BundleManifest.from_dict(on_disk) == manifest
    assert store.budget.used_bytes == 300 + 33
    assert store.budget.reserved_bytes == 0


def test_a_write_over_the_allowance_abandons_the_generation(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path, _limits())
    incident_id = store.new_incident_id()
    writer = store.new_bundle(incident_id, 100)
    assert writer is not None
    out = writer.file("incident.jsonl")
    with pytest.raises(BudgetExceeded):
        out.write(b"z" * 101)
    out.close(sync=False)
    writer.abandon()
    assert not writer.directory.exists()
    assert store.budget.used_bytes == 0 and store.budget.reserved_bytes == 0
    assert store.manifest(incident_id) is None


def _no_cross_device_links(monkeypatch: pytest.MonkeyPatch) -> None:
    import errno

    def link(source: Any, target: Any) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(store_module.os, "link", link)


def test_a_cross_filesystem_adoption_copies_within_the_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A link across devices (EXDEV, also between two bind mounts of one
    device) falls back to a copy charged chunk by chunk."""
    _no_cross_device_links(monkeypatch)
    store = IncidentStore(tmp_path, _limits())
    big = tmp_path / "big.gz"
    big.write_bytes(b"b" * 5000)
    writer = store.new_bundle(store.new_incident_id(), 4000)
    assert writer is not None
    with pytest.raises(BudgetExceeded):
        writer.adopt(big, "traces/big.gz")
    assert big.exists()  # the source is removed only after a complete copy
    writer.abandon()
    writer = store.new_bundle(store.new_incident_id(), 6000)
    assert writer is not None
    assert writer.adopt(big, "traces/big.gz") == 5000
    assert big.exists()  # until the generation is published
    writer.publish()
    assert not big.exists()
    assert (writer.directory / "traces/big.gz").read_bytes() == b"b" * 5000


def test_an_abandoned_generation_leaves_adopted_traces_where_they_were(
    tmp_path: Path,
) -> None:
    """A seal over its allowance used to delete the trace it had moved in."""
    store = IncidentStore(tmp_path, _limits())
    trace = tmp_path / "rank0.pt.trace.json.gz"
    trace.write_bytes(b"t" * 8 * KIB)
    writer = store.new_bundle(store.new_incident_id(), 16 * KIB)
    assert writer is not None
    writer.adopt(trace, "traces/rank0.pt.trace.json.gz")
    with pytest.raises(BudgetExceeded):
        with writer.file("incident.jsonl") as out:
            out.write(b"x" * 16 * KIB)
    writer.abandon()
    assert trace.read_bytes() == b"t" * 8 * KIB
    assert store.budget.used_bytes == 0 == store._scan_bytes()


def test_an_adopted_file_that_grows_is_charged_its_growth(tmp_path: Path) -> None:
    """A producer still writing to the trace kept appending, uncharged."""
    store = IncidentStore(tmp_path, _limits())
    trace = tmp_path / "rank0.pt.trace.json"
    trace.write_bytes(b"t" * 100)
    writer = store.new_bundle(store.new_incident_id(), 2 * KIB)
    assert writer is not None
    assert writer.adopt(trace, "traces/rank0.pt.trace.json") == 100
    with trace.open("ab") as producer:  # the exporter's still-open file
        producer.write(b"g" * 900)
    writer.publish()
    assert store.budget.used_bytes == 1000 == store._scan_bytes()

    trace = tmp_path / "rank1.pt.trace.json"
    trace.write_bytes(b"t" * 100)
    writer = store.new_bundle(store.new_incident_id(), KIB)
    assert writer is not None
    writer.adopt(trace, "traces/rank1.pt.trace.json")
    with trace.open("ab") as producer:
        producer.write(b"g" * 4 * KIB)
    with pytest.raises(BudgetExceeded):
        writer.publish()
    writer.abandon()
    assert trace.exists()
    assert store.budget.used_bytes == 1000 == store._scan_bytes()


def test_a_new_generation_links_traces_and_frees_only_what_it_replaced(
    tmp_path: Path,
) -> None:
    store = IncidentStore(tmp_path, _limits())
    trace = tmp_path / "rank0.pt.trace.json.gz"
    trace.write_bytes(b"t" * 1000)
    incident_id = store.new_incident_id()
    first = store.new_bundle(incident_id, 4 * KIB)
    assert first is not None
    with first.file("incident.jsonl") as out:
        out.write(b"a" * 100)
    first.adopt(trace, "traces/rank0.pt.trace.json.gz")
    sealed = first.publish()

    second = store.next_generation(incident_id, 4 * KIB)
    assert second is not None
    second.link_previous("traces/rank0.pt.trace.json.gz")
    with second.file("incident.jsonl") as out:
        out.write(b"b" * 250)
    with second.file("report.json") as out:
        out.write(b"{}")
    manifest = second.publish(finalized=True)

    bundle = tmp_path / "incidents" / incident_id
    assert manifest.current == "gen-1" and manifest.finalized
    assert manifest.sealed_at_ns == sealed.sealed_at_ns  # the seal time is kept
    assert not (bundle / "gen-0").exists()
    # The trace is one inode: charged once, and still charged.
    assert store.budget.used_bytes == 1000 + 250 + 2
    # What the budget charged is what a fresh scan of the generations finds.
    store.close()  # a restart: the process that held it is gone
    assert IncidentStore(tmp_path, _limits()).budget.used_bytes == 1000 + 250 + 2
    assert bytes_on_disk([bundle / "gen-1"], seen=set()) == 1000 + 250 + 2


def test_a_new_bundle_never_replaces_an_existing_one(tmp_path: Path) -> None:
    """A retried seal with a persisted id used to delete the published bundle."""
    store = IncidentStore(tmp_path, _limits())
    incident_id = _gen0(store, b"kept\n")
    with pytest.raises(FileExistsError):
        store.new_bundle(incident_id, KIB)
    with open_incident_bundle(tmp_path / "incidents" / incident_id) as view:
        assert view.file("incident.jsonl").read_bytes() == b"kept\n"
    assert store.budget.reserved_bytes == 0
    assert store.budget.used_bytes == store._scan_bytes()


def test_max_incident_bytes_bounds_a_bundle_across_its_generations(
    tmp_path: Path,
) -> None:
    """gen-1 links gen-0's trace at no charge and could add another full
    max_incident_bytes: a bundle settled at almost twice the limit."""
    store = IncidentStore(tmp_path, _limits(max_incident_bytes=1000))
    incident_id = store.new_incident_id()
    first = store.new_bundle(incident_id, 1000)
    assert first is not None
    with first.file("traces/t.json") as out:
        out.write(b"t" * 900)
    first.publish()

    second = store.next_generation(incident_id, 1000)
    assert second is not None
    second.link_previous("traces/t.json")
    with pytest.raises(BudgetExceeded):
        with second.file("diagnosis.json") as out:
            out.write(b"d" * 900)
    second.abandon()

    third = store.next_generation(incident_id, 1000)
    assert third is not None
    third.link_previous("traces/t.json")
    with third.file("diagnosis.json") as out:
        out.write(b"d" * 100)  # 900 linked + 100 new: at the limit
    third.publish()
    bundle = tmp_path / "incidents" / incident_id
    assert store_module._payload_bytes(bundle, seen=set()) == 1000
    assert store.budget.used_bytes == store._scan_bytes()


def test_one_store_owns_a_root(tmp_path: Path) -> None:
    """A second process's recover() deleted a live writer's generation, and
    that writer then published a manifest naming nothing."""
    store = IncidentStore(tmp_path, _limits())
    with pytest.raises(StoreInUse, match="another process"):
        IncidentStore(tmp_path, _limits())
    store.close()
    again = IncidentStore(tmp_path, _limits())
    again.close()


def test_a_generation_being_written_is_never_deleted_under_it(
    tmp_path: Path,
) -> None:
    store = IncidentStore(tmp_path, _limits())
    incident_id = _gen0(store, b"first\n")
    bundle = tmp_path / "incidents" / incident_id
    writer = store.next_generation(incident_id, KIB)
    assert writer is not None
    with writer.file("incident.jsonl") as out:
        out.write(b"second\n")
    # Whoever tries (a recovery, a retention pass) defers while it is written.
    assert store._try_delete(writer.directory, bundle) is False
    store._deferred.clear()
    writer.publish()
    with open_incident_bundle(bundle) as view:
        assert view.file("incident.jsonl").read_bytes() == b"second\n"


def test_a_generation_whose_files_vanished_is_never_published(
    tmp_path: Path,
) -> None:
    store = IncidentStore(tmp_path, _limits())
    incident_id = _gen0(store, b"first\n")
    writer = store.next_generation(incident_id, KIB)
    assert writer is not None
    with writer.file("incident.jsonl") as out:
        out.write(b"second\n")
    (writer.directory / "incident.jsonl").unlink()  # removed behind its back
    with pytest.raises(FileNotFoundError, match="incident.jsonl"):
        writer.publish()
    writer.abandon()
    assert store.manifest(incident_id).current == "gen-0"  # type: ignore[union-attr]


def _fd_name(fd: int) -> str:
    """The last path component an open descriptor refers to."""
    import fcntl

    if os.path.isdir("/proc/self/fd"):
        return os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[-1]
    raw = fcntl.fcntl(fd, getattr(fcntl, "F_GETPATH"), b"\0" * 1024)  # macOS
    return bytes(raw).rstrip(b"\0").decode().rsplit("/", 1)[-1]


def test_publication_syncs_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every file and directory reaches the disk before the manifest names
    it, and a new bundle's own entry in the store before anything inside."""
    synced: list[str] = []
    real_fsync = store_module.os.fsync

    def fsync(fd: int) -> None:
        synced.append(_fd_name(fd))
        real_fsync(fd)

    store = IncidentStore(tmp_path, _limits())
    monkeypatch.setattr(store_module.os, "fsync", fsync)
    incident_id = store.new_incident_id()
    writer = store.new_bundle(incident_id, KIB)
    assert writer is not None
    assert synced == ["incidents"]  # the new bundle's directory entry
    with writer.file("incident.jsonl") as out:
        out.write(b"x\n")
    writer.publish()
    assert synced == [
        "incidents",
        "incident.jsonl",  # closing the file
        "incident.jsonl",  # the generation, file by file
        "gen-0",
        incident_id,  # the generation's entry in the bundle
        "manifest.json.tmp",
        incident_id,  # the manifest's rename
    ]


# ------------------------------------------------------------------ readers


def test_a_pinned_reader_keeps_its_generation_until_it_lets_go(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path, _limits())
    incident_id = _gen0(store, b"first\n")
    bundle = tmp_path / "incidents" / incident_id

    with open_incident_bundle(bundle) as view:
        assert view.manifest.current == "gen-0"
        second = store.next_generation(incident_id, KIB)
        assert second is not None
        with second.file("incident.jsonl") as out:
            out.write(b"second\n")
        second.publish(finalized=True)
        # Published, but the reader's generation is still there to read.
        assert view.file("incident.jsonl").read_bytes() == b"first\n"
        assert store.deferred == 1

    assert store.reclaim_deferred() == 1
    assert not (bundle / "gen-0").exists()
    assert store.budget.used_bytes == len(b"second\n")
    with open_incident_bundle(bundle) as view:
        assert view.file("incident.jsonl").read_bytes() == b"second\n"


def test_a_reader_without_the_lock_retries_on_a_vanished_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = IncidentStore(tmp_path, _limits())
    incident_id = _gen0(store, b"first\n")
    bundle = tmp_path / "incidents" / incident_id
    stale = store_module._read_manifest(bundle)
    second = store.next_generation(incident_id, KIB)
    assert second is not None
    with second.file("incident.jsonl") as out:
        out.write(b"second\n")
    second.publish()
    stale_reads = [stale]
    real = store_module._read_manifest

    def read(path: Path) -> BundleManifest:
        # The first read sees the manifest from before the new generation.
        return stale_reads.pop() if stale_reads else real(path)

    monkeypatch.setattr(store_module, "_read_manifest", read)
    view = read_manifest_snapshot(bundle)
    assert view.manifest.current == "gen-1"


# ----------------------------------------------------------------- recovery


def _unpublished(store: IncidentStore, payload: bytes) -> GenerationWriter:
    writer = store.new_bundle(store.new_incident_id(), KIB)
    assert writer is not None
    with writer.file("incident.jsonl") as out:
        out.write(payload)
    return writer


def test_recovery_seals_a_bundle_left_without_a_manifest(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path, _limits())
    writer = _unpublished(store, b"partial\n")  # the watcher died before sealing

    store.close()  # a restart: the process that held it is gone
    report = IncidentStore(tmp_path, _limits()).recover()

    assert report.sealed_interrupted == [writer.incident_id]
    manifest = store.manifest(writer.incident_id)
    assert manifest is not None
    assert manifest.status == STATUS_INTERRUPTED
    assert not manifest.complete and not manifest.finalized
    assert [f.path for f in manifest.files] == ["gen-0/incident.jsonl"]


def test_recovery_removes_old_empty_bundles_and_keeps_young_ones(
    tmp_path: Path,
) -> None:
    store = IncidentStore(tmp_path, _limits())
    old = store.root / store.new_incident_id()
    young = store.root / store.new_incident_id()
    old.mkdir()
    young.mkdir()
    long_ago = time.time() - 2 * 3600
    os.utime(old, (long_ago, long_ago))

    report = store.recover()

    assert report.junk_removed == [old.name]
    assert young.exists() and not old.exists()


@pytest.mark.parametrize("crash_point", ["before_manifest", "before_reclaim"])
def test_a_crash_at_each_publication_boundary_leaves_one_whole_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_point: str
) -> None:
    store = IncidentStore(tmp_path, _limits())
    incident_id = _gen0(store, b"first\n")
    bundle = tmp_path / "incidents" / incident_id
    second = store.next_generation(incident_id, KIB)
    assert second is not None
    with second.file("incident.jsonl") as out:
        out.write(b"second\n")

    class Crash(Exception):
        pass

    def crash(*_args: Any, **_kwargs: Any) -> None:
        raise Crash

    target = "_stage_manifest" if crash_point == "before_manifest" else None
    if target:
        monkeypatch.setattr(store_module, target, crash)
    else:
        monkeypatch.setattr(IncidentStore, "_reclaim_old_generations", crash)
    with pytest.raises(Crash):
        second.publish()
    monkeypatch.undo()
    second._unpin()  # the process died, and its locks with it

    # A reader at this moment sees one whole generation.
    view = read_manifest_snapshot(bundle)
    expected = b"first\n" if crash_point == "before_manifest" else b"second\n"
    assert view.file("incident.jsonl").read_bytes() == expected

    store.close()  # a restart: the process that held it is gone
    report = IncidentStore(tmp_path, _limits()).recover()
    assert report.generations_removed == 1
    assert [p.name for p in sorted(bundle.glob("gen-*"))] == [view.manifest.current]


def test_recovery_keeps_the_deletions_a_reader_defers(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path, _limits())
    incident_id = _gen0(store, b"first\n")
    bundle = tmp_path / "incidents" / incident_id
    left = store.next_generation(incident_id, KIB)  # a crash left it unnamed
    assert left is not None
    with left.file("incident.jsonl") as out:
        out.write(b"orphan\n")
    left._unpin()  # the process died, and its locks with it
    store.close()  # a restart: the process that held it is gone
    restarted = IncidentStore(tmp_path, _limits())
    with open_incident_bundle(bundle):
        report = restarted.recover()
        assert report.generations_removed == 0
        assert restarted.deferred == 1
    assert restarted.reclaim_deferred() == 1
    assert [p.name for p in bundle.glob("gen-*")] == ["gen-0"]
    assert restarted.budget.used_bytes == restarted._scan_bytes()


def test_a_publish_that_fails_after_the_rename_stays_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The directory fsync after the manifest's rename can fail (EIO,
    EMFILE). The caller's abandon() then deleted the generation the
    manifest named, and recover() deleted the other one."""
    store = IncidentStore(tmp_path, _limits())
    incident_id = _gen0(store, b"first\n")
    bundle = tmp_path / "incidents" / incident_id
    second = store.next_generation(incident_id, KIB)
    assert second is not None
    with second.file("incident.jsonl") as out:
        out.write(b"second\n")
    real_fsync_dir = store_module._fsync_dir

    def fsync_dir(directory: Path) -> None:
        if directory == bundle and (bundle / "manifest.json").exists():
            if store_module._read_manifest(bundle).current == "gen-1":
                raise OSError(5, "Input/output error")
        real_fsync_dir(directory)

    monkeypatch.setattr(store_module, "_fsync_dir", fsync_dir)
    with pytest.raises(OSError, match="Input/output"):
        second.publish()
    second.abandon()  # as the API tells a caller whose publish raised
    monkeypatch.undo()

    with open_incident_bundle(bundle) as view:
        assert view.manifest.current == "gen-1"
        assert view.file("incident.jsonl").read_bytes() == b"second\n"
    store.close()  # a restart: the process that held it is gone
    IncidentStore(tmp_path, _limits()).recover()
    assert (bundle / "gen-1" / "incident.jsonl").read_bytes() == b"second\n"


def test_recovery_keeps_every_generation_when_the_named_one_is_missing(
    tmp_path: Path,
) -> None:
    store = IncidentStore(tmp_path, _limits())
    incident_id = _gen0(store, b"first\n")
    bundle = tmp_path / "incidents" / incident_id
    (bundle / "gen-0").rename(bundle / "gen-7")  # the manifest names gen-0
    report = store.recover()
    assert report.unreadable == [incident_id]
    assert report.generations_removed == 0
    assert (bundle / "gen-7" / "incident.jsonl").exists()


def test_a_generation_left_by_a_crash_is_forgotten_when_replaced(
    tmp_path: Path,
) -> None:
    store = IncidentStore(tmp_path, _limits())
    incident_id = _gen0(store, b"first\n")
    left = store.next_generation(incident_id, KIB)
    assert left is not None
    with left.file("incident.jsonl") as out:
        out.write(b"orphan" * 100)  # never published: the watcher died
    store.close()  # a restart: the process that held it is gone
    restarted = IncidentStore(tmp_path, _limits())  # charges it, from disk
    retry = restarted.next_generation(incident_id, KIB)
    assert retry is not None
    with retry.file("incident.jsonl") as out:
        out.write(b"second\n")
    retry.publish()
    assert restarted.budget.used_bytes == restarted._scan_bytes()


def test_recovery_removes_a_half_written_manifest(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path, _limits())
    incident_id = _gen0(store, b"x\n")
    bundle = tmp_path / "incidents" / incident_id
    (bundle / "manifest.json.tmp").write_text("{trunc")
    report = store.recover()
    assert report.temporaries_removed == 1
    assert not (bundle / "manifest.json.tmp").exists()
    assert store.manifest(incident_id) is not None


# ---------------------------------------------------------------- retention


def test_retention_prunes_oldest_seal_first_and_spares_protected(
    tmp_path: Path,
) -> None:
    store = IncidentStore(tmp_path, _limits(max_incidents=2))
    base = time.time_ns()
    ids = [_gen0(store, b"r\n", now_ns=base + i) for i in range(4)]

    pruned = store.prune(now_ns=base + 10, protected=frozenset({ids[0]}))

    assert [p.incident_id for p in pruned] == [ids[1], ids[2]]
    assert {p.reason for p in pruned} == {"max_incidents"}
    assert [m.incident_id for _p, m in store.bundles()] == [ids[0], ids[3]]


def test_retention_by_age_and_by_bytes(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path, _limits(max_age_hours=1.0))
    now = time.time_ns()
    old = _gen0(store, b"o\n", now_ns=now - 2 * 3600 * 10**9)
    fresh = _gen0(store, b"f\n", now_ns=now)
    pruned = store.prune(now_ns=now)
    assert [(p.incident_id, p.reason) for p in pruned] == [(old, "max_age_hours")]
    assert store.manifest(fresh) is not None

    store.limits = _limits(max_total_bytes=1, max_incident_bytes=1)
    pruned = store.prune(now_ns=now)
    assert [(p.incident_id, p.reason) for p in pruned] == [(fresh, "max_total_bytes")]
    assert store.budget.used_bytes == 0


def test_a_full_store_makes_room_for_the_newest_incident(tmp_path: Path) -> None:
    """A flight recorder keeps the newest: the oldest unprotected bundles go
    to fit a reservation, and only a reservation that still cannot fit is
    refused."""
    store = IncidentStore(
        tmp_path, _limits(max_total_bytes=1000, max_incident_bytes=600)
    )
    base = time.time_ns()
    oldest = _gen0_exact(store, b"a" * 500, now_ns=base)
    kept = _gen0_exact(store, b"b" * 400, now_ns=base + 1)

    writer = store.new_bundle(store.new_incident_id(), 300)
    assert writer is not None
    pruned = store.take_pruned()
    assert [(p.incident_id, p.reason, p.bytes) for p in pruned] == [
        (oldest, "max_total_bytes", 500)
    ]
    assert store.take_pruned() == []
    writer.abandon()

    # Never a protected bundle: open, or being finalized.
    assert (
        store.new_bundle(store.new_incident_id(), 700, protected=frozenset({kept}))
        is None
    )
    assert store.manifest(kept) is not None
    assert store.budget.used_bytes == store._scan_bytes()


def _gen0_exact(store: IncidentStore, payload: bytes, *, now_ns: int) -> str:
    incident_id = store.new_incident_id(now_ns)
    writer = store.new_bundle(incident_id, len(payload))
    assert writer is not None
    with writer.file("incident.jsonl") as out:
        out.write(payload)
    writer.publish(sealed_at_ns=now_ns)
    return incident_id


def test_a_crash_while_pruning_never_brings_a_bundle_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rmtree deletes in directory order; with the manifest first (ext4's
    hash order can do that), a crash left gen-0 behind and recover() sealed
    the deleted bundle as interrupted."""
    store = IncidentStore(tmp_path, _limits(max_incidents=1))
    base = time.time_ns()
    old = _gen0(store, b"o\n", now_ns=base)
    new = _gen0(store, b"n\n", now_ns=base + 1)
    real_rmtree = store_module.shutil.rmtree

    class Crash(BaseException):
        pass

    def crash_after_the_manifest(path: Any, ignore_errors: bool = False) -> None:
        manifest = Path(path) / "manifest.json"
        if manifest.exists():
            manifest.unlink()  # the manifest went first...
            raise Crash  # ...and the process died
        real_rmtree(path, ignore_errors=ignore_errors)

    monkeypatch.setattr(store_module.shutil, "rmtree", crash_after_the_manifest)
    with pytest.raises(Crash):
        store.prune(now_ns=base + 2)
    monkeypatch.undo()
    store.close()

    restarted = IncidentStore(tmp_path, _limits(max_incidents=1))
    report = restarted.recover()
    assert report.sealed_interrupted == []
    assert [m.incident_id for _p, m in restarted.bundles()] == [new]
    assert not any(
        p.name.startswith(".trash") for p in (tmp_path / "incidents").iterdir()
    )
    assert old not in {p.name for p in (tmp_path / "incidents").iterdir()}


def test_bytes_a_deferred_deletion_will_free_count_as_freed(tmp_path: Path) -> None:
    """With the oldest bundle's deletion deferred by a reader, retention
    must not also remove the next one to make up the same bytes."""
    store = IncidentStore(tmp_path, _limits())
    base = time.time_ns()
    oldest = _gen0_exact(store, b"a" * 6000, now_ns=base)
    newer = _gen0_exact(store, b"b" * 3000, now_ns=base + 1)
    store.limits = _limits(max_total_bytes=7000, max_incident_bytes=7000)
    with open_incident_bundle(tmp_path / "incidents" / oldest):
        assert store.prune(now_ns=base + 2) == []
        assert store.deferred == 1
        assert store.manifest(newer) is not None
    assert store.reclaim_deferred() == 1
    assert [m.incident_id for _p, m in store.bundles()] == [newer]


def test_a_bundle_being_read_is_pruned_once_the_reader_leaves(
    tmp_path: Path,
) -> None:
    store = IncidentStore(tmp_path, _limits(max_incidents=1))
    base = time.time_ns()
    first = _gen0(store, b"1\n", now_ns=base)
    _gen0(store, b"2\n", now_ns=base + 1)
    bundle = tmp_path / "incidents" / first
    entered = threading.Event()
    leave = threading.Event()

    def reader() -> None:
        with open_incident_bundle(bundle):
            entered.set()
            leave.wait(5)

    thread = threading.Thread(target=reader)
    thread.start()
    entered.wait(5)
    try:
        assert store.prune(now_ns=base + 2) == []
        assert bundle.exists() and store.deferred == 1
    finally:
        leave.set()
        thread.join(5)
    assert store.reclaim_deferred() == 1
    assert not bundle.exists()


def _assert_charged_as_held(store: IncidentStore) -> None:
    assert store.budget.used_bytes == store._scan_bytes()


@pytest.mark.parametrize("reader_at_prune", [False, True])
def test_a_deferred_generation_is_never_forgotten_twice(
    tmp_path: Path, reader_at_prune: bool
) -> None:
    """A reader pins A while its finalizer publishes gen-1, so gen-0's
    deletion is deferred; retention then removes all of A. Before, the
    deferred gen-0 was forgotten again, and the store held more than its cap."""
    store = IncidentStore(tmp_path, _limits(max_incidents=4))
    base = time.time_ns()
    first = _gen0(store, b"A" * 20_000, now_ns=base)
    bundle = tmp_path / "incidents" / first
    entered, leave = threading.Event(), threading.Event()

    def reader() -> None:
        with open_incident_bundle(bundle):
            entered.set()
            leave.wait(5)

    thread = threading.Thread(target=reader)
    thread.start()
    entered.wait(5)
    try:
        writer = store.next_generation(first, 2 * KIB)
        assert writer is not None
        with writer.file("incident.jsonl") as out:
            out.write(b"a" * 1000)
        writer.publish(finalized=True)
        assert store.deferred == 1
        _assert_charged_as_held(store)
        if not reader_at_prune:
            leave.set()
            thread.join(5)
        for offset in range(4):
            _gen0(store, b"o" * 5_000, now_ns=base + 1 + offset)
        _assert_charged_as_held(store)
        pruned = store.prune(now_ns=base + 10)
        assert [p.incident_id for p in pruned] == ([] if reader_at_prune else [first])
        _assert_charged_as_held(store)
    finally:
        leave.set()
        thread.join(5)
    store.reclaim_deferred()
    _assert_charged_as_held(store)
    assert not bundle.exists() and store.deferred == 0
    store.reclaim_deferred()  # again: nothing left to forget
    _assert_charged_as_held(store)


def test_manifest_parsing_rejects_foreign_and_malformed_documents() -> None:
    with pytest.raises(ValueError, match="not an incident bundle"):
        BundleManifest.from_dict({"format": "other"})
    with pytest.raises(ValueError, match="schema_version"):
        BundleManifest.from_dict(
            {"format": "stormlog.infer.incident_bundle", "schema_version": 2}
        )
    with pytest.raises(ValueError, match="malformed"):
        BundleManifest.from_dict(
            {"format": "stormlog.infer.incident_bundle", "schema_version": 1}
        )
