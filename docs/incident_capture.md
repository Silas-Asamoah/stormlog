[← Back to docs](index.md)

# Inference incident capture

`stormlog infer watch` keeps a bounded record of a vLLM server's recent past
beside it. When a condition has been bad for long enough, it seals what it has
into an incident bundle, and can open one bounded profiler window. This page
grows with the command (#219). It starts with the bundle format, which the
rest builds on.

## Incident bundles

Each incident is one directory under `<watch root>/incidents/`:

```text
inc-20261003T120000Z-0001-1f2e3d4c/
  .lock            readers, and the generation's writer, hold it shared;
                   deletion takes it exclusively
  gen-0/           written when the incident is sealed
  gen-1/           written by finalization, next to gen-0 until it is published
  manifest.json    names the current generation and lists its files
```

One process owns a store: it holds `incidents/.store.lock` exclusively for
its lifetime, and a second one on the same root fails at once
(`StoreInUse`). A generation is pinned while it is written, so no deletion
can take it from under its writer, and it is published only if every file
the writer put in it is still there.

### Publication

A bundle is written in generations, so a reader always sees one complete
generation:

1. The new generation is written whole, and its files and directories are
   fsynced.
2. `manifest.json` is replaced: written to a temporary file, fsynced, renamed
   over the old one, and the bundle directory fsynced. The manifest names the
   current generation (`current`), and lists each of its files with its size
   and SHA-256.
3. Only then is the previous generation deleted, and only when no reader
   holds the bundle. A later generation hard-links unchanged files, such as
   raw traces, from the previous one instead of copying them.

A crash leaves either the old manifest with its whole generation, or the new
one with its whole generation. When the watcher next starts, before it prunes
anything, it removes any generation the manifest does not name. A bundle a
crash left with no manifest is sealed as `interrupted`, with `complete:
false`, from whatever its `gen-0` holds. One that holds nothing is removed
after an hour.

An incident's membership is frozen when it is sealed: its manifest has
`membership_frozen: true`. A trigger that relates to it later is recorded in
the watcher's ledger, and the bundle does not change for it.

### Reading a bundle

From Python, pin the bundle for the whole read:

```python
from stormlog.infer.watch.store import open_incident_bundle

with open_incident_bundle("watch/incidents/inc-20261003T120000Z-0001-1f2e3d4c") as view:
    print(view.manifest.status, view.manifest.current)
    records = view.file("incident.jsonl").read_text().splitlines()
```

While the bundle is pinned, no generation it names is deleted. A tool that
cannot take the lock reads `manifest.json`, then the files it names. If one
of them has gone, a newer generation was published in between, so it reads
the manifest again: `read_manifest_snapshot` does this up to three times.

### Disk limits

Every byte a bundle's generations hold is charged to the store's budget
before it is written:
- A raw trace is hard-linked in (copied, where no link can be made) and
  charged its actual size first. Its original name is removed only once the
  generation is published, so an abandoned generation leaves it where it
  was, and anything it grew by meanwhile is charged at publication.
- A file written by the watcher is charged chunk by chunk, before each chunk
  reaches the disk.

A write that would go over its allowance abandons the generation before
anything is published, and frees what it had charged. A file hard-linked
between generations is counted once. The manifest and lock file, a few KiB
per bundle, are not charged.

| Limit | Default | When it is reached |
| --- | --- | --- |
| `max_total_bytes` | 4 GiB | the oldest sealed bundles are removed to make room for a new bundle or generation, never one still open or being finalized; one that still does not fit is not written |
| `max_incident_bytes` | 1 GiB | one bundle's bytes across its generations, a file linked forward counted once: a reservation over it is refused, and a write or link that would take the bundle over it fails before anything is published |
| `max_incidents` | 50 | the oldest sealed bundles are removed |
| `max_age_hours` | 72 | bundles sealed earlier are removed |

Retention removes the oldest seal first, never a bundle that is still open
or being finalized. A bundle a reader holds is skipped, and removed once the
reader lets go.

These limits apply only to what the watcher keeps under its root. vLLM
writes each profiler trace into its own trace directory before the watcher
can measure it, so nothing here bounds that write; see the watcher's
trace-volume settings.
