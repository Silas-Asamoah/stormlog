"""Place server telemetry timestamps on the client artifact's clock.

The server join uses the inference clock-alignment model: an
``infer.clock_alignment`` record from the client artifact, or one built from
``--clock-offset-ns`` and ``--clock-uncertainty-ns``, applied through
``align_timestamp``. Clock domains name a host boot (see
:func:`~stormlog.infer.host_clock.wall_clock_domain`), so equal names mean one
clock.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .correlation_accounting import (
    AlignedTimestamp,
    align_timestamp,
    covering_alignments,
    resolve_inference_events,
)
from .correlation_codec import expand_inference_records
from .correlation_events import (
    ClockAlignmentEvent,
    CorrelationContext,
    parse_inference_record,
)
from .host_clock import is_boot_qualified, wall_clock_domain
from .telemetry import TelemetrySample

CLI_ALIGNMENT_ID = "cli:clock-offset"
# Marks the server side of an equal, boot-less pair of clock domains.
UNVERIFIED_SERVER_SUFFIX = "#server"
UNCOVERED = "uncovered"
AMBIGUOUS = "ambiguous"

# A client-clock time and the alignment record used (None for one shared clock).
Placement = tuple[AlignedTimestamp, ClockAlignmentEvent | None]


@dataclass(frozen=True)
class ServerClock:
    """How one server clock domain maps onto the client artifact's domain."""

    server_domain: str
    client_domain: str
    evidence: str
    alignments: tuple[ClockAlignmentEvent, ...] = ()
    same_clock_uncertainty_ns: int = 0
    overridden: tuple[str, ...] = ()
    ignored: tuple[str, ...] = ()

    def align(self, timestamp_ns: int) -> Placement | str:
        """Return the client-clock time and the record used, or why none applies.

        The record is ``None`` for one shared clock. A timestamp covered by no
        alignment, or by several, yields ``UNCOVERED`` or ``AMBIGUOUS``.
        """
        if self.server_domain == self.client_domain:
            same = align_timestamp(
                timestamp_ns,
                from_clock_domain=self.server_domain,
                to_clock_domain=self.client_domain,
                alignments=(),
            )
            placed = AlignedTimestamp(
                same.value_ns, self.same_clock_uncertainty_ns, same.clock_domain
            )
            return placed, None
        matches = covering_alignments(
            timestamp_ns,
            from_clock_domain=self.server_domain,
            to_clock_domain=self.client_domain,
            alignments=self.alignments,
        )
        if len(matches) != 1:
            return UNCOVERED if not matches else AMBIGUOUS
        placed = align_timestamp(
            timestamp_ns,
            from_clock_domain=self.server_domain,
            to_clock_domain=self.client_domain,
            alignments=matches,
        )
        return placed, matches[0]


@dataclass(frozen=True)
class SampleAlignment:
    """Client-clock times for the samples one clock could place."""

    aligned: dict[TelemetrySample, AlignedTimestamp]
    unaligned: dict[str, int]
    applied: list[dict[str, Any]]
    # Samples no single alignment placed, with UNCOVERED or AMBIGUOUS.
    unplaced: dict[TelemetrySample, str]


def align_samples(
    samples: Iterable[TelemetrySample], clock: ServerClock
) -> SampleAlignment:
    """Place every sample; count the ones no single alignment covers."""
    aligned: dict[TelemetrySample, AlignedTimestamp] = {}
    unplaced: dict[TelemetrySample, str] = {}
    unaligned = {UNCOVERED: 0, AMBIGUOUS: 0}
    records: dict[int, ClockAlignmentEvent | None] = {}
    counts: dict[int, int] = {}
    for sample in samples:
        placement = clock.align(sample.observed_at_ns)
        if isinstance(placement, str):
            unaligned[placement] += 1
            unplaced[sample] = placement
            continue
        aligned[sample], record = placement
        records[id(record)] = record
        counts[id(record)] = counts.get(id(record), 0) + 1
    applied = [_applied(clock, records[key], counts[key]) for key in records]
    return SampleAlignment(aligned, unaligned, applied, unplaced)


def _applied(
    clock: ServerClock, record: ClockAlignmentEvent | None, samples: int
) -> dict[str, Any]:
    if record is None:
        return {
            "source": "same_host",
            "event_id": None,
            "from_clock_domain": clock.server_domain,
            "to_clock_domain": clock.client_domain,
            "offset_ns": 0,
            "uncertainty_ns": clock.same_clock_uncertainty_ns,
            "valid_from_ns": None,
            "valid_to_ns": None,
            "aligned_samples": samples,
        }
    return {
        "source": "operator" if record.event_id == CLI_ALIGNMENT_ID else "artifact",
        "event_id": record.event_id,
        "from_clock_domain": record.from_clock_domain,
        "to_clock_domain": record.to_clock_domain,
        "offset_ns": record.offset_ns,
        "uncertainty_ns": record.uncertainty_ns,
        "valid_from_ns": record.valid_from_ns,
        "valid_to_ns": record.valid_to_ns,
        "aligned_samples": samples,
    }


def client_clock_domain(artifact: Mapping[str, Any]) -> str | None:
    """Return the client artifact's wall clock domain, naming the boot if known.

    The input must be expanded by the artifact/sequence decoder first.
    Artifacts written before domains carried a boot ID record
    ``{host}/unix_epoch_ns`` and keep the boot ID in ``metadata``; both parts
    together name the same clock as a boot-qualified domain.
    """
    context = artifact.get("context")
    if not isinstance(context, Mapping):
        return None
    recorded = context.get("clock_domain")
    if isinstance(recorded, str) and is_boot_qualified(recorded):
        return recorded
    host = _domain_part(context.get("host"))
    if host is not None:
        return wall_clock_domain(host, _domain_part(_metadata_boot_id(artifact)))
    return recorded if isinstance(recorded, str) and recorded else None


def _domain_part(value: object) -> str | None:
    """Return ``value`` if it can be one segment of a clock domain name."""
    return value if isinstance(value, str) and value and "/" not in value else None


def _metadata_boot_id(artifact: Mapping[str, Any]) -> object:
    metadata = artifact.get("metadata")
    return metadata.get("boot_id") if isinstance(metadata, Mapping) else None


def artifact_alignments(
    records: Iterable[Mapping[str, Any]], run_id: str
) -> tuple[ClockAlignmentEvent, ...]:
    """Parse the artifact's ``infer.clock_alignment`` records.

    This run's records are deduplicated by the correlation resolver; records
    that name another run follow them unresolved, so a copied calibration can
    be reported instead of stopping the analysis. ``build_server_clock`` never
    places samples with them.
    """
    records = expand_inference_records(records)
    alignments = [
        event
        for event in (
            parse_inference_record(record)
            for record in records
            if record.get("event_type") == ClockAlignmentEvent.EVENT_TYPE
        )
        if isinstance(event, ClockAlignmentEvent)
    ]
    own = [event for event in alignments if event.context.run_id == run_id]
    foreign = tuple(event for event in alignments if event.context.run_id != run_id)
    return resolve_inference_events(own).alignments + foreign


def build_server_clock(
    *,
    run_id: str,
    server_domain: str,
    client_domain: str,
    recorded: tuple[ClockAlignmentEvent, ...],
    offset_ns: int | None,
    uncertainty_ns: int | None,
) -> ServerClock | str:
    """Choose the clock evidence for one server domain, or say what is missing.

    Operator-supplied flags take precedence over artifact records for the same
    pair of domains; the replaced records are listed in ``overridden``. Records
    from another run never place samples; they are listed in ``ignored``.
    """
    shared = server_domain == client_domain and is_boot_qualified(server_domain)
    issue = _flag_issue(offset_ns, uncertainty_ns, shared)
    if issue is not None:
        return issue
    if shared:
        return _same_clock(
            run_id, server_domain, client_domain, offset_ns, uncertainty_ns
        )
    aligned_domain = _alignable_domain(server_domain, client_domain, offset_ns)
    if aligned_domain is None:
        return "clock_domain_unverified"
    server_domain = aligned_domain
    pair = tuple(
        item
        for item in recorded
        if (item.from_clock_domain, item.to_clock_domain)
        == (server_domain, client_domain)
    )
    if offset_ns is None:
        return _recorded_clock(run_id, server_domain, client_domain, pair)
    operator = cli_alignment(
        run_id, server_domain, client_domain, offset_ns, uncertainty_ns or 0
    )
    return ServerClock(
        server_domain,
        client_domain,
        "operator_supplied",
        (operator,),
        overridden=tuple(item.event_id for item in pair),
    )


def _alignable_domain(
    server_domain: str, client_domain: str, offset_ns: int | None
) -> str | None:
    """Name the server clock an alignment can start from, or None if none can.

    Equal names without a boot ID could still be two machines. Only the
    operator's offset connects them, under a distinct name for the server side
    so ``align_timestamp`` does not treat the equal names as one clock.
    """
    if server_domain != client_domain:
        return server_domain
    if offset_ns is None:
        return None
    return server_domain + UNVERIFIED_SERVER_SUFFIX


def _flag_issue(
    offset_ns: int | None, uncertainty_ns: int | None, shared_clock: bool
) -> str | None:
    """Reject clock flags that cannot describe this pair of clocks."""
    if offset_ns is not None and uncertainty_ns is None:
        return "clock_uncertainty_required"
    if not shared_clock and offset_ns is None and uncertainty_ns is not None:
        # An uncertainty alone describes one shared clock; never drop it silently.
        return "clock_offset_required"
    return None


def _recorded_clock(
    run_id: str,
    server_domain: str,
    client_domain: str,
    pair: tuple[ClockAlignmentEvent, ...],
) -> ServerClock | str:
    """Use this run's alignment records; a calibration from another run is ignored."""
    own = tuple(item for item in pair if item.context.run_id == run_id)
    if not own:
        return (
            "clock_alignment_from_another_run" if pair else "clock_alignment_required"
        )
    return ServerClock(
        server_domain,
        client_domain,
        "artifact_record",
        own,
        ignored=tuple(item.event_id for item in pair if item not in own),
    )


def _same_clock(
    run_id: str,
    server_domain: str,
    client_domain: str,
    offset_ns: int | None,
    uncertainty_ns: int | None,
) -> ServerClock | str:
    # Validate operator values exactly as an alignment record would be.
    supplied = cli_alignment(
        run_id, server_domain, client_domain, offset_ns or 0, uncertainty_ns or 0
    )
    if supplied.offset_ns != 0:
        # One host and boot is one clock, so a nonzero offset contradicts it.
        return "clock_offset_on_shared_clock"
    return ServerClock(
        server_domain,
        client_domain,
        "same_host",
        same_clock_uncertainty_ns=supplied.uncertainty_ns,
    )


def cli_alignment(
    run_id: str,
    server_domain: str,
    client_domain: str,
    offset_ns: int,
    uncertainty_ns: int,
) -> ClockAlignmentEvent:
    """Represent operator-supplied flags as an in-memory alignment record."""
    return ClockAlignmentEvent(
        context=CorrelationContext(
            run_id=run_id,
            session_id="stormlog.infer.analyze",
            producer_id="stormlog.infer.analyze",
            source="stormlog.infer.analyze",
            clock_domain=client_domain,
            clock_kind="wall",
            collection_mode="imported",
            provenance="reported",
        ),
        event_id=CLI_ALIGNMENT_ID,
        from_clock_domain=server_domain,
        to_clock_domain=client_domain,
        offset_ns=offset_ns,
        uncertainty_ns=uncertainty_ns,
    )
