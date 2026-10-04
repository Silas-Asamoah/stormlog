"""What became of every span offered for export: settled once, counted exactly.

A batch is sent in one or more transmissions, each confirmed, refused,
ambiguous (sent, outcome unknown) or never sent. When the batch is done, or
when the exporter freezes at close, its history settles it into exactly one
of: exported, rejected (the collector said so), refused, dropped (never
sent) or unknown (it may have been stored). So at every instant

    offered = exported + rejected + refused + dropped + unknown + queued + in_flight

and after the freeze nothing is queued or in flight. A transmission whose
answer was lost may still have been stored, so a collector can receive a
batch more than once; ``max_extra_copies`` bounds how many extra spans that
can add.

The breaker marks a destination down after three batches in a row settle
without a confirmed transmission; while it is down, the head batch is
retried once per probe interval, and one confirmation brings it back up.
"""

from __future__ import annotations

import random
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field

from .otlp_http import AMBIGUOUS, CONFIRMED, NOT_SENT, REFUSED, Transmission

# Settlement reasons beyond a transmission's own category.
QUEUE_FULL = "queue_full"
CLOSED = "closed"
SHUTDOWN = "shutdown"
SHUTDOWN_IN_FLIGHT = "shutdown_in_flight"
ENCODE_ERROR = "encode_error"
REJECTED_AFTER_AMBIGUOUS = "rejected_after_ambiguous"
REFUSED_AFTER_AMBIGUOUS = "refused_after_ambiguous"

# Breaker transitions.
FIRST_FAILURE = "first_failure"
BREAKER_OPEN = "breaker_open"
FIRST_SUCCESS = "first_success"
BREAKER_CLOSED = "breaker_closed"
MAX_TRANSITIONS = 64


@dataclass(frozen=True)
class Settlement:
    """Where one batch's spans ended up."""

    exported: int = 0
    rejected: int = 0
    refused: tuple[tuple[str, int], ...] = ()
    dropped: tuple[tuple[str, int], ...] = ()
    unknown: tuple[tuple[str, int], ...] = ()
    extra_copies: int = 0

    @property
    def spans(self) -> int:
        return (
            self.exported
            + self.rejected
            + sum(n for _, n in self.refused + self.dropped + self.unknown)
        )


def dropped(reason: str, spans: int) -> Settlement:
    return Settlement(dropped=((reason, spans),))


class BatchHistory:
    """The transmissions of one batch, and the settlement they lead to."""

    def __init__(self, spans: int, *, started_at: float | None = None) -> None:
        self.spans = spans
        self.started_at = time.monotonic() if started_at is None else started_at
        self.attempts = 0
        self.ambiguous = 0
        self.last_kind: str | None = None
        self.last_ambiguous: str | None = None
        self.last_refused: str | None = None
        self.last_not_sent: str | None = None
        self.confirmed: Transmission | None = None

    def record(self, transmission: Transmission) -> None:
        self.attempts += 1
        kind = transmission.kind
        self.last_kind = kind
        category = transmission.category or kind
        if kind == CONFIRMED:
            self.confirmed = transmission
        elif kind == AMBIGUOUS:
            self.ambiguous += 1
            self.last_ambiguous = category
        elif kind == REFUSED:
            self.last_refused = category
        else:
            self.last_not_sent = category

    def settle(self) -> Settlement:
        """The settlement of a batch whose last transmission is final."""
        n = self.spans
        copies = n * max(0, self.ambiguous + (self.confirmed is not None) - 1)
        if self.confirmed is not None:
            result = self.confirmed.result
            rejected = result.rejected if result is not None else 0
            if self.ambiguous:
                # An earlier transmission may have stored what this one rejected.
                unknown = ((REJECTED_AFTER_AMBIGUOUS, rejected),) if rejected else ()
                return Settlement(
                    exported=n - rejected, unknown=unknown, extra_copies=copies
                )
            return Settlement(exported=n - rejected, rejected=rejected)
        if self.ambiguous:
            reason = (
                REFUSED_AFTER_AMBIGUOUS
                if self.last_kind == REFUSED
                else self.last_ambiguous or AMBIGUOUS
            )
            return Settlement(unknown=((reason, n),), extra_copies=copies)
        if self.last_refused is not None:
            return Settlement(refused=((self.last_refused, n),))
        return dropped(self.last_not_sent or SHUTDOWN, n)

    def settle_at_freeze(self, *, sending: bool) -> Settlement:
        """The settlement when the exporter freezes with this batch unfinished.

        ``sending`` is whether a transmission had begun to send its body: it
        may yet be stored, so it counts as one more ambiguous transmission.
        A batch never sent is dropped at shutdown.
        """
        if sending:
            self.ambiguous += 1
            self.last_kind = AMBIGUOUS
            self.last_ambiguous = SHUTDOWN_IN_FLIGHT
        elif self.ambiguous == 0 and self.last_refused is None:
            return dropped(SHUTDOWN, self.spans)
        return self.settle()


@dataclass
class _Totals:
    taken: int = 0
    exported: int = 0
    rejected: int = 0
    refused: Counter[str] = field(default_factory=Counter)
    dropped: Counter[str] = field(default_factory=Counter)
    unknown: Counter[str] = field(default_factory=Counter)
    extra_copies: int = 0
    late: Counter[str] = field(default_factory=Counter)

    def add(self, settlement: Settlement) -> None:
        self.exported += settlement.exported
        self.rejected += settlement.rejected
        self.refused.update(dict(settlement.refused))
        self.dropped.update(dict(settlement.dropped))
        self.unknown.update(dict(settlement.unknown))
        self.extra_copies += settlement.extra_copies

    @property
    def settled(self) -> int:
        return (
            self.exported
            + self.rejected
            + sum(self.refused.values())
            + sum(self.dropped.values())
            + sum(self.unknown.values())
        )


class DeliveryLedger:
    """Running totals of settlements, and the batch in flight; frozen once.

    The worker takes spans from the queue (``take``), builds them into a
    batch (``begin``), records each transmission (``attempting``,
    ``record``) and settles the batch (``finish``). Every step holds the
    ledger's lock, as does ``freeze``, so a batch is settled by the worker
    or by the freeze, never both. After the freeze the worker's steps
    return False and its results are counted as late.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals = _Totals()
        self._pending = 0
        self._current: BatchHistory | None = None
        self._attempting = False
        self.frozen = False

    def take(self, spans: int) -> bool:
        with self._lock:
            if self.frozen:
                return False
            self._totals.taken += spans
            self._pending += spans
            return True

    def drop_pending(self, reason: str, spans: int) -> None:
        """Settle spans taken but never batched, such as one that failed to encode."""
        with self._lock:
            if self.frozen:
                return
            self._pending -= spans
            self._totals.add(dropped(reason, spans))

    def begin(self, history: BatchHistory) -> bool:
        with self._lock:
            if self.frozen:
                return False
            self._pending -= history.spans
            self._current = history
            return True

    def attempting(self) -> bool:
        """Mark a transmission as started; False once frozen."""
        with self._lock:
            self._attempting = not self.frozen
            return self._attempting

    def record(self, history: BatchHistory, transmission: Transmission) -> bool:
        with self._lock:
            self._attempting = False
            if self.frozen:
                self._totals.late[transmission.kind] += 1
                return False
            history.record(transmission)
            return True

    def finish(self, history: BatchHistory) -> Settlement | None:
        """Settle the batch in flight; None once frozen."""
        with self._lock:
            if self.frozen or self._current is not history:
                return None
            settlement = history.settle()
            self._totals.add(settlement)
            self._current = None
            return settlement

    def freeze(self, *, drained: int, sending: Callable[[], bool]) -> None:
        """Settle everything unfinished at shutdown, then freeze.

        ``drained`` spans were still queued. ``sending`` says whether the
        transmission in progress, if any, had begun to send its body.
        """
        with self._lock:
            if self.frozen:
                return
            totals = self._totals
            totals.taken += drained
            unbatched = drained + self._pending
            if unbatched:
                totals.add(dropped(SHUTDOWN, unbatched))
            self._pending = 0
            if self._current is not None:
                in_flight = self._attempting and sending()
                totals.add(self._current.settle_at_freeze(sending=in_flight))
                self._current = None
            self._attempting = False
            self.frozen = True

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            totals = self._totals
            return {
                "exported": totals.exported,
                "rejected": totals.rejected,
                "refused": dict(totals.refused),
                "dropped": dict(totals.dropped),
                "unknown": dict(totals.unknown),
                "in_flight": totals.taken - totals.settled,
                "max_extra_copies": totals.extra_copies,
                "late_results": dict(totals.late),
                "frozen": self.frozen,
            }


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter, within a per-batch budget."""

    initial_seconds: float = 0.5
    max_seconds: float = 8.0
    max_attempts: int = 5
    budget_seconds: float = 30.0

    def delay(
        self, attempt: int, *, rng: Callable[[float, float], float] = random.uniform
    ) -> float:
        """The wait before attempt ``attempt + 1``, after ``attempt`` attempts."""
        ceiling = min(self.max_seconds, self.initial_seconds * 2 ** (attempt - 1))
        return rng(0.0, ceiling)

    def remaining(self, history: BatchHistory, now: float) -> float:
        """Seconds left in the batch's budget; 0 once its attempts are used up."""
        if history.attempts >= self.max_attempts:
            return 0.0
        return max(0.0, history.started_at + self.budget_seconds - now)


class Breaker:
    """Whether a destination is up, and the transitions that said so."""

    def __init__(
        self,
        *,
        threshold: int = 3,
        probe_interval: float = 8.0,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.threshold = threshold
        self.probe_interval = probe_interval
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self.up = True
        self._failed_batches = 0
        self._last_attempt_ok = True
        self.last_success_ns: int | None = None
        self.transitions: deque[dict[str, object]] = deque(maxlen=MAX_TRANSITIONS)
        self.transitions_dropped = 0

    def attempted(self, transmission: Transmission) -> None:
        """Note one transmission: a confirmation brings the destination up."""
        ok = transmission.kind == CONFIRMED
        with self._lock:
            if ok:
                self.last_success_ns = self._clock_ns()
                if not self._last_attempt_ok:
                    self._note(FIRST_SUCCESS, None)
                if not self.up:
                    self.up = True
                    self._note(BREAKER_CLOSED, None)
                self._failed_batches = 0
            elif self._last_attempt_ok:
                self._note(FIRST_FAILURE, transmission.category or transmission.kind)
            self._last_attempt_ok = ok

    def settled(self, history: BatchHistory) -> None:
        """Note a settled batch; three without a confirmation open the breaker."""
        if history.attempts == 0 or history.confirmed is not None:
            return
        with self._lock:
            self._failed_batches += 1
            if self.up and self._failed_batches >= self.threshold:
                self.up = False
                self._note(BREAKER_OPEN, history.last_kind or NOT_SENT)

    def _note(self, event: str, reason: str | None) -> None:
        if len(self.transitions) == self.transitions.maxlen:
            self.transitions_dropped += 1
        self.transitions.append(
            {"at_ns": self._clock_ns(), "event": event, "reason": reason}
        )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "up": self.up,
                "last_success_ns": self.last_success_ns,
                "transitions": list(self.transitions),
                "transitions_dropped": self.transitions_dropped,
            }
