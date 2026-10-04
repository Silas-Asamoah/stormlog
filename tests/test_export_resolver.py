"""The destination resolver: a few candidates, one thread, a bounded wait."""

import socket
import threading
import time
from typing import Any

from stormlog._export.resolver import Candidate, Resolver

V6 = (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 4318, 0, 0))
V4 = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 4318))


def _entry(address: str) -> tuple[Any, ...]:
    return (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 4318))


def test_candidates_keep_the_system_order_without_duplicates_up_to_four() -> None:
    found = [V6, V4, V4] + [_entry(f"10.0.0.{n}") for n in range(1, 5)]
    resolver = Resolver("collector", 4318, getaddrinfo=lambda *a, **k: found)
    assert resolver.resolve(wait=5)
    addresses = [candidate.address for candidate in resolver.candidates()]
    assert addresses == [V6[4], V4[4], ("10.0.0.1", 4318), ("10.0.0.2", 4318)]


def test_the_last_address_that_worked_is_tried_first() -> None:
    resolver = Resolver("localhost", 4318, getaddrinfo=lambda *a, **k: [V6, V4])
    resolver.resolve(wait=5)
    ipv4 = resolver.candidates()[1]
    resolver.mark_good(ipv4)
    assert resolver.candidates()[0] == ipv4
    resolver.resolve(wait=5)  # re-resolution keeps the preference
    assert resolver.candidates() == [ipv4, Candidate(*V6[:3], V6[4])]


def test_a_stuck_resolution_bounds_the_wait_and_is_never_repeated() -> None:
    release = threading.Event()
    calls: list[int] = []

    def stuck(*_args: object, **_kwargs: object) -> list[Any]:
        calls.append(1)
        release.wait(10)
        return [V4]

    resolver = Resolver("collector", 4318, stall_seconds=0.15, getaddrinfo=stuck)
    started = time.monotonic()
    assert not resolver.resolve(wait=0.1)
    stalled_early = resolver.stalled
    assert not resolver.resolve(wait=0.1)
    stalled_later = resolver.stalled
    assert not stalled_early and stalled_later
    assert time.monotonic() - started < 2
    assert calls == [1]  # the second call waited for the first resolution
    release.set()
    assert resolver.resolve(wait=5)
    assert [c.address for c in resolver.candidates()] == [V4[4]]


def test_a_failed_resolution_is_counted_and_keeps_earlier_addresses() -> None:
    answers: list[Any] = [[V4], socket.gaierror("Name or service not known")]

    def flaky(*_args: object, **_kwargs: object) -> list[Any]:
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer  # type: ignore[no-any-return]

    resolver = Resolver("collector", 4318, getaddrinfo=flaky)
    assert resolver.resolve(wait=5)
    assert resolver.resolve(wait=5)  # the failure leaves the known address
    assert resolver.stats.failures == 1
    assert "not known" in (resolver.stats.last_error or "")


def test_localhost_resolves_for_real() -> None:
    resolver = Resolver("localhost", 4318)
    assert resolver.resolve(wait=5)
    hosts = {candidate.address[0] for candidate in resolver.candidates()}
    assert hosts & {"127.0.0.1", "::1"}
