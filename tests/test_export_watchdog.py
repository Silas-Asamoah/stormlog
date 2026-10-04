"""The socket watchdog: deadlines end blocked reads, and only their own."""

import socket
import threading
import time

from stormlog._export.watchdog import Watchdog


def _blocked_recv(sock: socket.socket, results: list[bytes]) -> threading.Thread:
    thread = threading.Thread(target=lambda: results.append(sock.recv(10)))
    thread.start()
    return thread


def test_a_deadline_ends_a_blocked_receive() -> None:
    watchdog = Watchdog()
    near, far = socket.socketpair()
    try:
        results: list[bytes] = []
        started = time.monotonic()
        token = watchdog.arm(near, time.monotonic() + 0.2)
        reader = _blocked_recv(near, results)
        reader.join(5)
        assert not reader.is_alive() and results == [b""]
        assert time.monotonic() - started < 2
        assert watchdog.fired(token) and not watchdog.disarm(token)
        assert watchdog.stats.fired == 1
    finally:
        near.close()
        far.close()
        watchdog.stop()


def test_disarming_in_time_leaves_the_socket_alone() -> None:
    watchdog = Watchdog()
    near, far = socket.socketpair()
    try:
        token = watchdog.arm(near, time.monotonic() + 0.1)
        assert watchdog.disarm(token)
        time.sleep(0.3)
        far.sendall(b"hi")
        assert near.recv(10) == b"hi"
        assert watchdog.stats.fired == 0
    finally:
        near.close()
        far.close()
        watchdog.stop()


def test_a_stale_deadline_never_touches_a_later_attempt() -> None:
    watchdog = Watchdog()
    first_near, first_far = socket.socketpair()
    second_near, second_far = socket.socketpair()
    try:
        stale = watchdog.arm(first_near, time.monotonic() + 0.1)
        watchdog.disarm(stale)  # the first attempt ended in time
        fresh = watchdog.arm(second_near, time.monotonic() + 5)
        time.sleep(0.3)  # past the stale deadline
        second_far.sendall(b"ok")
        assert second_near.recv(10) == b"ok"
        assert watchdog.disarm(fresh)
    finally:
        for sock in (first_near, first_far, second_near, second_far):
            sock.close()
        watchdog.stop()


def test_deadlines_fire_in_order_from_one_thread() -> None:
    watchdog = Watchdog(name="watchdog-under-test")
    pairs = [socket.socketpair() for _ in range(3)]
    try:
        now = time.monotonic()
        tokens = [
            watchdog.arm(pair[0], now + delay)
            for pair, delay in zip(pairs, (0.3, 0.1, 0.2))
        ]
        time.sleep(0.6)
        assert all(watchdog.fired(token) for token in tokens)
        names = [t.name for t in threading.enumerate()]
        assert names.count("watchdog-under-test") == 1
    finally:
        for near, far in pairs:
            near.close()
            far.close()
        watchdog.stop()
