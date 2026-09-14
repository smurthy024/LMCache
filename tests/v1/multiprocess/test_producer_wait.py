# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the abortable producer-event wait.

Exercises :mod:`lmcache.v1.multiprocess.producer_wait` through its public
surface with a fake event backend and injected clock/sleep, so no device is
needed. The behaviour under test is the docstring contract: ``auto`` resolves
per runtime, the host wait returns ``READY`` once the event fires, aborts as
soon as the producer is reported dead, honours the ceiling, and never enqueues
anything itself.
"""

# Standard
import weakref

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import producer_wait as pw
from lmcache.v1.multiprocess.producer_wait import (
    AbandonedEventRegistry,
    ProducerWaitMode,
    ProducerWaitOutcome,
    ProducerWaitPolicy,
    wait_for_producer_event,
)


class _FakeBackend:
    """Event backend whose query answers are scripted; records wait_event."""

    def __init__(self, answers: list[bool]):
        self._answers = list(answers)
        self.queries = 0
        self.stream_waits = 0

    def query_event(self, event: object) -> bool:
        self.queries += 1
        if self._answers:
            return self._answers.pop(0)
        return False

    def wait_event(self, event: object, stream: object) -> None:
        self.stream_waits += 1


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        # Every poll advances time a little even for sleep(0), so a ceiling
        # is reachable in a bounded number of iterations.
        self.now += max(seconds, 0.001)


# -- Policy -------------------------------------------------------------------


def test_auto_resolves_to_host_on_hip(monkeypatch):
    monkeypatch.setattr(pw, "is_hip_runtime", lambda: True)
    assert ProducerWaitPolicy.from_config("auto", 1.0).mode is ProducerWaitMode.HOST


def test_auto_resolves_to_device_elsewhere(monkeypatch):
    monkeypatch.setattr(pw, "is_hip_runtime", lambda: False)
    policy = ProducerWaitPolicy.from_config("auto", 1.0)
    assert policy.mode is ProducerWaitMode.DEVICE
    assert policy.timeout_s == 1.0


@pytest.mark.parametrize("mode", ["device", "host"])
def test_explicit_mode_ignores_runtime(monkeypatch, mode):
    monkeypatch.setattr(pw, "is_hip_runtime", lambda: mode != "host")
    assert ProducerWaitPolicy.from_config(mode, 0).mode is ProducerWaitMode(mode)


def test_policy_rejects_unknown_mode_and_negative_timeout():
    with pytest.raises(ValueError):
        ProducerWaitPolicy.from_config("spin", 1.0)
    with pytest.raises(ValueError, match=">= 0"):
        ProducerWaitPolicy.from_config("host", -1.0)


def test_policy_must_be_resolved():
    with pytest.raises(ValueError, match="AUTO"):
        ProducerWaitPolicy(mode=ProducerWaitMode.AUTO, timeout_s=1.0)


# -- wait_for_producer_event --------------------------------------------------


def test_ready_immediately_does_not_sleep():
    backend = _FakeBackend([True])
    clock = _FakeClock()
    slept: list[float] = []

    outcome = wait_for_producer_event(
        backend,
        object(),
        producer_is_dead=lambda: False,
        timeout_s=10.0,
        clock=clock,
        sleep=slept.append,
    )

    assert outcome is ProducerWaitOutcome.READY
    assert slept == []
    assert backend.stream_waits == 0


def test_ready_after_polling():
    backend = _FakeBackend([False, False, False, True])
    clock = _FakeClock()

    outcome = wait_for_producer_event(
        backend,
        object(),
        producer_is_dead=lambda: False,
        timeout_s=10.0,
        clock=clock,
        sleep=clock.sleep,
    )

    assert outcome is ProducerWaitOutcome.READY
    assert backend.queries == 4
    assert backend.stream_waits == 0


def test_dead_producer_aborts_before_event_fires():
    backend = _FakeBackend([False] * 1000)
    clock = _FakeClock()
    polls = {"n": 0}

    def dead() -> bool:
        polls["n"] += 1
        return polls["n"] >= 3

    outcome = wait_for_producer_event(
        backend,
        object(),
        producer_is_dead=dead,
        timeout_s=0,  # no ceiling: only death can end the wait
        clock=clock,
        sleep=clock.sleep,
    )

    assert outcome is ProducerWaitOutcome.PRODUCER_DEAD
    assert polls["n"] == 3
    # The event was never reported ready, so no query returned True.
    assert backend.queries < 1000


def test_ceiling_aborts_live_but_silent_producer():
    backend = _FakeBackend([])  # never ready
    clock = _FakeClock()

    outcome = wait_for_producer_event(
        backend,
        object(),
        producer_is_dead=lambda: False,
        timeout_s=2.0,
        clock=clock,
        sleep=clock.sleep,
    )

    assert outcome is ProducerWaitOutcome.TIMEOUT
    assert clock.now - 1000.0 > 2.0


def test_backs_off_from_yield_to_sleep_after_spin_phase():
    backend = _FakeBackend([])
    clock = _FakeClock()
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock.sleep(seconds)

    wait_for_producer_event(
        backend,
        object(),
        producer_is_dead=lambda: len(slept) >= 250,
        timeout_s=0,
        clock=clock,
        sleep=sleep,
    )

    assert slept[:200] == [0.0] * 200
    assert all(s > 0 for s in slept[200:])


# -- AbandonedEventRegistry ---------------------------------------------------


class _Ev:
    pass


def test_registry_keeps_events_alive_and_ignores_none():
    registry = AbandonedEventRegistry()
    ev = _Ev()
    ref = weakref.ref(ev)

    registry.retain(ev, None)
    del ev

    assert ref() is not None, "retained event must not be collected"
    assert len(registry) == 1
