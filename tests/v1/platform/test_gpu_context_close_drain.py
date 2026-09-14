# SPDX-License-Identifier: Apache-2.0
"""Bounded stream drain used by ``GPUCacheContext.close``.

``drain_stream`` is the piece of ``close()`` that decides between "the stream
finished, release normally" and "the stream will never finish (its work waits
on a dead producer), park it". Tested with a fake stream and injected clock so
no device is needed.
"""

# Third Party
import pytest

# First Party
from lmcache.v1.platform.cuda.cache_context import drain_stream


class _FakeStream:
    def __init__(self, ready_after_queries: int | None):
        self._ready_after = ready_after_queries
        self.queries = 0
        self.synchronized = 0

    def query(self) -> bool:
        self.queries += 1
        return self._ready_after is not None and self.queries >= self._ready_after

    def synchronize(self) -> None:
        self.synchronized += 1


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.mark.parametrize("timeout", [None, 0, 0.0])
def test_unbounded_uses_synchronize(timeout):
    stream = _FakeStream(ready_after_queries=None)
    assert drain_stream(stream, timeout) is True
    assert stream.synchronized == 1
    assert stream.queries == 0


def test_bounded_returns_true_once_stream_is_idle():
    stream = _FakeStream(ready_after_queries=3)
    clock = _FakeClock()
    assert drain_stream(stream, 5.0, clock=clock, sleep=clock.sleep) is True
    assert stream.queries == 3
    assert stream.synchronized == 0


def test_bounded_returns_false_when_stream_never_drains():
    stream = _FakeStream(ready_after_queries=None)
    clock = _FakeClock()
    assert drain_stream(stream, 1.0, clock=clock, sleep=clock.sleep) is False
    assert stream.synchronized == 0, "a stuck stream must never be synchronized"
    assert clock.now > 1.0
