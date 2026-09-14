# SPDX-License-Identifier: Apache-2.0
"""Waiting for an engine worker's "KV is ready" IPC event before a transfer.

Every LMCache-driven store/retrieve is ordered after an interprocess event the
engine worker recorded on its own stream once the KV blocks are written. The
server has two ways to honour that ordering:

``device`` (upstream behaviour)
    ``event_backend.wait_event(producer_event, stream)`` enqueues the wait on
    the transfer stream and the handler thread moves on immediately.

``host``
    The handler thread polls ``event_backend.query_event`` and only enqueues
    the copies once the event has fired. The wait is abortable: it stops as
    soon as the owning worker is reaped, or after a ceiling.

Why ``host`` exists
-------------------
On ROCm/HIP an interprocess event wait is not a GPU-side dependency. The HIP
runtime implements the importer's ``streamWait`` as a host callback that spins
until the exporter writes a flag into shared memory. If the exporter (the vLLM
worker) dies between sending the request and recording the event, that flag
never arrives and, on HIP 7.x, *every* later runtime call touching the stream,
the imported event or the imported KV mapping blocks forever:
``hipEventDestroy`` (at handler frame exit, holding the entry's transfer pin),
``hipStreamSynchronize`` (in ``GPUCacheContext.close``) and
``hipIpcCloseMemHandle`` (torch's storage deleter). The reaper then cannot
release the dead worker's context and its KV pool -- tens of GiB -- stays
mapped until the server process restarts, so the restarted engine fails to
allocate its KV cache and crash-loops. There is no teardown-side fix; the
doomed wait must never be enqueued. ``host`` mode achieves that. On CUDA the
stream wait is a real device-side dependency and the upstream path is kept.

Cost of ``host`` mode: the affinity thread serving an instance is occupied
while that instance's event is pending (normally well under a millisecond,
the tail of the layer that recorded it). With ``--max-gpu-workers`` below the
number of registered instances, a second instance sharing the thread queues
behind it for that long. On HIP the runtime was already host-spinning for the
same duration on its own callback thread, so per-request latency is unchanged.
"""

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.event_ipc import EventIPCBackend

logger = init_logger(__name__)

# Spin with ``sleep(0)`` (yield) for this many polls before backing off to
# ``_POLL_SLEEP_S``. Events normally fire within a few hundred microseconds of
# the request arriving, so most waits finish in the yield phase.
_SPIN_POLLS = 200
_POLL_SLEEP_S = 0.0002


class ProducerWaitMode(str, Enum):
    """How the server waits for the producer's IPC event."""

    AUTO = "auto"
    """``HOST`` on ROCm/HIP, ``DEVICE`` elsewhere."""
    DEVICE = "device"
    """Enqueue the wait on the transfer stream (upstream behaviour)."""
    HOST = "host"
    """Poll on the handler thread; abort when the producer is reaped."""


class ProducerWaitOutcome(Enum):
    """Result of :func:`wait_for_producer_event`."""

    READY = "ready"
    """The event fired; device work may be enqueued."""
    PRODUCER_DEAD = "producer_dead"
    """The owning worker was reaped/unregistered before the event fired."""
    TIMEOUT = "timeout"
    """The ceiling elapsed with the worker still considered alive."""


def is_hip_runtime() -> bool:
    """Return True when torch is built against ROCm/HIP."""
    return getattr(torch.version, "hip", None) is not None


@dataclass(frozen=True)
class ProducerWaitPolicy:
    """Resolved producer-wait behaviour for one server process.

    Args:
        mode: ``DEVICE`` or ``HOST`` (``AUTO`` must already be resolved; see
            :meth:`from_config`).
        timeout_s: Ceiling for a ``HOST`` wait when the owning worker has not
            been reaped. ``0`` disables the ceiling so only reaping aborts the
            wait. Ignored in ``DEVICE`` mode.
    """

    mode: ProducerWaitMode
    timeout_s: float

    def __post_init__(self) -> None:
        if self.mode is ProducerWaitMode.AUTO:
            raise ValueError("ProducerWaitPolicy.mode must be resolved (not AUTO)")

    @classmethod
    def from_config(
        cls, mode: str | ProducerWaitMode, timeout_s: float
    ) -> "ProducerWaitPolicy":
        """Build a policy from configuration values, resolving ``auto``.

        Args:
            mode: ``"auto"``, ``"device"`` or ``"host"`` (or the enum).
            timeout_s: Host-wait ceiling in seconds; ``0`` disables it.

        Returns:
            The resolved policy.

        Raises:
            ValueError: If ``mode`` is not a known mode or ``timeout_s`` is
                negative.
        """
        resolved = ProducerWaitMode(mode)
        if resolved is ProducerWaitMode.AUTO:
            resolved = (
                ProducerWaitMode.HOST if is_hip_runtime() else ProducerWaitMode.DEVICE
            )
        if timeout_s < 0:
            raise ValueError(f"producer wait timeout must be >= 0; got {timeout_s}")
        return cls(mode=resolved, timeout_s=timeout_s)

    @classmethod
    def default(cls) -> "ProducerWaitPolicy":
        """Policy equivalent to the default server configuration (``auto``)."""
        return cls.from_config(ProducerWaitMode.AUTO, 600.0)


def wait_for_producer_event(
    event_backend: EventIPCBackend,
    event: object,
    *,
    producer_is_dead: Callable[[], bool],
    timeout_s: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ProducerWaitOutcome:
    """Poll ``event`` on the calling thread until it fires or the wait is moot.

    Args:
        event_backend: Backend used to query the event.
        event: The imported producer event.
        producer_is_dead: Returns True once the owning worker has been reaped
            or unregistered; checked on every poll.
        timeout_s: Give up after this many seconds even if the worker is still
            considered alive. ``0`` means no ceiling.
        clock: Monotonic clock (injectable for tests).
        sleep: Sleep function (injectable for tests).

    Returns:
        :attr:`ProducerWaitOutcome.READY` once the event has fired, otherwise
        the reason the wait was abandoned. On a non-``READY`` outcome the
        caller must not enqueue any device work ordered after the event.
    """
    if event_backend.query_event(event):
        return ProducerWaitOutcome.READY
    start = clock()
    polls = 0
    while True:
        if producer_is_dead():
            return ProducerWaitOutcome.PRODUCER_DEAD
        if timeout_s > 0 and clock() - start > timeout_s:
            return ProducerWaitOutcome.TIMEOUT
        polls += 1
        sleep(0.0 if polls <= _SPIN_POLLS else _POLL_SLEEP_S)
        if event_backend.query_event(event):
            return ProducerWaitOutcome.READY


class AbandonedEventRegistry:
    """Keeps event objects of dead producers alive for the process lifetime.

    Destroying an imported HIP IPC event whose exporter died can block the
    calling thread indefinitely in ``hipEventDestroy``. Parking the Python
    objects here means their destructors never run. Memory cost is a few
    hundred bytes per abandoned transfer, bounded by the number of transfers
    that were in flight when workers died over the server's lifetime.
    """

    def __init__(self) -> None:
        self._events: list[object] = []
        self._lock = threading.Lock()

    def retain(self, *events: object) -> None:
        """Take ownership of ``events`` so they are never destroyed.

        Args:
            events: Event objects (``None`` entries are ignored).
        """
        with self._lock:
            self._events.extend(e for e in events if e is not None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)
