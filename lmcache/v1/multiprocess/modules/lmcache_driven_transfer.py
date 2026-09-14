# SPDX-License-Identifier: Apache-2.0
"""LMCache-driven KV cache transfer operations for the MPCacheServer."""

# Standard
from dataclasses import dataclass
from typing import Any, Sequence
import gc
import threading
import time

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import (
    EngineType,
    _lmcache_nvtx_annotate,
)
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.kv_layer_groups import ObjectGroupInfo
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType, next_transfer_key
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    KVCache,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import InstanceLivenessTarget
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules.lookup import resolve_prefetched_obj_keys
from lmcache.v1.multiprocess.native_completion import (
    DeviceHostFuncDispatcher,
    submit_callback_to_stream,
)
from lmcache.v1.multiprocess.object_group_transfer import (
    downsample_and_stage_block_ids,
    transfer_kv_per_object_group,
)
from lmcache.v1.multiprocess.producer_wait import (
    AbandonedEventRegistry,
    ProducerWaitMode,
    ProducerWaitOutcome,
    ProducerWaitPolicy,
    wait_for_producer_event,
)
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.platform.base.cache_context import BaseCacheContext
from lmcache.v1.platform.base.event_ipc import (
    EventIPCBackend,
    get_event_ipc_backend,
)
from lmcache.v1.platform.cache_context import create_cache_context
import lmcache.lmcache_native as lmcache_native
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    PeriodicThreadRegistry,
    ThreadLevel,
    ThreadRunSummary,
    create_periodic_thread,
)

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# IPC memory management constants
# ---------------------------------------------------------------------------

# How often (seconds) the periodic IPC-collection thread runs.  Between
# cycles, IPC handles orphaned by normal operation (e.g. a worker that
# crashed before its unregister) accumulate.  This thread reclaims them
# by calling ``empty_cache()`` + ``ipc_collect()`` so the handles do not
# pin GPU memory indefinitely.  60 s is a reasonable default: frequent
# enough to keep handle count bounded, infrequent enough to avoid
# interfering with active transfers.
_IPC_COLLECT_INTERVAL_S = 60.0

# Grace period (seconds) before a dead-but-in-use context entry is
# force-released.  When a worker is reaped/unregistered while a store or
# retrieve is in-flight, the entry is marked ``dead=True`` and its release
# is deferred until the transfer finishes and calls ``checkin_context_entry``.
# If the transfer handler raises before reaching the ``finally`` block
# (or the worker process is killed mid-transfer), the checkin never
# happens and the entry -- and its IPC handles -- leak permanently.
# This timeout is a safety net: after the grace period the entry is
# force-released regardless, accepting the small risk of unmapping a
# segment that a zombie CUDA stream might still reference.
_CHECKIN_TIMEOUT_S = 120.0


def _free_device_memory_bytes() -> int | None:
    """Return free device memory in bytes, or None if the backend cannot say."""
    mem_get_info = getattr(torch_dev, "mem_get_info", None)
    if mem_get_info is None:
        return None
    try:
        return int(mem_get_info()[0])
    except Exception:
        return None


def get_layout_desc(
    cache_context: BaseCacheContext,
    num_tokens: int,
    object_group_id: int,
) -> MemoryLayoutDesc:
    """Get the memory layout description for a specific object group.

    The returned layout describes the single memory object that backs
    ``object_group_id``: one (shape, dtype) entry per kernel group in that
    object group, in the kernel groups' declared layout order. Kernel groups
    may have different shapes and dtypes.

    Args:
        cache_context: The cache context containing the KV cache information.
        num_tokens: The number of tokens to determine the layout for.
        object_group_id: Index of the object group whose layout to build.

    Returns:
        MemoryLayoutDesc: The memory layout description containing shapes and
        dtypes, one entry per kernel group in the object group.
    """
    object_group = cache_context.kv_layer_groups_manager.object_groups[object_group_id]
    shapes_and_dtypes = [
        cache_context.get_kernel_group_shape_dtype(num_tokens, kernel_group_idx)
        for kernel_group_idx in object_group.kernel_group_indices
    ]
    shapes, dtypes = zip(*shapes_and_dtypes, strict=False)
    return MemoryLayoutDesc(shapes=list(shapes), dtypes=list(dtypes))


def all_null_chunk_masks(
    block_ids: Sequence[Sequence[int]],
    object_groups: Sequence[ObjectGroupInfo],
    blocks_per_chunk: Sequence[int],
    num_chunks: int,
    null_block_id: int = 0,
) -> list[list[bool]]:
    """Mark, per object group, the chunks whose engine block ids are all null.

    A chunk is null for an object group when every block ID of every kernel
    group equals the server's null marker. Align-mode Mamba/linear
    layers produce such chunks: only the block holding the last recurrent state
    is real, so every earlier chunk is null. These chunks must not be stored --
    the null block carries no valid KV, and object keys are content hashes, so
    committing them would serve garbage to a later prefix hit.

    Args:
        block_ids: Raw per-kernel-group engine block ids (before any downsample),
            indexed by kernel-group index.
        object_groups: The object groups, indexed by object-group id.
        blocks_per_chunk: Blocks in one chunk per kernel group, indexed by
            kernel-group index.
        num_chunks: Number of chunks in the request.
        null_block_id: Server-wide block ID denoting absent data. Defaults to
            the historical vLLM null block zero.

    Returns:
        ``mask[g][i]`` is True iff chunk ``i`` is all-null for object group ``g``.
    """
    masks: list[list[bool]] = []
    for group in object_groups:
        chunk_null: list[bool] = []
        for i in range(num_chunks):
            is_null = True
            for kg in group.kernel_group_indices:
                bpc = blocks_per_chunk[kg]
                if any(
                    block != null_block_id
                    for block in block_ids[kg][i * bpc : (i + 1) * bpc]
                ):
                    is_null = False
                    break
            chunk_null.append(is_null)
        masks.append(chunk_null)
    return masks


# Hashable by identity: ContextEntry is used as a key in _dead_entries
# (the dead-entry tracker populated by _retire_entries). A plain @dataclass
# sets __hash__ = None, which would crash _retire_entries the first time a
# busy (in_use > 0) entry is retired and added to the tracker. eq=False keeps
# the default identity-based hash/equality, which is correct here since the
# tracker keys entries by object identity, not by field value.
@dataclass(eq=False)
class ContextEntry:
    """Registered cache context metadata for a single worker instance.

    The concrete type is whatever :func:`create_cache_context` returned
    for the wrapper list at registration time -- a
    :class:`GPUCacheContext` for CUDA-IPC wrappers, a
    :class:`CPUCacheContext` for POSIX-SHM wrappers. Both expose
    the same ``kv_tensors`` / ``engine_kv_format`` / ``num_layers`` / ...
    duck-typed surface, so downstream consumers stay agnostic.

    Args:
        cache_context: Platform cache context (GPU or CPU) managing
            shape and pointers to the registered KV cache tensors.
        model_name: The name of the model associated with this KV cache.
        world_size: The world size associated with this KV cache.
        last_seen: ``time.monotonic()`` of the most recent activity from
            this instance (register, PING, store, or retrieve). Drives reaping.
        has_liveness_signal: True once the instance has sent at least one
            PING. Selects the reap window (timeout vs registration grace).
            Latched only by PING, never by traffic.
        event_backend: Cached event backend selected for this context's device.
    """

    cache_context: BaseCacheContext
    model_name: str
    world_size: int
    last_seen: float = 0.0
    has_liveness_signal: bool = False
    event_backend: EventIPCBackend | None = None
    # Transfer-vs-teardown guard: ``in_use`` counts store/retrieve operations
    # currently holding this entry's cache_context; ``dead`` marks an entry
    # removed from the registry (reap / unregister) whose release must wait
    # for the last such operation. Closing the context mid-copy unmaps the
    # IPC/SHM segments under an active tensor copy and the server dies with
    # SIGSEGV in the copy kernel (observed 2026-07-21 on every engine
    # death/restart under load). Both fields are guarded by the module lock.
    in_use: int = 0
    dead: bool = False
    # Guard against double-release: set to True the first time
    # _release_entries runs for this entry.  Both checkin_context_entry and
    # _ipc_collect_cycle can reach _release_entries for the same dead entry;
    # without this flag, cache_context.close() and layout_desc_registry
    # .unregister() would be called twice.
    released: bool = False


class LMCacheDrivenTransferModule(InstanceLivenessTarget):
    """Handles LMCache-driven KV cache transfer operations.

    Owns GPU context registrations and provides handlers for
    register, unregister, store, and retrieve of GPU KV caches.

    Args:
        ctx: The shared engine context.
        producer_wait: How store/retrieve wait for the worker's "KV is ready"
            event. Defaults to the ``auto`` policy (host-side wait on
            ROCm/HIP, stream wait elsewhere); see
            :mod:`lmcache.v1.multiprocess.producer_wait`.
        close_drain_timeout_s: Bound on draining a released context's stream
            in ``cache_context.close()``; ``0`` means unbounded.
    """

    def __init__(
        self,
        ctx: MPCacheServerContext,
        *,
        producer_wait: ProducerWaitPolicy | None = None,
        close_drain_timeout_s: float = 20.0,
    ) -> None:
        self._ctx = ctx
        self._producer_wait = (
            producer_wait if producer_wait is not None else ProducerWaitPolicy.default()
        )
        self._close_drain_timeout_s = close_drain_timeout_s
        # Events of producers that died mid-transfer; kept alive for the
        # process lifetime because destroying them can block on HIP.
        self._abandoned_events = AbandonedEventRegistry()
        self._cache_contexts: dict[int, ContextEntry] = {}
        # Guards all reads/writes of _cache_contexts. The reaper mutates it
        # off the MQ main loop, so register/unregister/store/retrieve and
        # report_status all serialize through this lock. Held only for dict
        # ops -- never across context creation, layout-registry calls, or
        # empty_cache (leaf-lock invariant: no thread holds two locks).
        self._lock = threading.Lock()

        # Route finish_write / finish_read_prefetched through a C++ host
        # callback so the driver thread doesn't acquire the GIL.
        self._device_host_func_dispatcher = DeviceHostFuncDispatcher()
        self._device_host_func_dispatcher.register(
            "finish_write",
            self._ctx.storage_manager.finish_write,
            payload_type=list[ObjectKey],
        )
        self._device_host_func_dispatcher.register(
            "finish_read_prefetched",
            self._ctx.storage_manager.finish_read_prefetched,
            payload_type=list[ObjectKey],
        )
        self._device_host_func_dispatcher.start()

        # ------------------------------------------------------------------
        # Periodic IPC memory management
        # ------------------------------------------------------------------
        # Dead entries deferred by _retire_entries (in_use > 0 at reap time)
        # keyed by the entry itself → monotonic timestamp when marked dead.
        # The periodic thread scans this dict and force-releases entries
        # that have been stuck past _CHECKIN_TIMEOUT_S.
        self._dead_entries: dict[ContextEntry, float] = {}
        self._ipc_collector: PeriodicThread | None = create_periodic_thread(
            name="lmcache-ipc-collector",
            interval=_IPC_COLLECT_INTERVAL_S,
            execute_fn=self._ipc_collect_cycle,
            level=ThreadLevel.LOW,
        )
        self._ipc_collector.start()

    def register_host_func(self, kind: str, handler: Any, payload_type: Any) -> None:
        """Register *handler* for *kind* on the per-process device host-func
        dispatcher (stream-ordered callbacks without a driver-thread GIL
        acquire); pair with ``submit_callback_to_stream``."""
        self._device_host_func_dispatcher.register(kind, handler, payload_type)

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def get_and_touch_context_entry(self, instance_id: int) -> ContextEntry | None:
        """Return the entry for ``instance_id``, refreshing its last-seen time.

        The refresh keeps an actively transferring worker from being reaped
        even if its PINGs are briefly delayed. Does not latch the
        ping-proven flag -- only PINGs do that.

        Args:
            instance_id: The worker instance ID.

        Returns:
            The entry, or None if the instance is not (or no longer) tracked.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._cache_contexts.get(instance_id)
            if entry is not None:
                entry.last_seen = now
            return entry

    def _release_failed_retrieve_locks(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> None:
        """Release only one failed instance's unconsumed lookup locks.

        The lookup session is the ownership record.  If it is absent or does
        not match the RETRIEVE range, no release is attempted: L1 locks are
        anonymous refcounts, so guessing could consume a concurrent reader's
        lock.  ``claim_failed_retrieve_release`` also makes duplicate failure
        responses idempotent.
        """
        session = self._ctx.session_manager.get(key.request_id)
        if session is None:
            logger.warning(
                "Cannot release RETRIEVE locks for unregistered instance %d: "
                "request %s has no lookup session",
                instance_id,
                key.request_id,
            )
            return

        lock_state = session.prepare_failed_retrieve_release(key)
        if lock_state is None:
            return
        hit_chunks, locked_gids, group_windows, lookup_generation = lock_state
        obj_keys = resolve_prefetched_obj_keys(
            self._ctx,
            key,
            hit_chunks,
            locked_gids,
            group_windows=group_windows,
        )
        if not session.claim_failed_retrieve_release(
            instance_id, key, lookup_generation
        ):
            return
        if obj_keys:
            # One failed RETRIEVE owns one read lock per key.  In
            # particular, do not release the scheduler's whole MLA
            # reservation here: the remaining TP workers and concurrent
            # requests still own their independent read locks.
            self._ctx.storage_manager.finish_read_prefetched(obj_keys, read_locks=1)

    def checkout_context_entry(self, instance_id: int) -> ContextEntry | None:
        """Like get_and_touch_context_entry, but pins the entry against release.

        The caller MUST pair this with :meth:`checkin_context_entry` (via
        try/finally) once its transfer no longer touches the entry's
        cache_context. While pinned, reap/unregister defer the context
        close instead of unmapping segments under the caller's copy.

        Args:
            instance_id: The worker instance ID.

        Returns:
            The pinned entry, or None if the instance is not tracked.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._cache_contexts.get(instance_id)
            if entry is None:
                return None
            entry.last_seen = now
            entry.in_use += 1
            return entry

    def checkin_context_entry(self, entry: ContextEntry) -> None:
        """Release a pin taken by :meth:`checkout_context_entry`.

        The last checkin of a dead (reaped/unregistered) entry performs the
        deferred context release.

        Args:
            entry: The entry returned by checkout_context_entry.
        """
        with self._lock:
            if entry.released:
                # Already force-released by the periodic collector's timeout
                # sweep.  The checkout pin is consumed; don't decrement
                # in_use (the collector already reset it to 0).
                return
            entry.in_use -= 1
            release_now = entry.dead and entry.in_use <= 0 and not entry.released
            if release_now:
                entry.released = True
                self._dead_entries.pop(entry, None)
        if not release_now:
            return
        logger.info(
            "Deferred release of instance context (model=%s) after last "
            "in-flight transfer drained",
            entry.model_name,
        )
        # Hand the sole reference to the release list — a lingering local
        # binding would keep the entry alive through ipc_collect and the
        # IPC segments would never unmap (LMCache#4014).
        to_release = [entry]
        del entry
        self._release_entries(to_release)

    def _retire_entries(self, entries: list[ContextEntry]) -> None:
        """Mark removed entries dead; release the idle ones immediately.

        Busy entries (in_use > 0) are released by the final
        checkin_context_entry instead — closing them here would unmap
        IPC/SHM segments under an in-flight tensor copy (SIGSEGV).

        Args:
            entries: Entries already popped from the registry. The list is
                cleared before release so no binding pins a released entry.
        """
        idle: list[ContextEntry] = []
        busy_entries: list[ContextEntry] = []
        with self._lock:
            for entry in entries:
                entry.dead = True
                if entry.in_use <= 0:
                    idle.append(entry)
                else:
                    busy_entries.append(entry)
            if entries:
                # Drop the last local binding so the released entry is not
                # kept alive through ipc_collect (LMCache#4014).
                del entry
            entries.clear()
        if busy_entries:
            logger.warning(
                "Deferring release of %d instance context(s) with in-flight "
                "transfers; they close when the last transfer drains",
                len(busy_entries),
            )
            # Track dead entries so the periodic IPC-collector thread can
            # force-release them if checkin never happens (handler crash /
            # worker kill mid-transfer).  Without this, busy dead entries
            # leak permanently — the store/retrieve finally block never runs.
            now = time.monotonic()
            for entry in busy_entries:
                self._dead_entries[entry] = now
        self._release_entries(idle)

    def context_entries_snapshot(self) -> dict[int, ContextEntry]:
        """Return a shallow copy of the registry for iteration or status.

        Returns:
            A new dict mapping instance ID to entry; does not refresh
            last-seen times.
        """
        with self._lock:
            return dict(self._cache_contexts)

    def touch_instance(self, instance_id: int) -> None:
        """Refresh the worker's last-seen time and mark it ping-proven.

        A no-op if the instance is not tracked.

        Args:
            instance_id: The worker instance ID.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._cache_contexts.get(instance_id)
            if entry is not None:
                entry.last_seen = now
                entry.has_liveness_signal = True

    def tracked_instance_count(self) -> int:
        """Return the number of currently registered instances."""
        with self._lock:
            return len(self._cache_contexts)

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        """Reap GPU registrations that have gone silent.

        A ping-proven instance is judged against ``reap_timeout_s``; one
        that has never pinged against the larger ``registration_grace_s``.

        Args:
            reap_timeout_s: Silence budget for ping-proven instances.
            registration_grace_s: Silence budget for never-pinged instances.

        Returns:
            The instance IDs reaped this scan.
        """
        now = time.monotonic()
        reaped: list[tuple[int, ContextEntry]] = []
        with self._lock:
            stale_ids = [
                iid
                for iid, entry in self._cache_contexts.items()
                if now - entry.last_seen
                > (
                    reap_timeout_s
                    if entry.has_liveness_signal
                    else registration_grace_s
                )
            ]
            for iid in stale_ids:
                reaped.append((iid, self._cache_contexts.pop(iid)))
        reaped_ids: list[int] = []
        entries: list[ContextEntry] = []
        for iid, e in reaped:
            logger.warning(
                "Reaped GPU instance %d: silent for %.1fs (pinged=%s)",
                iid,
                now - e.last_seen,
                e.has_liveness_signal,
            )
            reaped_ids.append(iid)
            entries.append(e)
        if reaped:
            del e  # a bound name would pin the final entry (see _release_entries)
            reaped.clear()
            self._retire_entries(entries)
        return reaped_ids

    def _release_entries(self, entries: list[ContextEntry]) -> None:
        """Release a batch of entries and reclaim their device memory.

        Args:
            entries: The only remaining references to the released entries.
                The list is cleared before memory is reclaimed.
        """
        if not entries:
            return
        num_entries = len(entries)
        free_before = _free_device_memory_bytes()
        for entry in entries:
            entry.cache_context.close(drain_timeout_s=self._close_drain_timeout_s)
            self._ctx.layout_desc_registry.unregister(
                entry.model_name, entry.world_size
            )
        del entry
        entries.clear()
        # ipc_collect() only unmaps a CUDA-IPC-imported segment once its last
        # tensor reference is gone (LMCache#4014), hence the clear() above.
        # A full gc pass first: the context holds CuPy ExternalStream / tensor
        # refcycles (#4261) that plain refcounting does not break, and a
        # cycle-held KV tensor keeps the worker's pool mapped.
        gc.collect()
        torch_dev.empty_cache()
        ipc_collect = getattr(torch_dev, "ipc_collect", None)
        if ipc_collect is not None:
            # Backends without IPC collection omit this optional operation.
            ipc_collect()
        free_after = _free_device_memory_bytes()
        if free_before is not None and free_after is not None:
            logger.info(
                "Released %d context(s): free device memory %.2f -> %.2f GiB "
                "(%+.2f GiB)",
                num_entries,
                free_before / 2**30,
                free_after / 2**30,
                (free_after - free_before) / 2**30,
            )

    def _ipc_collect_cycle(self) -> ThreadRunSummary:
        """Periodic IPC handle reclamation and dead-entry timeout sweep.

        Runs every ``_IPC_COLLECT_INTERVAL_S`` seconds.  Two jobs:

        1. **Dead-entry sweep** — force-release context entries that were
           marked dead by ``_retire_entries`` but never checked in (the
           store/retrieve handler raised or the worker was killed
           mid-transfer).  Without this safety net, such entries and
           their IPC handles leak permanently.

        2. **Global IPC collection** — call ``empty_cache()`` +
           ``ipc_collect()`` to reclaim any CUDA-IPC-imported segments
           whose Python references were dropped since the last cycle
           but whose underlying handles have not yet been collected by
           the CUDA driver.  This bounds handle growth between explicit
           release events (reap / unregister / close).

        Returns:
            A summary recording how many dead entries were force-released.
        """
        # 1. Sweep dead entries past the checkin timeout.
        now = time.monotonic()
        expired: list[ContextEntry] = []
        with self._lock:
            # Snapshot to avoid mutating while iterating; entries to release
            # are removed from _dead_entries before _release_entries is called
            # (which itself acquires no lock — leaf-lock invariant).
            for entry, dead_since in list(self._dead_entries.items()):
                if now - dead_since > _CHECKIN_TIMEOUT_S:
                    self._dead_entries.pop(entry, None)
                    # Force the in_use counter to zero so _release_entries
                    # proceeds; the transfer that pinned it is certainly dead
                    # by now (it has been _CHECKIN_TIMEOUT_S seconds).
                    entry.in_use = 0
                    entry.released = True
                    expired.append(entry)

        force_released = 0
        if expired:
            logger.warning(
                "Force-releasing %d dead context entry(s) that exceeded "
                "checkin timeout of %.0fs",
                len(expired),
                _CHECKIN_TIMEOUT_S,
            )
            # _release_entries clears the list (drops refs before ipc_collect),
            # so capture the count before the call.
            force_released = len(expired)
            self._release_entries(expired)

        # 2. Global IPC collection (best-effort, never fatal).
        #    Skip when force_released > 0 because _release_entries already
        #    ran empty_cache() + ipc_collect() for the expired entries.
        if force_released == 0:
            try:
                torch_dev.empty_cache()
                ipc_collect = getattr(torch_dev, "ipc_collect", None)
                if ipc_collect is not None:
                    ipc_collect()
            except Exception as e:
                logger.debug("Periodic ipc_collect failed (non-fatal): %s", e)

        return ThreadRunSummary(
            success=True,
            message=f"force_released={force_released}",
        )

    def report_status(self) -> dict:
        """Return GPU transfer module status information.

        Returns:
            A dict containing registered GPU instance IDs and
            per-instance KV cache layout metadata.
        """
        registered_gpu_ids: list[int] = []
        cache_context_meta: dict[str, dict] = {}

        for instance_id, entry in self.context_entries_snapshot().items():
            registered_gpu_ids.append(instance_id)
            ctx = entry.cache_context
            cache_context_meta[str(instance_id)] = {
                "model_name": entry.model_name,
                "world_size": entry.world_size,
                "kv_cache_layout": ctx.report_status(),
            }

        return {
            "registered_gpu_ids": registered_gpu_ids,
            "cache_context_meta": cache_context_meta,
        }

    def close(self) -> None:
        """Release GPU resources owned by this module."""
        # Stop the periodic IPC collector before releasing entries so it
        # does not race with the final cleanup.
        if self._ipc_collector is not None:
            PeriodicThreadRegistry.get_instance().unregister(
                self._ipc_collector.name
            )
            self._ipc_collector.stop()
            self._ipc_collector = None

        # Stop the drain thread before storage_manager.close() so any
        # in-flight completions reach a live storage manager.
        self._device_host_func_dispatcher.stop()

        with self._lock:
            entries = list(self._cache_contexts.values())
            self._cache_contexts.clear()
            # Also grab any dead entries that were deferred.
            entries.extend(self._dead_entries.keys())
            self._dead_entries.clear()
        self._retire_entries(entries)

    @request_handler()
    def register_kv_cache(
        self,
        instance_id: int,
        kv_caches: KVCache,
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
        engine_group_infos: list[EngineGroupInfo],
    ) -> None:
        """Register the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id: The GPU instance ID (such as PID).
            kv_caches: The KV cache tensor wrappers from the
                serving engine.
            model_name: The name of the model associated with this KV cache.
            world_size: The world size associated with this KV cache.
            engine_type: Which serving engine produced the caches.
                Forwarded to GPUCacheContext for format detection.
            layout_hints: See LayoutHints.  Forwarded to
                GPUCacheContext for GPU KV format detection.
            engine_group_infos: Engine-neutral KV cache group metadata
                (already msgspec-decoded by the message queue).
        """
        now = time.monotonic()
        # NOOP-register: an already-registered instance (e.g. a recovering
        # worker re-registering on its first ping) refreshes its last-seen
        # time so a stale entry is not reaped right after recovery. REGISTER
        # is SYNC-serialized on the MQ main loop, so it is the sole inserter.
        with self._lock:
            existing = self._cache_contexts.get(instance_id)
            if existing is not None:
                existing.last_seen = now
                logger.info(
                    "Instance %d already registered; refreshing liveness",
                    instance_id,
                )
                return

        # Build the context and layout descriptor outside the lock.
        cache_context = create_cache_context(
            kv_caches,
            self._ctx.chunk_size,
            layout_hints=layout_hints or None,
            engine_group_infos=engine_group_infos,
            engine_type=engine_type,
            separate_object_groups=self._ctx.separate_object_groups,
            full_sw_kv=self._ctx.full_sw_kv,
            max_batch_size=self._ctx.max_batch_size,
        )
        kv_groups_manager = cache_context.kv_layer_groups_manager
        num_object_groups = kv_groups_manager.num_object_groups
        event_backend = get_event_ipc_backend(cache_context.device)
        event_backend.check_event_support(cache_context.device)
        layout_desc = get_layout_desc(
            cache_context, self._ctx.chunk_size, object_group_id=0
        )
        # One layout per object group, also in the single-group case: no
        # None special-casing downstream (group 0 maps to the merged layout).
        group_layout_descs = {
            gid: get_layout_desc(
                cache_context, self._ctx.chunk_size, object_group_id=gid
            )
            for gid in range(num_object_groups)
        }
        attn_desc = kv_groups_manager.get_attn_desc()
        self._ctx.layout_desc_registry.register(
            model_name,
            world_size,
            layout_desc,
            attn_desc,
            group_layout_descs=group_layout_descs,
        )

        with self._lock:
            self._cache_contexts[instance_id] = ContextEntry(
                cache_context=cache_context,
                model_name=model_name,
                world_size=world_size,
                last_seen=now,
                has_liveness_signal=False,
                event_backend=event_backend,
            )

        logger.info(
            "Registered KV cache for GPU ID %d with %d layers",
            instance_id,
            cache_context.num_layers,
        )

    @request_handler()
    def unregister_kv_cache(self, instance_id: int) -> None:
        """Unregister the KV cache tensors for a given GPU instance ID.

        Args:
            instance_id: The GPU instance ID (such as PID).
        """
        with self._lock:
            popped = [
                e
                for e in (self._cache_contexts.pop(instance_id, None),)
                if e is not None
            ]
        if not popped:
            logger.warning(
                "No registered GPU context found for instance ID %d", instance_id
            )
            return

        # No scalar binding: `popped` must stay the only reference so
        # the release's reclaim actually unmaps the IPC segments.
        self._retire_entries(popped)
        logger.info("Unregistered KV cache for GPU ID %d", instance_id)

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    @_lmcache_nvtx_annotate
    def store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        """Store the GPU KV cache blocks to CPU.

        Args:
            key: The IPC key for the KV cache blocks.
                Must have worker_id != None (worker store operation).
            instance_id: The GPU instance ID (such as PID).
            gpu_block_ids: GPU block IDs to store, indexed by LMCache KV
                group index.
            event_ipc_handle: The IPC handle of the event to wait on.

        Returns:
            A tuple where the first element is the IPC handle of the event
            that signals the completion of the store operation, and the second
            element indicates whether the store operation completed without a
            fatal error (not whether every requested chunk was stored; see
            Notes). The event handle is empty when no device work was submitted.

        Raises:
            RuntimeError: If the backend does not support IPC event handles.

        Notes:
            All-or-nothing. If ``gpu_block_ids`` do not fully cover every chunk
            ``key`` resolves to for every LMCache group (e.g. a caller/protocol
            bug), or a copy fails, the whole store is skipped and nothing is
            committed (logged at WARNING); a subsequent retrieve simply misses
            and the engine recomputes. The boolean result reports whether the
            store completed without such a failure.

            In ``host`` producer-wait mode (default on ROCm/HIP) the store is
            also skipped, with an empty event handle, when the worker is
            reaped before its producer event fires; nothing is enqueued.
        """
        # Pin the entry for the whole operation: a reap/unregister in this
        # window then defers the context close to the checkin below instead of
        # unmapping IPC/SHM segments under our tensor copies (SIGSEGV).
        entry = self.checkout_context_entry(instance_id)
        if entry is None:
            # The worker can reconnect to a replacement server before its next
            # registration probe. No device work was submitted in that window,
            # so return an empty completion-event handle and a terminal False
            # response instead of leaving the MQ future unanswered. Echoing the
            # producer handle would make the originating process import its own
            # IPC event, which is invalid on HIP.
            logger.warning(
                "Rejecting STORE for unregistered GPU instance ID %d",
                instance_id,
            )
            return b"", False
        try:
            return self._store_pinned(
                key, instance_id, gpu_block_ids, event_ipc_handle, entry
            )
        finally:
            self.checkin_context_entry(entry)

    def _store_pinned(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        entry: ContextEntry,
    ) -> tuple[bytes, bool]:
        """Body of :meth:`store`, running with *entry* pinned against release."""
        st = time.perf_counter()

        cache_context = entry.cache_context
        model_name = entry.model_name
        event_backend = entry.event_backend
        if event_backend is None:
            raise RuntimeError("Registered cache context has no event backend")

        num_object_groups = cache_context.kv_layer_groups_manager.num_object_groups
        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
            key, list(range(num_object_groups))
        )
        num_chunks = len(obj_keys_per_obj_group[0])

        # NOTE: different engine groups may have different block sizes, so
        # ``blocks_per_chunk[i]`` is the number of blocks in one chunk for
        # group ``i``.
        blocks_per_chunk = [
            cache_context.calculate_num_blocks(self._ctx.chunk_size, group_idx)
            for group_idx in range(
                cache_context.kv_layer_groups_manager.num_kernel_groups
            )
        ]

        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            event = event_backend.create_event(cache_context.device)

            # Fail closed: every LMCache group must have block IDs covering all
            # chunks. A short list (e.g. a caller/protocol bug) would otherwise
            # drive the transfer kernel to read out-of-bounds GPU memory, so skip
            # the whole store and commit nothing rather than caching a partial or
            # garbage entry. A later request can store it once the block IDs are
            # complete. Checked on the raw block ids, before cutting drops the
            # per-chunk blocks that sliding-window groups do not need.
            if any(
                len(group_block_ids) < num_chunks * bpc
                for group_block_ids, bpc in zip(
                    gpu_block_ids, blocks_per_chunk, strict=True
                )
            ):
                logger.warning(
                    "STORE block ID underflow for request_id=%s: each group needs "
                    "num_chunks * blocks_per_chunk block IDs for %d chunks "
                    "(per-group blocks_per_chunk=%s); skipping the store.",
                    key.request_id,
                    num_chunks,
                    blocks_per_chunk,
                )
                event_backend.record_event(event, cache_context.stream)
                return event_backend.export_event(event, cache_context.device), False

            # Chunks whose block ids are all the null block (e.g. align-mode
            # Mamba chunks holding no real state) carry no valid KV and must not
            # be committed. Computed on the raw block ids before downsampling
            # mutates them.
            null_block_id = self._ctx.null_block_id
            skipped_chunks = all_null_chunk_masks(
                gpu_block_ids,
                cache_context.kv_layer_groups_manager.object_groups,
                blocks_per_chunk,
                num_chunks,
                null_block_id,
            )

            block_ids_per_group_gpu = downsample_and_stage_block_ids(
                cache_context, gpu_block_ids
            )

            producer_event = event_backend.import_event(
                event_ipc_handle, cache_context.device
            )
            if not self._wait_for_producer(
                entry,
                producer_event,
                event,
                event_backend,
                cache_context,
                key,
                "STORE",
            ):
                # Nothing was enqueued: no completion event, terminal False.
                return b"", False

            # CPU-synchronous sentinel: a GPU store is about to be enqueued.
            # Must be published via publish() (not publish_on_stream) so the
            # drain thread sees it before MP_REQUEST_END can race MP_STORE_END.
            self._ctx.event_bus.publish(
                Event(
                    event_type=EventType.MP_STORE_SUBMITTED,
                    session_id=key.request_id,
                    metadata={"device": str(cache_context.device)},
                )
            )

            # Worker 0 only: bindings depend on token content alone, so one
            # report covers every rank's keys. Published before finish_write
            # is enqueued so the token bindings precede the write-finished
            # events on the bus.
            if key.worker_id == 0 and self._ctx.event_bus.has_subscribers(
                EventType.MP_TOKENS
            ):
                self._publish_token_bindings(key, obj_keys_per_obj_group[0])

            transfer_key = next_transfer_key(key.request_id)
            self._ctx.event_bus.publish_on_stream(
                cache_context.cupy_stream,
                Event(
                    event_type=EventType.MP_STORE_START,
                    session_id=key.request_id,
                    metadata={
                        "device": str(cache_context.device),
                        "engine_id": instance_id,
                        "model_name": model_name,
                        "transfer_key": transfer_key,
                    },
                ),
            )

            reserved_dict: dict[ObjectKey, MemoryObj] = {}
            all_dict: dict[ObjectKey, MemoryObj] = {}
            total_bytes: int = 0
            store_succeeded = False
            try:
                for obj_group_id in range(num_object_groups):
                    obj_keys = obj_keys_per_obj_group[obj_group_id]
                    skip_mask = skipped_chunks[obj_group_id]
                    keys_to_reserve = [
                        k for i, k in enumerate(obj_keys) if not skip_mask[i]
                    ]
                    layout_desc = get_layout_desc(
                        cache_context,
                        self._ctx.chunk_size,
                        object_group_id=obj_group_id,
                    )
                    reserved_dict = self._ctx.storage_manager.reserve_write(
                        keys_to_reserve, layout_desc
                    )
                    all_dict.update(reserved_dict)
                    if reserved_dict:
                        total_bytes += next(
                            iter(reserved_dict.values())
                        ).get_size() * len(reserved_dict)

                    # Keys not in reserved_dict (all-null chunks skipped above, or
                    # skipped by the storage manager) become None entries; the
                    # helper skips them for D2H.
                    memory_objs: list[MemoryObj | None] = [
                        reserved_dict.get(obj_key) for obj_key in obj_keys
                    ]

                    # NOTE: batch_size must stay 1 for store.
                    transfer_kv_per_object_group(
                        cache_context,
                        block_ids_per_group_gpu,
                        memory_objs,
                        object_group_id=obj_group_id,
                        batch_size=1,
                        skip_first_n_tokens=0,
                        direction=lmcache_native.TransferDirection.D2H,
                        transfer_key=transfer_key,
                    )

                store_succeeded = True
            except Exception:
                logger.exception("Cannot store keys due to exception")
            finally:
                event_backend.record_event(event, cache_context.stream)
                # Fail closed: commit the reserved objects only when every chunk
                # copied successfully; otherwise the whole store is skipped.
                stored_count = len(all_dict) if store_succeeded else 0
                if stored_count:
                    submit_callback_to_stream(
                        cache_context.cupy_stream,
                        "finish_write",
                        list(all_dict.keys()),
                    )
                else:
                    total_bytes = 0
                num_tokens = num_chunks * self._ctx.chunk_size if stored_count else 0
                self._ctx.event_bus.publish_on_stream(
                    cache_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_STORE_END,
                        session_id=key.request_id,
                        metadata={
                            "stored_count": stored_count,
                            "device": str(cache_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "total_bytes": total_bytes,
                            "num_tokens": num_tokens,
                            "transfer_key": transfer_key,
                        },
                    ),
                )

        ed = time.perf_counter()
        if stored_count:
            logger.info(
                "Stored %d tokens in %.3f seconds",
                num_chunks * self._ctx.chunk_size,
                ed - st,
            )
        return (
            event_backend.export_event(event, cache_context.device),
            store_succeeded,
        )

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    @_lmcache_nvtx_annotate
    def retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int = 0,
    ) -> tuple[bytes, bool]:
        """Retrieve the CPU KV cache and put into GPU blocks.

        Args:
            key: The IPC key for the KV cache blocks.
                Must have worker_id != None (worker retrieve operation).
            instance_id: The GPU instance ID (such as PID).
            gpu_block_ids: GPU block IDs to retrieve into, indexed by LMCache
                KV group index.
            event_ipc_handle: The IPC handle of the event to wait on.
            skip_first_n_tokens: Number of tokens to skip writing at
                the start of the retrieve range. This avoids overwriting
                APC-shared GPU blocks that may be read concurrently by other
                requests.

        Returns:
            A tuple where the first element is the IPC handle of the event
            that signals the completion of the retrieve operation, and the
            second element indicates whether the key was successfully retrieved.
            The event handle is empty when no device work was submitted.

        Raises:
            RuntimeError: If the backend does not support IPC event handles.

        Notes:
            In ``host`` producer-wait mode (default on ROCm/HIP) the retrieve
            returns ``(b"", False)`` without enqueuing anything, and releases
            the lookup's prefetch locks, when the worker is reaped before its
            producer event fires.
        """
        # See store(): the pin defers a concurrent reap/unregister's context
        # close until the checkin below.
        entry = self.checkout_context_entry(instance_id)
        if entry is None:
            # See store(): there is no completion event because no device work
            # was submitted. The False result lets the caller recover or
            # recompute without importing its own producer event.
            logger.warning(
                "Rejecting RETRIEVE for unregistered GPU instance ID %d",
                instance_id,
            )
            try:
                self._release_failed_retrieve_locks(key, instance_id)
            except Exception:
                # A cleanup failure must never suppress the terminal response:
                # the client otherwise waits forever because blocking-handler
                # exceptions are only logged by the MQ server.
                logger.exception(
                    "Failed to release RETRIEVE locks for unregistered "
                    "GPU instance ID %d",
                    instance_id,
                )
            return b"", False
        try:
            return self._retrieve_pinned(
                key,
                instance_id,
                gpu_block_ids,
                event_ipc_handle,
                skip_first_n_tokens,
                entry,
            )
        finally:
            self.checkin_context_entry(entry)

    def _retrieve_pinned(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int,
        entry: ContextEntry,
    ) -> tuple[bytes, bool]:
        """Body of :meth:`retrieve`, running with *entry* pinned against release."""
        st = time.perf_counter()

        cache_context = entry.cache_context
        model_name = entry.model_name
        event_backend = entry.event_backend
        if event_backend is None:
            raise RuntimeError("Registered cache context has no event backend")

        num_object_groups = cache_context.kv_layer_groups_manager.num_object_groups
        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
            key, list(range(num_object_groups))
        )
        num_chunks = len(obj_keys_per_obj_group[0])

        # CPU-synchronous sentinel: a GPU retrieve is about to be enqueued.
        # Must be published via publish() (not publish_on_stream) so the
        # drain thread sees it before MP_REQUEST_END can race MP_RETRIEVE_END.
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_RETRIEVE_SUBMITTED,
                session_id=key.request_id,
                metadata={"device": str(cache_context.device)},
            )
        )

        transfer_key = next_transfer_key(key.request_id)
        self._ctx.event_bus.publish_on_stream(
            cache_context.cupy_stream,
            Event(
                event_type=EventType.MP_RETRIEVE_START,
                session_id=key.request_id,
                metadata={
                    "device": str(cache_context.device),
                    "engine_id": instance_id,
                    "model_name": model_name,
                    "transfer_key": transfer_key,
                },
            ),
        )

        blocks_per_chunk = [
            cache_context.calculate_num_blocks(self._ctx.chunk_size, group_idx)
            for group_idx in range(
                cache_context.kv_layer_groups_manager.num_kernel_groups
            )
        ]

        with (
            torch_dev.device(cache_context.device),
            torch_dev.stream(cache_context.stream),
        ):
            event = event_backend.create_event(cache_context.device)

            # Fail closed: a short block-id list would drive the transfer
            # kernel to write out-of-bounds GPU memory. Checked on the raw
            # block ids, before cutting drops the per-chunk blocks that
            # sliding-window groups do not need.
            if any(
                len(group_block_ids) < num_chunks * bpc
                for group_block_ids, bpc in zip(
                    gpu_block_ids, blocks_per_chunk, strict=True
                )
            ):
                logger.error(
                    "RETRIEVE block ID underflow for request_id=%s: each group "
                    "needs num_chunks * blocks_per_chunk block IDs for %d "
                    "chunks (per-group blocks_per_chunk=%s); skipping the "
                    "retrieve.",
                    key.request_id,
                    num_chunks,
                    blocks_per_chunk,
                )
                event_backend.record_event(event, cache_context.stream)
                return event_backend.export_event(event, cache_context.device), False

            # Cut and stage all block_ids to GPU once before the transfer
            block_ids_per_group_gpu = downsample_and_stage_block_ids(
                cache_context, gpu_block_ids
            )
            producer_event = event_backend.import_event(
                event_ipc_handle, cache_context.device
            )
            if not self._wait_for_producer(
                entry,
                producer_event,
                event,
                event_backend,
                cache_context,
                key,
                "RETRIEVE",
            ):
                # Nothing was enqueued. Release the locks the lookup's prefetch
                # took for this request and close the observability span that
                # MP_RETRIEVE_SUBMITTED opened above.
                try:
                    self._release_failed_retrieve_locks(key, instance_id)
                except Exception:
                    logger.exception(
                        "Failed to release RETRIEVE locks after abandoning the "
                        "producer wait for GPU instance ID %d",
                        instance_id,
                    )
                self._ctx.event_bus.publish(
                    Event(
                        event_type=EventType.MP_RETRIEVE_END,
                        session_id=key.request_id,
                        metadata={
                            "retrieved_count": 0,
                            "device": str(cache_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "cache_salt": key.cache_salt,
                            "total_bytes": 0,
                            "num_tokens": 0,
                        },
                    )
                )
                return b"", False

            # Per object group, the prefetch only locked the in-window suffix
            # (the last ``num_chunks_in_sw`` chunks; the whole prefix for full
            # attention, where the value is < 0). Read and transfer only those.
            # Aux (connector-private) groups are never served by the
            # std retrieve: the lookup does not lock their keys and their
            # block-id entry is a placeholder -- reading them would be an
            # unlocked read of a plane nobody consumes here.
            attn_desc = cache_context.kv_layer_groups_manager.get_attn_desc()
            skipped_groups = {
                g for g, kind in enumerate(attn_desc.group_kinds) if kind == "aux"
            }
            group_skips = [
                0 if window < 0 else max(0, num_chunks - window)
                for window in attn_desc.num_chunks_in_sw
            ]
            expected_retained = sum(
                num_chunks - skip
                for g, skip in enumerate(group_skips)
                if g not in skipped_groups
            )

            prefetched_keys: list[ObjectKey] = []
            total_bytes = 0
            retrieve_succeeded = True
            try:
                for obj_group_id in range(num_object_groups):
                    if obj_group_id in skipped_groups:
                        continue
                    skip = group_skips[obj_group_id]
                    in_window_keys = obj_keys_per_obj_group[obj_group_id][skip:]
                    with self._ctx.storage_manager.read_prefetched_results(
                        in_window_keys
                    ) as window_objs:
                        if not window_objs or len(window_objs) != len(in_window_keys):
                            logger.error("Some keys not found during retrieve!")
                            retrieve_succeeded = False
                            break

                        total_bytes += sum(mo.get_size() for mo in window_objs)

                        # None-pad the skipped prefix to full length so the
                        # transfer's ``num_objects_to_skip`` and block-id slicing
                        # line up unchanged; the None entries are never read.
                        memory_objs: list[MemoryObj | None] = [None] * skip + list(
                            window_objs
                        )

                        transfer_kv_per_object_group(
                            cache_context,
                            block_ids_per_group_gpu,
                            memory_objs,
                            object_group_id=obj_group_id,
                            batch_size=cache_context.max_batch_size,
                            skip_first_n_tokens=skip_first_n_tokens,
                            direction=lmcache_native.TransferDirection.H2D,
                            transfer_key=transfer_key,
                        )
                        # Extend only after the copy is enqueued: on exception,
                        # read_prefetched_results releases this group's locks
                        # itself, and a key must not be released twice.
                        prefetched_keys.extend(in_window_keys)
            except Exception:
                logger.exception("Cannot retrieve keys due to exception")
                retrieve_succeeded = False
            finally:
                event_backend.record_event(event, cache_context.stream)
                if prefetched_keys:
                    submit_callback_to_stream(
                        cache_context.cupy_stream,
                        "finish_read_prefetched",
                        prefetched_keys,
                    )
                num_tokens = (
                    num_chunks * self._ctx.chunk_size
                    if len(prefetched_keys) == expected_retained
                    else 0
                )
                self._ctx.event_bus.publish_on_stream(
                    cache_context.cupy_stream,
                    Event(
                        event_type=EventType.MP_RETRIEVE_END,
                        session_id=key.request_id,
                        metadata={
                            "retrieved_count": len(prefetched_keys),
                            "device": str(cache_context.device),
                            "engine_id": instance_id,
                            "model_name": model_name,
                            "cache_salt": key.cache_salt,
                            "total_bytes": total_bytes,
                            "num_tokens": num_tokens,
                            "transfer_key": transfer_key,
                        },
                    ),
                )
        if retrieve_succeeded:
            tokens_retrieved = num_chunks * self._ctx.chunk_size
            ed = time.perf_counter()
            logger.info(
                "Retrieved %d tokens in %.3f seconds",
                tokens_retrieved,
                ed - st,
            )

        return (
            event_backend.export_event(event, cache_context.device),
            retrieve_succeeded,
        )

    def _publish_token_bindings(
        self, key: IPCCacheServerKey, obj_keys: list[ObjectKey]
    ) -> None:
        """Publish one ``MP_TOKENS`` event for ``key``'s chunks.

        Pairs each complete chunk in ``[key.start, key.end)`` with its
        ObjectKey chunk hash and token position. Must be called at store
        submission, before the write-finished events reach the bus, so the
        cache-event subscriber can stamp them onto the STORE entries. A
        store that later fails leaves only unused cache entries.

        Args:
            key: The IPC key of the store being submitted.
            obj_keys: One ObjectKey per complete chunk, in chunk order.
        """
        # Complete chunks in [key.start, key.end) paired with the absolute
        # position of each chunk's first token. Prefix-chained chunk hashes
        # imply a position without revealing it, so it is reported here. A
        # trailing partial chunk has no stored KV to bind to.
        chunk_size = self._ctx.chunk_size
        token_ids = list(key.token_ids)
        effective_len = min(len(token_ids), key.end)
        num_complete = effective_len - effective_len % chunk_size
        token_offsets = list(range(key.start, num_complete, chunk_size))
        token_chunks = [
            token_ids[offset : offset + chunk_size] for offset in token_offsets
        ]
        if not token_chunks:
            return
        if len(obj_keys) != len(token_chunks):
            logger.warning(
                "Skipping token bindings for request %s: %d resolved keys "
                "vs %d complete chunks in [%d, %d)",
                key.request_id,
                len(obj_keys),
                len(token_chunks),
                key.start,
                key.end,
            )
            return
        self._ctx.event_bus.publish(
            Event(
                event_type=EventType.MP_TOKENS,
                session_id=key.request_id,
                metadata={
                    "chunk_hashes": [obj_key.chunk_hash for obj_key in obj_keys],
                    "token_chunks": token_chunks,
                    "token_offsets": token_offsets,
                },
            )
        )

    def _wait_for_producer(
        self,
        entry: ContextEntry,
        producer_event: object,
        completion_event: object,
        event_backend: EventIPCBackend,
        cache_context: BaseCacheContext,
        key: IPCCacheServerKey,
        op: str,
    ) -> bool:
        """Order the upcoming transfer after the worker's producer event.

        In ``DEVICE`` mode this enqueues the wait on the context stream and
        returns immediately. In ``HOST`` mode it polls the event on this
        thread and returns False -- without enqueuing anything -- if the
        owning worker is reaped first or the ceiling elapses; the caller then
        answers the request with an empty completion handle and ``False``.

        Args:
            entry: The pinned context entry (its ``dead`` flag aborts the wait).
            producer_event: The imported producer event.
            completion_event: This transfer's own, not-yet-recorded event.
            event_backend: Event backend for the context's device.
            cache_context: The worker's cache context.
            key: The request key (for logging).
            op: ``"STORE"`` or ``"RETRIEVE"`` (for logging).

        Returns:
            True if the transfer may be enqueued, False if it must be skipped.
        """
        if self._producer_wait.mode is ProducerWaitMode.DEVICE:
            event_backend.wait_event(producer_event, cache_context.stream)
            return True

        started = time.perf_counter()
        outcome = wait_for_producer_event(
            event_backend,
            producer_event,
            producer_is_dead=lambda: entry.dead,
            timeout_s=self._producer_wait.timeout_s,
        )
        if outcome is ProducerWaitOutcome.READY:
            return True

        waited_s = time.perf_counter() - started
        if outcome is ProducerWaitOutcome.PRODUCER_DEAD:
            logger.warning(
                "%s for request_id=%s skipped: GPU instance was reaped %.1fs "
                "into the wait for its producer event; nothing was enqueued",
                op,
                key.request_id,
                waited_s,
            )
        else:
            logger.warning(
                "%s for request_id=%s skipped: producer event not signaled "
                "within %.0fs while the instance still looks alive; nothing "
                "was enqueued",
                op,
                key.request_id,
                waited_s,
            )
        # Neither event may be destroyed now: the imported one belongs to a
        # producer that will never signal it (hipEventDestroy would block),
        # and the local one is kept with it so the pair's lifetime is simple.
        self._abandoned_events.retain(producer_event, completion_event)
        return False
