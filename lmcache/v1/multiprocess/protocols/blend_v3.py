# SPDX-License-Identifier: Apache-2.0
"""Blend V3 protocol definitions."""

# First Party
from lmcache.v1.multiprocess.custom_types import (
    CBMatchResult,
    CBUnifiedLookupResult,
    DeviceIPCWrapper,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.protocols.base import HandlerType, ProtocolDefinition

REQUEST_NAMES = [
    "CB_REGISTER_ROPE_V3",
    "CB_UNREGISTER_ROPE_V3",
    "CB_RETRIEVE_PRE_COMPUTED_V3",
    "CB_UNIFIED_LOOKUP",
    "AUX_PUT",
    "AUX_GET_BY_HASH_IPC",
    "CB_SNAPSHOT_PUT",
    "CB_SNAPSHOT_GET",
]


def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
    """Return V3 blend protocol definitions."""
    return {
        # Register rope state on a previously-registered instance.
        # Payload: (instance_id, cos_sin_caches_ipc, head_size, is_neox_style,
        #           group_to_cache, group_rot).
        # cos_sin_caches_ipc: one IPC handle per distinct rope (dual-RoPE
        # models send two); group_to_cache maps engine group
        # idx -> cache idx (empty = all groups use cache 0).
        # group_rot: per-engine-group rope window [offset_elems, width_elems]
        # ([] entry = skip that group's re-RoPE; empty list = legacy
        # inference). MLA models must declare it — see cb_register_rope.
        # Returns: None.
        "CB_REGISTER_ROPE_V3": ProtocolDefinition(
            payload_classes=[
                int,
                list[DeviceIPCWrapper],
                int,
                bool,
                list[int],
                list[list[int]],
            ],
            response_class=None,
            handler_type=HandlerType.SYNC,
        ),
        # Drop rope state (paged KV cache lives on; use UNREGISTER_KV_CACHE).
        # Payload: (instance_id,). Returns: None.
        "CB_UNREGISTER_ROPE_V3": ProtocolDefinition(
            payload_classes=[int],
            response_class=None,
            handler_type=HandlerType.SYNC,
        ),
        # Retrieve pre-computed chunks into the request's paged blocks.
        # Payload: (key, cb_match_result, gpu_block_ids, instance_id,
        #           event_ipc_handle).
        # gpu_block_ids is per engine group (list[list[int]]).
        "CB_RETRIEVE_PRE_COMPUTED_V3": ProtocolDefinition(
            payload_classes=[
                IPCCacheServerKey,
                list[CBMatchResult],
                list[list[int]],
                int,
                bytes,
            ],
            response_class=tuple[bytes, bool],
            handler_type=HandlerType.BLOCKING,
        ),
        # Unified lookup: server runs prefix lookup + non-prefix fingerprint
        # match in one RPC, reconciles, and prefetches only the complement.
        # Payload:
        #   - key: IPCCacheServerKey carrying the query token IDs.
        #   - tp_size: tensor-parallel size (for MLA multi-reader locking,
        #     mirrors LOOKUP).
        # Returns: CBUnifiedLookupResult(prefix_coverage_tokens,
        #          non_prefix_segments).
        "CB_UNIFIED_LOOKUP": ProtocolDefinition(
            payload_classes=[IPCCacheServerKey, int],
            # Nullable: handler returns None to defer until both the prefix and
            # the sparse chunks are in L1 (mirrors dense QUERY_PREFETCH_STATUS).
            response_class=CBUnifiedLookupResult | None,
            handler_type=HandlerType.BLOCKING,
        ),
        # Generic opaque per-chunk blob store. One blob per cacheable chunk of
        # the request range, stored under the chunk's content hash in object
        # group ``group`` (disjoint from KV groups) so it is reusable across
        # requests. The server NEVER interprets the bytes: the caller packs
        # whatever it wants per chunk (e.g. all layers' compressed projections),
        # so one call covers all chunks (hence all layers) at once. ``sizes``
        # gives the per-chunk byte length used to split ``blob`` and to lay out
        # the read. Payload: (key, group, sizes, blob_ipc). Returns: success.
        "AUX_PUT": ProtocolDefinition(
            payload_classes=[
                IPCCacheServerKey,
                int,
                list[int],
                DeviceIPCWrapper,
            ],
            response_class=bool,
            handler_type=HandlerType.BLOCKING,
        ),
        # GPU-IPC retrieve: the worker exports a GPU receive buffer
        # (dst_ipc) and a forward-fence CUDA event; the server copies each
        # matched chunk straight into that buffer on its stream (same physical
        # GPU, no D2H/H2D/ZMQ bytes), then returns its completion-event handle.
        # Payload: (key, group, chunk_hashes, sizes, dst_ipc, instance_id,
        #           event_ipc_handle). Returns: (event_ipc_handle, ok).
        "AUX_GET_BY_HASH_IPC": ProtocolDefinition(
            payload_classes=[
                IPCCacheServerKey,
                int,
                list[bytes],
                list[int],
                DeviceIPCWrapper,
                int,
                bytes,
            ],
            response_class=tuple[bytes, bool],
            handler_type=HandlerType.BLOCKING,
        ),
        # Prompt-end state snapshot store (recurrent-state hybrids). The blob
        # is opaque: the caller packs the per-layer conv/SSM state at end of
        # prefill plus the unaligned tail-sliver KV rows; ``sizes`` are its
        # section byte lengths, echoed back verbatim on a lookup hit. Keyed
        # by the prompt's last full-chunk chain hash + the tail tokens (and
        # the worker id — state is TP-sharded).
        # Payload: (key [full prompt tokens], sizes, blob_ipc, instance_id,
        #           event_ipc_handle). Returns: success.
        "CB_SNAPSHOT_PUT": ProtocolDefinition(
            payload_classes=[
                IPCCacheServerKey,
                list[int],
                DeviceIPCWrapper,
                int,
                bytes,
            ],
            response_class=bool,
            handler_type=HandlerType.BLOCKING,
        ),
        # Fetch the snapshot blob promised by a CB_UNIFIED_LOOKUP hit into a
        # worker-exported GPU buffer (CUDA IPC, event-gated — mirrors
        # AUX_GET_BY_HASH_IPC). Payload: (key [same request id + tokens as
        # the lookup], dst_ipc, instance_id, event_ipc_handle).
        # Returns: (completion_event_ipc_handle, ok) — ok False on a lost
        # entry so the worker degrades to full recompute of the prefix.
        "CB_SNAPSHOT_GET": ProtocolDefinition(
            payload_classes=[
                IPCCacheServerKey,
                DeviceIPCWrapper,
                int,
                bytes,
            ],
            response_class=tuple[bytes, bool],
            handler_type=HandlerType.BLOCKING,
        ),
    }
