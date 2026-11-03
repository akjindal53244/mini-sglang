"""KV storage, physical page allocation, and reusable prompt prefixes."""

from __future__ import annotations

import heapq
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import count

import torch
from minisgl.core import Request
from minisgl.distributed import get_tp_info
from minisgl.kernel import fast_compare_key
from minisgl.utils import align_down, div_ceil, div_even


class MHAKVCache:
    """Paged storage: [K/V, layer, page, token, local KV head, head dimension]."""

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        local_heads = div_even(num_kv_heads, get_tp_info().size, allow_replicate=True)
        # One big allocation. self.k[layer] is [page, token, local head, head_dim].
        buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_heads, head_dim), device=device, dtype=dtype
        )
        self.k, self.v = buffer.unbind(0)
        self.device, self.dtype = device, dtype

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int):
        from minisgl.kernel import store_cache

        # Flatten page and token dimensions: out_loc contains physical token
        # slots even when attention kernels address this storage by whole pages.
        store_cache(self.k[layer_id].flatten(0, 1), self.v[layer_id].flatten(0, 1), out_loc, k, v)


# A logical clock orders LRU eviction. Wall time could tie differently per process,
# and every TP rank must evict the same nodes to keep identical batches.
_lru_clock = count()


@dataclass(eq=False)
class RadixTreeNode:
    key: torch.Tensor  # CPU token IDs on this compressed edge.
    value: torch.Tensor  # Corresponding physical KV token slots.
    parent: RadixTreeNode | None = None
    children: dict = field(default_factory=dict)
    ref_count: int = 0
    timestamp: int = field(default_factory=lambda: next(_lru_clock))

    @property
    def length(self) -> int:
        return len(self.key)

    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class CacheHandle:
    cached_len: int
    node: RadixTreeNode

    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        value_list: list[torch.Tensor] = []
        while node.parent is not None:
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        return torch.cat(value_list) if value_list else node.value


class RadixPrefixCache:
    """Share full-page prefixes and evict the least-recently-used unlocked leaves.

    With enabled=False, every match is empty and KV is freed when a request ends.

    Locks cover the entire path to the root, protecting KV still read by active
    requests. Background: https://lmsys.org/blog/2024-01-17-sglang/
    """

    def __init__(self, device: torch.device, page_size: int, enabled: bool = True):
        self.enabled = enabled
        self.device = device
        self.page_size = page_size
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(
            torch.empty(0, dtype=torch.int32), self.empty_tensor, ref_count=1
        )  # The empty root is always protected.

    def _page_key(self, ids: torch.Tensor):
        return ids[0].item() if self.page_size == 1 else tuple(ids[: self.page_size].tolist())

    def _split_edge(self, node: RadixTreeNode, length: int) -> RadixTreeNode:
        assert 0 < length < node.length
        prefix = RadixTreeNode(
            node.key[:length],
            node.value[:length],
            node.parent,
            ref_count=node.ref_count,
            timestamp=node.timestamp,
        )
        node.parent.children[self._page_key(prefix.key)] = prefix
        node.key, node.value = node.key[length:], node.value[length:]
        node.parent = prefix
        prefix.children[self._page_key(node.key)] = node
        # Existing handles still point to node. Its new parent preserves the
        # same full prefix and inherits locks; no KV storage moves during a split.
        return prefix

    def lock_handle(self, handle: CacheHandle, unlock: bool = False) -> None:
        node, delta = handle.node, -1 if unlock else 1
        while node.parent is not None:
            old_count = node.ref_count
            node.ref_count += delta
            assert node.ref_count >= 0
            # Only 0 <-> 1 transitions change capacity. Several active requests
            # can share one path; every ancestor must remain protected.
            if old_count == 0:
                self.evictable_size -= node.length
                self.protected_size += node.length
            elif node.ref_count == 0:
                self.evictable_size += node.length
                self.protected_size -= node.length
            node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> CacheHandle:
        if not self.enabled:
            return CacheHandle(0, self.root_node)
        node, prefix_len = self._tree_walk(input_ids)
        return CacheHandle(prefix_len, node)

    def insert_prefix(
        self, input_ids: torch.Tensor, indices: torch.Tensor
    ) -> tuple[int, CacheHandle]:
        if not self.enabled:
            return 0, CacheHandle(0, self.root_node)
        assert len(input_ids) == len(indices), "Each cached token needs one KV location"
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:
            # Clone locations: the request's mutable page-table row will be reused.
            new_node = RadixTreeNode(input_ids[prefix_len:], indices[prefix_len:].clone(), node)
            node.children[self._page_key(new_node.key)] = new_node
            self.evictable_size += new_node.length
            node = new_node
        return prefix_len, CacheHandle(insert_len, node)

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        leaves = self._collect_evictable_leaves()
        heapq.heapify(leaves)
        evicted_indices: list[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert leaves, f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leaves)
            assert node.ref_count == 0 and not node.children and node.parent is not None
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self._page_key(node.key)]
            # The empty root stays protected and never enters the eviction heap.
            if (not parent.children) and parent.ref_count == 0:
                heapq.heappush(leaves, parent)

        return torch.cat(evicted_indices)

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size

    def check_integrity(self) -> None:
        """Check tree links, page alignment, locks, and token accounting on CPU.

        Inspect tensor metadata only: copying all KV indices to CPU here would
        turn an idle-time diagnostic into a costly GPU synchronization.
        """
        evictable = protected = 0
        seen = {id(self.root_node)}
        pending = [self.root_node]
        while pending:
            parent = pending.pop()
            for key, node in parent.children.items():
                if id(node) in seen or node.parent is not parent:
                    raise RuntimeError("Radix cache contains a cycle or a broken parent link")
                seen.add(id(node))
                if node.length <= 0 or node.length % self.page_size:
                    raise RuntimeError("Radix node length must contain complete pages")
                if len(node.key) != node.length or len(node.value) != node.length:
                    raise RuntimeError("Radix token IDs and KV locations have different lengths")
                if key != self._page_key(node.key):
                    raise RuntimeError("Radix child is stored under the wrong prefix key")
                if node.ref_count < 0 or (
                    parent.parent is not None and node.ref_count > parent.ref_count
                ):
                    raise RuntimeError("Radix path lock counts are inconsistent")
                if node.ref_count:
                    protected += node.length
                else:
                    evictable += node.length
                pending.append(node)
        if (evictable, protected) != (self.evictable_size, self.protected_size):
            raise RuntimeError("Radix cache token accounting does not match its tree")

    def _collect_evictable_leaves(self) -> list[RadixTreeNode]:
        nodes: list[RadixTreeNode] = [self.root_node]
        leaves: list[RadixTreeNode] = []

        while nodes:
            node = nodes.pop()
            if not node.children and node.ref_count == 0:
                leaves.append(node)
            nodes.extend(node.children.values())

        return leaves

    def _tree_walk(self, input_ids: torch.Tensor) -> tuple[RadixTreeNode, int]:
        prefix_len = 0
        input_len = len(input_ids)
        node = self.root_node
        tic = next(_lru_clock)

        while prefix_len < input_len:
            child_node = node.children.get(self._page_key(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len
            node = child_node  # walk to child node

            # The child key guarantees at least the first page matches.
            match_len = fast_compare_key(node.key, input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                node = self._split_edge(node, match_len)
                node.timestamp = tic
                return node, prefix_len

            # update timestamp for accessed node
            node.timestamp = tic

        return node, prefix_len


class KVCacheManager:
    """Allocate physical pages and coordinate ownership with the prefix cache.

    Free-list entries are page-aligned *token offsets*, not page numbers.
    All capacity properties exposed to admission are measured in tokens.
    """

    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor, cache_type: str):
        # For page_size=4, [0, 4, 8] denotes three free pages, not three tokens.
        device = page_table.device
        self.free_pages = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        if cache_type not in ("radix", "naive"):
            raise ValueError(f"Unknown prefix cache: {cache_type}")
        self.prefix_cache = RadixPrefixCache(
            device=device, page_size=page_size, enabled=cache_type == "radix"
        )
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size
        self._deferred_frees: list[torch.Tensor] | None = None

    @property
    def available_tokens(self) -> int:
        return self.prefix_cache.evictable_size + len(self.free_pages) * self.page_size

    def lock(self, handle: CacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: CacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: list[Request]) -> None:
        needed_pages = 0
        allocation_info: list[tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.scheduled_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def cache_req(self, req: Request, *, finished: bool) -> None:
        """Retain reusable KV, free duplicate pages, and release an unshared tail.

        The old handle protects prefixes shared at admission. Another request
        may have inserted additional matching tokens since then; free our
        duplicate allocation for that interval. Only full pages enter the tree.
        """
        # Cancellation can arrive before an in-flight token reaches CPU. Cache
        # only tokens whose IDs are known; release the remaining KV on finish.
        known_len = min(req.cached_len, len(req.input_ids))
        insert_ids = req.input_ids[:known_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        existing_len, new_handle = self.prefix_cache.insert_prefix(
            insert_ids, page_indices[:known_len]
        )
        # Keep the old prefix protected through insertion and tree splitting.
        self.unlock(old_handle)
        duplicate = slice(old_handle.cached_len, existing_len)
        self._free(page_indices[duplicate])
        if finished:  # this tail part should be freed
            self._free(page_indices[new_handle.cached_len :])
        else:  # keep the tail part, update the handle
            req.cache_handle = new_handle
            self.lock(new_handle)
            # This request keeps decoding, so its row must stop naming the pages
            # freed above: a later allocation would overwrite KV it still reads.
            # An in-flight lookahead already holds the old slots, but it runs
            # before any new owner on the engine stream, so that copy stays valid.
            if existing_len > old_handle.cached_len:
                page_indices[duplicate] = new_handle.get_matched_indices()[duplicate]

    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.total_size // self.page_size
        if len(self.free_pages) + cache_pages != self.num_pages:
            raise RuntimeError(
                "KVCacheManager integrity check failed:"
                f" free_pages({len(self.free_pages)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )

    @contextmanager
    def defer_page_frees(self):
        """Append returned pages once per result batch instead of once per request."""
        if self._deferred_frees is not None:
            raise RuntimeError("Nested deferred page frees are not supported")
        self._deferred_frees = []
        try:
            yield
        finally:
            returned_pages = self._deferred_frees
            self._deferred_frees = None
            if returned_pages:
                self.free_pages = torch.cat([self.free_pages, *returned_pages])

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if needed_pages > (free_pages := len(self.free_pages)):
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self.free_pages = torch.cat([self.free_pages, evicted[:: self.page_size]])
            assert len(self.free_pages) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_pages[:needed_pages]
        self.free_pages = self.free_pages[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            pages = indices[:: self.page_size]
            if self._deferred_frees is None:
                self.free_pages = torch.cat([self.free_pages, pages])
            else:
                # indices may view a page-table row that changes before the flush.
                self._deferred_frees.append(pages.clone())

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: list[tuple[int, int, int]],
    page_size: int,
) -> None:
    needed_tokens = len(allocated)
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    page_table[table_idxs, offsets] = allocated
