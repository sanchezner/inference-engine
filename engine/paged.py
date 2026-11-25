from dataclasses import dataclass, field

import torch

from engine.config import ModelConfig
from engine.kernels.triton.paged_attn import decode_supported, paged_attn_decode
from engine.kernels.triton.reshape_cache import cache_supported, reshape_and_cache
from engine.model import attention_cached, attention_prefill, merge_heads, mlp, project_qkv, rms_norm
from engine.weights import Weights


# KV lives in pages so a finished sequence can return storage without moving
# anyone else. k and v are [layer, page, kv_head, slot, dim].
# CUDA writes the new K/V with one scatter. One new token reads the pages.
# A prefill with no prefix uses SDPA. A prefix still copies pages (page_gather).
# refcount is how many sequences hold the page. Zero is free, or cached when a prefix still names it.
@dataclass
class PagePool:
    k: torch.Tensor
    v: torch.Tensor
    free: list[int]
    block_size: int
    refcount: list[int] = field(default_factory=list)
    cache: object = None

    def __post_init__(self):
        if self.k.shape != self.v.shape or self.k.ndim != 5:
            raise ValueError(f"k and v pools must share a 5D shape, got {tuple(self.k.shape)} and {tuple(self.v.shape)}")
        if self.k.dtype != self.v.dtype or self.k.device != self.v.device:
            raise ValueError("k and v pools must share a device and dtype")
        n_pages = self.k.shape[1]
        if sorted(self.free) != list(range(n_pages)):
            raise ValueError(f"free list must start as every page id, got {self.free}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.refcount:
            if len(self.refcount) != n_pages:
                raise ValueError(f"refcount length {len(self.refcount)} does not match {n_pages} pages")
        else:
            self.refcount = [0] * n_pages


@dataclass
class Sequence:
    pool: PagePool
    blocks: list[int] = field(default_factory=list)
    seq_len: int = 0


def new_page_pool(cfg: ModelConfig, n_pages: int, block_size: int, *, device, dtype) -> PagePool:
    device = torch.device(device)
    shape = (cfg.n_layers, n_pages, cfg.n_kv_heads, block_size, cfg.head_dim)
    return PagePool(
        k=torch.zeros(shape, device=device, dtype=dtype),
        v=torch.zeros(shape, device=device, dtype=dtype),
        free=list(range(n_pages)),
        block_size=block_size,
    )


def new_sequence(pool: PagePool) -> Sequence:
    return Sequence(pool=pool)


def new_page_cache(cfg: ModelConfig, n_pages: int, block_size: int, *, device, dtype) -> Sequence:
    return new_sequence(new_page_pool(cfg, n_pages, block_size, device=device, dtype=dtype))


def pages_for(block_size: int, total_len: int) -> int:
    # Admission counts with this same ceiling, so a seated request cannot pop an empty free list.
    if total_len <= 0:
        return 0
    return (total_len + block_size - 1) // block_size


def reserve_pages(seq: Sequence, total_len: int) -> None:
    pool = seq.pool
    need = pages_for(pool.block_size, total_len)
    while len(seq.blocks) < need:
        seq.blocks.append(take_page(pool))


def release_pages(seq: Sequence) -> None:
    # Sort so the next pop is the lowest id. The block-table checks rely on that,
    # including the out-of-order reuse [4, 5, 1].
    pool = seq.pool
    recycled = []
    for page in seq.blocks:
        pool.refcount[page] -= 1
        if pool.refcount[page] > 0:
            continue
        if pool.refcount[page] < 0:
            raise RuntimeError(f"page {page} refcount went negative")
        cache = pool.cache
        if cache is not None and cache.holds(page):
            # A full block someone might reuse stays out of the free list until it is the oldest.
            cache.mark_evictable(page)
            continue
        recycled.append(page)
    seq.blocks = []
    if recycled:
        pool.free.extend(recycled)
        pool.free.sort()


def take_page(pool: PagePool) -> int:
    # Free pages first. A cached prefix is only overwritten once nothing free is left.
    if not pool.free:
        cache = pool.cache
        evicted = None if cache is None else cache.evict()
        if evicted is None:
            raise RuntimeError("page pool has no free page and no evictable prefix block")
        if pool.refcount[evicted] != 0:
            raise RuntimeError(f"evicted page {evicted} still has refcount {pool.refcount[evicted]}")
        pool.free.append(evicted)
    page = pool.free.pop(0)
    if pool.refcount[page] != 0:
        raise RuntimeError(f"free page {page} has refcount {pool.refcount[page]}")
    pool.refcount[page] = 1
    return page


def pin_page(pool: PagePool, page: int) -> None:
    cache = pool.cache
    if cache is None or not cache.holds(page):
        raise RuntimeError(f"page {page} is not a cached prefix block")
    if pool.refcount[page] == 0 and page in pool.free:
        raise RuntimeError(f"page {page} is on the free list")
    pool.refcount[page] += 1
    cache.unmark(page)


def available_pages(pool: PagePool) -> int:
    # Pages nobody is reading can be taken. Cached ones are included because allocation evicts them.
    cached = 0 if pool.cache is None else pool.cache.evictable_count()
    return len(pool.free) + cached


def page_write(seq: Sequence, layer: int, k_new, v_new, start: int) -> None:
    pool = seq.pool
    bs = pool.block_size
    for t in range(k_new.shape[2]):
        pos = start + t
        page = seq.blocks[pos // bs]
        slot = pos % bs
        pool.k[layer, page, :, slot] = k_new[0, :, t]
        pool.v[layer, page, :, slot] = v_new[0, :, t]


def page_gather(seq: Sequence, layer: int, n: int):
    pool = seq.pool
    bs = pool.block_size
    blocks = seq.blocks
    k_pages = pool.k[layer, blocks]
    v_pages = pool.v[layer, blocks]
    k = k_pages.transpose(0, 1).reshape(k_pages.shape[1], len(blocks) * bs, k_pages.shape[-1])[:, :n].unsqueeze(0)
    v = v_pages.transpose(0, 1).reshape(v_pages.shape[1], len(blocks) * bs, v_pages.shape[-1])[:, :n].unsqueeze(0)
    return k, v


def forward_paged(weights: Weights, cfg: ModelConfig, rope, ids: torch.Tensor, seq: Sequence, *, graphs=None):
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError(f"paged forward takes one sequence [1, time], got {tuple(ids.shape)}")
    _same_device(weights, seq.pool)
    device = weights.embed.device
    ids = ids.to(device=device)
    past = seq.seq_len
    new_ids = ids[:, past:]
    if new_ids.shape[1] == 0:
        raise ValueError("paged forward was given no new tokens")
    T_new = new_ids.shape[1]
    # Off unless the caller passes one, so the token checks stay on the eager path below.
    if graphs is not None and _use_decode_kernel(weights, cfg, seq.pool, T_new):
        ran = graphs.try_decode(weights, cfg, rope, new_ids, [seq])
        if ran is not None:
            logits, _seqs = ran
            return logits, seq
    x = weights.embed[new_ids]
    pos = torch.arange(past, past + T_new, device=device).unsqueeze(0)
    cos, sin = rope.gather(pos)
    if _use_decode_kernel(weights, cfg, seq.pool, T_new):
        x = _decode_layers(weights, cfg, x, [seq], [past], cos, sin)
    elif past == 0:
        x = _prefill_layers(weights, cfg, x, [seq], cos, sin)
    else:
        x = _gather_one(weights, cfg, x, seq, past, T_new, cos, sin)
    return _logits(weights, cfg, x), seq


def gather_padded(seqs: list[Sequence], layer: int, lens: list[int], width: int):
    ks = []
    vs = []
    for seq, n in zip(seqs, lens):
        k, v = page_gather(seq, layer, n)
        n_pad = width - n
        if n_pad:
            k = torch.cat([k.new_zeros(1, k.shape[1], n_pad, k.shape[3]), k], dim=2)
            v = torch.cat([v.new_zeros(1, v.shape[1], n_pad, v.shape[3]), v], dim=2)
        ks.append(k)
        vs.append(v)
    return torch.cat(ks, dim=0), torch.cat(vs, dim=0)


def forward_paged_batch(
    weights: Weights, cfg: ModelConfig, rope, ids: torch.Tensor, seqs: list[Sequence], *, graphs=None,
):
    # One new token on CUDA reads each sequence's own length.
    # A fresh batch uses SDPA. A batch that already has pages still left-pads.
    if ids.ndim != 2:
        raise ValueError(f"ids must have shape [batch, time], got {tuple(ids.shape)}")
    if len(seqs) != ids.shape[0]:
        raise ValueError(f"{len(seqs)} sequences for a batch of {ids.shape[0]}")
    if not seqs:
        raise ValueError("paged batch forward needs at least one sequence")
    for seq in seqs:
        _same_device(weights, seq.pool)
    device = weights.embed.device
    ids = ids.to(device=device)
    lens = [seq.seq_len for seq in seqs]
    T_new = ids.shape[1]
    if graphs is not None and _use_decode_kernel(weights, cfg, seqs[0].pool, T_new):
        ran = graphs.try_decode(weights, cfg, rope, ids, seqs)
        if ran is not None:
            return ran
    x = weights.embed[ids]
    pos = torch.tensor(lens, device=device).unsqueeze(1) + torch.arange(T_new, device=device)
    cos, sin = rope.gather(pos)
    if _use_decode_kernel(weights, cfg, seqs[0].pool, T_new):
        x = _decode_layers(weights, cfg, x, seqs, lens, cos, sin)
    elif all(n == 0 for n in lens):
        x = _prefill_layers(weights, cfg, x, seqs, cos, sin)
    else:
        x = _gather_batch(weights, cfg, x, seqs, lens, T_new, cos, sin)
    return _logits(weights, cfg, x), seqs


def _use_decode_kernel(weights: Weights, cfg: ModelConfig, pool: PagePool, t_new: int) -> bool:
    # One query per sequence. A longer span is prefill and does not use this kernel.
    return t_new == 1 and decode_supported(weights.embed.device, weights.embed.dtype, cfg.head_dim, pool.block_size)


def _decode_layers(weights, cfg, x, seqs, lens, cos, sin):
    # The attention kernel only reads the pool, so the new token is scattered in before the launch.
    _pool, tables, seq_lens, slots = _prepare_decode(seqs, lens, x.device)
    x = _decode_math(weights, cfg, x, seqs, lens, tables, seq_lens, slots, cos, sin)
    _advance(seqs, lens)
    return x


def _prepare_decode(seqs, lens, device, *, check=True):
    pool = seqs[0].pool
    for seq in seqs[1:]:
        if seq.pool is not pool:
            raise ValueError("a decode batch must share one page pool")
    for seq, n in zip(seqs, lens):
        reserve_pages(seq, n + 1)
    lengths = [n + 1 for n in lens]
    tables = _block_tables(seqs, device)
    width = tables.shape[1]
    for n in lengths:
        if n < 1 or (n + pool.block_size - 1) // pool.block_size > width:
            raise ValueError("each seq_len must be >= 1 and fit in block_tables")
    seq_lens = _stage(torch.tensor(lengths, dtype=torch.int32), device)
    # The graph path skips the slot readback. Its replay cannot stop for a host check.
    if check:
        slots = _slots_for(pool, tables, lens, 1)
    elif cache_supported(device, pool.k.dtype, pool.k.shape[-1], pool.block_size):
        slots = _slots_from_table(tables, lens, 1, pool.block_size)
    else:
        slots = None
    return pool, tables, seq_lens, slots


def _decode_math(weights, cfg, x, seqs, lens, tables, seq_lens, slots, cos, sin):
    pool = seqs[0].pool
    for i, layer in enumerate(weights.layers):
        h = rms_norm(x, layer.input_norm, cfg.rms_norm_eps)
        q, k_new, v_new = project_qkv(h, layer, cfg, cos, sin)
        _write_new(seqs, i, k_new, v_new, lens, slots)
        attn = paged_attn_decode(q[:, :, 0], pool.k[i], pool.v[i], tables, seq_lens, check=False)
        x = x + merge_heads(attn.unsqueeze(2), layer)
        h = rms_norm(x, layer.post_norm, cfg.rms_norm_eps)
        x = x + mlp(h, layer)
    return x


def _advance(seqs, lens) -> None:
    for seq, n in zip(seqs, lens):
        seq.seq_len = n + 1


def _prefill_layers(weights, cfg, x, seqs, cos, sin):
    # The new K and V are one contiguous span per row. SDPA on that span cannot see a prefix.
    pool = seqs[0].pool
    for seq in seqs[1:]:
        if seq.pool is not pool:
            raise ValueError("a prefill batch must share one page pool")
    t_new = x.shape[1]
    for seq in seqs:
        if seq.seq_len != 0:
            raise ValueError("a prefix has to be gathered; SDPA prefill only sees the new tokens")
        reserve_pages(seq, t_new)
    starts = [0] * len(seqs)
    slots = _slots_for(pool, _block_tables(seqs, x.device), starts, t_new)
    for i, layer in enumerate(weights.layers):
        h = rms_norm(x, layer.input_norm, cfg.rms_norm_eps)
        delta, k_new, v_new = attention_prefill(h, layer, cfg, cos, sin)
        _write_new(seqs, i, k_new, v_new, starts, slots)
        x = x + delta
        h = rms_norm(x, layer.post_norm, cfg.rms_norm_eps)
        x = x + mlp(h, layer)
    for seq in seqs:
        seq.seq_len = t_new
    return x


def _gather_one(weights, cfg, x, seq, past, t_new, cos, sin):
    reserve_pages(seq, past + t_new)
    slots = _slots_for(seq.pool, _block_tables([seq], x.device), [past], t_new)
    for i, layer in enumerate(weights.layers):
        h = rms_norm(x, layer.input_norm, cfg.rms_norm_eps)
        past_k, past_v = (None, None) if past == 0 else page_gather(seq, i, past)
        delta, k_new, v_new = attention_cached(h, layer, cfg, cos, sin, past_k, past_v)
        _write_new([seq], i, k_new, v_new, [past], slots)
        x = x + delta
        h = rms_norm(x, layer.post_norm, cfg.rms_norm_eps)
        x = x + mlp(h, layer)
    seq.seq_len = past + t_new
    return x


def _gather_batch(weights, cfg, x, seqs, lens, t_new, cos, sin):
    width = max(lens)
    pad_lens = torch.tensor([width - n for n in lens], device=x.device)
    for seq, n in zip(seqs, lens):
        reserve_pages(seq, n + t_new)
    slots = _slots_for(seqs[0].pool, _block_tables(seqs, x.device), lens, t_new)
    for i, layer in enumerate(weights.layers):
        h = rms_norm(x, layer.input_norm, cfg.rms_norm_eps)
        past_k, past_v = gather_padded(seqs, i, lens, width)
        delta, k_new, v_new = attention_cached(h, layer, cfg, cos, sin, past_k, past_v, pad_lens)
        _write_new(seqs, i, k_new, v_new, lens, slots)
        x = x + delta
        h = rms_norm(x, layer.post_norm, cfg.rms_norm_eps)
        x = x + mlp(h, layer)
    for seq, n in zip(seqs, lens):
        seq.seq_len = n + t_new
    return x


def _write_new(seqs, layer, k_new, v_new, starts, slots) -> None:
    if slots is None:
        for row, seq in enumerate(seqs):
            page_write(seq, layer, k_new[row:row + 1], v_new[row:row + 1], starts[row])
        return
    pool = seqs[0].pool
    reshape_and_cache(k_new, v_new, pool.k[layer], pool.v[layer], slots, check=False)


def _slots_for(pool, tables, starts, t_new):
    if not cache_supported(tables.device, pool.k.dtype, pool.k.shape[-1], pool.block_size):
        return None
    slots = _slots_from_table(tables, starts, t_new, pool.block_size)
    n_slots = pool.k.shape[1] * pool.block_size
    if bool(((slots < 0) | (slots >= n_slots)).any()):
        raise ValueError("each new token must land in a reserved page")
    return slots


def _slots_from_table(tables, starts, t_new, block_size):
    # The kernel splits this int32 back into a page and an offset by dividing by block_size.
    pos = _stage(torch.tensor(starts, dtype=torch.int64), tables.device)[:, None]
    pos = pos + torch.arange(t_new, device=pos.device)
    page = tables.gather(1, pos // block_size)
    return page * block_size + (pos % block_size).to(dtype=torch.int32)


def _block_tables(seqs, device):
    width = max(len(seq.blocks) for seq in seqs)
    table = torch.zeros(len(seqs), width, dtype=torch.int32)
    for row, seq in enumerate(seqs):
        if seq.blocks:
            table[row, :len(seq.blocks)] = torch.tensor(seq.blocks, dtype=torch.int32)
    return _stage(table, device)


def _stage(cpu, device):
    # One upload. A row-by-row device copy would sync once per sequence.
    device = torch.device(device)
    if device.type != "cuda":
        return cpu.to(device)
    return cpu.pin_memory().to(device, non_blocking=True)


def _logits(weights, cfg, x):
    x = rms_norm(x, weights.norm, cfg.rms_norm_eps)
    return x[:, -1] @ weights.lm_head.T


def _same_device(weights: Weights, pool: PagePool) -> None:
    got = pool.k
    if got.device != weights.embed.device or got.dtype != weights.embed.dtype:
        raise ValueError(
            f"page pool is {got.dtype} on {got.device}, weights are {weights.embed.dtype} on {weights.embed.device}"
        )
