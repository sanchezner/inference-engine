from dataclasses import dataclass, field

import torch

from engine.config import ModelConfig
from engine.kernels.triton.paged_attn import decode_supported, paged_attn_decode
from engine.kernels.triton.reshape_cache import cache_supported, reshape_and_cache
from engine.model import attention_cached, merge_heads, mlp, project_qkv, rms_norm
from engine.weights import Weights


# KV lives in pages so a finished sequence can return storage without moving
# anyone else. k and v are [layer, page, kv_head, slot, dim].
# CUDA writes the new K/V with one scatter, then a one-token step reads the pages.
# Prefill attention and CPU still copy pages into a dense tensor (page_gather).
@dataclass
class PagePool:
    k: torch.Tensor
    v: torch.Tensor
    free: list[int]
    block_size: int

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
        seq.blocks.append(pool.free.pop(0))


def release_pages(seq: Sequence) -> None:
    # Sort so the next pop is the lowest id. The block-table checks rely on that,
    # including the out-of-order reuse [4, 5, 1].
    pool = seq.pool
    pool.free.extend(seq.blocks)
    pool.free.sort()
    seq.blocks = []


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


def forward_paged(weights: Weights, cfg: ModelConfig, rope, ids: torch.Tensor, seq: Sequence):
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError(f"paged forward takes one sequence [1, time], got {tuple(ids.shape)}")
    _same_device(weights, seq.pool)
    device = weights.embed.device
    ids = ids.to(device=device)
    past = seq.seq_len
    new_ids = ids[:, past:]
    if new_ids.shape[1] == 0:
        raise ValueError("paged forward was given no new tokens")
    x = weights.embed[new_ids]
    T_new = new_ids.shape[1]
    pos = torch.arange(past, past + T_new, device=device).unsqueeze(0)
    cos, sin = rope.gather(pos)
    if _use_decode_kernel(weights, cfg, seq.pool, T_new):
        x = _decode_layers(weights, cfg, x, [seq], [past], cos, sin)
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


def forward_paged_batch(weights: Weights, cfg: ModelConfig, rope, ids: torch.Tensor, seqs: list[Sequence]):
    # One new token on CUDA reads each sequence's own length.
    # More than one new token still left-pads so one dense mask can run.
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
    x = weights.embed[ids]
    pos = torch.tensor(lens, device=device).unsqueeze(1) + torch.arange(T_new, device=device)
    cos, sin = rope.gather(pos)
    if _use_decode_kernel(weights, cfg, seqs[0].pool, T_new):
        x = _decode_layers(weights, cfg, x, seqs, lens, cos, sin)
    else:
        x = _gather_batch(weights, cfg, x, seqs, lens, T_new, cos, sin)
    return _logits(weights, cfg, x), seqs


def _use_decode_kernel(weights: Weights, cfg: ModelConfig, pool: PagePool, t_new: int) -> bool:
    # One query per sequence. Prefill has many queries and stays on the dense mask.
    return t_new == 1 and decode_supported(weights.embed.device, weights.embed.dtype, cfg.head_dim, pool.block_size)


def _decode_layers(weights, cfg, x, seqs, lens, cos, sin):
    # The attention kernel only reads the pool, so the new token is scattered in before the launch.
    pool = seqs[0].pool
    for seq in seqs[1:]:
        if seq.pool is not pool:
            raise ValueError("a decode batch must share one page pool")
    for seq, n in zip(seqs, lens):
        reserve_pages(seq, n + 1)
    lengths = [n + 1 for n in lens]
    tables = _block_tables(seqs, x.device)
    width = tables.shape[1]
    for n in lengths:
        if n < 1 or (n + pool.block_size - 1) // pool.block_size > width:
            raise ValueError("each seq_len must be >= 1 and fit in block_tables")
    seq_lens = torch.tensor(lengths, device=x.device, dtype=torch.int32)
    # Pages stay put for the whole step, so one slot map serves every layer.
    slots = _slots_for(pool, tables, lens, 1)
    for i, layer in enumerate(weights.layers):
        h = rms_norm(x, layer.input_norm, cfg.rms_norm_eps)
        q, k_new, v_new = project_qkv(h, layer, cfg, cos, sin)
        _write_new(seqs, i, k_new, v_new, lens, slots)
        attn = paged_attn_decode(q[:, :, 0], pool.k[i], pool.v[i], tables, seq_lens, check=False)
        x = x + merge_heads(attn.unsqueeze(2), layer)
        h = rms_norm(x, layer.post_norm, cfg.rms_norm_eps)
        x = x + mlp(h, layer)
    for seq, n in zip(seqs, lens):
        seq.seq_len = n + 1
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
    pos = torch.tensor(starts, device=tables.device, dtype=torch.int64)[:, None]
    pos = pos + torch.arange(t_new, device=tables.device)
    page = tables.gather(1, pos // block_size)
    return page * block_size + (pos % block_size).to(dtype=torch.int32)


def _block_tables(seqs, device):
    width = max(len(seq.blocks) for seq in seqs)
    table = torch.zeros(len(seqs), width, device=device, dtype=torch.int32)
    for row, seq in enumerate(seqs):
        if seq.blocks:
            table[row, :len(seq.blocks)] = torch.tensor(seq.blocks, dtype=torch.int32, device=device)
    return table


def _logits(weights, cfg, x):
    x = rms_norm(x, weights.norm, cfg.rms_norm_eps)
    return x[:, -1] @ weights.lm_head.T


def _same_device(weights: Weights, pool: PagePool) -> None:
    got = pool.k
    if got.device != weights.embed.device or got.dtype != weights.embed.dtype:
        raise ValueError(
            f"page pool is {got.dtype} on {got.device}, weights are {weights.embed.dtype} on {weights.embed.device}"
        )
