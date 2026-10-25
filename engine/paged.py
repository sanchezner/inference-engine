from dataclasses import dataclass, field

import torch

from engine.config import ModelConfig
from engine.model import attention_cached, mlp, rms_norm
from engine.weights import Weights


# KV lives in pages so a finished sequence can return storage without moving
# anyone else. k and v are [layer, page, kv_head, slot, dim]. Reading them
# still copies a sequence's pages into a dense tensor (page_gather); the
# kernel that reads the block table directly is M3, not this file.
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


def reserve_pages(seq: Sequence, total_len: int) -> None:
    pool = seq.pool
    need = (total_len + pool.block_size - 1) // pool.block_size
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
    reserve_pages(seq, past + T_new)

    for i, layer in enumerate(weights.layers):
        h = rms_norm(x, layer.input_norm, cfg.rms_norm_eps)
        past_k, past_v = (None, None) if past == 0 else page_gather(seq, i, past)
        delta, k_new, v_new = attention_cached(h, layer, cfg, cos, sin, past_k, past_v)
        page_write(seq, i, k_new, v_new, past)
        x = x + delta
        h = rms_norm(x, layer.post_norm, cfg.rms_norm_eps)
        x = x + mlp(h, layer)

    seq.seq_len = past + T_new
    x = rms_norm(x, weights.norm, cfg.rms_norm_eps)
    return x[:, -1] @ weights.lm_head.T, seq


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
    # One new-token row per live sequence. Past keys are left-padded to the
    # longest sequence so a single dense attention can run; M3 drops that pad.
    if ids.ndim != 2:
        raise ValueError(f"ids must have shape [batch, time], got {tuple(ids.shape)}")
    if len(seqs) != ids.shape[0]:
        raise ValueError(f"{len(seqs)} sequences for a batch of {ids.shape[0]}")
    for seq in seqs:
        _same_device(weights, seq.pool)
    device = weights.embed.device
    ids = ids.to(device=device)
    lens = [seq.seq_len for seq in seqs]
    width = max(lens)
    T_new = ids.shape[1]
    pad_lens = torch.tensor([width - n for n in lens], device=device)
    x = weights.embed[ids]
    pos = torch.tensor(lens, device=device).unsqueeze(1) + torch.arange(T_new, device=device)
    cos, sin = rope.gather(pos)

    for seq, n in zip(seqs, lens):
        reserve_pages(seq, n + T_new)

    for i, layer in enumerate(weights.layers):
        h = rms_norm(x, layer.input_norm, cfg.rms_norm_eps)
        past_k, past_v = gather_padded(seqs, i, lens, width)
        delta, k_new, v_new = attention_cached(
            h, layer, cfg, cos, sin, past_k, past_v, pad_lens
        )
        for row, seq in enumerate(seqs):
            page_write(seq, i, k_new[row:row + 1], v_new[row:row + 1], lens[row])
        x = x + delta
        h = rms_norm(x, layer.post_norm, cfg.rms_norm_eps)
        x = x + mlp(h, layer)

    for seq, n in zip(seqs, lens):
        seq.seq_len = n + T_new

    x = rms_norm(x, weights.norm, cfg.rms_norm_eps)
    return x[:, -1] @ weights.lm_head.T, seqs


def _same_device(weights: Weights, pool: PagePool) -> None:
    got = pool.k
    if got.device != weights.embed.device or got.dtype != weights.embed.dtype:
        raise ValueError(
            f"page pool is {got.dtype} on {got.device}, weights are {weights.embed.dtype} on {weights.embed.device}"
        )
