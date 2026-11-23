from functools import lru_cache

import torch
import triton
import triton.language as tl


# Each block size is its own compiled tile. The M3 sweep is these four.
_BLOCK_SIZES = (4, 8, 16, 32)
# 64 is SmolLM2. 128 is the other Llama head size, so the tile is not a one-model constant.
_HEAD_DIMS = (64, 128)
# Split-KV policy, tuned on the 3070's 46 SMs. Below FULL_WAVES programs per SM the grid leaves SMs idle.
_FULL_WAVES = 2
_TARGET_WAVES = 4
# A split shorter than this spends more on its partial write and the merge than it saves.
_MIN_SPLIT_TOKENS = 128


def decode_supported(device, dtype, head_dim, block_size) -> bool:
    return (
        device.type == "cuda"
        and dtype in (torch.float32, torch.bfloat16)
        and head_dim in _HEAD_DIMS
        and block_size in _BLOCK_SIZES
    )


@triton.jit
def _paged_decode(
    q_ptr, k_ptr, v_ptr, o_ptr,
    bt_ptr, sl_ptr,
    m_ptr, l_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kp, stride_kh, stride_ks, stride_kd,
    stride_vp, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_mb, stride_mh, stride_ms,
    stride_bt, stride_bt_col, stride_sl,
    scale, n_rep, split_tokens,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    # Heads in a group share one KV head, so the key is never copied out and repeated.
    kv_h = pid_h // n_rep
    seq_len = tl.load(sl_ptr + pid_b * stride_sl).to(tl.int32)
    # split_tokens is a whole number of pages, so a split never starts mid-page.
    lo = pid_s * split_tokens
    hi = tl.minimum(seq_len, lo + split_tokens)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_d * stride_qd).to(tl.float32)

    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    # -inf, not the pad stand-in: the first block's rescale must be a hard zero.
    m_i = tl.full([1], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)

    # A split past the end of a short row runs zero iterations and leaves m=-inf, l=0, acc=0.
    for start in tl.range(lo, hi, BLOCK_N):
        page = tl.load(bt_ptr + pid_b * stride_bt + (start // BLOCK_N) * stride_bt_col).to(tl.int64)
        slots = tl.arange(0, BLOCK_N)
        token_idx = start + slots
        mask = token_idx < seq_len
        k = tl.load(
            k_ptr + page * stride_kp + kv_h * stride_kh + slots[:, None] * stride_ks + offs_d[None, :] * stride_kd,
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + page * stride_vp + kv_h * stride_vh + slots[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=mask[:, None],
            other=0.0,
        ).to(tl.float32)
        score = tl.sum(q[None, :] * k, axis=1) * scale
        # Pad slots must lose the max. A scaled dot product never reaches -1e20.
        score = tl.where(mask, score, -1.0e20)
        m_ij = tl.max(score, axis=0, keep_dims=True)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(score - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0, keep_dims=True)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    o = o_ptr + pid_b * stride_ob + pid_h * stride_oh + pid_s * stride_os + offs_d * stride_od
    if SPLIT:
        # Unnormalized. The merge rescales each split by exp(m_s - max m) before it divides.
        tl.store(o, acc)
        one = tl.arange(0, 1)
        tl.store(m_ptr + pid_b * stride_mb + pid_h * stride_mh + pid_s * stride_ms + one, m_i)
        tl.store(l_ptr + pid_b * stride_mb + pid_h * stride_mh + pid_s * stride_ms + one, l_i)
    else:
        tl.store(o, acc / l_i)


@triton.jit
def _merge_splits(
    acc_ptr, m_ptr, l_ptr, o_ptr,
    stride_ab, stride_ah, stride_as, stride_ad,
    stride_mb, stride_mh, stride_ms,
    stride_ob, stride_oh, stride_od,
    n_splits,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_s = tl.arange(0, SPLITS)
    offs_d = tl.arange(0, HEAD_DIM)
    live = offs_s < n_splits
    stat = pid_b * stride_mb + pid_h * stride_mh + offs_s * stride_ms
    m = tl.load(m_ptr + stat, mask=live, other=float("-inf"))
    l = tl.load(l_ptr + stat, mask=live, other=0.0)
    # Split 0 always holds a token (seq_len >= 1), so the max is finite and empty splits weigh exp(-inf) = 0.
    w = tl.exp(m - tl.max(m, axis=0))
    acc = tl.load(
        acc_ptr + pid_b * stride_ab + pid_h * stride_ah + offs_s[:, None] * stride_as + offs_d[None, :] * stride_ad,
        mask=live[:, None],
        other=0.0,
    )
    out = tl.sum(w[:, None] * acc, axis=0) / tl.sum(w * l, axis=0)
    tl.store(o_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_d * stride_od, out)


@lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def choose_splits(batch: int, n_heads: int, max_tokens: int, n_sm: int) -> int:
    # max_tokens is the table width in tokens, known on the host, so choosing never syncs.
    programs = batch * n_heads
    if programs >= _FULL_WAVES * n_sm:
        return 1
    want = -(-_TARGET_WAVES * n_sm // programs)
    return max(1, min(want, max_tokens // _MIN_SPLIT_TOKENS))


def paged_attn_decode(q, k_pool, v_pool, block_tables, seq_lens, *, num_splits=None, check=True):
    # One query per sequence. The new token's K and V are already in the pages.
    # num_splits=None picks from the grid. A fixed count over the same table width is batch invariant.
    if q.ndim != 3:
        raise ValueError(f"q must have shape [batch, heads, dim], got {tuple(q.shape)}")
    if k_pool.ndim != 4 or v_pool.shape != k_pool.shape:
        raise ValueError(
            f"k and v pools must share shape [pages, kv_heads, block, dim], got {tuple(k_pool.shape)} and {tuple(v_pool.shape)}"
        )
    batch, n_heads, head_dim = q.shape
    n_kv = k_pool.shape[1]
    block_size = k_pool.shape[2]
    if k_pool.shape[3] != head_dim:
        raise ValueError(f"q dim {head_dim} does not match pool dim {k_pool.shape[3]}")
    if head_dim not in _HEAD_DIMS:
        raise ValueError(f"head_dim must be one of {_HEAD_DIMS}, got {head_dim}")
    if block_size not in _BLOCK_SIZES:
        raise ValueError(f"block_size must be one of {_BLOCK_SIZES}, got {block_size}")
    if n_heads < 1 or n_kv < 1 or n_heads % n_kv != 0:
        raise ValueError(f"n_heads ({n_heads}) must be a positive multiple of n_kv_heads ({n_kv})")
    if block_tables.ndim != 2 or block_tables.shape[0] != batch:
        raise ValueError(
            f"block_tables must have shape [{batch}, blocks], got {tuple(block_tables.shape)}"
        )
    if seq_lens.shape != (batch,):
        raise ValueError(f"seq_lens must have shape [{batch}], got {tuple(seq_lens.shape)}")
    if block_tables.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        raise ValueError(
            f"block_tables and seq_lens must be int32, got {block_tables.dtype} and {seq_lens.dtype}"
        )
    if q.dtype not in (torch.float32, torch.bfloat16) or k_pool.dtype != q.dtype or v_pool.dtype != q.dtype:
        raise ValueError(f"q, k, and v must share float32 or bfloat16, got {q.dtype}, {k_pool.dtype}, {v_pool.dtype}")
    device = q.device
    if device.type != "cuda":
        raise ValueError(f"paged attention decode runs on CUDA, got {device}")
    for name, tensor in (("k_pool", k_pool), ("v_pool", v_pool), ("block_tables", block_tables), ("seq_lens", seq_lens)):
        if tensor.device != device:
            raise ValueError(f"{name} is on {tensor.device}, q is on {device}")
    if num_splits is not None and (isinstance(num_splits, bool) or not isinstance(num_splits, int) or num_splits < 1):
        raise ValueError(f"num_splits must be a positive int or None, got {num_splits!r}")
    if batch == 0:
        return q.new_empty(q.shape)
    # One sync. A short table would otherwise read off the end of block_tables.
    # check=False skips it. The forward already checked, and this runs once per layer.
    if check:
        fits = ((seq_lens + block_size - 1) // block_size) <= block_tables.shape[1]
        if bool(((seq_lens < 1) | ~fits).any()):
            raise ValueError("each seq_len must be >= 1 and fit in block_tables")

    width = block_tables.shape[1]
    if num_splits is None:
        num_splits = choose_splits(batch, n_heads, width * block_size, _sm_count(device.index or 0))
    pages_per_split = max(1, -(-width // min(num_splits, max(width, 1))))
    splits = max(1, -(-width // pages_per_split))
    split = splits > 1
    out = torch.empty_like(q)
    if split:
        part = torch.empty(batch, n_heads, splits, head_dim, device=device, dtype=torch.float32)
        m = torch.empty(batch, n_heads, splits, device=device, dtype=torch.float32)
        l = torch.empty_like(m)
        dst, dst_strides, stat_strides = part, part.stride(), m.stride()
    else:
        m = l = out
        dst = out
        dst_strides = (out.stride(0), out.stride(1), 0, out.stride(2))
        stat_strides = (0, 0, 0)
    _paged_decode[(batch, n_heads, splits)](
        q, k_pool, v_pool, dst,
        block_tables, seq_lens,
        m, l,
        q.stride(0), q.stride(1), q.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2), v_pool.stride(3),
        *dst_strides,
        *stat_strides,
        block_tables.stride(0), block_tables.stride(1), seq_lens.stride(0),
        head_dim ** -0.5,
        n_heads // n_kv,
        pages_per_split * block_size,
        HEAD_DIM=head_dim,
        BLOCK_N=block_size,
        SPLIT=split,
    )
    if split:
        _merge_splits[(batch, n_heads)](
            part, m, l, out,
            *part.stride(),
            *m.stride(),
            out.stride(0), out.stride(1), out.stride(2),
            splits,
            HEAD_DIM=head_dim,
            SPLITS=triton.next_power_of_2(splits),
        )
    return out
