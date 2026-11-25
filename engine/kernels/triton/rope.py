import torch
import triton
import triton.language as tl

from engine.kernels.triton.paged_attn import _HEAD_DIMS


# The bench turns this off to time the eager rotation against the kernel.
use_fused = True

_DTYPES = (torch.float32, torch.bfloat16)


@triton.jit
def _rotate_head(
    in_ptr, out_ptr, cos_ptr, sin_ptr,
    stride_b, stride_h, stride_t,
    stride_ob, stride_oh, stride_ot,
    stride_cb, stride_ct, stride_sb, stride_st,
    row, time, head,
    HALF: tl.constexpr,
):
    offs = tl.arange(0, HALF)
    base = in_ptr + row * stride_b + head * stride_h + time * stride_t
    # Both halves are loaded. The tables repeat them, and a general cos does not.
    x0 = tl.load(base + offs).to(tl.float32)
    x1 = tl.load(base + HALF + offs).to(tl.float32)
    cbase = cos_ptr + row * stride_cb + time * stride_ct
    sbase = sin_ptr + row * stride_sb + time * stride_st
    c0 = tl.load(cbase + offs).to(tl.float32)
    c1 = tl.load(cbase + HALF + offs).to(tl.float32)
    s0 = tl.load(sbase + offs).to(tl.float32)
    s1 = tl.load(sbase + HALF + offs).to(tl.float32)
    y0 = x0 * c0 - x1 * s0
    y1 = x1 * c1 + x0 * s1
    obase = out_ptr + row * stride_ob + head * stride_oh + time * stride_ot
    tl.store(obase + offs, y0)
    tl.store(obase + HALF + offs, y1)


@triton.jit
def _rope_qk(
    q_ptr, k_ptr, cos_ptr, sin_ptr, oq_ptr, ok_ptr,
    stride_qb, stride_qh, stride_qt,
    stride_kb, stride_kh, stride_kt,
    stride_oqb, stride_oqh, stride_oqt,
    stride_okb, stride_okh, stride_okt,
    stride_cb, stride_ct, stride_sb, stride_st,
    n_kv_heads,
    HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    time = tl.program_id(1)
    head = tl.program_id(2)
    _rotate_head(
        q_ptr, oq_ptr, cos_ptr, sin_ptr,
        stride_qb, stride_qh, stride_qt,
        stride_oqb, stride_oqh, stride_oqt,
        stride_cb, stride_ct, stride_sb, stride_st,
        row, time, head,
        HALF=HEAD_DIM // 2,
    )
    if head < n_kv_heads:
        _rotate_head(
            k_ptr, ok_ptr, cos_ptr, sin_ptr,
            stride_kb, stride_kh, stride_kt,
            stride_okb, stride_okh, stride_okt,
            stride_cb, stride_ct, stride_sb, stride_st,
            row, time, head,
            HALF=HEAD_DIM // 2,
        )


def rope_supported(q, k, cos, sin) -> bool:
    if not use_fused:
        return False
    return _problem(q, k, cos, sin) is None


def fused_rope(q, k, cos, sin):
    problem = _problem(q, k, cos, sin)
    if problem is not None:
        raise ValueError(problem)
    if q.shape[0] == 0 or q.shape[2] == 0:
        return q, k
    out_q = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    out_k = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    _rope_qk[(q.shape[0], q.shape[2], q.shape[1])](
        q, k, cos, sin, out_q, out_k,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        out_q.stride(0), out_q.stride(1), out_q.stride(2),
        out_k.stride(0), out_k.stride(1), out_k.stride(2),
        0 if cos.shape[0] == 1 else cos.stride(0),
        cos.stride(1),
        0 if sin.shape[0] == 1 else sin.stride(0),
        sin.stride(1),
        k.shape[1],
        HEAD_DIM=q.shape[-1],
    )
    return out_q, out_k


def _problem(q, k, cos, sin):
    if q.ndim != 4 or k.ndim != 4:
        return f"q and k must have shape [batch, heads, time, dim], got {tuple(q.shape)} and {tuple(k.shape)}"
    if q.device.type != "cuda" or k.device != q.device or cos.device != q.device or sin.device != q.device:
        return f"rope runs on one CUDA device, got q {q.device}, k {k.device}, cos {cos.device}, sin {sin.device}"
    if q.dtype not in _DTYPES or k.dtype != q.dtype:
        return f"q and k must share float32 or bfloat16, got {q.dtype} and {k.dtype}"
    if cos.ndim != 3 or sin.ndim != 3:
        return f"cos and sin must have shape [batch, time, dim], got {tuple(cos.shape)} and {tuple(sin.shape)}"
    if cos.dtype != sin.dtype or (cos.dtype != torch.float32 and cos.dtype != q.dtype):
        return f"cos and sin must be float32 or the activation dtype, got {cos.dtype} and {sin.dtype}"
    batch, n_q, time, head_dim = q.shape
    n_kv = k.shape[1]
    if k.shape[0] != batch or k.shape[2] != time or k.shape[3] != head_dim:
        return f"k must share q's batch, time, and dim, got q {tuple(q.shape)} and k {tuple(k.shape)}"
    if head_dim not in _HEAD_DIMS:
        return f"head_dim must be one of {_HEAD_DIMS}, got {head_dim}"
    if n_q < 1 or n_kv < 1 or n_kv > n_q:
        return f"kv_heads ({n_kv}) must sit inside a positive query head count ({n_q})"
    if cos.shape != sin.shape or cos.shape[1] != time or cos.shape[2] != head_dim or cos.shape[0] not in (1, batch):
        return (
            f"cos and sin must be [{batch} or 1, {time}, {head_dim}], "
            f"got {tuple(cos.shape)} and {tuple(sin.shape)}"
        )
    if not _packed(q) or not _packed(k) or not _packed(cos) or not _packed(sin):
        return "the last dimension must be contiguous and strides must be non-negative"
    return None


def _packed(t) -> bool:
    return t.stride(-1) == 1 and all(s >= 0 for s in t.stride())
