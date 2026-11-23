import torch
import triton
import triton.language as tl

from engine.kernels.triton.paged_attn import _BLOCK_SIZES, _HEAD_DIMS


def cache_supported(device, dtype, head_dim, block_size) -> bool:
    return (
        device.type == "cuda"
        and dtype in (torch.float32, torch.bfloat16)
        and head_dim in _HEAD_DIMS
        and block_size in _BLOCK_SIZES
    )


@triton.jit
def _scatter_kv(
    k_ptr, v_ptr, kc_ptr, vc_ptr, slot_ptr,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_kcp, stride_kch, stride_kcs, stride_kcd,
    stride_vcp, stride_vch, stride_vcs, stride_vcd,
    stride_sb, stride_st,
    n_time,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    kv_h = tl.program_id(1)
    row = token // n_time
    col = token - row * n_time
    # Pointer offsets are int64. An int32 page walks off the pool once the cache is large.
    slot = tl.load(slot_ptr + row * stride_sb + col * stride_st).to(tl.int64)
    page = slot // BLOCK_SIZE
    offset = slot % BLOCK_SIZE
    offs = tl.arange(0, HEAD_DIM)
    k = tl.load(k_ptr + row * stride_kb + kv_h * stride_kh + col * stride_kt + offs * stride_kd)
    v = tl.load(v_ptr + row * stride_vb + kv_h * stride_vh + col * stride_vt + offs * stride_vd)
    tl.store(kc_ptr + page * stride_kcp + kv_h * stride_kch + offset * stride_kcs + offs * stride_kcd, k)
    tl.store(vc_ptr + page * stride_vcp + kv_h * stride_vch + offset * stride_vcs + offs * stride_vcd, v)


def reshape_and_cache(k, v, k_pool, v_pool, slot_mapping, *, check=True):
    # k is [batch, kv_heads, time, dim], the layout project_qkv already produced.
    if k.ndim != 4:
        raise ValueError(f"k must have shape [batch, kv_heads, time, dim], got {tuple(k.shape)}")
    if v.shape != k.shape:
        raise ValueError(f"v must match k, got {tuple(v.shape)} and {tuple(k.shape)}")
    if k_pool.ndim != 4 or v_pool.shape != k_pool.shape:
        raise ValueError(
            f"k and v pools must share shape [pages, kv_heads, block, dim], got {tuple(k_pool.shape)} and {tuple(v_pool.shape)}"
        )
    batch, n_kv, time, head_dim = k.shape
    n_pages, pool_kv, block_size, pool_dim = k_pool.shape
    if pool_kv != n_kv or pool_dim != head_dim:
        raise ValueError(
            f"pool heads and dim are {pool_kv} and {pool_dim}, k is {n_kv} and {head_dim}"
        )
    if head_dim not in _HEAD_DIMS:
        raise ValueError(f"head_dim must be one of {_HEAD_DIMS}, got {head_dim}")
    if block_size not in _BLOCK_SIZES:
        raise ValueError(f"block_size must be one of {_BLOCK_SIZES}, got {block_size}")
    if n_kv < 1:
        raise ValueError(f"kv_heads must be positive, got {n_kv}")
    if slot_mapping.shape != (batch, time):
        raise ValueError(f"slot_mapping must have shape [{batch}, {time}], got {tuple(slot_mapping.shape)}")
    if slot_mapping.dtype != torch.int32:
        raise ValueError(f"slot_mapping must be int32, got {slot_mapping.dtype}")
    if k.dtype not in (torch.float32, torch.bfloat16) or v.dtype != k.dtype or k_pool.dtype != k.dtype or v_pool.dtype != k.dtype:
        raise ValueError(
            f"k, v, and the pools must share float32 or bfloat16, got {k.dtype}, {v.dtype}, {k_pool.dtype}, {v_pool.dtype}"
        )
    device = k.device
    if device.type != "cuda":
        raise ValueError(f"cache write runs on CUDA, got {device}")
    for name, tensor in (("v", v), ("k_pool", k_pool), ("v_pool", v_pool), ("slot_mapping", slot_mapping)):
        if tensor.device != device:
            raise ValueError(f"{name} is on {tensor.device}, k is on {device}")
    if batch == 0 or time == 0:
        return
    # One sync. A short table would otherwise write off the end of the pool.
    # check=False skips it. The forward already checked, and this runs once per layer.
    if check:
        n_slots = n_pages * block_size
        if bool(((slot_mapping < 0) | (slot_mapping >= n_slots)).any()):
            raise ValueError("each slot must land inside the pool")

    _scatter_kv[(batch * time, n_kv)](
        k, v, k_pool, v_pool, slot_mapping,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2), v_pool.stride(3),
        slot_mapping.stride(0), slot_mapping.stride(1),
        time,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
    )
