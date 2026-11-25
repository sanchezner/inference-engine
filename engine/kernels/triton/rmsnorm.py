import torch
import triton
import triton.language as tl


# The bench turns this off to time the eager norm against the kernel.
use_fused = True

_DTYPES = (torch.float32, torch.bfloat16)
# A wider row would unroll a long static loop. These models are 576 and 2048.
_MAX_COLS = 8192


@triton.jit
def _rms_norm(
    x_ptr, w_ptr, y_ptr, eps, stride_row,
    N_COLS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * stride_row
    y_ptr += row * stride_row
    # Sum in fp32. The eager norm squares the cast row, so a bf16 mean would not match it.
    var = 0.0
    for start in range(0, N_COLS, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N_COLS
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        var += tl.sum(x * x, axis=0)
    rstd = tl.rsqrt(var / N_COLS + eps)
    for start in range(0, N_COLS, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N_COLS
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + offs, x * rstd * w, mask=mask)


def _block(n_cols: int) -> int:
    if n_cols <= 1024:
        return 1024
    if n_cols <= 2048:
        return 2048
    return 1024


def rms_supported(x, weight) -> bool:
    if not use_fused or not x.is_cuda or x.dtype not in _DTYPES or not x.is_contiguous():
        return False
    n_cols = x.shape[-1]
    if x.numel() == 0:
        return _weight_ok(x, weight, n_cols)
    return (
        1 <= n_cols <= _MAX_COLS
        and _weight_ok(x, weight, n_cols)
    )


def _weight_ok(x, weight, n_cols) -> bool:
    return (
        weight.is_cuda
        and weight.device == x.device
        and weight.dtype == x.dtype
        and weight.ndim == 1
        and weight.is_contiguous()
        and weight.numel() == n_cols
    )


def _row_ok(x, weight) -> bool:
    if not x.is_cuda or x.dtype not in _DTYPES or not x.is_contiguous() or x.ndim < 1:
        return False
    n_cols = x.shape[-1]
    if not _weight_ok(x, weight, n_cols):
        return False
    if x.numel() == 0:
        return True
    return 1 <= n_cols <= _MAX_COLS


def fused_rms_norm(x, weight, eps):
    if not _row_ok(x, weight):
        raise ValueError(
            f"fused rms wants a contiguous CUDA fp32 or bf16 row with a matching 1D weight, "
            f"got x {tuple(x.shape)} {x.dtype} {x.device} and weight {tuple(weight.shape)} {weight.dtype} {weight.device}"
        )
    if x.numel() == 0:
        return torch.empty_like(x)
    n_cols = x.shape[-1]
    flat = x.view(-1, n_cols)
    out = torch.empty_like(flat)
    _rms_norm[(flat.shape[0],)](
        flat, weight, out, float(eps), n_cols,
        N_COLS=n_cols, BLOCK=_block(n_cols),
    )
    return out.view(x.shape)
