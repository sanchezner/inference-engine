import torch

import engine.kernels.triton.rope as rope
from engine.kernels.triton.rope import fused_rope
from engine.model import apply_rope, apply_rope_ref

# RTX 3070, WSL2, triton 3.8. Measured max-abs on the sweep below: fp32 9.5e-7, bf16 1.5e-5.
# fp32 is fused multiply-add against the eager multiply, then add. bf16 is that gap after the cast.
_FP32_ATOL = 1e-5
_BF16_ATOL = 1e-4


def main():
    if not torch.cuda.is_available():
        raise SystemExit("cuda is required")
    torch.manual_seed(0)
    _rejects_bad_arguments()
    _empty_is_a_no_op()
    _identities()
    _sweep()
    _transposed_layout_matches()
    _broadcast_batch_matches()
    _inputs_are_not_mutated()
    _noncontiguous_last_dim_uses_the_reference()
    _flag_off_is_the_reference()


def _rejects_bad_arguments():
    q, k, cos, sin = _args(2, 4, 2, 3, 64, torch.float32)
    _raises(lambda: fused_rope(q.cpu(), k.cpu(), cos.cpu(), sin.cpu()), "CUDA")
    _raises(lambda: fused_rope(q[0], k[0], cos, sin), "shape")
    _raises(lambda: fused_rope(q[..., :32], k[..., :32], cos[..., :32], sin[..., :32]), "head_dim")
    _raises(lambda: fused_rope(q.to(torch.float16), k.to(torch.float16), cos, sin), "bfloat16")
    wide_k = torch.randn(2, 8, 3, 64, device="cuda")
    _raises(lambda: fused_rope(q, wide_k, cos, sin), "kv_heads")
    # A sliced last dim keeps the shape and breaks the unit stride the kernel stores through.
    sliced = torch.randn(2, 4, 3, 128, device="cuda")[..., ::2]
    _raises(lambda: fused_rope(sliced, k, cos, sin), "contiguous")


def _empty_is_a_no_op():
    q, k, cos, sin = _args(0, 4, 2, 3, 64, torch.bfloat16)
    got_q, got_k = fused_rope(q, k, cos, sin)
    assert got_q.shape == q.shape and got_k.shape == k.shape
    assert torch.equal(got_q, q) and torch.equal(got_k, k)


def _identities():
    for dtype in (torch.float32, torch.bfloat16):
        q, k, _, _ = _args(2, 4, 2, 5, 64, dtype, seed=3)
        ones = torch.ones(2, 5, 64, device="cuda")
        zeros = torch.zeros_like(ones)
        got_q, got_k = fused_rope(q, k, ones, zeros)
        assert torch.equal(got_q, q)
        assert torch.equal(got_k, k)
        got_q, got_k = fused_rope(q, k, zeros, ones)
        assert torch.equal(got_q, _half_turn(q))
        assert torch.equal(got_k, _half_turn(k))


def _sweep():
    shapes = (
        (1, 4, 4, 1, 64),
        (2, 8, 2, 7, 64),
        (4, 32, 32, 1, 64),
        (2, 9, 3, 4, 64),
        (1, 4, 4, 3, 128),
        (2, 8, 2, 2, 128),
    )
    for dtype, atol, name in (
        (torch.float32, _FP32_ATOL, "fp32"),
        (torch.bfloat16, _BF16_ATOL, "bf16"),
    ):
        for batch, n_q, n_kv, time, dim in shapes:
            q, k, cos, sin = _args(batch, n_q, n_kv, time, dim, dtype, seed=dim + time + n_q)
            got_q, got_k = fused_rope(q, k, cos, sin)
            ref_q, ref_k = apply_rope_ref(q, k, cos, sin)
            err = _max_err(got_q, got_k, ref_q, ref_k)
            assert err <= atol, f"{name} {(batch, n_q, n_kv, time, dim)}: max abs err {err} > {atol}"
            disp_q, disp_k = apply_rope(q, k, cos, sin)
            assert torch.equal(disp_q, got_q) and torch.equal(disp_k, got_k)


def _transposed_layout_matches():
    # project_qkv views [batch, time, heads, dim] and swaps time with heads.
    dtype = torch.bfloat16
    batch, time, n_q, n_kv, dim = 2, 5, 8, 2, 64
    g = torch.Generator(device="cuda")
    g.manual_seed(11)
    q = torch.randn(batch, time, n_q, dim, device="cuda", dtype=dtype, generator=g).transpose(1, 2)
    k = torch.randn(batch, time, n_kv, dim, device="cuda", dtype=dtype, generator=g).transpose(1, 2)
    cos = torch.randn(batch, time, dim, device="cuda", generator=g)
    sin = torch.randn(batch, time, dim, device="cuda", generator=g)
    assert not q.is_contiguous() and q.stride(-1) == 1
    got_q, got_k = fused_rope(q, k, cos, sin)
    ref_q, ref_k = apply_rope_ref(q, k, cos, sin)
    err = _max_err(got_q, got_k, ref_q, ref_k)
    assert err <= _BF16_ATOL, err
    assert got_q.is_contiguous() and got_k.is_contiguous()


def _broadcast_batch_matches():
    q, k, _, _ = _args(4, 8, 2, 3, 64, torch.float32, seed=19)
    g = torch.Generator(device="cuda")
    g.manual_seed(20)
    cos = torch.randn(1, 3, 64, device="cuda", generator=g)
    sin = torch.randn(1, 3, 64, device="cuda", generator=g)
    got_q, got_k = fused_rope(q, k, cos, sin)
    ref_q, ref_k = apply_rope_ref(q, k, cos, sin)
    err = _max_err(got_q, got_k, ref_q, ref_k)
    assert err <= _FP32_ATOL, err


def _inputs_are_not_mutated():
    q, k, cos, sin = _args(2, 4, 2, 2, 64, torch.bfloat16, seed=5)
    q0, k0 = q.clone(), k.clone()
    fused_rope(q, k, cos, sin)
    assert torch.equal(q, q0) and torch.equal(k, k0)


def _noncontiguous_last_dim_uses_the_reference():
    _, _, cos, sin = _args(2, 4, 2, 3, 64, torch.bfloat16, seed=7)
    q = torch.randn(2, 4, 3, 128, device="cuda", dtype=torch.bfloat16)[..., ::2]
    k = torch.randn(2, 2, 3, 128, device="cuda", dtype=torch.bfloat16)[..., ::2]
    assert q.stride(-1) != 1
    got_q, got_k = apply_rope(q, k, cos, sin)
    ref_q, ref_k = apply_rope_ref(q, k, cos, sin)
    assert torch.equal(got_q, ref_q) and torch.equal(got_k, ref_k)


def _flag_off_is_the_reference():
    saved = rope.use_fused
    try:
        rope.use_fused = False
        q, k, cos, sin = _args(2, 4, 2, 3, 64, torch.bfloat16, seed=9)
        got_q, got_k = apply_rope(q, k, cos, sin)
        ref_q, ref_k = apply_rope_ref(q, k, cos, sin)
        assert torch.equal(got_q, ref_q) and torch.equal(got_k, ref_k)
    finally:
        rope.use_fused = saved


def _args(batch, n_q, n_kv, time, dim, dtype, seed=0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    q = torch.randn(batch, n_q, time, dim, device="cuda", dtype=dtype, generator=g)
    k = torch.randn(batch, n_kv, time, dim, device="cuda", dtype=dtype, generator=g)
    cos = torch.randn(batch, time, dim, device="cuda", generator=g)
    sin = torch.randn(batch, time, dim, device="cuda", generator=g)
    return q, k, cos, sin


def _half_turn(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _max_err(got_q, got_k, ref_q, ref_k) -> float:
    q_err = (got_q.float() - ref_q.float()).abs().max().item()
    k_err = (got_k.float() - ref_k.float()).abs().max().item()
    return max(q_err, k_err)


def _raises(fn, text):
    try:
        fn()
    except ValueError as exc:
        assert text in str(exc), exc
        return
    raise AssertionError(f"expected ValueError containing {text!r}")


if __name__ == "__main__":
    main()
