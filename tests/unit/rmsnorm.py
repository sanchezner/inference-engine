import torch

import engine.kernels.triton.rmsnorm as rmsnorm
from engine.kernels.triton.rmsnorm import fused_rms_norm
from engine.model import rms_norm, rms_norm_ref

# RTX 3070, WSL2, triton 3.8. Measured max-abs on the sweep below: fp32 9.5e-7, bf16 0.
# fp32 is reduction order against the eager mean. bf16 matched the eager store on this sweep.
_FP32_ATOL = 1e-5
_BF16_ATOL = 1e-2


def main():
    if not torch.cuda.is_available():
        raise SystemExit("cuda is required")
    torch.manual_seed(0)
    _rejects_bad_arguments()
    _empty_row_is_empty()
    _ones_match()
    _sweep()
    _three_d_matches()
    _noncontiguous_uses_the_reference()
    _flag_off_is_the_reference()


def _rejects_bad_arguments():
    x = torch.randn(4, 64, device="cuda")
    w = torch.randn(64, device="cuda")
    _raises(lambda: fused_rms_norm(x.cpu(), w.cpu(), 1e-5), "CUDA")
    _raises(lambda: fused_rms_norm(x, w[:32], 1e-5), "weight")
    _raises(lambda: fused_rms_norm(x.to(torch.float16), w.to(torch.float16), 1e-5), "bf16")
    _raises(lambda: fused_rms_norm(x.t(), torch.randn(4, device="cuda"), 1e-5), "contiguous")


def _empty_row_is_empty():
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.empty(0, 64, device="cuda", dtype=dtype)
        w = torch.randn(64, device="cuda", dtype=dtype)
        got = fused_rms_norm(x, w, 1e-5)
        assert got.shape == x.shape
        assert got.dtype == dtype
        assert torch.equal(rms_norm(x, w, 1e-5), rms_norm_ref(x, w, 1e-5))


def _ones_match():
    for n in (1, 64, 576, 2048):
        x = torch.ones(3, n, device="cuda")
        w = torch.ones(n, device="cuda")
        got = fused_rms_norm(x, w, 1e-5)
        ref = rms_norm_ref(x, w, 1e-5)
        err = (got - ref).abs().max().item()
        assert err <= 1e-6, f"ones {n}: max abs err {err}"


def _sweep():
    shapes = ((1, 64), (8, 2048), (2, 576), (4, 70), (1, 4096), (8, 1, 2048))
    for dtype, atol, name in (
        (torch.float32, _FP32_ATOL, "fp32"),
        (torch.bfloat16, _BF16_ATOL, "bf16"),
    ):
        for shape in shapes:
            g = torch.Generator(device="cuda")
            g.manual_seed(shape[-1] + len(shape))
            x = torch.randn(*shape, device="cuda", dtype=dtype, generator=g)
            w = torch.randn(shape[-1], device="cuda", dtype=dtype, generator=g)
            got = fused_rms_norm(x, w, 1e-5)
            ref = rms_norm_ref(x, w, 1e-5)
            err = (got.float() - ref.float()).abs().max().item()
            assert err <= atol, f"{name} {shape}: max abs err {err} > {atol}"
            dispatched = rms_norm(x, w, 1e-5)
            assert torch.equal(dispatched, got)


def _three_d_matches():
    x = torch.randn(2, 5, 576, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(576, device="cuda", dtype=torch.bfloat16)
    got = fused_rms_norm(x, w, 1e-5)
    ref = rms_norm_ref(x, w, 1e-5)
    err = (got.float() - ref.float()).abs().max().item()
    assert got.shape == x.shape
    assert err <= _BF16_ATOL, err


def _noncontiguous_uses_the_reference():
    base = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    x = base[::2]
    w = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    assert not x.is_contiguous()
    got = rms_norm(x, w, 1e-5)
    ref = rms_norm_ref(x, w, 1e-5)
    assert torch.equal(got, ref)


def _flag_off_is_the_reference():
    saved = rmsnorm.use_fused
    try:
        rmsnorm.use_fused = False
        x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(256, device="cuda", dtype=torch.bfloat16)
        assert torch.equal(rms_norm(x, w, 1e-5), rms_norm_ref(x, w, 1e-5))
    finally:
        rmsnorm.use_fused = saved


def _raises(fn, text):
    try:
        fn()
    except ValueError as exc:
        assert text in str(exc), exc
        return
    raise AssertionError(f"expected ValueError containing {text!r}")


if __name__ == "__main__":
    main()
