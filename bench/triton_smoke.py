import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


def main():
    n = 128
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y = torch.arange(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    add_kernel[(1,)](x, y, out, n, BLOCK=128)
    torch.cuda.synchronize()
    err = (out - (x + y)).abs().max().item()
    if err != 0.0:
        raise SystemExit(f"triton smoke max abs err {err}")
    print("triton_ok")


if __name__ == "__main__":
    main()
