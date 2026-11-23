import torch

from engine.kernels.triton.reshape_cache import reshape_and_cache
from engine.paged import PagePool, Sequence, page_write

# A cache write is a copy. fp32 and bf16 must match page_write bit for bit.


def main():
    if not torch.cuda.is_available():
        raise SystemExit("cuda is required")
    torch.manual_seed(0)
    _rejects_bad_arguments()
    _empty_write_leaves_the_pool()
    _out_of_order_blocks_match_page_write()
    _sweep()
    _strides_are_honored()


def _rejects_bad_arguments():
    k, v, kc, vc, slots = _buffers(batch=2, n_kv=2, time=2, head_dim=64, n_pages=4, block_size=16)
    _raises(lambda: reshape_and_cache(k[0], v[0], kc, vc, slots[0]), "k must have shape")
    _raises(lambda: reshape_and_cache(k, v[:, :, :1], kc, vc, slots[:, :1]), "v must match")
    _raises(lambda: reshape_and_cache(k, v, kc[:, :, :2], vc[:, :, :2], slots), "block_size")
    _raises(lambda: reshape_and_cache(k[..., :32], v[..., :32], kc[..., :32], vc[..., :32], slots), "head_dim")
    _raises(lambda: reshape_and_cache(k, v, kc[:, :1], vc[:, :1], slots), "pool heads")
    _raises(lambda: reshape_and_cache(k.cpu(), v.cpu(), kc.cpu(), vc.cpu(), slots.cpu()), "CUDA")
    _raises(lambda: reshape_and_cache(k, v.to(torch.bfloat16), kc.to(torch.bfloat16), vc.to(torch.bfloat16), slots), "bfloat16")
    _raises(lambda: reshape_and_cache(k, v, kc, vc, slots.to(torch.int64)), "int32")
    _raises(lambda: reshape_and_cache(k, v, kc, vc, torch.full_like(slots, -1)), "slot")
    _raises(lambda: reshape_and_cache(k, v, kc, vc, torch.full_like(slots, 4 * 16)), "slot")


def _empty_write_leaves_the_pool():
    k, v, kc, vc, slots = _buffers(batch=0, n_kv=2, time=4, head_dim=64, n_pages=2, block_size=8)
    kc.fill_(1)
    vc.fill_(2)
    reshape_and_cache(k, v, kc, vc, slots)
    assert torch.equal(kc, torch.ones_like(kc))
    assert torch.equal(vc, torch.full_like(vc, 2))
    k0, v0, kc0, vc0, slots0 = _buffers(batch=2, n_kv=2, time=0, head_dim=64, n_pages=2, block_size=8)
    kc0.fill_(1)
    vc0.fill_(2)
    reshape_and_cache(k0, v0, kc0, vc0, slots0)
    assert torch.equal(kc0, torch.ones_like(kc0))
    assert torch.equal(vc0, torch.full_like(vc0, 2))


def _out_of_order_blocks_match_page_write():
    # The paged checks' reuse pattern: blocks [4, 5, 1], last page only partly full.
    for dtype in (torch.float32, torch.bfloat16):
        held = [[4, 5, 1]]
        _compare(n_kv=2, head_dim=64, block_size=4, held=held, starts=[0], time=9, dtype=dtype, seed=41, n_pages=8)


def _sweep():
    cases = []
    for dtype in (torch.float32, torch.bfloat16):
        for block_size in (4, 8, 16, 32):
            for n_kv in (1, 3, 8):
                # Starts one slot before a boundary so the write spills into the next page.
                held = [[0, 1], [3, 6]]
                cases.append((dtype, block_size, n_kv, 64, held, [block_size - 1, 0], 3, 10 + block_size + n_kv))
        cases.append((dtype, 16, 2, 64, [list(range(32))], [0], 512, 80))
        cases.append((dtype, 16, 32, 128, [[1], [4]], [0, 3], 2, 90))
    for dtype, block_size, n_kv, head_dim, held, starts, time, seed in cases:
        n_pages = max(page for row in held for page in row) + 2
        _compare(
            n_kv=n_kv,
            head_dim=head_dim,
            block_size=block_size,
            held=held,
            starts=starts,
            time=time,
            dtype=dtype,
            seed=seed,
            n_pages=n_pages,
        )


def _strides_are_honored():
    dtype = torch.bfloat16
    held = [[2, 7], [0, 3]]
    starts = [6, 1]
    time = 4
    block_size = 8
    k, v, slots = _inputs(n_kv=4, head_dim=64, time=time, held=held, starts=starts, block_size=block_size, dtype=dtype, seed=11)
    kc, vc = _poison_pools(n_pages=10, n_kv=4, block_size=block_size, head_dim=64, dtype=dtype)
    reshape_and_cache(k, v, kc, vc, slots)
    k_nc = k.permute(1, 0, 2, 3).contiguous().permute(1, 0, 2, 3)
    v_nc = v.permute(1, 0, 2, 3).contiguous().permute(1, 0, 2, 3)
    slots_nc = slots.t().contiguous().t()
    view_k = kc.new_empty(kc.shape).permute(1, 0, 2, 3).contiguous().permute(1, 0, 2, 3)
    view_v = vc.new_empty(vc.shape).permute(1, 0, 2, 3).contiguous().permute(1, 0, 2, 3)
    view_k.copy_(torch.full_like(kc, float("nan")))
    view_v.copy_(torch.full_like(vc, float("nan")))
    assert not k_nc.is_contiguous() and not slots_nc.is_contiguous() and not view_k.is_contiguous()
    reshape_and_cache(k_nc, v_nc, view_k, view_v, slots_nc)
    _same(view_k, kc, f"stride k {dtype}")
    _same(view_v, vc, f"stride v {dtype}")


def _compare(n_kv, head_dim, block_size, held, starts, time, dtype, seed, n_pages):
    k, v, slots = _inputs(n_kv, head_dim, time, held, starts, block_size, dtype, seed)
    kc, vc = _poison_pools(n_pages, n_kv, block_size, head_dim, dtype)
    ref_k, ref_v = _page_write(kc, vc, k, v, held, starts, block_size)
    reshape_and_cache(k, v, kc, vc, slots)
    name = f"{dtype} block {block_size} kv {n_kv} dim {head_dim} time {time}"
    _same(kc, ref_k, f"k {name}")
    _same(vc, ref_v, f"v {name}")


def _inputs(n_kv, head_dim, time, held, starts, block_size, dtype, seed):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    batch = len(held)
    k = torch.randn(batch, n_kv, time, head_dim, device="cuda", dtype=dtype, generator=g)
    v = torch.randn(batch, n_kv, time, head_dim, device="cuda", dtype=dtype, generator=g)
    rows = []
    for blocks, start in zip(held, starts):
        row = []
        for t in range(time):
            pos = start + t
            page = blocks[pos // block_size]
            row.append(page * block_size + (pos % block_size))
        rows.append(row)
    slots = torch.tensor(rows, device="cuda", dtype=torch.int32)
    return k, v, slots


def _poison_pools(n_pages, n_kv, block_size, head_dim, dtype):
    # NaN marks every slot the write must leave alone.
    k = torch.full((n_pages, n_kv, block_size, head_dim), float("nan"), device="cuda", dtype=dtype)
    v = torch.full((n_pages, n_kv, block_size, head_dim), float("nan"), device="cuda", dtype=dtype)
    return k, v


def _page_write(k_pool, v_pool, k_new, v_new, held, starts, block_size):
    pool = PagePool(
        k=k_pool.clone().unsqueeze(0),
        v=v_pool.clone().unsqueeze(0),
        free=list(range(k_pool.shape[0])),
        block_size=block_size,
    )
    for row, (blocks, start) in enumerate(zip(held, starts)):
        seq = Sequence(pool, blocks=list(blocks), seq_len=start)
        page_write(seq, 0, k_new[row:row + 1], v_new[row:row + 1], start)
    return pool.k[0], pool.v[0]


def _buffers(batch, n_kv, time, head_dim, n_pages, block_size):
    k = torch.zeros(batch, n_kv, time, head_dim, device="cuda")
    v = torch.zeros_like(k)
    kc = torch.zeros(n_pages, n_kv, block_size, head_dim, device="cuda")
    vc = torch.zeros_like(kc)
    slots = torch.zeros(batch, time, device="cuda", dtype=torch.int32)
    return k, v, kc, vc, slots


def _same(got, ref, name):
    nan_got = torch.isnan(got)
    nan_ref = torch.isnan(ref)
    assert torch.equal(nan_got, nan_ref), name
    if bool(nan_got.all()):
        return
    assert torch.equal(got[~nan_got], ref[~nan_ref]), name


def _raises(fn, text):
    try:
        fn()
    except ValueError as exc:
        assert text in str(exc), exc
    else:
        raise AssertionError(f"accepted an input that should fail ({text})")


if __name__ == "__main__":
    main()
