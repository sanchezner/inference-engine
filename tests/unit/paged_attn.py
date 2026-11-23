import torch

from engine.config import ModelConfig
from engine.kernels.triton.paged_attn import paged_attn_decode
from engine.model import RopeTables, repeat_kv
import engine.paged as paged
from engine.paged import PagePool, Sequence, forward_paged, forward_paged_batch, new_page_pool, new_sequence, page_gather
from engine.weights import LayerWeights, Weights

# RTX 3070, WSL2, triton 3.8. Measured max-abs on the sweep below: fp32 4.8e-7, bf16 7.8e-3.
# The bf16 gap is one ulp at magnitude 1, from rounding the fp32 accumulator on store.
_FP32_ATOL = 1e-5
_BF16_ATOL = 2e-2
# Gather softmax against SDPA, both in fp32. Measured disagreement 3.6e-7.
_ORACLE_ATOL = 1e-5


def main():
    if not torch.cuda.is_available():
        raise SystemExit("cuda is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    _rejects_bad_arguments()
    _empty_batch_is_empty()
    _one_token_is_the_value()
    _out_of_order_blocks_match_gather()
    _sweep()
    _batch_matches_each_row_alone()
    _strides_are_honored()
    _decode_forward_matches_gather()


def _rejects_bad_arguments():
    q, k, v, bt, sl = _buffers(batch=2, n_heads=4, n_kv=2, head_dim=64, block_size=16, width=2)
    _raises(lambda: paged_attn_decode(q[0], k, v, bt, sl), "q must have shape")
    _raises(lambda: paged_attn_decode(q, k[:, :, :2], v[:, :, :2], bt, sl), "block_size")
    _raises(lambda: paged_attn_decode(q[..., :32], k[..., :32], v[..., :32], bt, sl), "head_dim")
    _raises(lambda: paged_attn_decode(q[:, :3], k, v, bt, sl), "multiple")
    _raises(lambda: paged_attn_decode(q.cpu(), k.cpu(), v.cpu(), bt.cpu(), sl.cpu()), "CUDA")
    _raises(lambda: paged_attn_decode(q, k.to(torch.bfloat16), v.to(torch.bfloat16), bt, sl), "bfloat16")
    _raises(lambda: paged_attn_decode(q, k, v, bt.to(torch.int64), sl), "int32")
    _raises(lambda: paged_attn_decode(q, k, v, bt, torch.zeros_like(sl)), "seq_len")
    _raises(lambda: paged_attn_decode(q, k, v, bt[:, :1], torch.tensor([20, 20], device="cuda", dtype=torch.int32)), "fit")


def _empty_batch_is_empty():
    q, k, v, bt, sl = _buffers(batch=0, n_heads=4, n_kv=2, head_dim=64, block_size=16, width=1)
    out = paged_attn_decode(q, k, v, bt, sl)
    assert out.shape == (0, 4, 64)
    assert out.dtype == q.dtype and out.device == q.device


def _one_token_is_the_value():
    for dtype in (torch.float32, torch.bfloat16):
        for block_size in (4, 8, 16, 32):
            for n_kv, n_heads in ((1, 1), (2, 8), (4, 4)):
                got, _, _, bt, v = _compare(
                    n_heads=n_heads,
                    n_kv=n_kv,
                    head_dim=64,
                    block_size=block_size,
                    lengths=[1, 1],
                    dtype=dtype,
                    seed=block_size + n_heads,
                )
                n_rep = n_heads // n_kv
                for b in range(2):
                    page = int(bt[b, 0])
                    for h in range(n_heads):
                        expect = v[page, h // n_rep, 0]
                        assert torch.equal(got[b, h], expect), (dtype, block_size, h)


def _out_of_order_blocks_match_gather():
    # The paged checks' reuse pattern: blocks [4, 5, 1], last page only partly full.
    for dtype, atol in ((torch.float32, _FP32_ATOL), (torch.bfloat16, _BF16_ATOL)):
        n_heads, n_kv, head_dim, block_size = 6, 2, 64, 4
        n_pages = 8
        g = torch.Generator(device="cuda")
        g.manual_seed(41)
        k = torch.randn(n_pages, n_kv, block_size, head_dim, device="cuda", dtype=dtype, generator=g)
        v = torch.randn(n_pages, n_kv, block_size, head_dim, device="cuda", dtype=dtype, generator=g)
        q = torch.randn(1, n_heads, head_dim, device="cuda", dtype=dtype, generator=g)
        k[0] = float("nan")
        v[0] = float("nan")
        k[1, :, 1:] = float("nan")
        v[1, :, 1:] = float("nan")
        bt = torch.tensor([[4, 5, 1, 0]], device="cuda", dtype=torch.int32)
        sl = torch.tensor([9], device="cuda", dtype=torch.int32)
        got = paged_attn_decode(q, k, v, bt, sl)
        ref, sdpa = _oracles(q, k, v, bt, sl)
        _close(got, ref, atol, f"blocks[4,5,1] {dtype}")
        _close(sdpa, ref, _ORACLE_ATOL, f"sdpa blocks[4,5,1] {dtype}")


def _sweep():
    cases = []
    for dtype in (torch.float32, torch.bfloat16):
        for block_size in (4, 8, 16, 32):
            for ratio in (1, 8):
                lengths = [1, block_size - 1, block_size, block_size + 1, 128]
                cases.append((dtype, block_size, 2, 2 * ratio, 64, lengths, 10 + block_size + ratio))
            cases.append((dtype, block_size, 2, 8, 64, [4096, 4097], 80 + block_size))
        cases.append((dtype, 16, 2, 6, 64, [1, 7, 16, 17, 200], 90))
        cases.append((dtype, 16, 2, 8, 128, [3, 20], 91))
    for dtype, block_size, n_kv, n_heads, head_dim, lengths, seed in cases:
        _compare(
            n_heads=n_heads,
            n_kv=n_kv,
            head_dim=head_dim,
            block_size=block_size,
            lengths=lengths,
            dtype=dtype,
            seed=seed,
        )


def _batch_matches_each_row_alone():
    q, k, v, bt, sl = _layout(
        n_heads=8, n_kv=2, head_dim=64, block_size=16, lengths=[1, 17, 64, 200], dtype=torch.float32, seed=7,
    )
    batched = paged_attn_decode(q, k, v, bt, sl)
    for b in range(q.shape[0]):
        alone = paged_attn_decode(q[b:b + 1], k, v, bt[b:b + 1], sl[b:b + 1])
        assert torch.equal(alone, batched[b:b + 1])


def _strides_are_honored():
    q, k, v, bt, sl = _layout(
        n_heads=4, n_kv=2, head_dim=64, block_size=8, lengths=[3, 20], dtype=torch.bfloat16, seed=11,
    )
    base = paged_attn_decode(q, k, v, bt, sl)
    q_nc = q.transpose(0, 1).contiguous().transpose(0, 1)
    bt_nc = bt.transpose(0, 1).contiguous().transpose(0, 1)
    k_nc = k.permute(1, 0, 2, 3).contiguous().permute(1, 0, 2, 3)
    v_nc = v.permute(1, 0, 2, 3).contiguous().permute(1, 0, 2, 3)
    assert not q_nc.is_contiguous() and not bt_nc.is_contiguous() and not k_nc.is_contiguous()
    got = paged_attn_decode(q_nc, k_nc, v_nc, bt_nc, sl)
    assert torch.equal(got, base)


def _decode_forward_matches_gather():
    # The kernel replaces the gather on a one-token CUDA step. Same weights, both ways.
    for dtype, atol in ((torch.float32, _FP32_ATOL), (torch.bfloat16, _BF16_ATOL)):
        cfg, weights, rope = _tiny_model(dtype, n_heads=8, n_kv=2, seed=3)
        pool = new_page_pool(cfg, n_pages=16, block_size=4, device="cuda", dtype=dtype)
        seqs = []
        for n, seed in zip((1, 5, 8), (1, 2, 3)):
            seq = new_sequence(pool)
            ids = torch.randint(0, cfg.vocab_size, (1, n), device="cuda", generator=_gen(seed))
            _forcing_gather(lambda: forward_paged(weights, cfg, rope, ids, seq))
            seqs.append(seq)
        # Page 15 is not in any table. An over-read comes back as NaN.
        pool.k[:, 15] = float("nan")
        pool.v[:, 15] = float("nan")
        seqs[1].blocks = [seqs[1].blocks[1], seqs[1].blocks[0]]
        snap = _snap(seqs)
        new_ids = torch.randint(0, cfg.vocab_size, (3, 1), device="cuda", generator=_gen(9))
        got, _ = _assert_decode_does_not_gather(lambda: forward_paged_batch(weights, cfg, rope, new_ids, seqs))
        written = (pool.k.clone(), pool.v.clone(), [list(s.blocks) for s in seqs], [s.seq_len for s in seqs])
        _restore(seqs, snap)
        ref, _ = _forcing_gather(lambda: forward_paged_batch(weights, cfg, rope, new_ids, seqs))
        _close(got, ref, atol, f"batch decode {dtype}")
        assert [list(s.blocks) for s in seqs] == written[2]
        assert [s.seq_len for s in seqs] == written[3]
        # Layer 0's residual does not depend on attention, so that write matches exactly.
        _same_pages(pool.k[0], written[0][0], 0.0, f"k0 {dtype}")
        _same_pages(pool.v[0], written[1][0], 0.0, f"v0 {dtype}")
        _same_pages(pool.k, written[0], atol, f"k {dtype}")
        _same_pages(pool.v, written[1], atol, f"v {dtype}")


def _tiny_model(dtype, n_heads, n_kv, seed):
    g = _gen(seed)
    head_dim = 64
    hidden = n_heads * head_dim
    intermediate = 128
    cfg = ModelConfig(
        n_layers=2,
        n_heads=n_heads,
        n_kv_heads=n_kv,
        head_dim=head_dim,
        hidden_size=hidden,
        intermediate_size=intermediate,
        rms_norm_eps=1e-5,
        vocab_size=32,
        max_position=64,
        rope_theta=10_000.0,
        rope_type="default",
        tie_word_embeddings=True,
        eos_token_ids=(0,),
        bos_token_id=None,
    )

    def draw(*shape):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=g) * 0.02

    layers = tuple(
        LayerWeights(
            q=draw(n_heads * head_dim, hidden),
            k=draw(n_kv * head_dim, hidden),
            v=draw(n_kv * head_dim, hidden),
            o=draw(hidden, n_heads * head_dim),
            gate=draw(intermediate, hidden),
            up=draw(intermediate, hidden),
            down=draw(hidden, intermediate),
            input_norm=torch.ones(hidden, device="cuda", dtype=dtype),
            post_norm=torch.ones(hidden, device="cuda", dtype=dtype),
        )
        for _ in range(cfg.n_layers)
    )
    embed = draw(cfg.vocab_size, hidden)
    weights = Weights(embed=embed, layers=layers, norm=torch.ones(hidden, device="cuda", dtype=dtype), lm_head=embed)
    return cfg, weights, RopeTables.build(cfg, "cuda")


def _assert_decode_does_not_gather(fn):
    saved_gather = paged.page_gather
    saved_padded = paged.gather_padded

    def boom(*_args, **_kwargs):
        raise AssertionError("decode step gathered pages")

    paged.page_gather = boom
    paged.gather_padded = boom
    try:
        return fn()
    finally:
        paged.page_gather = saved_gather
        paged.gather_padded = saved_padded


def _forcing_gather(fn):
    saved = paged.decode_supported
    paged.decode_supported = lambda *_args, **_kwargs: False
    try:
        return fn()
    finally:
        paged.decode_supported = saved


def _snap(seqs):
    pool = seqs[0].pool
    return (
        pool.k.clone(),
        pool.v.clone(),
        list(pool.free),
        [(list(seq.blocks), seq.seq_len) for seq in seqs],
    )


def _restore(seqs, snap):
    k, v, free, states = snap
    pool = seqs[0].pool
    pool.k.copy_(k)
    pool.v.copy_(v)
    pool.free[:] = free
    for seq, (blocks, n) in zip(seqs, states):
        seq.blocks[:] = blocks
        seq.seq_len = n


def _gen(seed):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return g


def _compare(n_heads, n_kv, head_dim, block_size, lengths, dtype, seed):
    q, k, v, bt, sl = _layout(n_heads, n_kv, head_dim, block_size, lengths, dtype, seed)
    got = paged_attn_decode(q, k, v, bt, sl)
    ref, sdpa = _oracles(q, k, v, bt, sl)
    atol = _FP32_ATOL if dtype == torch.float32 else _BF16_ATOL
    name = f"{dtype} block {block_size} heads {n_heads}/{n_kv} lens {lengths}"
    _close(got, ref, atol, name)
    _close(sdpa, ref, _ORACLE_ATOL, f"sdpa {name}")
    return got, ref, sdpa, bt, v


def _layout(n_heads, n_kv, head_dim, block_size, lengths, dtype, seed):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    lengths = list(lengths)
    needs = [(n + block_size - 1) // block_size for n in lengths]
    n_pages = sum(needs) + 1
    k = torch.randn(n_pages, n_kv, block_size, head_dim, device="cuda", dtype=dtype, generator=g)
    v = torch.randn(n_pages, n_kv, block_size, head_dim, device="cuda", dtype=dtype, generator=g)
    q = torch.randn(len(lengths), n_heads, head_dim, device="cuda", dtype=dtype, generator=g)
    order = torch.randperm(n_pages - 1, device="cuda", generator=g).tolist()
    poison = n_pages - 1
    k[poison] = float("nan")
    v[poison] = float("nan")
    width = max(needs) + 1
    bt = torch.full((len(lengths), width), poison, device="cuda", dtype=torch.int32)
    for b, (n, need) in enumerate(zip(lengths, needs)):
        pages = order[:need]
        del order[:need]
        bt[b, :need] = torch.tensor(pages, dtype=torch.int32, device="cuda")
        rem = n % block_size
        if rem:
            k[pages[-1], :, rem:] = float("nan")
            v[pages[-1], :, rem:] = float("nan")
    sl = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    return q, k, v, bt, sl


def _oracles(q, k, v, bt, sl):
    block_size = k.shape[2]
    n_rep = q.shape[1] // k.shape[1]
    pool = PagePool(
        k=k.unsqueeze(0),
        v=v.unsqueeze(0),
        free=list(range(k.shape[0])),
        block_size=block_size,
    )
    manual = []
    sdpa = []
    scale = q.shape[-1] ** -0.5
    for b in range(q.shape[0]):
        n = int(sl[b].item())
        need = (n + block_size - 1) // block_size
        seq = Sequence(pool, blocks=[int(x) for x in bt[b, :need].tolist()], seq_len=n)
        kk, vv = page_gather(seq, 0, n)
        kk = repeat_kv(kk, n_rep)[0].float()
        vv = repeat_kv(vv, n_rep)[0].float()
        qf = q[b].float()
        # qf is [heads, dim] and kk is [heads, tokens, dim]. matmul would fold the token axis into the batch.
        scores = (qf[:, None, :] * kk).sum(dim=-1) * scale
        attn = torch.softmax(scores, dim=-1)
        manual.append((attn[:, :, None] * vv).sum(dim=1))
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            sdpa.append(torch.nn.functional.scaled_dot_product_attention(qf[:, None, :], kk, vv).squeeze(1))
    return torch.stack(manual), torch.stack(sdpa)


def _buffers(batch, n_heads, n_kv, head_dim, block_size, width):
    q = torch.zeros(batch, n_heads, head_dim, device="cuda")
    k = torch.zeros(1, n_kv, block_size, head_dim, device="cuda")
    v = torch.zeros_like(k)
    bt = torch.zeros(batch, width, device="cuda", dtype=torch.int32)
    sl = torch.ones(batch, device="cuda", dtype=torch.int32)
    return q, k, v, bt, sl


def _same_pages(got, ref, atol, name):
    nan_got = torch.isnan(got)
    nan_ref = torch.isnan(ref)
    assert torch.equal(nan_got, nan_ref), name
    if bool(nan_got.all()):
        return
    err = (got[~nan_got].float() - ref[~nan_ref].float()).abs().max().item()
    assert err <= atol, f"{name}: max abs err {err} > {atol}"


def _close(got, ref, atol, name):
    err = (got.float() - ref.float()).abs().max().item()
    assert err <= atol, f"{name}: max abs err {err} > {atol}"


def _raises(fn, text):
    try:
        fn()
    except ValueError as exc:
        assert text in str(exc), exc
    else:
        raise AssertionError(f"accepted an input that should fail ({text})")


if __name__ == "__main__":
    main()
