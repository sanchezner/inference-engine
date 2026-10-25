import torch
from functools import partial

from checks.agreement import assert_bf16, mean_kl_and_top1
from checks.oracle import load_engine, load_smollm
from checks.tolerances import LLAMA_LOGIT_ATOL, assert_logits
from engine.generate import generate
from engine.legacy.llama import forward_llama_cached
from engine.model import forward as forward_llama
from engine.paged import (
    forward_paged,
    forward_paged_batch,
    new_page_cache,
    new_page_pool,
    new_sequence,
    release_pages,
)
from engine.schedule import generate_continuous_paged


def _conserved(pool, seqs):
    n_pages = pool.k.shape[1]
    held = [block for seq in seqs for block in seq.blocks]
    assert len(held) == len(set(held)), held
    assert sorted(held + pool.free) == list(range(n_pages))


def main():
    torch.manual_seed(0)
    _fp32("cpu")
    if not torch.cuda.is_available():
        raise SystemExit("cuda is required: M1 checks the engine on the GPU")
    _fp32("cuda")
    _bf16()


def _fp32(device):
    tok, model = load_smollm(device=device, dtype=torch.float32)
    cfg, weights, rope = load_engine(device, torch.float32)
    eos = model.config.eos_token_id
    uncached = partial(forward_llama, weights, cfg, rope=rope)
    paged = partial(forward_paged, weights, cfg, rope)
    paged_batch = partial(forward_paged_batch, weights, cfg, rope)
    # The dense cache is the CPU ablation. On CUDA the oracle is HF itself:
    # batch invariance between two of our own paths is only exact on CPU fp32.
    cached = partial(forward_llama_cached, model) if device == "cpu" else None
    ids = tok("The capital of France is", return_tensors="pt").input_ids

    def reference(row, n):
        if cached is not None:
            return generate(cached, row, max_new_tokens=n, eos_id=eos, cache=[])
        return model.generate(
            row.to(device), max_new_tokens=n, do_sample=False, pad_token_id=eos
        ).cpu()

    with torch.no_grad():
        my_logits = uncached(ids)
        page = new_page_cache(cfg, n_pages=8, block_size=4, device=device, dtype=torch.float32)
        prefill_logits, page = paged(ids, page)
        assert_logits(f"page_prefill_{device}", prefill_logits, my_logits, LLAMA_LOGIT_ATOL)
        ref_logits = model(ids.to(device)).logits[:, -1]
        assert_logits(f"page_vs_hf_{device}", prefill_logits, ref_logits, LLAMA_LOGIT_ATOL)
        assert page.blocks == [0, 1]
        assert page.seq_len == 5
        assert page.pool.free == [2, 3, 4, 5, 6, 7]
        assert page.pool.k.device.type == torch.device(device).type
        _conserved(page.pool, [page])

        page = new_page_cache(cfg, n_pages=8, block_size=4, device=device, dtype=torch.float32)
        paged_ids = generate(paged, ids, max_new_tokens=10, eos_id=eos, cache=page)
        assert torch.equal(paged_ids.cpu(), reference(ids, 10).cpu())
        assert page.blocks == [0, 1, 2, 3]
        assert page.seq_len == 14
        assert page.pool.free == [4, 5, 6, 7]

        pool = new_page_pool(cfg, n_pages=8, block_size=4, device=device, dtype=torch.float32)
        france = new_sequence(pool)
        germany = new_sequence(pool)
        france_ids = ids
        germany_ids = tok("The capital of Germany is", return_tensors="pt").input_ids
        cat_ids = tok("The cat", return_tensors="pt").input_ids

        france_out = generate(paged, france_ids, max_new_tokens=10, eos_id=eos, cache=france)
        germany_out = generate(paged, germany_ids, max_new_tokens=4, eos_id=eos, cache=germany)
        assert torch.equal(france_out.cpu(), reference(france_ids, 10).cpu())
        assert torch.equal(germany_out.cpu(), reference(germany_ids, 4).cpu())
        assert france.blocks == [0, 1, 2, 3] and france.seq_len == 14
        assert germany.blocks == [4, 5] and germany.seq_len == 8

        release_pages(france)
        assert france.blocks == []
        assert germany.blocks == [4, 5]
        assert pool.free == [0, 1, 2, 3, 6, 7]
        _conserved(pool, [france, germany])

        cat = new_sequence(pool)
        cat_logits, cat = paged(cat_ids, cat)
        assert_logits(f"cat_prefill_{device}", cat_logits, uncached(cat_ids), LLAMA_LOGIT_ATOL)
        assert cat.blocks == [0]
        assert germany.blocks == [4, 5]
        assert pool.free == [1, 2, 3, 6, 7]

        germany_out = generate(paged, germany_out, max_new_tokens=1, eos_id=eos, cache=germany)
        assert torch.equal(germany_out.cpu(), reference(germany_ids, 5).cpu())
        # Page 1 was France's, then freed, then handed to Germany. The block
        # table is a playlist, not a counter, so the ids are not 4, 5, 6.
        assert germany.blocks == [4, 5, 1]
        assert germany.seq_len == 9
        assert pool.free == [2, 3, 6, 7]
        _conserved(pool, [france, germany, cat])

    _continuous(device, tok, model, cfg, paged, paged_batch, eos, cached, [
        "The capital of France is", "The cat", "The capital of Germany is",
    ])
    _continuous(device, tok, model, cfg, paged, paged_batch, eos, cached, [
        "The capital of France is", "The cat", "Once upon a time there was a",
    ])


def _continuous(device, tok, model, cfg, paged, paged_batch, eos, cached, texts):
    budgets = [6, 2, 4]
    rows = [tok(p, return_tensors="pt").input_ids[0] for p in texts]
    with torch.no_grad():
        if cached is not None:
            serial = [
                generate(cached, row.view(1, -1), max_new_tokens=b, eos_id=eos, cache=[])
                for row, b in zip(rows, budgets)
            ]
        else:
            serial = [
                model.generate(
                    row.view(1, -1).to(device), max_new_tokens=b, do_sample=False, pad_token_id=eos
                ).cpu()
                for row, b in zip(rows, budgets)
            ]
        pool = new_page_pool(cfg, n_pages=16, block_size=4, device=device, dtype=torch.float32)
        cont = generate_continuous_paged(paged, paged_batch, pool, rows, budgets, eos, max_batch=2)
    for got, ref in zip(cont, serial):
        assert torch.equal(got.cpu(), ref[0].cpu())
    assert pool.free == list(range(16))
    _conserved(pool, [])


def _bf16():
    tok, oracle = load_smollm(device="cuda", dtype=torch.float32)
    cfg, weights, rope = load_engine("cuda", torch.bfloat16)

    def start():
        return new_page_cache(cfg, n_pages=32, block_size=4, device="cuda", dtype=torch.bfloat16)

    def step(seq, prefix):
        return forward_paged(weights, cfg, rope, prefix, seq)

    with torch.no_grad():
        kl_value, top1 = mean_kl_and_top1(tok, oracle, start, step)
        # Prompt length, not the sampled tokens, decides the block table.
        # This still has to hold when bf16 would have emitted a different id.
        seq = new_page_cache(cfg, n_pages=8, block_size=4, device="cuda", dtype=torch.bfloat16)
        ids = tok("The capital of France is", return_tensors="pt").input_ids
        logits, seq = forward_paged(weights, cfg, rope, ids, seq)
    assert logits.dtype == torch.bfloat16
    assert seq.blocks == [0, 1] and seq.seq_len == ids.shape[1]
    assert seq.pool.k.dtype == torch.bfloat16 and seq.pool.k.device.type == "cuda"
    release_pages(seq)
    assert seq.pool.free == list(range(8))
    assert_bf16(kl_value, top1)


if __name__ == "__main__":
    main()
