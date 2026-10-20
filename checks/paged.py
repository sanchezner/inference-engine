import torch
from functools import partial
from engine.load import load_smollm
from engine.llama import forward_llama
from engine.legacy.llama import forward_llama_cached
from engine.paged import (
    forward_llama_paged,
    forward_llama_paged_batch,
    new_page_cache,
    new_page_pool,
    new_sequence,
    release_pages,
)
from engine.generate import generate
from engine.schedule import generate_continuous_paged
from checks.tolerances import LLAMA_LOGIT_ATOL, assert_logits


def _conserved(pool, seqs):
    n_pages = pool["k"].shape[1]
    held = [block for seq in seqs for block in seq["blocks"]]
    assert len(held) == len(set(held)), held
    assert sorted(held + pool["free"]) == list(range(n_pages))


def main():
    torch.manual_seed(0)
    llama_tok, llama = load_smollm()
    uncached = partial(forward_llama, llama)
    cached = partial(forward_llama_cached, llama)
    paged = partial(forward_llama_paged, llama)
    paged_batch = partial(forward_llama_paged_batch, llama)
    llama_ids = llama_tok("The capital of France is", return_tensors="pt").input_ids
    eos = llama.config.eos_token_id

    with torch.no_grad():
        my_llama_logits = uncached(llama_ids)

        page = new_page_cache(llama, n_pages=8, block_size=4)
        prefill_logits, page = paged(llama_ids, page)
        assert_logits("page_prefill", prefill_logits, my_llama_logits, LLAMA_LOGIT_ATOL)
        assert page["blocks"] == [0, 1]
        assert page["seq_len"] == 5
        assert page["pool"]["free"] == [2, 3, 4, 5, 6, 7]
        _conserved(page["pool"], [page])

        page = new_page_cache(llama, n_pages=8, block_size=4)
        paged_ids = generate(paged, llama_ids, max_new_tokens=10, eos_id=eos, cache=page)
        cached_ids = generate(cached, llama_ids, max_new_tokens=10, eos_id=eos, cache=[])
        assert torch.equal(paged_ids, cached_ids)
        assert page["blocks"] == [0, 1, 2, 3]
        assert page["seq_len"] == 14
        assert page["pool"]["free"] == [4, 5, 6, 7]

        pool = new_page_pool(llama, n_pages=8, block_size=4)
        france = new_sequence(pool)
        germany = new_sequence(pool)
        france_ids = llama_ids
        germany_ids = llama_tok("The capital of Germany is", return_tensors="pt").input_ids
        cat_ids = llama_tok("The cat", return_tensors="pt").input_ids

        france_out = generate(paged, france_ids, max_new_tokens=10, eos_id=eos, cache=france)
        france_serial = generate(cached, france_ids, max_new_tokens=10, eos_id=eos, cache=[])
        germany_out = generate(paged, germany_ids, max_new_tokens=4, eos_id=eos, cache=germany)
        germany_serial = generate(cached, germany_ids, max_new_tokens=4, eos_id=eos, cache=[])
        assert torch.equal(france_out, france_serial)
        assert torch.equal(germany_out, germany_serial)
        assert france["blocks"] == [0, 1, 2, 3] and france["seq_len"] == 14
        assert germany["blocks"] == [4, 5] and germany["seq_len"] == 8

        release_pages(france)
        assert france["blocks"] == []
        assert germany["blocks"] == [4, 5]
        assert pool["free"] == [0, 1, 2, 3, 6, 7]
        _conserved(pool, [france, germany])

        cat = new_sequence(pool)
        cat_logits, cat = paged(cat_ids, cat)
        assert_logits("cat_prefill", cat_logits, uncached(cat_ids), LLAMA_LOGIT_ATOL)
        assert cat["blocks"] == [0]
        assert germany["blocks"] == [4, 5]
        assert pool["free"] == [1, 2, 3, 6, 7]

        germany_out = generate(paged, germany_out, max_new_tokens=1, eos_id=eos, cache=germany)
        germany_serial = generate(cached, germany_ids, max_new_tokens=5, eos_id=eos, cache=[])
        assert torch.equal(germany_out, germany_serial)
        # Page 1 was France's, then freed, then handed to Germany. The block
        # table is a playlist, not a counter, so the ids are not 4, 5, 6.
        assert germany["blocks"] == [4, 5, 1]
        assert germany["seq_len"] == 9
        assert pool["free"] == [2, 3, 6, 7]
        _conserved(pool, [france, germany, cat])

    _continuous(llama, llama_tok, cached, paged, paged_batch, eos, [
        "The capital of France is", "The cat", "The capital of Germany is",
    ])
    _continuous(llama, llama_tok, cached, paged, paged_batch, eos, [
        "The capital of France is", "The cat", "Once upon a time there was a",
    ])


def _continuous(llama, llama_tok, cached, paged, paged_batch, eos, texts):
    budgets = [6, 2, 4]
    rows = [llama_tok(p, return_tensors="pt").input_ids[0] for p in texts]
    with torch.no_grad():
        serial = [
            generate(cached, row.view(1, -1), max_new_tokens=b, eos_id=eos, cache=[])
            for row, b in zip(rows, budgets)
        ]
        pool = new_page_pool(llama, n_pages=16, block_size=4)
        cont = generate_continuous_paged(paged, paged_batch, pool, rows, budgets, eos, max_batch=2)
    for got, ref in zip(cont, serial):
        assert torch.equal(got, ref[0])
    assert pool["free"] == list(range(16))


if __name__ == "__main__":
    main()
