import torch
from functools import partial
from engine.load import load_smollm
from engine.llama import forward_llama, forward_llama_cached
from engine.paged import new_page_cache, new_page_pool, new_sequence, release_pages, forward_llama_paged, forward_llama_paged_batch
from engine.generate import generate
from engine.schedule import generate_continuous_paged

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
    print("page_prefill_err", (prefill_logits - my_llama_logits).abs().max().item())
    print("blocks", page["blocks"], "seq_len", page["seq_len"], "free", page["pool"]["free"])

    page = new_page_cache(llama, n_pages=8, block_size=4)
    paged_ids = generate(paged, llama_ids, max_new_tokens=10, eos_id=eos, cache=page)
    cached_ids = generate(cached, llama_ids, max_new_tokens=10, eos_id=eos, cache=[])
    print(torch.equal(paged_ids, cached_ids))
    print("blocks", page["blocks"], "seq_len", page["seq_len"], "free", page["pool"]["free"])

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
    print(torch.equal(france_out, france_serial), torch.equal(germany_out, germany_serial))
    print("france", france["blocks"], france["seq_len"], "germany", germany["blocks"], germany["seq_len"])

    release_pages(france)
    print("after release", "france", france["blocks"], "germany", germany["blocks"], "free", pool["free"])

    cat = new_sequence(pool)
    cat_logits, cat = paged(cat_ids, cat)
    cat_ref = uncached(cat_ids)
    print("cat_prefill_err", (cat_logits - cat_ref).abs().max().item())
    print("cat", cat["blocks"], "germany", germany["blocks"], "free", pool["free"])

    germany_out = generate(paged, germany_out, max_new_tokens=1, eos_id=eos, cache=germany)
    germany_serial = generate(cached, germany_ids, max_new_tokens=5, eos_id=eos, cache=[])
    print(torch.equal(germany_out, germany_serial))
    print("germany", germany["blocks"], germany["seq_len"], "free", pool["free"])


def run_continuous_paged(texts, budgets):
    rows = [llama_tok(p, return_tensors="pt").input_ids[0] for p in texts]

    with torch.no_grad():
        serial = [generate(cached, row.view(1, -1), max_new_tokens=b, eos_id=eos, cache=[]) for row, b in zip(rows, budgets)]
        cont_pool = new_page_pool(llama, n_pages=16, block_size=4)
        cont = generate_continuous_paged(paged, paged_batch, cont_pool, rows, budgets, eos, max_batch=2)

    for i, s in enumerate(serial):
        print(torch.equal(cont[i], s[0]), llama_tok.decode(s[0]))

    print("free", cont_pool["free"])


run_continuous_paged(["The capital of France is", "The cat", "The capital of Germany is"], [6, 2, 4])
run_continuous_paged(["The capital of France is", "The cat", "Once upon a time there was a"], [6, 2, 4])