import time
import torch
from functools import partial
from engine.load import load_smollm
from engine.llama import forward_llama, forward_llama_cached
from engine.cache import left_pad
from engine.generate import generate, generate_batch
from engine.schedule import generate_continuous


torch.manual_seed(0)
llama_tok, llama = load_smollm()
uncached = partial(forward_llama, llama)
cached = partial(forward_llama_cached, llama)
llama_ids = llama_tok("The capital of France is", return_tensors="pt").input_ids
eos = llama.config.eos_token_id

with torch.no_grad():
    my_llama_logits = uncached(llama_ids)

prompts = [
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Spain is",
    "The capital of Italy is",
]

batch_ids = torch.cat([llama_tok(p, return_tensors="pt").input_ids for p in prompts], dim=0)

with torch.no_grad():
    prefill_b, cache_b = cached(batch_ids, [])
    print(tuple(prefill_b.shape), tuple(cache_b[0][0].shape))
    print("row0_max_abs_err", (prefill_b[0] - my_llama_logits).abs().max().item())

    t0 = time.perf_counter()
    serial_ids = [generate(cached, row.view(1, -1), max_new_tokens=10, eos_id=eos, cache=[]) for row in batch_ids]
    serial_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    batched_ids = generate_batch(cached, batch_ids, max_new_tokens=10, eos_id=eos, cache=[])
    batched_s = time.perf_counter() - t0

for i, serial in enumerate(serial_ids):
    stop = serial.shape[1]
    print(torch.equal(batched_ids[i, :stop], serial[0]), llama_tok.decode(serial[0]))

print("serial_s", serial_s, "batched_s", batched_s)

mixed_prompts = [
    "The capital of France is",
    "The capital of Germany is",
    "The cat",
]

mixed_rows = [llama_tok(p, return_tensors="pt").input_ids[0] for p in mixed_prompts]
mixed_ids, pad_lens = left_pad(mixed_rows, eos)

with torch.no_grad():
    prefill_m, _ = cached(mixed_ids, [], pad_lens)
    cat_logits = uncached(mixed_rows[2].view(1, -1))
    print("france_pad_err", (prefill_m[0] - my_llama_logits).abs().max().item())
    print("cat_pad_err", (prefill_m[2] - cat_logits).abs().max().item())

    serial_m = [generate(cached, row.view(1, -1), max_new_tokens=10, eos_id=eos, cache=[]) for row in mixed_rows]
    batched_m = generate_batch(cached, mixed_ids, max_new_tokens=10, eos_id=eos, cache=[], pad_lens=pad_lens)

for i, serial in enumerate(serial_m):
    start = int(pad_lens[i])
    stop = serial.shape[1]
    print(torch.equal(batched_m[i, start:start + stop], serial[0]), llama_tok.decode(serial[0]))


def run_continuous(texts, budgets):
    rows = [llama_tok(p, return_tensors="pt").input_ids[0] for p in texts]

    with torch.no_grad():
        t0 = time.perf_counter()
        serial = [generate(cached, row.view(1, -1), max_new_tokens=b, eos_id=eos, cache=[]) for row, b in zip(rows, budgets)]
        serial_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        cont = generate_continuous(cached, rows, budgets, eos, max_batch=2)
        cont_s = time.perf_counter() - t0
    
    for i, s in enumerate(serial):
        print(torch.equal(cont[i], s[0]), llama_tok.decode(s[0]))

    print("serial_c_s", serial_s, "cont_s", cont_s)


run_continuous(["The capital of France is", "The cat", "The capital of Germany is"], [6, 2, 4])
run_continuous(["The capital of France is", "The cat", "Once upon a time there was a"], [6, 2, 4])