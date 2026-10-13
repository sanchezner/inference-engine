import time
import torch
from functools import partial
from engine.load import load_smollm
from engine.llama import forward_llama, forward_llama_cached
from engine.generate import generate

torch.manual_seed(0)
llama_tok, llama = load_smollm()
uncached = partial(forward_llama, llama)
cached = partial(forward_llama_cached, llama)
llama_ids = llama_tok("The capital of France is", return_tensors="pt").input_ids
eos = llama.config.eos_token_id

with torch.no_grad():
    my_llama_logits = uncached(llama_ids)
    ref_ids = llama.generate(llama_ids, max_new_tokens=10, do_sample=False, pad_token_id=eos)

    prefill_logits, prefill_cache = cached(llama_ids, [])
    print("prefill_logit_max_abs_err", (prefill_logits - my_llama_logits).abs().max().item())
    print("cache_k_shape", tuple(prefill_cache[0][0].shape), "layers", len(prefill_cache))

    generate(cached, llama_ids, max_new_tokens=10, eos_id=eos, cache=[])

    t0 = time.perf_counter()
    uncached_ids = generate(uncached, llama_ids, max_new_tokens=10, eos_id=eos)
    uncached_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    cached_ids = generate(cached, llama_ids, max_new_tokens=10, eos_id=eos, cache=[])
    cached_s = time.perf_counter() - t0

print(torch.equal(cached_ids, ref_ids))
print(torch.equal(cached_ids, uncached_ids))
print("uncached_s", uncached_s, "cached_s", cached_s)