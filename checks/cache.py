import time
import torch
from functools import partial
from checks.oracle import load_engine, load_smollm
from checks.tolerances import LLAMA_LOGIT_ATOL, assert_logits
from engine.generate import generate
from engine.legacy.llama import forward_llama_cached
from engine.model import forward


def main():
    torch.manual_seed(0)
    llama_tok, llama = load_smollm()
    cfg, weights, rope = load_engine("cpu", torch.float32)
    uncached = partial(forward, weights, cfg, rope=rope)
    cached = partial(forward_llama_cached, llama)
    llama_ids = llama_tok("The capital of France is", return_tensors="pt").input_ids
    eos = llama.config.eos_token_id

    with torch.no_grad():
        ref_logits = llama(llama_ids).logits[:, -1]
        ref_ids = llama.generate(llama_ids, max_new_tokens=10, do_sample=False, pad_token_id=eos)

        prefill_logits, prefill_cache = cached(llama_ids, [])
        assert_logits("prefill", prefill_logits, ref_logits, LLAMA_LOGIT_ATOL)
        assert tuple(prefill_cache[0][0].shape) == (1, 3, 5, 64)
        assert len(prefill_cache) == 30

        t0 = time.perf_counter()
        uncached_ids = generate(uncached, llama_ids, max_new_tokens=10, eos_id=eos)
        uncached_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        cached_ids = generate(cached, llama_ids, max_new_tokens=10, eos_id=eos, cache=[])
        cached_s = time.perf_counter() - t0

    assert torch.equal(cached_ids, ref_ids)
    assert torch.equal(cached_ids, uncached_ids)
    return {"uncached_generate_10_s": uncached_s, "cached_generate_10_s": cached_s}


if __name__ == "__main__":
    main()
