import time
import torch
from functools import partial
from checks.oracle import load_smollm
from engine.legacy.llama import forward_llama_cached
from engine.legacy.cache import left_pad
from engine.generate import generate
from engine.legacy.generate import generate_batch
from engine.legacy.schedule import generate_continuous
from checks.tolerances import LLAMA_LOGIT_ATOL, assert_logits


def _equal_rows(batched, serial_ids, pad_lens=None):
    for i, serial in enumerate(serial_ids):
        start = 0 if pad_lens is None else int(pad_lens[i])
        stop = serial.shape[1]
        assert torch.equal(batched[i, start:start + stop], serial[0])


def main():
    torch.manual_seed(0)
    llama_tok, llama = load_smollm()
    cached = partial(forward_llama_cached, llama)
    llama_ids = llama_tok("The capital of France is", return_tensors="pt").input_ids
    eos = llama.config.eos_token_id
    timings = {}

    with torch.no_grad():
        my_llama_logits = llama(llama_ids).logits[0, -1]

        prompts = [
            "The capital of France is",
            "The capital of Germany is",
            "The capital of Spain is",
            "The capital of Italy is",
        ]
        batch_ids = torch.cat([llama_tok(p, return_tensors="pt").input_ids for p in prompts], dim=0)
        prefill_b, cache_b = cached(batch_ids, [])
        assert tuple(prefill_b.shape) == (4, 49152)
        assert tuple(cache_b[0][0].shape) == (4, 3, 5, 64)
        assert_logits("row0", prefill_b[0], my_llama_logits, LLAMA_LOGIT_ATOL)

        t0 = time.perf_counter()
        serial_ids = [generate(cached, row.view(1, -1), max_new_tokens=10, eos_id=eos, cache=[]) for row in batch_ids]
        timings["serial_batch_4x10_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        batched_ids = generate_batch(cached, batch_ids, max_new_tokens=10, eos_id=eos, cache=[])
        timings["static_batch_4x10_s"] = time.perf_counter() - t0
        _equal_rows(batched_ids, serial_ids)

        mixed_rows = [llama_tok(p, return_tensors="pt").input_ids[0] for p in (
            "The capital of France is",
            "The capital of Germany is",
            "The cat",
        )]
        mixed_ids, pad_lens = left_pad(mixed_rows, eos)
        prefill_m, _ = cached(mixed_ids, [], pad_lens)
        cat_logits = llama(mixed_rows[2].view(1, -1)).logits[0, -1]
        assert_logits("france_pad", prefill_m[0], my_llama_logits, LLAMA_LOGIT_ATOL)
        assert_logits("cat_pad", prefill_m[2], cat_logits, LLAMA_LOGIT_ATOL)

        serial_m = [generate(cached, row.view(1, -1), max_new_tokens=10, eos_id=eos, cache=[]) for row in mixed_rows]
        batched_m = generate_batch(cached, mixed_ids, max_new_tokens=10, eos_id=eos, cache=[], pad_lens=pad_lens)
        _equal_rows(batched_m, serial_m, pad_lens)

    timings.update(_continuous(cached, llama_tok, eos, "The capital of France is", "The cat", "The capital of Germany is", "a"))
    timings.update(_continuous(cached, llama_tok, eos, "The capital of France is", "The cat", "Once upon a time there was a", "b"))
    return timings


def _continuous(cached, llama_tok, eos, first, second, third, tag):
    texts = [first, second, third]
    budgets = [6, 2, 4]
    rows = [llama_tok(p, return_tensors="pt").input_ids[0] for p in texts]
    with torch.no_grad():
        t0 = time.perf_counter()
        serial = [generate(cached, row.view(1, -1), max_new_tokens=b, eos_id=eos, cache=[]) for row, b in zip(rows, budgets)]
        serial_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        cont = generate_continuous(cached, rows, budgets, eos, max_batch=2)
        cont_s = time.perf_counter() - t0
    for got, ref in zip(cont, serial):
        assert torch.equal(got, ref[0])
    return {f"continuous_{tag}_serial_s": serial_s, f"continuous_{tag}_s": cont_s}


if __name__ == "__main__":
    main()
