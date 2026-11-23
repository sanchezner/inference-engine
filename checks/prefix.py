import torch
from functools import partial

from checks.oracle import load_engine
from engine.kv.prefix_cache import attach_prefix_cache
from engine.paged import forward_paged, forward_paged_batch, new_page_pool
from engine.sched.engine_core import EngineCore
from engine.sched.request import Request


# A hit has to match a miss in fp32, where greedy tokens are exact.
PREFIX = tuple(range(1, 13))
LONGER = PREFIX + (40, 41, 42, 43)
OTHER = tuple(range(80, 92))
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"


def main():
    cfg, weights, rope = load_engine("cpu", torch.float32)
    one = partial(forward_paged, weights, cfg, rope)
    batch = partial(forward_paged_batch, weights, cfg, rope)
    prompts = [PREFIX, LONGER, OTHER]
    with torch.no_grad():
        cold = _run(cfg, one, batch, prompts, cache=False)
        warm = _run(cfg, one, batch, prompts, cache=True)
    assert cold["tokens"] == warm["tokens"]
    assert warm["cached"] == {"0": 0, "1": 12, "2": 0}
    assert warm["hit_tokens"] == 12
    assert warm["query_tokens"] == sum(len(prompt) for prompt in prompts)
    assert warm["blocks_free"] == warm["blocks_total"]
    assert warm["free_list"] < warm["blocks_total"]


def _run(cfg, one, batch, prompts, cache):
    pool = new_page_pool(cfg, n_pages=32, block_size=4, device="cpu", dtype=torch.float32)
    if cache:
        attach_prefix_cache(pool, MODEL_ID)
    engine = EngineCore(one, batch, pool, max_batch=1)
    for index, prompt in enumerate(prompts):
        engine.add_request(Request(str(index), prompt, max_new_tokens=4))
    tokens = {str(index): [] for index in range(len(prompts))}
    cached = {}
    for _ in range(64):
        outs = engine.step()
        for out in outs:
            if out.request_id not in cached:
                cached[out.request_id] = out.cached_tokens
            if out.token_id is not None:
                tokens[out.request_id].append(out.token_id)
        stats = engine.stats()
        if stats.requests_waiting == 0 and stats.requests_running == 0:
            break
    else:
        raise AssertionError("prefix check did not finish")
    stats = engine.stats()
    return {
        "tokens": tokens,
        "cached": cached,
        "hit_tokens": stats.counters.prefix_cache_hit_tokens,
        "query_tokens": stats.counters.prefix_cache_query_tokens,
        "blocks_free": stats.blocks_free,
        "blocks_total": stats.blocks_total,
        "free_list": len(pool.free),
    }


if __name__ == "__main__":
    main()
