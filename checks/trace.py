import torch
from functools import partial

from bench.e2e.trace import TraceRequest, replay
from checks.oracle import load_engine
from engine.generate import generate
from engine.paged import forward_paged, forward_paged_batch, new_page_cache, new_page_pool
from engine.sched.engine_core import EngineCore


# Milestone acceptance for the replayer: real weights, not the stub.
N = 20
ABORTED = 5
MAX_NEW = 4
BLOCK = 16


def main():
    cfg, weights, rope = load_engine("cpu", torch.float32)
    one = partial(forward_paged, weights, cfg, rope)
    batch = partial(forward_paged_batch, weights, cfg, rope)
    pool = new_page_pool(cfg, n_pages=8, block_size=BLOCK, device="cpu", dtype=torch.float32)
    engine = EngineCore(one, batch, pool, max_batch=4)
    prompts = [(i + 1, i + 3, 7) for i in range(N)]
    trace = [TraceRequest(0.0, prompt, MAX_NEW, str(i)) for i, prompt in enumerate(prompts)]
    abort_ids = {str(i) for i in range(ABORTED)}

    with torch.no_grad():
        summary, by_request = replay(engine, trace, abort_ids=abort_ids)
        for i, prompt in enumerate(prompts):
            rid = str(i)
            got = [out.token_id for out in by_request[rid] if out.token_id is not None]
            offline = _greedy(one, prompt, cfg)
            if rid in abort_ids:
                assert by_request[rid][-1].finish_reason == "abort"
                assert by_request[rid][-1].token_id is None
                assert 0 < len(got) < len(offline)
                assert got == offline[:len(got)]
            else:
                assert got == offline
                assert by_request[rid][-1].finish_reason == "length"

    stats = engine.stats()
    assert stats.requests_waiting == 0 and stats.requests_running == 0
    assert stats.blocks_free == stats.blocks_total
    assert summary["finished"] == N - ABORTED
    assert summary["aborted"] == ABORTED
    assert summary["rejected"] == 0
    assert stats.counters.requests_finished == N - ABORTED
    assert stats.counters.requests_aborted == ABORTED


def _greedy(forward, prompt, cfg):
    cache = new_page_cache(cfg, n_pages=4, block_size=BLOCK, device="cpu", dtype=torch.float32)
    ids = torch.tensor(prompt, dtype=torch.long).view(1, -1)
    # eos -1 never matches, so this runs out the budget the same way the engine does.
    out = generate(forward, ids, max_new_tokens=MAX_NEW, eos_id=-1, cache=cache)
    return out[0, len(prompt):].tolist()


if __name__ == "__main__":
    main()
