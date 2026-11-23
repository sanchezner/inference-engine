# Same burst, cache off then on. The TTFT gap is the shared prefix, not a different arrival pattern.

import argparse
import json
import statistics
import sys
from datetime import date
from functools import partial
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from bench.env import environment, nvidia_smi, snapshot_revision
from bench.e2e.trace import replay, shared_prefix_trace
from checks.oracle import load_engine
from engine.kv.prefix_cache import attach_prefix_cache
from engine.paged import forward_paged, forward_paged_batch, new_page_pool, pages_for
from engine.sched.engine_core import EngineCore


MODEL = "HuggingFaceTB/SmolLM2-1.7B"
# 512 divides the serving block, so the whole system prompt is shareable and the tail is not.
PREFIX_LEN = 512
SUFFIX_LEN = 32
MAX_TOKENS = 8
REQUESTS = 8
MAX_BATCH = 8
BLOCK_SIZE = 16
SEED = 0
WARMUP = 1
REPS = 5

def _methodology(args) -> str:
    prompt = args.prefix_len + args.suffix_len
    return (
        "Open-loop burst through EngineCore on CUDA bf16. "
        f"{args.requests} requests arrive together at t=0. "
        f"Each prompt is one shared {args.prefix_len}-token prefix plus a {args.suffix_len}-token tail; "
        "the first tail token is the request index, so two tails cannot collide and share a block past the prefix. "
        f"max_new_tokens is {args.max_tokens}, max_batch is {args.max_batch}, block size is {args.block_size}. "
        "The pool is max_batch times one request's reserve-max budget, so the cache-off run fits without a rejection "
        "and the cache-on run is not faster merely because the cold run would not fit. "
        f"One warmup replay of each mode is discarded. {args.reps} timed repetitions alternate which mode runs first. "
        "Each replay builds a new pool, so the cache starts empty and the first request is always a miss. "
        "TTFT is the replayer's stamp, arrival to the first generated token, taken at the end of that request's prefill "
        "and before the step decodes. p50 and p99 are nearest-rank over the requests in one replay. "
        "The reported TTFT is the median of those p50s, with the min and max. "
        f"Prefill tokens scored are all {prompt} prompt tokens of every request with the cache off, "
        "and query tokens minus hit tokens with it on. "
        "A hit still gathers the cached pages to score the tail. The skipped work is the prefix's projections and MLP. "
        "Clocks are not locked. The card also drives the display. Host is WSL2. HTTP is not in these numbers."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Shared-prefix TTFT, cache on versus cache off.")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--prefix-len", type=int, default=PREFIX_LEN)
    parser.add_argument("--suffix-len", type=int, default=SUFFIX_LEN)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--requests", type=int, default=REQUESTS)
    parser.add_argument("--max-batch", type=int, default=MAX_BATCH)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("bench-prefix needs CUDA")

    cfg, weights, rope = load_engine("cuda", torch.bfloat16, name=args.model)
    prompt_len = args.prefix_len + args.suffix_len
    per_request = pages_for(args.block_size, prompt_len + args.max_tokens - 1)
    pages = args.max_batch * per_request
    trace = shared_prefix_trace(
        args.requests,
        args.prefix_len,
        args.suffix_len,
        args.max_tokens,
        args.seed,
        vocab=cfg.vocab_size,
    )
    aligned = (args.prefix_len // args.block_size) * args.block_size
    expected_hit = (args.requests - 1) * aligned
    expected_query = args.requests * prompt_len
    one = partial(forward_paged, weights, cfg, rope)
    batch = partial(forward_paged_batch, weights, cfg, rope)

    def once(cache):
        pool = new_page_pool(cfg, pages, args.block_size, device="cuda", dtype=torch.bfloat16)
        if cache:
            attach_prefix_cache(pool, args.model)
        engine = EngineCore(one, batch, pool, args.max_batch)
        torch.cuda.synchronize()
        summary, by_request = replay(engine, trace)
        torch.cuda.synchronize()
        stats = engine.stats()
        hit = stats.counters.prefix_cache_hit_tokens
        query = stats.counters.prefix_cache_query_tokens
        cached_sum = _first_cached(by_request)
        if summary["finished"] != args.requests or summary["rejected"] or summary["aborted"]:
            raise RuntimeError(f"trace did not finish cleanly: {summary}")
        if cached_sum != hit:
            raise RuntimeError(f"first-output cached_tokens sum {cached_sum} != hit counter {hit}")
        if cache and (hit != expected_hit or query != expected_query):
            raise RuntimeError(
                f"cache hit {hit} tokens over {query} queries, expected {expected_hit} over {expected_query}"
            )
        if not cache and (hit or query or cached_sum):
            raise RuntimeError(f"cache-off run recorded hits ({hit} hit, {query} query)")
        scored = (query - hit) if cache else expected_query
        return {
            "ttft_p50_s": summary["ttft_s"]["p50"],
            "ttft_p99_s": summary["ttft_s"]["p99"],
            "itl_p50_s": summary["itl_s"]["p50"],
            "elapsed_s": summary["elapsed_s"],
            "tokens": summary["tokens"],
            "hit_tokens": hit,
            "query_tokens": query,
            "prefill_tokens_scored": scored,
            "per_request": _per_request(by_request),
        }

    for _ in range(WARMUP):
        once(False)
        once(True)
    clocks = nvidia_smi()
    runs = {"off": [], "on": []}
    for rep in range(args.reps):
        # Odd reps run the cache first, so a warm card does not always land on the same side.
        order = (False, True) if rep % 2 == 0 else (True, False)
        for cache in order:
            runs["on" if cache else "off"].append(once(cache))

    off = _column(runs["off"], expected_query)
    on = _column(runs["on"], expected_query - expected_hit)
    off_p50 = off["ttft_p50_s"]["median"]
    on_p50 = on["ttft_p50_s"]["median"]
    report = {
        "milestone": "M3b",
        "date": date.today().isoformat(),
        "host": "WSL2",
        "command": _command(args),
        "methodology": _methodology(args),
        "model": args.model,
        "model_revision": snapshot_revision(args.model),
        "dtype": "bfloat16",
        "device": "cuda",
        "block_size": args.block_size,
        "prefix_len": args.prefix_len,
        "suffix_len": args.suffix_len,
        "max_tokens": args.max_tokens,
        "requests": args.requests,
        "max_batch": args.max_batch,
        "pages": pages,
        "pages_per_request_cold": per_request,
        "seed": args.seed,
        "reps": args.reps,
        "warmup_discarded": WARMUP,
        "expected_hit_tokens": expected_hit,
        "expected_query_tokens": expected_query,
        "environment": environment(),
        "nvidia_smi_after_warmup": clocks,
        "off": off,
        "on": on,
        "ttft_p50_reduction": (off_p50 - on_p50) / off_p50,
        "ttft_p99_reduction": (off["ttft_p99_s"]["median"] - on["ttft_p99_s"]["median"]) / off["ttft_p99_s"]["median"],
        "prefill_tokens_skipped": expected_hit,
    }
    text = json.dumps(report, indent=2) + "\n"
    out = args.out
    if out is None:
        out = _ROOT / "bench" / "results" / f"{report['date']}-prefix.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    if on_p50 >= off_p50:
        raise SystemExit(f"cache did not cut TTFT p50: on {on_p50:.6f}s, off {off_p50:.6f}s")


def _command(args) -> str:
    default = (
        args.model == MODEL
        and args.prefix_len == PREFIX_LEN
        and args.suffix_len == SUFFIX_LEN
        and args.max_tokens == MAX_TOKENS
        and args.requests == REQUESTS
        and args.max_batch == MAX_BATCH
        and args.block_size == BLOCK_SIZE
        and args.seed == SEED
        and args.reps == REPS
        and args.out is None
    )
    if default:
        return "make bench-prefix"
    return " ".join(sys.argv)


def _per_request(by_request):
    rows = []
    for rid, outs in by_request.items():
        first = next(out for out in outs if out.token_id is not None)
        rows.append((int(rid), first.timestamps.emitted - first.timestamps.arrival, first.cached_tokens))
    rows.sort()
    return [{"id": str(rid), "ttft_s": ttft, "cached_tokens": cached} for rid, ttft, cached in rows]


def _first_cached(by_request) -> int:
    total = 0
    for outs in by_request.values():
        first = next(out for out in outs if out.token_id is not None)
        total += first.cached_tokens
        for out in outs:
            if out is not first and out.cached_tokens:
                raise RuntimeError(f"request {out.request_id} carried cached_tokens after its first output")
    return total


def _column(runs, scored):
    for run in runs:
        if run["prefill_tokens_scored"] != scored:
            raise RuntimeError(f"prefill tokens scored {run['prefill_tokens_scored']} != {scored}")
    return {
        "ttft_p50_s": _spread(run["ttft_p50_s"] for run in runs),
        "ttft_p99_s": _spread(run["ttft_p99_s"] for run in runs),
        "elapsed_s": _spread(run["elapsed_s"] for run in runs),
        "hit_tokens": runs[0]["hit_tokens"],
        "query_tokens": runs[0]["query_tokens"],
        "prefill_tokens_scored": scored,
        "runs": runs,
    }


def _spread(values):
    values = list(values)
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


if __name__ == "__main__":
    main()
