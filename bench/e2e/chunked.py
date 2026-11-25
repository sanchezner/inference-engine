# Same burst, whole prompt then a token budget. The short-chat gap is the long prefill, not a different arrival pattern.

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
from bench.e2e.trace import mixed_trace, percentile, replay
from checks.oracle import load_engine
from engine.paged import forward_paged, forward_paged_batch, new_page_pool, pages_for
from engine.sched.engine_core import EngineCore


MODEL = "HuggingFaceTB/SmolLM2-1.7B"
SHORT_REQUESTS = 4
SHORT_LEN = 8
SHORT_TOKENS = 32
LONG_REQUESTS = 2
LONG_LEN = 2048
LONG_TOKENS = 4
PREFILL_BUDGET = 256
MAX_BATCH = 8
BLOCK_SIZE = 16
SEED = 0
WARMUP = 1
REPS = 5


def _methodology(args) -> str:
    return (
        "Open-loop burst through EngineCore on CUDA bf16. "
        f"{args.short_requests} short chats and {args.long_requests} long prompts arrive together at t=0. "
        "The chats are listed first, so admission seats them before a long prompt. "
        f"A chat prompt is {args.short_len} tokens and asks for {args.short_tokens} new tokens. "
        f"A long prompt is {args.long_len} tokens and asks for {args.long_tokens}. "
        f"max_batch is {args.max_batch}, block size is {args.block_size}, and the growth margin is one page. "
        "The pool is every request's full page demand plus one spare page per batch slot, so both modes fit "
        "without preemption and the budgeted run is not faster because the other ran out of pages. "
        "The prefix cache is off. "
        "Whole mode scores a prompt in the step that seats it. "
        f"Chunked mode scores at most {args.prefill_budget} prompt tokens per step, after decode, "
        "and a prompt already in progress spends that budget before a new one is seated. "
        f"One warmup replay of each mode is discarded. {args.reps} timed repetitions alternate which mode runs first. "
        "Each replay builds a new pool. "
        "Inter-token latency is the gap between consecutive generated tokens of one request, stamped when the token is sampled. "
        "p50 and p99 are nearest-rank over one replay. "
        "The short-chat figures use only the chats. A long prompt's own gaps are decode steps and would hide the stall. "
        "The reported value is the median of those per-replay percentiles, with the min and max. "
        "A later chunk still gathers the pages it already holds, so a chunk costs more than a fresh prefill of that many tokens. "
        "Clocks are not locked. The card also drives the display. Host is WSL2. HTTP is not in these numbers."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Short-chat inter-token latency, whole prompt versus a prefill budget.")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--short-requests", type=int, default=SHORT_REQUESTS)
    parser.add_argument("--short-len", type=int, default=SHORT_LEN)
    parser.add_argument("--short-tokens", type=int, default=SHORT_TOKENS)
    parser.add_argument("--long-requests", type=int, default=LONG_REQUESTS)
    parser.add_argument("--long-len", type=int, default=LONG_LEN)
    parser.add_argument("--long-tokens", type=int, default=LONG_TOKENS)
    parser.add_argument("--prefill-budget", type=int, default=PREFILL_BUDGET)
    parser.add_argument("--max-batch", type=int, default=MAX_BATCH)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("bench-chunked needs CUDA")
    if args.prefill_budget >= args.long_len:
        raise SystemExit("prefill budget must be shorter than a long prompt")
    if args.short_requests + args.long_requests > args.max_batch:
        raise SystemExit("max_batch must seat every request, or the long prompt waits for a chat to finish")

    cfg, weights, rope = load_engine("cuda", torch.bfloat16, name=args.model)
    short_ids = [f"s{i}" for i in range(args.short_requests)]
    long_ids = [f"l{i}" for i in range(args.long_requests)]
    n_requests = args.short_requests + args.long_requests
    pages = _pages(args)
    trace = mixed_trace(
        args.short_requests,
        args.short_len,
        args.short_tokens,
        args.long_requests,
        args.long_len,
        args.long_tokens,
        args.seed,
        vocab=cfg.vocab_size,
    )
    forward_one = partial(forward_paged, weights, cfg, rope)
    forward_batch = partial(forward_paged_batch, weights, cfg, rope)
    expected = {}

    def once(budget):
        pool = new_page_pool(cfg, pages, args.block_size, device="cuda", dtype=torch.bfloat16)
        calls = {"n": 0}

        def one(ids, seq):
            calls["n"] += 1
            return forward_one(ids, seq)

        engine = EngineCore(one, forward_batch, pool, args.max_batch, prefill_budget=budget)
        torch.cuda.synchronize()
        summary, by_request = replay(engine, trace)
        torch.cuda.synchronize()
        stats = engine.stats()
        if summary["finished"] != n_requests or summary["rejected"] or summary["aborted"]:
            raise RuntimeError(f"trace did not finish cleanly: {summary}")
        if stats.blocks_free != stats.blocks_total:
            raise RuntimeError(f"{stats.blocks_free} pages free of {stats.blocks_total}")
        tokens = _tokens(by_request)
        _check_lengths(tokens, short_ids, args.short_tokens)
        _check_lengths(tokens, long_ids, args.long_tokens)
        if not expected:
            expected.update(tokens)
        elif tokens != expected:
            raise RuntimeError("tokens diverged from the whole-prompt run")
        if budget is None and calls["n"] != n_requests:
            raise RuntimeError(
                f"whole run scored a prompt {calls['n']} times for {n_requests} requests; a second pass is a recompute"
            )
        if budget is not None and calls["n"] <= n_requests:
            raise RuntimeError(f"budgeted run scored prompts {calls['n']} times, which did not split the long prompt")
        short_ttft, _, short_p50, short_p99 = _latency(by_request, short_ids)
        long_ttft, _, _, _ = _latency(by_request, long_ids)
        return {
            "short_itl_p50_s": short_p50,
            "short_itl_p99_s": short_p99,
            "short_ttft_p50_s": short_ttft,
            "long_ttft_p50_s": long_ttft,
            "itl_p50_s": summary["itl_s"]["p50"],
            "itl_p99_s": summary["itl_s"]["p99"],
            "elapsed_s": summary["elapsed_s"],
            "tokens": summary["tokens"],
            "prefill_forwards": calls["n"],
        }

    for _ in range(WARMUP):
        once(None)
        once(args.prefill_budget)
    clocks = nvidia_smi()
    runs = {"whole": [], "chunked": []}
    for rep in range(args.reps):
        # Odd reps run the budget first, so a warm card does not always land on the same side.
        order = (None, args.prefill_budget) if rep % 2 == 0 else (args.prefill_budget, None)
        for budget in order:
            runs["chunked" if budget else "whole"].append(once(budget))

    whole = _column(runs["whole"])
    chunked = _column(runs["chunked"])
    whole_p99 = whole["short_itl_p99_s"]["median"]
    chunked_p99 = chunked["short_itl_p99_s"]["median"]
    report = {
        "part": "chunked-prefill",
        "date": date.today().isoformat(),
        "host": "WSL2",
        "command": _command(args),
        "methodology": _methodology(args),
        "model": args.model,
        "model_revision": snapshot_revision(args.model),
        "dtype": "bfloat16",
        "device": "cuda",
        "block_size": args.block_size,
        "prefill_budget": args.prefill_budget,
        "short_requests": args.short_requests,
        "short_len": args.short_len,
        "short_tokens": args.short_tokens,
        "long_requests": args.long_requests,
        "long_len": args.long_len,
        "long_tokens": args.long_tokens,
        "max_batch": args.max_batch,
        "pages": pages,
        "seed": args.seed,
        "reps": args.reps,
        "warmup_discarded": WARMUP,
        "environment": environment(),
        "nvidia_smi_after_warmup": clocks,
        "whole": whole,
        "chunked": chunked,
        "short_itl_p99_reduction": (whole_p99 - chunked_p99) / whole_p99,
    }
    text = json.dumps(report, indent=2) + "\n"
    out = args.out
    if out is None:
        out = _ROOT / "bench" / "results" / f"{report['date']}-chunked.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    if chunked_p99 >= whole_p99:
        raise SystemExit(
            f"budget did not cut short-chat inter-token p99: chunked {chunked_p99:.6f}s, whole {whole_p99:.6f}s"
        )


def _pages(args) -> int:
    shorts = args.short_requests * pages_for(args.block_size, args.short_len + args.short_tokens - 1)
    longs = args.long_requests * pages_for(args.block_size, args.long_len + args.long_tokens - 1)
    # The spare covers the one page each running request keeps in reserve.
    return shorts + longs + args.max_batch


def _tokens(by_request):
    return {
        rid: [out.token_id for out in outs if out.token_id is not None]
        for rid, outs in by_request.items()
    }


def _check_lengths(tokens, ids, n):
    for rid in ids:
        if len(tokens[rid]) != n:
            raise RuntimeError(f"request {rid} produced {len(tokens[rid])} tokens, expected {n}")


def _latency(by_request, ids):
    ttft = []
    gaps = []
    for rid in ids:
        outs = by_request[rid]
        times = [out.timestamps.emitted for out in outs if out.token_id is not None]
        ttft.append(times[0] - outs[0].timestamps.arrival)
        gaps.extend(b - a for a, b in zip(times, times[1:]))
    return percentile(ttft, 50), percentile(ttft, 99), percentile(gaps, 50), percentile(gaps, 99)


def _column(runs):
    forwards = {run["prefill_forwards"] for run in runs}
    if len(forwards) != 1:
        raise RuntimeError(f"prefill forwards changed across repetitions: {sorted(forwards)}")
    keys = (
        "short_itl_p50_s",
        "short_itl_p99_s",
        "short_ttft_p50_s",
        "long_ttft_p50_s",
        "itl_p50_s",
        "itl_p99_s",
        "elapsed_s",
    )
    column = {key: _spread(run[key] for run in runs) for key in keys}
    column["prefill_forwards"] = runs[0]["prefill_forwards"]
    column["runs"] = runs
    return column


def _spread(values):
    values = list(values)
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def _command(args) -> str:
    default = (
        args.model == MODEL
        and args.short_requests == SHORT_REQUESTS
        and args.short_len == SHORT_LEN
        and args.short_tokens == SHORT_TOKENS
        and args.long_requests == LONG_REQUESTS
        and args.long_len == LONG_LEN
        and args.long_tokens == LONG_TOKENS
        and args.prefill_budget == PREFILL_BUDGET
        and args.max_batch == MAX_BATCH
        and args.block_size == BLOCK_SIZE
        and args.seed == SEED
        and args.reps == REPS
        and args.out is None
    )
    if default:
        return "make bench-chunked"
    return " ".join(sys.argv)


if __name__ == "__main__":
    main()
