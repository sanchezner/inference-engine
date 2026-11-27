# Decode through EngineCore, graphs on against graphs off. The ceiling is the whole step.

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from datetime import date
from functools import partial
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.profiler import ProfilerActivity, profile

from bench.env import environment, nvidia_smi, snapshot_revision
from bench.e2e.trace import TraceRequest, replay
from bench.roofline import decode_ceiling_tokens_per_s, decode_step_bytes
from checks.oracle import load_engine
from engine.paged import forward_paged, forward_paged_batch, new_page_pool, pages_for
from engine.sched.engine_core import EngineCore
from engine.sched.request import Request


MODEL = "HuggingFaceTB/SmolLM2-1.7B"
PROMPT_LEN = 128
MAX_TOKENS = 32
BATCHES = (1, 8)
BLOCK_SIZE = 16
SEED = 0
WARMUP = 1
REPS = 5
BANDWIDTH_FILE = "bench/results/2026-10-31-bandwidth.json"


def _methodology(args) -> str:
    mid = args.prompt_len + (args.max_tokens - 1) // 2
    return (
        "Open-loop burst through EngineCore on CUDA bf16, greedy. "
        f"A case is {args.prompt_len}-token prompts, {args.max_tokens} new tokens, block size {args.block_size}. "
        "Every request arrives at t=0. max_batch equals the case batch, so after the first step they decode together. "
        "Graphs off is the eager decode forward. Graphs on passes DecodeGraphs into that forward. "
        "The pool is one spare page per request past the final length, and at least 32 pages, "
        "so a table-width bucket is not clipped by a tiny pool and the run does not preempt. "
        "One warmup replay per mode is discarded. It also captures the graph: the first decode already spans "
        "nine pages, which buckets to 16, and the run never reaches a wider bucket. "
        f"{args.reps} timed replays follow on the same pool and the same runner. "
        "Decode tokens/s is batch / inter-token p50. The requests share each decode step, so that p50 is the step. "
        "p50 is nearest-rank over one replay. The reported value is the median of those p50s, with the min and max. "
        f"The ceiling is decode_step_bytes at context {mid}, the cached length before the middle decode step, "
        "over the median copy in bench/results/2026-10-31-bandwidth.json. "
        "Context at the first decode is the prompt length; at the last it is the prompt plus the new tokens, minus one. "
        "The trace is the second step of a fresh burst after those replays: the first step prefills, the second only decodes. "
        "It is one engine step, with no extra timer sync inside the profiled region. "
        "Device-to-host copies are chrome events whose name contains DtoH. "
        "The intended copy is the one pageable read of the batch's token ids. "
        "Graphs off also copies once to pinned memory: eager decode checks that every new token landed in a reserved page, "
        "and that check reads a device bool. The graph path skips the check, because a replay cannot stop on the host. "
        "Each trace has one cudaDeviceSynchronize. It is on both paths and does not grow with the batch. "
        "A prefill token is read on its own, because finishing it frees a seat before the next admission. "
        "Clocks are not locked. The card also drives the display. Host is WSL2. HTTP is not in these numbers."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Engine decode, graphs on versus graphs off.")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--prompt-len", type=int, default=PROMPT_LEN)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("bench-engine-graph needs CUDA")
    if args.max_tokens < 2:
        raise SystemExit("max_tokens must be at least 2 so a decode gap exists")

    cfg, weights, rope = load_engine("cuda", torch.bfloat16, name=args.model)
    bandwidth = _copy_bandwidth()
    clocks = {"under_load": None}
    cases = []
    with torch.no_grad():
        warm = torch.empty(4096, 4096, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            warm @ warm
        torch.cuda.synchronize()
        del warm
        for batch in BATCHES:
            case = _case(weights, cfg, rope, batch, args, bandwidth, clocks)
            cases.append(case)
            print(
                f"b={batch}  "
                f"off {case['off']['decode_tokens_per_s']:.1f} tok/s "
                f"({case['off']['fraction_of_ceiling']:.3f})  "
                f"on {case['on']['decode_tokens_per_s']:.1f} tok/s "
                f"({case['on']['fraction_of_ceiling']:.3f})  "
                f"tokens_equal {case['tokens_equal']}  "
                f"dtoh off {case['trace']['off']['device_to_host_count']} "
                f"on {case['trace']['on']['device_to_host_count']}",
                flush=True,
            )
    report = {
        "part": "engine-graph",
        "date": date.today().isoformat(),
        "host": "WSL2",
        "command": _command(args),
        "methodology": _methodology(args),
        "model": args.model,
        "model_revision": snapshot_revision(args.model),
        "dtype": "bfloat16",
        "device": "cuda",
        "block_size": args.block_size,
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "reps": args.reps,
        "warmup_discarded": WARMUP,
        "bandwidth_source": BANDWIDTH_FILE,
        "bandwidth_median_bytes_per_s": bandwidth,
        "environment": environment(),
        "nvidia_smi_under_load": clocks["under_load"],
        "nvidia_smi_after": nvidia_smi(),
        "cases": cases,
    }
    text = json.dumps(report, indent=2) + "\n"
    out = args.out
    if out is None:
        out = _ROOT / "bench" / "results" / f"{report['date']}-engine-graph.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(out)
    for case in cases:
        on_copies = case["trace"]["on"]["device_to_host_count"]
        if on_copies != 1:
            raise SystemExit(f"batch {case['batch']} graph step copied device-to-host {on_copies} times")
        if not case["tokens_equal"]:
            raise SystemExit(f"batch {case['batch']} graph tokens diverged from eager")


def _case(weights, cfg, rope, batch, args, bandwidth, clocks) -> dict:
    trace = _burst(batch, args.prompt_len, args.max_tokens, args.seed, cfg.vocab_size)
    final = pages_for(args.block_size, args.prompt_len + args.max_tokens - 1)
    pages = max(32, batch * (final + 1))
    one = partial(forward_paged, weights, cfg, rope)
    forward_batch = partial(forward_paged_batch, weights, cfg, rope)
    modes = {}
    eager_tokens = None
    tokens_equal = True
    captured = None
    traced = {}
    for graphs in (False, True):
        label = "on" if graphs else "off"
        pool = new_page_pool(cfg, pages, args.block_size, device="cuda", dtype=torch.bfloat16)
        calls = {"n": 0}

        def wrapped(ids, seq, _calls=calls, _one=one):
            _calls["n"] += 1
            return _one(ids, seq)

        engine = EngineCore(wrapped, forward_batch, pool, batch, graphs=graphs)
        runs = []
        for _ in range(WARMUP):
            _once(engine, trace, calls, batch)
        if clocks["under_load"] is None and batch == BATCHES[-1] and graphs:
            clocks["under_load"] = nvidia_smi()
        for _ in range(args.reps):
            runs.append(_once(engine, trace, calls, batch))
        if eager_tokens is None:
            eager_tokens = runs[0]["tokens"]
        elif runs[0]["tokens"] != eager_tokens:
            tokens_equal = False
        if graphs:
            captured = len(engine._graphs._graphs)
        traced[label] = _profile_decode(engine, trace, calls, batch)
        modes[label] = _column(runs)
        del engine
        del pool
        torch.cuda.empty_cache()
    context = args.prompt_len + (args.max_tokens - 1) // 2
    ceiling = decode_ceiling_tokens_per_s(
        bandwidth, decode_step_bytes(cfg, 2, batch, context), batch,
    )
    for label in ("off", "on"):
        rate = modes[label]["decode_tokens_per_s"]
        modes[label]["fraction_of_ceiling"] = rate / ceiling
    return {
        "batch": batch,
        "pages": pages,
        "context_start": args.prompt_len,
        "context_mid": context,
        "context_end": args.prompt_len + args.max_tokens - 1,
        "ceiling_tokens_per_s": ceiling,
        "graphs_captured": captured,
        "tokens_equal": tokens_equal,
        "off": modes["off"],
        "on": modes["on"],
        "trace": traced,
    }


def _once(engine, trace, calls, batch) -> dict:
    before = calls["n"]
    summary, by_request = replay(engine, trace)
    torch.cuda.synchronize()
    if calls["n"] - before != batch:
        raise RuntimeError(f"prefill ran {calls['n'] - before} times for a batch of {batch}")
    if summary["finished"] != batch or summary["rejected"] or summary["aborted"]:
        raise RuntimeError(f"trace did not finish cleanly: {summary}")
    if summary["peak_running"] != batch:
        raise RuntimeError(f"peak running {summary['peak_running']} != batch {batch}")
    stats = engine.stats()
    if stats.blocks_free != stats.blocks_total:
        raise RuntimeError(f"{stats.blocks_free} pages free of {stats.blocks_total}")
    itl = summary["itl_s"]["p50"]
    if not itl or itl <= 0:
        raise RuntimeError(f"inter-token p50 was {itl}")
    return {
        "itl_p50_s": itl,
        "itl_p99_s": summary["itl_s"]["p99"],
        "ttft_p50_s": summary["ttft_s"]["p50"],
        "elapsed_s": summary["elapsed_s"],
        "decode_tokens_per_s": batch / itl,
        "tokens": _tokens(by_request),
    }


def _profile_decode(engine, trace, calls, batch) -> dict:
    # First step seats and prefills. The second step is decode only.
    for item in trace:
        engine.add_request(Request(item.request_id, item.prompt, item.max_tokens))
    before = calls["n"]
    engine.step()
    if calls["n"] - before != batch:
        raise RuntimeError(f"profile burst prefilled {calls['n'] - before} times for a batch of {batch}")
    before = calls["n"]
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        outs = engine.step()
    if calls["n"] != before:
        raise RuntimeError("profiled step ran a prefill")
    if len(outs) != batch:
        raise RuntimeError(f"profiled step returned {len(outs)} outputs for a batch of {batch}")
    for out in outs:
        if out.token_id is None or out.finish_reason is not None:
            raise RuntimeError("profiled step was not a continuing decode")
    for item in trace:
        engine.abort(item.request_id)
    engine.step()
    return _trace_summary(prof)


def _trace_summary(prof) -> dict:
    syncs = Counter()
    for event in prof.key_averages():
        label = str(event.key)
        if _is_sync(label):
            syncs[label] += event.count
    copies = Counter()
    for event in _chrome(prof):
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        label = str(event.get("name", ""))
        if "DtoH" in label or "dtoh" in label.lower():
            copies[label] += 1
    named = {key: copies[key] for key in sorted(copies)}
    return {
        "syncs": {key: syncs[key] for key in sorted(syncs)},
        "device_to_host": named,
        "device_to_host_count": sum(named.values()),
    }


def _chrome(prof):
    path = Path("/tmp/engine-graph-trace.json")
    prof.export_chrome_trace(str(path))
    try:
        data = json.loads(path.read_text())
    finally:
        path.unlink(missing_ok=True)
    return data["traceEvents"] if isinstance(data, dict) else data


def _is_sync(name: str) -> bool:
    lowered = name.lower()
    return "item" in lowered or "tolist" in lowered or "synchronize" in lowered


def _column(runs) -> dict:
    rate = statistics.median(run["decode_tokens_per_s"] for run in runs)
    kept = []
    for run in runs:
        kept.append({key: run[key] for key in run if key != "tokens"})
    return {
        "itl_p50_s": _spread(run["itl_p50_s"] for run in runs),
        "itl_p99_s": _spread(run["itl_p99_s"] for run in runs),
        "ttft_p50_s": _spread(run["ttft_p50_s"] for run in runs),
        "elapsed_s": _spread(run["elapsed_s"] for run in runs),
        "decode_tokens_per_s": rate,
        "runs": kept,
    }


def _spread(values):
    values = list(values)
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def _tokens(by_request):
    return {
        rid: [out.token_id for out in outs if out.token_id is not None]
        for rid, outs in sorted(by_request.items(), key=lambda item: int(item[0]))
    }


def _burst(n, prompt_len, max_tokens, seed, vocab):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        prompt = tuple(rng.randrange(vocab) for _ in range(prompt_len))
        rows.append(TraceRequest(0.0, prompt, max_tokens, str(i)))
    return rows


def _copy_bandwidth() -> float:
    path = _ROOT / BANDWIDTH_FILE
    return json.loads(path.read_text())["bandwidth"]["median_bytes_per_s"]


def _command(args) -> str:
    default = (
        args.model == MODEL
        and args.prompt_len == PROMPT_LEN
        and args.max_tokens == MAX_TOKENS
        and args.block_size == BLOCK_SIZE
        and args.seed == SEED
        and args.reps == REPS
        and args.out is None
    )
    if default:
        return "make bench-engine-graph"
    return " ".join(sys.argv)


if __name__ == "__main__":
    main()
