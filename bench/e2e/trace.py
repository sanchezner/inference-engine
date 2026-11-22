# Arrivals follow the clock. Waiting for each reply would hide a slow engine's latency.

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine.sched.engine_core import EngineCore
from engine.sched.request import Request


@dataclass(frozen=True)
class TraceRequest:
    t: float
    prompt: tuple[int, ...]
    max_tokens: int
    request_id: str


def poisson_trace(n, rate, prompt_len, max_tokens, seed, vocab=1024):
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be an int >= 1, got {n!r}")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or rate <= 0:
        raise ValueError(f"rate must be a positive number of requests per second, got {rate!r}")
    if isinstance(prompt_len, bool) or not isinstance(prompt_len, int) or prompt_len < 1:
        raise ValueError(f"prompt_len must be an int >= 1, got {prompt_len!r}")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError(f"max_tokens must be an int >= 1, got {max_tokens!r}")
    rng = random.Random(seed)
    t = 0.0
    rows = []
    for i in range(n):
        if i:
            # expovariate(rate) has mean 1/rate, so `rate` is requests per second.
            t += rng.expovariate(rate)
        prompt = tuple(rng.randrange(vocab) for _ in range(prompt_len))
        rows.append(TraceRequest(t, prompt, max_tokens, str(i)))
    return rows


def write_trace(path, requests):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for item in requests:
            row = {
                "t": item.t,
                "prompt": list(item.prompt),
                "max_tokens": item.max_tokens,
                "id": item.request_id,
            }
            handle.write(json.dumps(row) + "\n")


def read_trace(path):
    requests = []
    seen = set()
    last_t = 0.0
    with Path(path).open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            item = _parse_row(row, line_no, f"r{len(requests)}")
            if item.request_id in seen:
                raise ValueError(f"trace repeats request id {item.request_id!r}")
            if requests and item.t < last_t:
                raise ValueError(f"trace times go backwards at id {item.request_id!r}")
            seen.add(item.request_id)
            last_t = item.t
            requests.append(item)
    return requests


def replay(engine, trace, *, clock=None, sleep=None, step_cost=0.0, abort_ids=()):
    if step_cost < 0:
        raise ValueError(f"step_cost must be >= 0, got {step_cost}")
    if clock is None:
        clock = time.perf_counter
    if sleep is None:
        sleep = time.sleep
    abort_ids = frozenset(abort_ids)
    items = list(trace)
    start = clock()
    index = 0
    by_request = {}
    # Between steps, not inside one. A request that starts and ends in one step does not move these.
    peak_running = 0
    peak_waiting = 0
    steps = 0
    budget = sum(item.max_tokens for item in items) + len(items) + len(abort_ids) + 2

    def note():
        nonlocal peak_running, peak_waiting
        stats = engine.stats()
        peak_running = max(peak_running, stats.requests_running)
        peak_waiting = max(peak_waiting, stats.requests_waiting)
        return stats

    def record(outs):
        for out in outs:
            by_request.setdefault(out.request_id, []).append(out)

    def abort_seen(outs):
        for out in outs:
            if out.request_id in abort_ids and out.token_id is not None:
                engine.abort(out.request_id)

    while True:
        now = clock() - start
        while index < len(items) and items[index].t <= now + 1e-9:
            _submit(engine, items[index])
            index += 1
        stats = note()
        idle = stats.requests_waiting == 0 and stats.requests_running == 0
        if idle and index >= len(items):
            outs = engine.step()
            record(outs)
            abort_seen(outs)
            note()
            if not outs:
                break
            continue
        if idle:
            delay = items[index].t - (clock() - start)
            if delay > 0:
                before = clock()
                sleep(delay)
                if clock() <= before:
                    raise RuntimeError("clock did not advance while waiting for the next arrival")
            continue
        outs = engine.step()
        steps += 1
        if steps > budget:
            raise RuntimeError("trace replay exceeded its step budget")
        record(outs)
        abort_seen(outs)
        note()
        if step_cost > 0:
            sleep(step_cost)

    elapsed = clock() - start
    summary = summarize(by_request, elapsed)
    summary["peak_running"] = peak_running
    summary["peak_waiting"] = peak_waiting
    return summary, by_request


def summarize(by_request, elapsed):
    ttft = []
    itl = []
    tokens = 0
    finished = aborted = rejected = 0
    for outputs in by_request.values():
        token_times = []
        for out in outputs:
            if out.token_id is not None:
                tokens += 1
                token_times.append(out.timestamps.emitted)
        if token_times:
            ttft.append(token_times[0] - outputs[0].timestamps.arrival)
            itl.extend(b - a for a, b in zip(token_times, token_times[1:]))
        reason = outputs[-1].finish_reason
        if reason == "length" or reason == "stop":
            finished += 1
        elif reason == "abort":
            aborted += 1
        elif reason == "rejected":
            rejected += 1
    throughput = None if elapsed <= 0 else tokens / elapsed
    return {
        "requests": len(by_request),
        "finished": finished,
        "aborted": aborted,
        "rejected": rejected,
        "tokens": tokens,
        "elapsed_s": elapsed,
        "throughput_tok_s": throughput,
        "ttft_s": {"p50": percentile(ttft, 50), "p99": percentile(ttft, 99)},
        "itl_s": {"p50": percentile(itl, 50), "p99": percentile(itl, 99)},
    }


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    # Nearest rank. A short trace should not invent a value between two samples.
    rank = math.ceil(pct / 100 * len(ordered)) - 1
    return ordered[max(0, min(len(ordered) - 1, rank))]


def build_engine(model, device, dtype_name, max_batch, pages, block_size):
    from functools import partial

    import torch

    from checks.oracle import load_engine
    from engine.paged import forward_paged, forward_paged_batch, new_page_pool

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    cfg, weights, rope = load_engine(device, dtype, name=model)
    pool = new_page_pool(cfg, pages, block_size, device=device, dtype=dtype)
    one = partial(forward_paged, weights, cfg, rope)
    batch = partial(forward_paged_batch, weights, cfg, rope)
    return EngineCore(one, batch, pool, max_batch)


def _submit(engine, item):
    engine.add_request(Request(
        request_id=item.request_id,
        prompt_token_ids=item.prompt,
        max_new_tokens=item.max_tokens,
    ))


def _parse_row(row, line_no, default_id):
    if not isinstance(row, dict):
        raise ValueError(f"line {line_no} is not an object")
    try:
        t = row["t"]
        prompt = row["prompt"]
        max_tokens = row["max_tokens"]
    except KeyError as exc:
        raise ValueError(f"line {line_no} is missing {exc.args[0]}") from exc
    if isinstance(t, bool) or not isinstance(t, (int, float)):
        raise ValueError(f"line {line_no} t must be a number")
    t = float(t)
    if t < 0:
        raise ValueError(f"line {line_no} t must be >= 0")
    if not isinstance(prompt, list) or len(prompt) == 0:
        raise ValueError(f"line {line_no} prompt must be a non-empty list of token ids")
    tokens = []
    for token in prompt:
        if isinstance(token, bool) or not isinstance(token, int):
            raise ValueError(f"line {line_no} prompt must be token ids")
        tokens.append(token)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError(f"line {line_no} max_tokens must be an int >= 1")
    request_id = row.get("id", default_id)
    if not isinstance(request_id, str) or request_id == "":
        raise ValueError(f"line {line_no} id must be a non-empty string")
    return TraceRequest(t, tuple(tokens), max_tokens, request_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Replay an open-loop request trace through EngineCore.")
    parser.add_argument("trace", nargs="?", type=Path, help="JSONL trace to replay")
    parser.add_argument("--poisson", action="store_true", help="write a seeded Poisson trace to --out and exit")
    parser.add_argument("--out", type=Path, help="where --poisson writes the JSONL")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--rate", type=float, default=1.0, help="Poisson arrivals per second")
    parser.add_argument("--prompt-len", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--pages", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--abort", action="append", default=[], help="request id to abort once it has emitted a token; repeatable")
    parser.add_argument("--summary", type=Path, help="also write the JSON summary here")
    args = parser.parse_args(argv)
    if args.poisson:
        if args.out is None:
            parser.error("--poisson requires --out")
        write_trace(args.out, poisson_trace(
            args.requests, args.rate, args.prompt_len, args.max_tokens, args.seed,
        ))
        return
    if args.trace is None:
        parser.error("give a trace path to replay, or --poisson --out to only write one")
    items = read_trace(args.trace)
    engine = build_engine(args.model, args.device, args.dtype, args.max_batch, args.pages, args.block_size)
    summary, _ = replay(engine, items, abort_ids=args.abort)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text + "\n")


if __name__ == "__main__":
    main()
