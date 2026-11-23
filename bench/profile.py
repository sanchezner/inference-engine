# where one decode step of the current engine spends time
# make bench-profile
#
# this measures the gather-then-dense path
# CUDA events around a whole step say how long the GPU is occupied, including idle gaps between launches
# a second pass puts events around each named region; the profiler pass is only for the kernel list and for those gaps; its own overhead is not the step time

import json
import statistics
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch.profiler import ProfilerActivity, profile

from bench.env import environment, nvidia_smi, snapshot_revision
from bench.roofline import decode_ceiling_tokens_per_s, decode_step_bytes, kv_bytes_per_token
from engine.config import ModelConfig
from engine.model import repeat_kv as _repeat_kv
from engine.paged import (
    forward_paged,
    forward_paged_batch,
    gather_padded as _gather_padded,
    new_page_pool,
    new_sequence,
    page_gather as _page_gather,
    page_write as _page_write,
)
from engine.weights import Weights

import engine.model as model
import engine.paged as paged


MODELS = ("HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-1.7B")
DTYPE = torch.bfloat16
BLOCK_SIZE = 16
WARMUP_STEPS = 5
TIMING_STEPS = 20
BREAKDOWN_STEPS = 4
PROFILE_STEPS = 2
# Equal lengths, then one mixed batch whose longest row matches the uniform 512 case.
CASES = (
    {"batch": 1, "lengths": (128,)},
    {"batch": 8, "lengths": (128,) * 8},
    {"batch": 1, "lengths": (512,)},
    {"batch": 8, "lengths": (512,) * 8},
    {"batch": 4, "lengths": (32, 32, 32, 512)},
)

METHODOLOGY = (
    "Decode steps of forward_paged_batch on CUDA bf16, block size 16. Each sequence is prefilled once to its target length; every measured step then rewinds seq_len so the next step sees the same context and overwrites the same KV slot. The step matches the body of generate_continuous_paged: one new token per sequence, argmax, then a per-sequence .to(cpu) and .item(). A second finish path reads the whole argmax row with one .tolist(), so the extra syncs are visible. Step time is CUDA events around the forward plus the host time through the token read, after a sync so the card is idle at the start of the sample. Five warmup steps, then twenty timed steps; the reported step time is the median. A separate pass of four steps puts two CUDA events around each named call. That span includes the GPU waiting for Python, so it is not kernel time; the profiler attributes kernel time to the same names. page_gather is inside gather_padded; the pad cost is the difference, and it is not added on top. weight_gemm is a matmul whose inputs are both rank <= 3 (the projections, the MLP, the lm_head). attn_gemm is a rank-4 matmul (scores and the attention output). The profiler runs two further steps. Kernel time is summed from the trace. A kernel is charged to the innermost GPU annotation that covers its timestamps, so page_gather is not also added into gather_padded. Gaps are the idle time between kernels on a stream. The profiler's own overhead widens those gaps; the uninstrumented CUDA-event span is the step time, and idle_inside_span_ms subtracts the profiled kernel-busy time from that span. Host is WSL2. Clocks are not locked. fp32 1.7B does not fit beside a CUDA context on this 8 GB card, so the profile is bf16."
)

REGIONS = ("gather_padded", "page_gather", "page_write", "repeat_kv", "weight_gemm", "attn_gemm")


class Probe:
    # two switches that are never on together
    # `enabled` wraps each call in CUDA events
    # elapsed time includes the GPU idle while Python queues the next kernel
    # `mark` only names the call for the profiler, which attributes device time to the kernels launched inside the name

    def __init__(self):
        self.enabled = False
        self.mark = False
        self.pending = []
        self.cpu_s = defaultdict(float)
        self.calls = defaultdict(int)

    def call(self, name, fn, *args, **kwargs):
        if self.mark:
            with torch.profiler.record_function(name):
                return fn(*args, **kwargs)
        if not self.enabled:
            return fn(*args, **kwargs)
        cpu0 = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            return fn(*args, **kwargs)
        finally:
            end.record()
            self.cpu_s[name] += time.perf_counter() - cpu0
            self.calls[name] += 1
            self.pending.append((name, start, end))

    def take(self, steps: int) -> dict:
        torch.cuda.synchronize()
        gpu_ms = defaultdict(float)
        for name, start, end in self.pending:
            gpu_ms[name] += start.elapsed_time(end)
        self.pending.clear()
        out = {}
        for name in set(gpu_ms) | set(self.calls):
            out[name] = {
                "calls": self.calls[name],
                "calls_per_step": self.calls[name] / steps,
                # span includes the GPU idle while Python queues next kernel
                "span_ms_per_step": gpu_ms[name] / steps,
                "cpu_ms_per_step": self.cpu_s[name] * 1e3 / steps,
            }
        self.cpu_s.clear()
        self.calls.clear()
        return out


probe = Probe()
_orig_matmul = torch.Tensor.__matmul__


def _matmul(self, other):
    if not probe.enabled and not probe.mark:
        return _orig_matmul(self, other)
    name = "attn_gemm" if self.ndim >= 4 or other.ndim >= 4 else "weight_gemm"
    return probe.call(name, _orig_matmul, self, other)


def _install():
    def gather_padded(seqs, layer, lens, width):
        return probe.call("gather_padded", _gather_padded, seqs, layer, lens, width)

    def page_gather(seq, layer, n):
        return probe.call("page_gather", _page_gather, seq, layer, n)

    def page_write(seq, layer, k_new, v_new, start):
        return probe.call("page_write", _page_write, seq, layer, k_new, v_new, start)

    def repeat_kv(x, n_rep):
        return probe.call("repeat_kv", _repeat_kv, x, n_rep)

    paged.gather_padded = gather_padded
    paged.page_gather = page_gather
    paged.page_write = page_write
    model.repeat_kv = repeat_kv
    torch.Tensor.__matmul__ = _matmul


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _prefill(weights, cfg, rope, seq, n_tokens: int, generator) -> None:
    ids = torch.randint(0, cfg.vocab_size, (1, n_tokens), device=weights.embed.device, generator=generator)
    forward_paged(weights, cfg, rope, ids, seq)


def _rewind(seqs, lengths) -> None:
    for seq, n in zip(seqs, lengths):
        seq.seq_len = n


def _forward(weights, cfg, rope, seqs):
    last = torch.randint(0, cfg.vocab_size, (len(seqs), 1))
    return forward_paged_batch(weights, cfg, rope, last, seqs)


def _finish_per_sequence(next_id) -> None:
    # same shape as generate_continuous_paged when token stream lives on CPU
    for i in range(next_id.shape[0]):
        next_id[i].to(device="cpu").item()


def _finish_once(next_id) -> None:
    next_id.tolist()


def _time_finish(weights, cfg, rope, seqs, lengths, finish, steps: int) -> dict:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    walls = []
    spans = []
    syncs = []
    for _ in range(steps):
        torch.cuda.synchronize()
        wall0 = time.perf_counter()
        start.record()
        logits, _ = _forward(weights, cfg, rope, seqs)
        end.record()
        next_id = torch.argmax(logits, dim=-1)
        sync0 = time.perf_counter()
        finish(next_id)
        sync_s = time.perf_counter() - sync0
        _rewind(seqs, lengths)
        end.synchronize()
        walls.append((time.perf_counter() - wall0) * 1e3)
        spans.append(start.elapsed_time(end))
        syncs.append(sync_s * 1e3)
    return {
        "wall_ms": _median(walls),
        "wall_ms_min": min(walls),
        "wall_ms_max": max(walls),
        "gpu_span_ms": _median(spans),
        "token_sync_cpu_ms": _median(syncs),
        "samples": steps,
    }


def _breakdown(weights, cfg, rope, seqs, lengths) -> dict:
    probe.enabled = True
    try:
        for _ in range(BREAKDOWN_STEPS):
            logits, _ = _forward(weights, cfg, rope, seqs)
            _finish_per_sequence(torch.argmax(logits, dim=-1))
            _rewind(seqs, lengths)
        regions = probe.take(BREAKDOWN_STEPS)
    finally:
        probe.enabled = False
        probe.pending.clear()
        probe.cpu_s.clear()
        probe.calls.clear()
    return regions


def _kernel_table(prof) -> list[dict]:
    rows = []
    for evt in prof.key_averages():
        self_us = getattr(evt, "self_device_time_total", None)
        if self_us is None:
            self_us = getattr(evt, "self_cuda_time_total", 0)
        if self_us <= 0 and evt.self_cpu_time_total <= 0:
            continue
        rows.append(
            {
                "name": _short_kernel(evt.key),
                "count": evt.count,
                "self_device_us": self_us,
                "self_cpu_us": evt.self_cpu_time_total,
            }
        )
    rows.sort(key=lambda row: row["self_device_us"], reverse=True)
    return rows[:25]


def _stream_gaps(events) -> dict:
    by_stream = defaultdict(list)
    for evt in events:
        if evt.get("ph") != "X":
            continue
        if evt.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        by_stream[evt.get("tid")].append((evt["ts"], evt["dur"]))
    busy = 0.0
    gaps = 0.0
    span = 0.0
    count = 0
    for stream in by_stream.values():
        stream.sort()
        count += len(stream)
        busy += sum(dur for _, dur in stream)
        span += (stream[-1][0] + stream[-1][1]) - stream[0][0]
        for (ts, dur), nxt in zip(stream, stream[1:]):
            gap = nxt[0] - (ts + dur)
            if gap > 0:
                gaps += gap
    return {
        "streams": len(by_stream),
        "kernel_count": count,
        "busy_us": busy,
        "gap_us": gaps,
        "span_us": span,
        "per_step": PROFILE_STEPS,
    }


def _short_kernel(name: str) -> str:
    if name.startswith("aten::") or name.startswith("Memcpy"):
        return name
    head = name.split("<", 1)[0]
    return head if len(head) <= 140 else head[:140]


def _attribute(events) -> dict:
    # kernel time per region
    # a kernel is charged to innermost GPU annotation that covers it
    # page_gather is inside gather_padded; a copy launched by page_gather counts once under page_gather
    # kernels with no annotation (norms, RoPE, softmax, casts) goes into `other`
    spans = defaultdict(list)
    kernels = defaultdict(list)
    for evt in events:
        if evt.get("ph") != "X":
            continue
        cat = evt.get("cat")
        if cat == "gpu_user_annotation":
            spans[evt["tid"]].append((evt["ts"], evt["ts"] + evt["dur"], evt["name"]))
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels[evt["tid"]].append((evt["ts"], evt["dur"], evt["name"]))

    busy = defaultdict(float)
    counts = defaultdict(int)
    other = defaultdict(float)
    other_n = defaultdict(int)
    for tid, rows in kernels.items():
        covered = spans.get(tid, [])
        for ts, dur, name in rows:
            end = ts + dur
            best = None
            best_w = None
            for start, stop, region in covered:
                if start <= ts and end <= stop + 0.05:
                    width = stop - start
                    if best_w is None or width < best_w:
                        best_w = width
                        best = region
            if best is None:
                other[name] += dur
                other_n[name] += 1
            else:
                busy[best] += dur
                counts[best] += 1

    def per_step(us):
        return us / 1e3 / PROFILE_STEPS

    regions = {}
    for name in REGIONS:
        regions[name] = {
            "kernel_count": counts[name],
            "busy_ms_per_step": per_step(busy[name]),
        }
    other_rows = sorted(other.items(), key=lambda item: item[1], reverse=True)[:8]
    return {
        "regions": regions,
        "other_busy_ms_per_step": per_step(sum(other.values())),
        "other_top": [
            {"name": _short_kernel(name), "busy_ms_per_step": per_step(us), "count": other_n[name]}
            for name, us in other_rows
        ],
    }


def _profile_kernels(weights, cfg, rope, seqs, lengths) -> dict:
    probe.mark = True
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(PROFILE_STEPS):
                logits, _ = _forward(weights, cfg, rope, seqs)
                _finish_per_sequence(torch.argmax(logits, dim=-1))
                _rewind(seqs, lengths)
    finally:
        probe.mark = False
    path = Path("/tmp/decode-profile-trace.json")
    prof.export_chrome_trace(str(path))
    try:
        data = json.loads(path.read_text())
    finally:
        path.unlink(missing_ok=True)
    events = data["traceEvents"] if isinstance(data, dict) else data
    timeline = _stream_gaps(events)
    return {
        "top_self_device": _kernel_table(prof),
        "timeline": timeline,
        "busy_by_region": _attribute(events),
    }


def _case(weights, cfg, rope, lengths: tuple[int, ...], clocks: dict) -> dict:
    device = weights.embed.device
    pages = sum((n + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE for n in lengths)
    pool = new_page_pool(cfg, n_pages=pages, block_size=BLOCK_SIZE, device=device, dtype=weights.embed.dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    seqs = []
    for n in lengths:
        seq = new_sequence(pool)
        _prefill(weights, cfg, rope, seq, n, generator)
        seqs.append(seq)
    for _ in range(WARMUP_STEPS):
        logits, _ = _forward(weights, cfg, rope, seqs)
        _finish_per_sequence(torch.argmax(logits, dim=-1))
        _rewind(seqs, lengths)
    # queue several steps and sample while they drain; one step can finish before nvidia-smi starts, and the card falls back to idle clock
    if clocks["under_load"] is None and cfg.n_kv_heads >= 32 and len(lengths) >= 8 and max(lengths) >= 512:
        for _ in range(8):
            _forward(weights, cfg, rope, seqs)
            _rewind(seqs, lengths)
        clocks["under_load"] = nvidia_smi()
        torch.cuda.synchronize()
    per_sequence = _time_finish(weights, cfg, rope, seqs, lengths, _finish_per_sequence, TIMING_STEPS)
    once = _time_finish(weights, cfg, rope, seqs, lengths, _finish_once, TIMING_STEPS)
    regions = _breakdown(weights, cfg, rope, seqs, lengths)
    kernels = _profile_kernels(weights, cfg, rope, seqs, lengths)
    busy_ms = kernels["timeline"]["busy_us"] / 1e3 / PROFILE_STEPS
    kernels["busy_ms_per_step"] = busy_ms
    kernels["idle_inside_span_ms"] = per_sequence["gpu_span_ms"] - busy_ms
    del pool
    torch.cuda.empty_cache()
    width = max(lengths)
    gathered = sum(lengths)
    return {
        "batch": len(lengths),
        "lengths": list(lengths),
        "max_len": width,
        "sum_len": gathered,
        "padded_over_sum": (len(lengths) * width) / gathered,
        "pages": pages,
        "block_size": BLOCK_SIZE,
        "per_sequence_sync": per_sequence,
        "one_sync": once,
        "regions": regions,
        "kernels": kernels,
    }


def _load(repo: str):
    cfg = ModelConfig.from_json(hf_hub_download(repo, "config.json", local_files_only=True))
    weights = Weights.from_safetensors(
        hf_hub_download(repo, "model.safetensors", local_files_only=True),
        cfg,
        device="cuda",
        dtype=DTYPE,
    )
    from engine.model import RopeTables

    rope = RopeTables.build(cfg, device=weights.embed.device)
    return cfg, weights, rope


def main():
    if not torch.cuda.is_available():
        raise SystemExit("cuda is not available")
    _install()
    before = environment()
    clocks = {"under_load": None}
    models = {}
    try:
        with torch.inference_mode():
            for repo in MODELS:
                cfg, weights, rope = _load(repo)
                cases = [_case(weights, cfg, rope, case["lengths"], clocks) for case in CASES]
                dtype_bytes = torch.empty((), dtype=DTYPE).element_size()
                ceilings = []
                for case in cases:
                    step = decode_step_bytes(cfg, dtype_bytes, case["batch"], case["max_len"])
                    ceilings.append({"lengths": case["lengths"], "step_bytes_at_max_len": step})
                models[repo] = {
                    "revision": snapshot_revision(repo),
                    "n_layers": cfg.n_layers,
                    "n_heads": cfg.n_heads,
                    "n_kv_heads": cfg.n_kv_heads,
                    "head_dim": cfg.head_dim,
                    "gqa_rep": cfg.n_heads // cfg.n_kv_heads,
                    "dtype": "bf16",
                    "kv_bytes_per_token": kv_bytes_per_token(cfg, dtype_bytes),
                    "ceilings_need_bandwidth": ceilings,
                    "cases": cases,
                }
                del weights, rope
                torch.cuda.empty_cache()
    finally:
        torch.Tensor.__matmul__ = _orig_matmul
    report = {
        "milestone": "M2",
        "part": "profile",
        "date": date.today().isoformat(),
        "host": "WSL2",
        "command": "make bench-profile",
        "methodology": METHODOLOGY,
        "environment": before,
        "nvidia_smi_under_load": clocks["under_load"],
        "nvidia_smi_after": nvidia_smi(),
        "models": models,
    }
    out = Path(__file__).resolve().parents[1] / "bench" / "results" / f"{report['date']}-profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out)
    for repo, model_report in models.items():
        print(repo)
        for case in model_report["cases"]:
            sync = case["per_sequence_sync"]
            print(
                f"  b={case['batch']} lens={case['lengths']}  "
                f"wall {sync['wall_ms']:.2f} ms  gpu_span {sync['gpu_span_ms']:.2f} ms  "
                f"token_sync {sync['token_sync_cpu_ms']:.2f} ms"
            )


if __name__ == "__main__":
    main()
