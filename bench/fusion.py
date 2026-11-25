# Decode step with the fused rotation against the eager one, plus the launch counts that picked it.
# make bench-fusion
# The norm fusion's numbers stay in bench/results/2026-11-24-fusion.json. This run writes the rope file.

import json
import statistics
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch.profiler import ProfilerActivity, profile

from bench.env import environment, nvidia_smi
from engine.config import ModelConfig
from engine.kernels.triton.rope import fused_rope
from engine.model import RopeTables, apply_rope_ref
from engine.paged import forward_paged, forward_paged_batch, new_page_pool, new_sequence
from engine.weights import Weights

import engine.kernels.triton.rope as rope_kernel
import engine.model as model
import engine.paged as paged


REPO = "HuggingFaceTB/SmolLM2-1.7B"
DTYPE = torch.bfloat16
BLOCK_SIZE = 16
WARMUP = 5
SAMPLES = 20
PROFILE_STEPS = 2
CASES = (
    (128,),
    (128,) * 8,
    (512,) * 8,
)
PROFILE_CASE = (512,) * 8
BANDWIDTH_FILE = "bench/results/2026-10-31-bandwidth.json"

METHODOLOGY = (
    "SmolLM2-1.7B, CUDA bf16, block size 16, WSL2, clocks not locked. "
    "A decode step is forward_paged_batch plus argmax and a per-sequence .item(), timed with CUDA events after a sync. "
    "Each sequence is prefilled once with the fused rotation; timed steps rewind seq_len. "
    "Eager and fused alternate step by step after five warmup steps of each, twenty samples, median. "
    "Both sides use the fused RMSNorm. Eager rotation is apply_rope_ref. Fused rotation is one kernel for q and k. "
    "Launch counts are a two-step profiler trace of batch 8, context 512. A kernel is charged to the innermost annotation. "
    "RoPE is the fusion because the fused-norm profile still had it at 384 launches. "
    "The microbench is one rotation of q and k, shape [16, 32, 512, 64], with fp32 cos and sin. "
    "Each program reads one contiguous head, so a transposed query is the same stream as this contiguous one. "
    "Streaming bytes are one read and one write of q and of k, plus one read of cos and of sin. "
    "GB/s uses decimal 1e9 and the median of five windows. The fraction uses the median copy in "
    + BANDWIDTH_FILE
    + "."
)


class Probe:
    def __init__(self):
        self.mark = False

    def call(self, name, fn, *args, **kwargs):
        if self.mark:
            with torch.profiler.record_function(name):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)


probe = Probe()
_orig_matmul = torch.Tensor.__matmul__
_orig_add = torch.Tensor.__add__
_orig_mul = torch.Tensor.__mul__
_saved = {}


def _matmul(self, other):
    if not probe.mark or self.ndim >= 4 or other.ndim >= 4:
        return _orig_matmul(self, other)
    return probe.call("weight_gemm", _orig_matmul, self, other)


def _add(self, other):
    if not probe.mark:
        return _orig_add(self, other)
    if self.ndim == 3 and torch.is_tensor(other) and other.ndim == 3 and other.shape == self.shape:
        return probe.call("residual", _orig_add, self, other)
    return _orig_add(self, other)


def _mul(self, other):
    if not probe.mark:
        return _orig_mul(self, other)
    if self.ndim == 3 and torch.is_tensor(other) and other.ndim == 3 and other.shape == self.shape:
        return probe.call("swiglu", _orig_mul, self, other)
    return _orig_mul(self, other)


def _install():
    _saved["rms"] = paged.rms_norm
    _saved["rope"] = model.apply_rope
    _saved["cache"] = paged.reshape_and_cache
    _saved["attn"] = paged.paged_attn_decode

    def rms_norm(x, weight, eps):
        return probe.call("rms_norm", _saved["rms"], x, weight, eps)

    def apply_rope(q, k, cos, sin):
        return probe.call("rope", _saved["rope"], q, k, cos, sin)

    def reshape_and_cache(*args, **kwargs):
        return probe.call("cache_write", _saved["cache"], *args, **kwargs)

    def paged_attn_decode(*args, **kwargs):
        return probe.call("attn", _saved["attn"], *args, **kwargs)

    paged.rms_norm = rms_norm
    model.apply_rope = apply_rope
    paged.reshape_and_cache = reshape_and_cache
    paged.paged_attn_decode = paged_attn_decode
    torch.Tensor.__matmul__ = _matmul
    torch.Tensor.__add__ = _add
    torch.Tensor.__mul__ = _mul


def _restore():
    if not _saved:
        return
    paged.rms_norm = _saved["rms"]
    model.apply_rope = _saved["rope"]
    paged.reshape_and_cache = _saved["cache"]
    paged.paged_attn_decode = _saved["attn"]
    torch.Tensor.__matmul__ = _orig_matmul
    torch.Tensor.__add__ = _orig_add
    torch.Tensor.__mul__ = _orig_mul
    _saved.clear()


def _mode(name: str):
    rope_kernel.use_fused = name == "fused"


def _prefill(weights, cfg, rope, seq, n_tokens, generator):
    ids = torch.randint(0, cfg.vocab_size, (1, n_tokens), device=weights.embed.device, generator=generator)
    forward_paged(weights, cfg, rope, ids, seq)


def _rewind(seqs, lengths):
    for seq, n in zip(seqs, lengths):
        seq.seq_len = n


def _forward(weights, cfg, rope, seqs):
    last = torch.randint(0, cfg.vocab_size, (len(seqs), 1), device=weights.embed.device)
    return forward_paged_batch(weights, cfg, rope, last, seqs)


def _one_step(weights, cfg, rope, seqs, lengths, start, end):
    torch.cuda.synchronize()
    host0 = time.perf_counter()
    start.record()
    logits, _ = _forward(weights, cfg, rope, seqs)
    end.record()
    next_id = torch.argmax(logits, dim=-1)
    for i in range(next_id.shape[0]):
        next_id[i].to(device="cpu").item()
    _rewind(seqs, lengths)
    end.synchronize()
    return (time.perf_counter() - host0) * 1e3, start.elapsed_time(end)


def _time_pair(weights, cfg, rope, seqs, lengths) -> dict:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    modes = ("eager", "fused")
    for name in modes:
        _mode(name)
        for _ in range(WARMUP):
            _one_step(weights, cfg, rope, seqs, lengths, start, end)
    walls = {name: [] for name in modes}
    spans = {name: [] for name in modes}
    for _ in range(SAMPLES):
        for name in modes:
            _mode(name)
            wall, span = _one_step(weights, cfg, rope, seqs, lengths, start, end)
            walls[name].append(wall)
            spans[name].append(span)
    out = {}
    for name in modes:
        wall = statistics.median(walls[name])
        out[name] = {
            "wall_ms": wall,
            "wall_ms_min": min(walls[name]),
            "wall_ms_max": max(walls[name]),
            "gpu_span_ms": statistics.median(spans[name]),
            "samples": SAMPLES,
            "tokens_per_s": len(lengths) / (wall / 1e3),
        }
    out["wall_ratio_eager_over_fused"] = out["eager"]["wall_ms"] / out["fused"]["wall_ms"]
    return out


def _prepare(weights, cfg, rope, lengths):
    device = weights.embed.device
    pages = sum((n + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE for n in lengths)
    pool = new_page_pool(cfg, n_pages=pages, block_size=BLOCK_SIZE, device=device, dtype=weights.embed.dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    seqs = []
    _mode("fused")
    for n in lengths:
        seq = new_sequence(pool)
        _prefill(weights, cfg, rope, seq, n, generator)
        seqs.append(seq)
    return seqs


def _attribute(events) -> dict:
    spans = defaultdict(list)
    kernels = defaultdict(list)
    for evt in events:
        if evt.get("ph") != "X":
            continue
        cat = evt.get("cat")
        if cat == "gpu_user_annotation":
            spans[evt["tid"]].append((evt["ts"], evt["ts"] + evt["dur"], evt["name"]))
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels[evt["tid"]].append((evt["ts"], evt["dur"]))
    counts = defaultdict(int)
    busy = defaultdict(float)
    other_n = 0
    other_busy = 0.0
    for tid, rows in kernels.items():
        covered = spans.get(tid, [])
        for ts, dur in rows:
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
                other_n += 1
                other_busy += dur
            else:
                counts[best] += 1
                busy[best] += dur
    per = PROFILE_STEPS
    names = sorted(set(counts) | set(busy))
    return {
        "per_step": {
            name: {"launches": counts[name] / per, "busy_ms": busy[name] / 1e3 / per}
            for name in names
        },
        "other_launches_per_step": other_n / per,
        "other_busy_ms_per_step": other_busy / 1e3 / per,
        "total_launches_per_step": (sum(counts.values()) + other_n) / per,
    }


def _profile(weights, cfg, rope, seqs, lengths, name: str) -> dict:
    _mode(name)
    probe.mark = True
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(PROFILE_STEPS):
                logits, _ = _forward(weights, cfg, rope, seqs)
                torch.argmax(logits, dim=-1)
                _rewind(seqs, lengths)
        path = Path("/tmp/fusion-profile-trace.json")
        prof.export_chrome_trace(str(path))
        try:
            data = json.loads(path.read_text())
        finally:
            path.unlink(missing_ok=True)
    finally:
        probe.mark = False
    events = data["traceEvents"] if isinstance(data, dict) else data
    return _attribute(events)


def _copy_bandwidth() -> float:
    path = Path(__file__).resolve().parents[1] / BANDWIDTH_FILE
    return json.loads(path.read_text())["bandwidth"]["median_bytes_per_s"]


def _time_calls(fn) -> dict:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    one = max(start.elapsed_time(end), 0.01)
    iters = max(1, int(40 / one))
    windows = []
    for _ in range(5):
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        windows.append(start.elapsed_time(end) / iters)
    return {"median_ms": statistics.median(windows), "min_ms": min(windows), "max_ms": max(windows), "iters_per_window": iters}


def _micro(bandwidth: float) -> dict:
    batch, n_heads, time, dim = 16, 32, 512, 64
    gen = torch.Generator(device="cuda")
    gen.manual_seed(0)
    q = torch.randn(batch, n_heads, time, dim, device="cuda", dtype=DTYPE, generator=gen)
    k = torch.randn(batch, n_heads, time, dim, device="cuda", dtype=DTYPE, generator=gen)
    cos = torch.randn(batch, time, dim, device="cuda", dtype=torch.float32, generator=gen)
    sin = torch.randn(batch, time, dim, device="cuda", dtype=torch.float32, generator=gen)
    # Read and write q and k. Cos and sin are read once. A head is contiguous, so the transpose in project_qkv is the same stream.
    nbytes = (q.numel() + k.numel()) * q.element_size() * 2 + (cos.numel() + sin.numel()) * cos.element_size()
    eager = _time_calls(lambda: apply_rope_ref(q, k, cos, sin))
    fused = _time_calls(lambda: fused_rope(q, k, cos, sin))
    for row in (eager, fused):
        row["gbps"] = (nbytes / 1e9) / (row["median_ms"] / 1e3)
        row["fraction_of_copy"] = (row["gbps"] * 1e9) / bandwidth
    return {
        "batch": batch,
        "heads": n_heads,
        "time": time,
        "dim": dim,
        "dtype": "bf16",
        "bytes": nbytes,
        "eager": eager,
        "fused": fused,
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit("cuda is not available")
    bandwidth = _copy_bandwidth()
    cfg = ModelConfig.from_json(hf_hub_download(REPO, "config.json", local_files_only=True))
    weights = Weights.from_safetensors(
        hf_hub_download(REPO, "model.safetensors", local_files_only=True),
        cfg,
        device="cuda",
        dtype=DTYPE,
    )
    rope = RopeTables.build(cfg, device=weights.embed.device)
    before = environment()
    clocks = {"under_load": None}
    cases = []
    launches = {}
    try:
        with torch.inference_mode():
            # A couple of large gemms take the card out of its idle clock before the first sample.
            warm = torch.empty(4096, 4096, device="cuda", dtype=DTYPE)
            for _ in range(3):
                warm @ warm
            torch.cuda.synchronize()
            del warm
            for lengths in CASES:
                seqs = _prepare(weights, cfg, rope, lengths)
                if clocks["under_load"] is None and lengths == PROFILE_CASE:
                    for _ in range(4):
                        _mode("fused")
                        _forward(weights, cfg, rope, seqs)
                        _rewind(seqs, lengths)
                    clocks["under_load"] = nvidia_smi()
                    torch.cuda.synchronize()
                pair = _time_pair(weights, cfg, rope, seqs, lengths)
                case = {"batch": len(lengths), "context": max(lengths), "block_size": BLOCK_SIZE, **pair}
                cases.append(case)
                print(
                    f"b={case['batch']} ctx={case['context']}  "
                    f"eager {pair['eager']['wall_ms']:.2f} ms  fused {pair['fused']['wall_ms']:.2f} ms  "
                    f"{pair['wall_ratio_eager_over_fused']:.2f}x",
                    flush=True,
                )
                if lengths == PROFILE_CASE:
                    _install()
                    try:
                        for name in ("eager", "fused"):
                            launches[name] = _profile(weights, cfg, rope, seqs, lengths, name)
                            row = launches[name]
                            print(
                                f"  {name} launches {row['total_launches_per_step']:.0f}  "
                                f"rope {row['per_step'].get('rope', {}).get('launches', 0):.0f}",
                                flush=True,
                            )
                    finally:
                        _restore()
            micro = _micro(bandwidth)
    finally:
        rope_kernel.use_fused = True
    print(
        f"micro  eager {micro['eager']['gbps']:.1f} GB/s  fused {micro['fused']['gbps']:.1f} GB/s",
        flush=True,
    )
    report = {
        "part": "rope",
        "date": date.today().isoformat(),
        "host": "WSL2",
        "command": "make bench-fusion",
        "methodology": METHODOLOGY,
        "model": REPO,
        "n_layers": cfg.n_layers,
        "hidden_size": cfg.hidden_size,
        "bandwidth_source": BANDWIDTH_FILE,
        "bandwidth_median_bytes_per_s": bandwidth,
        "environment": before,
        "nvidia_smi_under_load": clocks["under_load"],
        "nvidia_smi_after": nvidia_smi(),
        "cases": cases,
        "launches": launches,
        "micro": micro,
    }
    out = Path(__file__).resolve().parents[1] / "bench" / "results" / f"{report['date']}-rope.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
