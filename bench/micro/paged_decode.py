# paged-attention decode: kernel bandwidth, and the decode step against gather
# make bench-paged

import json
import statistics
import time
from datetime import date
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from bench.env import environment, nvidia_smi, snapshot_revision
from bench.roofline import (
    decode_ceiling_tokens_per_s,
    decode_step_bytes,
    embedding_lookup_bytes,
    kv_bytes_per_token,
    paged_decode_kernel_bytes,
    streamed_weight_bytes,
)
from engine.config import ModelConfig
from engine.kernels.triton.paged_attn import _paged_decode
from engine.model import RopeTables
from engine.paged import forward_paged, forward_paged_batch, new_page_pool, new_sequence
from engine.weights import Weights

import engine.paged as paged


ROOT = Path(__file__).resolve().parents[2]
BANDWIDTH_PATH = ROOT / "bench" / "results" / "2026-10-31-m2-bandwidth.json"
# Head counts from checks/roofline.py. The kernel does not need the weights.
GEOMETRIES = (
    {"name": "135M", "n_heads": 9, "n_kv_heads": 3, "head_dim": 64},
    {"name": "1.7B", "n_heads": 32, "n_kv_heads": 32, "head_dim": 64},
)
BLOCK_SIZES = (8, 16, 32)
BATCHES = (1, 4, 8, 32)
CONTEXTS = (128, 512, 2048)
# Same grid, so a shorter context is less memory work and not a smaller launch.
RAGGED = (
    {"name": "uniform-512", "lengths": (512,) * 8},
    {"name": "uniform-64", "lengths": (64,) * 8},
    {"name": "mixed", "lengths": (64,) * 7 + (512,)},
)
STEP_MODELS = ("HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-1.7B")
STEP_BLOCK = 16
# The four M2 profile shapes, plus a short row and a mixed row that share the long row's max.
STEP_CASES = (
    (128,),
    (128,) * 8,
    (512,),
    (512,) * 8,
    (64,) * 8,
    (64,) * 7 + (512,),
)
DTYPE = torch.bfloat16
KERNEL_WINDOW_MS = 40.0
KERNEL_REPS = 5
STEP_WARMUP = 5
STEP_TIMING = 10

METHODOLOGY = (
    "Two clocks. The kernel clock is a CUDA graph of one paged_attn_decode launch, replayed with no Python between launches, so the span is device time of the kernel. Bytes are q read plus the output written plus K and V read, for the live tokens only, one layer. GQA does not multiply the KV bytes by the group size. Contexts and the ragged lengths are multiples of the block size, so the last page is full and the mask does not drop slots. Block tables are a seeded shuffle of the pages, which is the paged case; one sequential table is timed at the same bytes to show the shuffle. Five windows, each aimed at 40 ms; the reported time is the median window. The fraction is that GB/s over the median device-to-device copy in bench/results/2026-10-31-m2-bandwidth.json (378.7 GB/s, read plus write, decimal 1e9). The step clock is forward_paged_batch on CUDA bf16, block size 16, the same quantity as bench/profile.py: CUDA events around the forward, then argmax and a per-sequence .item(), after a sync so the card is idle at the start. Five warmup steps, then ten timed steps; seq_len is rewound so every step sees the same context and overwrites the same KV slot. The gather path is the same forward with the decode kernel switched off, so attention gathers pages and the cache write stays the scatter kernel. Tokens/s is batch / median wall seconds. The ceiling uses that same copy bandwidth and decode_step_bytes at the longest row, which is how the M2 table was computed. Host is WSL2. Clocks are not locked. There is no split-KV in this kernel: the grid is (batch, query heads)."
)


def _copy_bandwidth() -> float:
    report = json.loads(BANDWIDTH_PATH.read_text())
    return float(report["bandwidth"]["median_bytes_per_s"])


def _launch(q, k, v, out, bt, sl, n_rep, head_dim, block_size):
    _paged_decode[(q.shape[0], q.shape[1])](
        q, k, v, out, bt, sl,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        bt.stride(0), bt.stride(1), sl.stride(0),
        head_dim ** -0.5,
        n_rep,
        HEAD_DIM=head_dim,
        BLOCK_N=block_size,
    )


def _graph(launch):
    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _window_ms(graph, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def _choose_iters(graph) -> int:
    sample = _window_ms(graph, 1)
    if sample <= 0:
        return 50
    return min(500, max(20, int(KERNEL_WINDOW_MS / sample)))


def _time_graph(graph) -> dict:
    iters = _choose_iters(graph)
    samples = [_window_ms(graph, iters) for _ in range(KERNEL_REPS)]
    median = statistics.median(samples)
    return {
        "iters_per_window": iters,
        "windows": KERNEL_REPS,
        "samples_ms": samples,
        "median_ms": median,
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _fill(n_pages, n_kv, block_size, head_dim, generator):
    shape = (n_pages, n_kv, block_size, head_dim)
    k = torch.randn(shape, device="cuda", dtype=DTYPE, generator=generator)
    v = torch.randn(shape, device="cuda", dtype=DTYPE, generator=generator)
    return k, v


def _uniform_buffers(geo, block_size, batch, context, order, seed):
    pages_each = context // block_size
    n_pages = batch * pages_each
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    k, v = _fill(n_pages, geo["n_kv_heads"], block_size, geo["head_dim"], generator)
    q = torch.randn(batch, geo["n_heads"], geo["head_dim"], device="cuda", dtype=DTYPE, generator=generator)
    out = torch.empty_like(q)
    if order == "sequential":
        perm = torch.arange(n_pages, device="cuda")
    else:
        perm = torch.randperm(n_pages, device="cuda", generator=generator)
    bt = perm.to(dtype=torch.int32).view(batch, pages_each)
    sl = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    return q, k, v, out, bt, sl


def _ragged_buffers(geo, block_size, lengths, seed):
    needs = [n // block_size for n in lengths]
    n_pages = sum(needs)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    k, v = _fill(n_pages, geo["n_kv_heads"], block_size, geo["head_dim"], generator)
    batch = len(lengths)
    q = torch.randn(batch, geo["n_heads"], geo["head_dim"], device="cuda", dtype=DTYPE, generator=generator)
    out = torch.empty_like(q)
    perm = torch.randperm(n_pages, device="cuda", generator=generator)
    width = max(needs)
    bt = torch.zeros(batch, width, device="cuda", dtype=torch.int32)
    cursor = 0
    for row, need in enumerate(needs):
        bt[row, :need] = perm[cursor:cursor + need].to(dtype=torch.int32)
        cursor += need
    sl = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    return q, k, v, out, bt, sl


def _measure(q, k, v, out, bt, sl, geo, block_size, bandwidth, clocks) -> dict:
    n_rep = geo["n_heads"] // geo["n_kv_heads"]

    def launch():
        _launch(q, k, v, out, bt, sl, n_rep, geo["head_dim"], block_size)

    graph = _graph(launch)
    try:
        if clocks["under_load"] is None and geo["name"] == "1.7B" and q.shape[0] >= 8 and int(sl.max()) >= 2048:
            for _ in range(20):
                graph.replay()
            clocks["under_load"] = nvidia_smi()
            torch.cuda.synchronize()
        timed = _time_graph(graph)
    finally:
        del graph
    seq_lens = [int(n) for n in sl.tolist()]
    moved = paged_decode_kernel_bytes(geo["n_heads"], geo["n_kv_heads"], geo["head_dim"], 2, seq_lens)
    seconds = timed["median_ms"] / 1e3
    gbps = (moved / seconds) / 1e9
    return {
        "n_heads": geo["n_heads"],
        "n_kv_heads": geo["n_kv_heads"],
        "head_dim": geo["head_dim"],
        "block_size": block_size,
        "batch": len(seq_lens),
        "seq_lens": seq_lens,
        "sum_len": sum(seq_lens),
        "max_len": max(seq_lens),
        "grid": [len(seq_lens), geo["n_heads"]],
        "bytes": moved,
        "gbps": gbps,
        "fraction_of_copy": (moved / seconds) / bandwidth,
        **timed,
    }


def _kernel_grid(bandwidth, clocks) -> list[dict]:
    rows = []
    seed = 0
    for geo in GEOMETRIES:
        for block_size in BLOCK_SIZES:
            for batch in BATCHES:
                for context in CONTEXTS:
                    seed += 1
                    buffers = _uniform_buffers(geo, block_size, batch, context, "shuffled", seed)
                    row = _measure(*buffers, geo, block_size, bandwidth, clocks)
                    row["model"] = geo["name"]
                    row["order"] = "shuffled"
                    rows.append(row)
                    print(
                        f"kernel {geo['name']} b={batch} L={context} block={block_size}  "
                        f"{row['gbps']:.1f} GB/s  {row['fraction_of_copy'] * 100:.1f}%",
                        flush=True,
                    )
                    del buffers
                    torch.cuda.empty_cache()
        # One contiguous table, same bytes as the shuffled batch-8 context-512 row.
        buffers = _uniform_buffers(geo, STEP_BLOCK, 8, 512, "sequential", 900 + geo["n_heads"])
        row = _measure(*buffers, geo, STEP_BLOCK, bandwidth, clocks)
        row["model"] = geo["name"]
        row["order"] = "sequential"
        rows.append(row)
        print(
            f"kernel {geo['name']} b=8 L=512 block={STEP_BLOCK} sequential  "
            f"{row['gbps']:.1f} GB/s  {row['fraction_of_copy'] * 100:.1f}%",
            flush=True,
        )
        del buffers
        torch.cuda.empty_cache()
    return rows


def _kernel_ragged(bandwidth, clocks) -> list[dict]:
    rows = []
    for geo in GEOMETRIES:
        for spec in RAGGED:
            buffers = _ragged_buffers(geo, STEP_BLOCK, spec["lengths"], seed=400 + geo["n_heads"] + spec["lengths"][0])
            row = _measure(*buffers, geo, STEP_BLOCK, bandwidth, clocks)
            row["model"] = geo["name"]
            row["order"] = "shuffled"
            row["case"] = spec["name"]
            rows.append(row)
            print(
                f"ragged {geo['name']} {spec['name']} sum={row['sum_len']}  "
                f"{row['median_ms']:.4f} ms  {row['gbps']:.1f} GB/s",
                flush=True,
            )
            del buffers
            torch.cuda.empty_cache()
    return rows


def _prefill(weights, cfg, rope, seq, n_tokens, generator):
    ids = torch.randint(0, cfg.vocab_size, (1, n_tokens), device=weights.embed.device, generator=generator)
    forward_paged(weights, cfg, rope, ids, seq)


def _rewind(seqs, lengths):
    for seq, n in zip(seqs, lengths):
        seq.seq_len = n


def _time_steps(weights, cfg, rope, seqs, lengths) -> dict:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    walls = []
    spans = []
    for _ in range(STEP_TIMING):
        torch.cuda.synchronize()
        # Wall time, because the M2 table's step runs through the token read.
        host0 = time.perf_counter()
        start.record()
        logits, _ = forward_paged_batch(
            weights, cfg, rope, torch.randint(0, cfg.vocab_size, (len(seqs), 1)), seqs
        )
        end.record()
        next_id = torch.argmax(logits, dim=-1)
        for i in range(next_id.shape[0]):
            next_id[i].to(device="cpu").item()
        _rewind(seqs, lengths)
        end.synchronize()
        walls.append((time.perf_counter() - host0) * 1e3)
        spans.append(start.elapsed_time(end))
    return {
        "wall_ms": statistics.median(walls),
        "wall_ms_min": min(walls),
        "wall_ms_max": max(walls),
        "gpu_span_ms": statistics.median(spans),
        "gpu_span_ms_min": min(spans),
        "gpu_span_ms_max": max(spans),
        "samples": STEP_TIMING,
    }


def _warmup(weights, cfg, rope, seqs, lengths):
    for _ in range(STEP_WARMUP):
        forward_paged_batch(weights, cfg, rope, torch.randint(0, cfg.vocab_size, (len(seqs), 1)), seqs)
        _rewind(seqs, lengths)
    torch.cuda.synchronize()


def _step_case(weights, cfg, rope, lengths, bandwidth, clocks) -> dict:
    device = weights.embed.device
    pages = sum((n + 1 + STEP_BLOCK - 1) // STEP_BLOCK for n in lengths)
    pool = new_page_pool(cfg, n_pages=pages, block_size=STEP_BLOCK, device=device, dtype=weights.embed.dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    seqs = []
    for n in lengths:
        seq = new_sequence(pool)
        _prefill(weights, cfg, rope, seq, n, generator)
        seqs.append(seq)
    _warmup(weights, cfg, rope, seqs, lengths)
    kernel = _time_steps(weights, cfg, rope, seqs, lengths)
    saved = paged.decode_supported
    paged.decode_supported = lambda *_args, **_kwargs: False
    try:
        _warmup(weights, cfg, rope, seqs, lengths)
        gather = _time_steps(weights, cfg, rope, seqs, lengths)
    finally:
        paged.decode_supported = saved
    if clocks["step"] is None and cfg.n_kv_heads >= 32 and len(lengths) >= 8 and max(lengths) >= 512:
        for _ in range(4):
            forward_paged_batch(weights, cfg, rope, torch.randint(0, cfg.vocab_size, (len(seqs), 1)), seqs)
            _rewind(seqs, lengths)
        clocks["step"] = nvidia_smi()
        torch.cuda.synchronize()
    del pool
    torch.cuda.empty_cache()
    batch = len(lengths)
    dtype_bytes = 2
    at_max = decode_step_bytes(cfg, dtype_bytes, batch, max(lengths))
    attended = sum(n + 1 for n in lengths)
    at_sum = (
        streamed_weight_bytes(cfg, dtype_bytes)
        + embedding_lookup_bytes(cfg, dtype_bytes, batch)
        + attended * kv_bytes_per_token(cfg, dtype_bytes)
    )
    kernel_s = kernel["wall_ms"] / 1e3
    gather_s = gather["wall_ms"] / 1e3
    return {
        "batch": batch,
        "lengths": list(lengths),
        "max_len": max(lengths),
        "sum_len": sum(lengths),
        "padded_tokens": batch * max(lengths),
        "block_size": STEP_BLOCK,
        "kernel": kernel,
        "gather": gather,
        "kernel_tokens_per_s": batch / kernel_s,
        "gather_tokens_per_s": batch / gather_s,
        "kernel_over_gather": gather_s / kernel_s,
        "step_bytes_at_max_len": at_max,
        "step_bytes_at_attended_sum": at_sum,
        "kernel_fraction_of_ceiling_at_max_len": (batch / kernel_s) / decode_ceiling_tokens_per_s(bandwidth, at_max, batch),
    }


def _load(repo):
    cfg = ModelConfig.from_json(hf_hub_download(repo, "config.json", local_files_only=True))
    weights = Weights.from_safetensors(
        hf_hub_download(repo, "model.safetensors", local_files_only=True),
        cfg,
        device="cuda",
        dtype=DTYPE,
    )
    rope = RopeTables.build(cfg, device=weights.embed.device)
    return cfg, weights, rope


def _steps(bandwidth, clocks) -> dict:
    models = {}
    with torch.inference_mode():
        for repo in STEP_MODELS:
            cfg, weights, rope = _load(repo)
            cases = []
            for lengths in STEP_CASES:
                case = _step_case(weights, cfg, rope, lengths, bandwidth, clocks)
                cases.append(case)
                print(
                    f"step {repo.split('/')[-1]} lens={list(lengths)}  "
                    f"kernel {case['kernel']['wall_ms']:.1f} ms  gather {case['gather']['wall_ms']:.1f} ms  "
                    f"{case['kernel_over_gather']:.2f}x",
                    flush=True,
                )
            models[repo] = {
                "revision": snapshot_revision(repo),
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "n_kv_heads": cfg.n_kv_heads,
                "head_dim": cfg.head_dim,
                "cases": cases,
            }
            del weights, rope
            torch.cuda.empty_cache()
    return models


def main():
    if not torch.cuda.is_available():
        raise SystemExit("cuda is not available")
    bandwidth = _copy_bandwidth()
    before = environment()
    clocks = {"under_load": None, "step": None}
    kernel_rows = _kernel_grid(bandwidth, clocks)
    ragged_rows = _kernel_ragged(bandwidth, clocks)
    steps = _steps(bandwidth, clocks)
    report = {
        "milestone": "M3",
        "part": "paged-decode",
        "date": date.today().isoformat(),
        "host": "WSL2",
        "command": "make bench-paged",
        "methodology": METHODOLOGY,
        "bandwidth_source": "bench/results/2026-10-31-m2-bandwidth.json",
        "bandwidth_median_bytes_per_s": bandwidth,
        "environment": before,
        "nvidia_smi_under_load": clocks["under_load"],
        "nvidia_smi_during_step": clocks["step"],
        "nvidia_smi_after": nvidia_smi(),
        "kernel": kernel_rows,
        "ragged": ragged_rows,
        "steps": steps,
    }
    out = ROOT / "bench" / "results" / f"{report['date']}-m3-paged.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
