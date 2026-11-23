# paged-attention decode: kernel bandwidth, and the decode step against gather
# make bench-paged

import json
import statistics
import time
from contextlib import contextmanager
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
from engine.kernels.triton.paged_attn import _sm_count, choose_splits, paged_attn_decode
from engine.model import RopeTables
from engine.paged import forward_paged, forward_paged_batch, new_page_pool, new_sequence
from engine.weights import Weights

import engine.kernels.triton.paged_attn as paged_attn
import engine.paged as paged


ROOT = Path(__file__).resolve().parents[2]
BANDWIDTH_PATH = ROOT / "bench" / "results" / "2026-10-31-bandwidth.json"
# Head counts from checks/roofline.py. The kernel does not need the weights.
GEOMETRIES = (
    {"name": "135M", "n_heads": 9, "n_kv_heads": 3, "head_dim": 64},
    {"name": "1.7B", "n_heads": 32, "n_kv_heads": 32, "head_dim": 64},
)
BLOCK_SIZES = (8, 16, 32)
BATCHES = (1, 2, 4, 8, 32)
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
STEP_MODES = ("kernel", "unsplit", "gather")
DTYPE = torch.bfloat16
KERNEL_WINDOW_MS = 40.0
KERNEL_REPS = 5
CALLS_PER_REPLAY = 16
L2_MULTIPLE = 4
STEP_WARMUP = 5
STEP_TIMING = 10

METHODOLOGY = (
    "Two clocks. The kernel clock is a CUDA graph of 16 paged_attn_decode calls back to back, so a short kernel is not waiting on the host to submit the next replay; the reported time is per call. The calls rotate across copies of the K/V pool until the copies total at least four times the L2 cache (4 MiB on the 3070), so a small pool is read from DRAM and not from L2. With split-KV the call is two launches (the split pass and the merge) plus the fp32 partials; both are inside the graph and inside the time. Split counts come from choose_splits on the table width; when it picks more than one, the same buffers are also timed at num_splits=1 under 'unsplit', with the windows of the two alternating. Bytes are q read plus the output written plus K and V read, for the live tokens only, one layer. The partials are not counted as bytes, so split-KV pays for them in time. GQA does not multiply the KV bytes by the group size. Contexts and the ragged lengths are multiples of the block size, so the last page is full and the mask does not drop slots. Block tables are a seeded shuffle of the pages, which is the paged case; one sequential table is timed at the same bytes to show the shuffle. Five windows, each aimed at 40 ms; the reported time is the median window. The fraction is that GB/s over the median device-to-device copy in bench/results/2026-10-31-bandwidth.json (378.7 GB/s, read plus write, decimal 1e9). The step clock is forward_paged_batch on CUDA bf16, block size 16, the same quantity as bench/profile.py: CUDA events around the forward, then argmax and a per-sequence .item(), after a sync so the card is idle at the start. Three modes per case (kernel, unsplit, gather): five warmup steps each, then ten timed rounds that rotate through the modes one step at a time, so background load lands on every column alike; seq_len is rewound so every step sees the same context and overwrites the same KV slot. The gather path is the same forward with the decode kernel switched off, so attention gathers pages and the cache write stays the scatter kernel. The unsplit step is the kernel path with choose_splits forced to 1. Tokens/s is batch / median wall seconds. The ceiling uses that same copy bandwidth and decode_step_bytes at the longest row, which is how the M2 table was computed. Host is WSL2. Clocks are not locked."
)


def _copy_bandwidth() -> float:
    report = json.loads(BANDWIDTH_PATH.read_text())
    return float(report["bandwidth"]["median_bytes_per_s"])


def _graph(launch):
    # Several calls per replay. One call per replay lets the host's replay cost starve a 10 us kernel.
    for i in range(CALLS_PER_REPLAY):
        launch(i)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(CALLS_PER_REPLAY):
            launch(i)
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _pool_copies(k, v):
    # Consecutive calls read different copies, so a pool smaller than L2 is still read from DRAM.
    l2 = torch.cuda.get_device_properties(k.device).L2_cache_size
    n = min(CALLS_PER_REPLAY, max(1, -(-L2_MULTIPLE * l2 // (k.nbytes + v.nbytes))))
    return [(k, v)] + [(k.clone(), v.clone()) for _ in range(n - 1)]


def _window_ms(graph, iters: int) -> float:
    # Milliseconds per call, not per replay.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / (iters * CALLS_PER_REPLAY)


def _choose_iters(graph) -> int:
    sample = _window_ms(graph, 1) * CALLS_PER_REPLAY
    if sample <= 0:
        return 50
    return min(500, max(20, int(KERNEL_WINDOW_MS / sample)))


def _time_graphs(graphs) -> list[dict]:
    # Windows alternate between graphs, so a display redraw lands on both sides of a comparison.
    iters = [_choose_iters(graph) for graph in graphs]
    samples = [[] for _ in graphs]
    for _ in range(KERNEL_REPS):
        for graph, n, out in zip(graphs, iters, samples):
            out.append(_window_ms(graph, n))
    return [
        {
            "iters_per_window": n,
            "windows": KERNEL_REPS,
            "samples_ms": s,
            "median_ms": statistics.median(s),
            "min_ms": min(s),
            "max_ms": max(s),
        }
        for n, s in zip(iters, samples)
    ]


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
    if order == "sequential":
        perm = torch.arange(n_pages, device="cuda")
    else:
        perm = torch.randperm(n_pages, device="cuda", generator=generator)
    bt = perm.to(dtype=torch.int32).view(batch, pages_each)
    sl = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    return q, k, v, bt, sl


def _ragged_buffers(geo, block_size, lengths, seed):
    needs = [n // block_size for n in lengths]
    n_pages = sum(needs)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    k, v = _fill(n_pages, geo["n_kv_heads"], block_size, geo["head_dim"], generator)
    batch = len(lengths)
    q = torch.randn(batch, geo["n_heads"], geo["head_dim"], device="cuda", dtype=DTYPE, generator=generator)
    perm = torch.randperm(n_pages, device="cuda", generator=generator)
    width = max(needs)
    bt = torch.zeros(batch, width, device="cuda", dtype=torch.int32)
    cursor = 0
    for row, need in enumerate(needs):
        bt[row, :need] = perm[cursor:cursor + need].to(dtype=torch.int32)
        cursor += need
    sl = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    return q, k, v, bt, sl


def _time_calls(q, k, v, bt, sl, split_counts, clocks=None) -> tuple[list[dict], int]:
    copies = _pool_copies(k, v)
    graphs = [
        _graph(lambda i, n=n: paged_attn_decode(q, *copies[i % len(copies)], bt, sl, num_splits=n, check=False))
        for n in split_counts
    ]
    try:
        if clocks is not None:
            for _ in range(20):
                graphs[0].replay()
            clocks["under_load"] = nvidia_smi()
            torch.cuda.synchronize()
        return _time_graphs(graphs), len(copies)
    finally:
        del graphs, copies


def _measure(q, k, v, bt, sl, geo, block_size, bandwidth, clocks) -> dict:
    batch = q.shape[0]
    splits = choose_splits(batch, geo["n_heads"], bt.shape[1] * block_size, _sm_count(q.device.index or 0))
    sample_clocks = clocks["under_load"] is None and geo["name"] == "1.7B" and batch >= 8 and int(sl.max()) >= 2048
    timings, n_copies = _time_calls(q, k, v, bt, sl, (splits, 1) if splits > 1 else (1,), clocks if sample_clocks else None)
    timed = timings[0]
    seq_lens = [int(n) for n in sl.tolist()]
    moved = paged_decode_kernel_bytes(geo["n_heads"], geo["n_kv_heads"], geo["head_dim"], 2, seq_lens)

    def score(t):
        seconds = t["median_ms"] / 1e3
        return {"gbps": (moved / seconds) / 1e9, "fraction_of_copy": (moved / seconds) / bandwidth}

    row = {
        "n_heads": geo["n_heads"],
        "n_kv_heads": geo["n_kv_heads"],
        "head_dim": geo["head_dim"],
        "block_size": block_size,
        "batch": len(seq_lens),
        "seq_lens": seq_lens,
        "sum_len": sum(seq_lens),
        "max_len": max(seq_lens),
        "num_splits": splits,
        "grid": [len(seq_lens), geo["n_heads"], splits],
        "pool_copies": n_copies,
        "calls_per_replay": CALLS_PER_REPLAY,
        "bytes": moved,
        **score(timed),
        **timed,
    }
    if splits > 1:
        row["unsplit"] = {**score(timings[1]), **timings[1]}
    return row


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
                    unsplit = f"  unsplit {row['unsplit']['gbps']:.1f} GB/s" if "unsplit" in row else ""
                    print(
                        f"kernel {geo['name']} b={batch} L={context} block={block_size} splits={row['num_splits']}  "
                        f"{row['gbps']:.1f} GB/s  {row['fraction_of_copy'] * 100:.1f}%{unsplit}",
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


@contextmanager
def _mode(name):
    saved_choose = paged_attn.choose_splits
    saved_supported = paged.decode_supported
    if name == "unsplit":
        paged_attn.choose_splits = lambda *_args, **_kwargs: 1
    elif name == "gather":
        paged.decode_supported = lambda *_args, **_kwargs: False
    try:
        yield
    finally:
        paged_attn.choose_splits = saved_choose
        paged.decode_supported = saved_supported


def _one_step(weights, cfg, rope, seqs, lengths, start, end):
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
    return (time.perf_counter() - host0) * 1e3, start.elapsed_time(end)


def _time_steps(weights, cfg, rope, seqs, lengths) -> dict:
    # Modes rotate step by step, so background load lands on all three columns alike.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for name in STEP_MODES:
        with _mode(name):
            for _ in range(STEP_WARMUP):
                _one_step(weights, cfg, rope, seqs, lengths, start, end)
    walls = {name: [] for name in STEP_MODES}
    spans = {name: [] for name in STEP_MODES}
    for _ in range(STEP_TIMING):
        for name in STEP_MODES:
            with _mode(name):
                wall, span = _one_step(weights, cfg, rope, seqs, lengths, start, end)
            walls[name].append(wall)
            spans[name].append(span)
    return {
        name: {
            "wall_ms": statistics.median(walls[name]),
            "wall_ms_min": min(walls[name]),
            "wall_ms_max": max(walls[name]),
            "gpu_span_ms": statistics.median(spans[name]),
            "gpu_span_ms_min": min(spans[name]),
            "gpu_span_ms_max": max(spans[name]),
            "samples": STEP_TIMING,
        }
        for name in STEP_MODES
    }


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
    timed = _time_steps(weights, cfg, rope, seqs, lengths)
    kernel, unsplit, gather = timed["kernel"], timed["unsplit"], timed["gather"]
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
    width = max((n + 1 + STEP_BLOCK - 1) // STEP_BLOCK for n in lengths)
    return {
        "batch": batch,
        "lengths": list(lengths),
        "max_len": max(lengths),
        "sum_len": sum(lengths),
        "padded_tokens": batch * max(lengths),
        "block_size": STEP_BLOCK,
        "num_splits": choose_splits(batch, cfg.n_heads, width * STEP_BLOCK, _sm_count(device.index or 0)),
        "kernel": kernel,
        "kernel_unsplit": unsplit,
        "gather": gather,
        "kernel_tokens_per_s": batch / kernel_s,
        "kernel_unsplit_tokens_per_s": batch / (unsplit["wall_ms"] / 1e3),
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
                    f"kernel {case['kernel']['wall_ms']:.1f} ms (splits {case['num_splits']})  "
                    f"unsplit {case['kernel_unsplit']['wall_ms']:.1f} ms  gather {case['gather']['wall_ms']:.1f} ms  "
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
        "bandwidth_source": "bench/results/2026-10-31-bandwidth.json",
        "bandwidth_median_bytes_per_s": bandwidth,
        "environment": before,
        "nvidia_smi_under_load": clocks["under_load"],
        "nvidia_smi_during_step": clocks["step"],
        "nvidia_smi_after": nvidia_smi(),
        "kernel": kernel_rows,
        "ragged": ragged_rows,
        "steps": steps,
    }
    out = ROOT / "bench" / "results" / f"{report['date']}-paged.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
