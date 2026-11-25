# Decode step replayed from a CUDA graph against the eager step.
# make bench-graph

import json
import statistics
import time
from collections import Counter
from datetime import date
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch.profiler import ProfilerActivity, profile

from bench.env import environment, nvidia_smi
from bench.roofline import decode_ceiling_tokens_per_s, decode_step_bytes
from engine.config import ModelConfig
from engine.decode_graph import DecodeGraphs, greedy_tokens
from engine.model import RopeTables
from engine.paged import forward_paged, forward_paged_batch, new_page_pool, new_sequence
from engine.weights import Weights


REPO = "HuggingFaceTB/SmolLM2-1.7B"
DTYPE = torch.bfloat16
BLOCK_SIZE = 16
WARMUP = 5
SAMPLES = 10
CASES = (
    (128,),
    (128,) * 8,
    (512,) * 8,
)
PROFILE_CASE = (512,) * 8
BANDWIDTH_FILE = "bench/results/2026-10-31-bandwidth.json"
MODES = ("eager_item", "eager_once", "graph")

METHODOLOGY = (
    "SmolLM2-1.7B, CUDA bf16, block size 16, WSL2, clocks not locked. "
    "A decode step is one new token per sequence. Each sequence is prefilled once; timed steps rewind seq_len "
    "so every step sees the same context and overwrites the same KV slot. "
    "eager_item is forward_paged_batch, argmax, then one .item() per sequence, which is the sync the profile measured. "
    "eager_once is the same forward with one .tolist() for the whole batch. "
    "graph is DecodeGraphs around that forward, same one .tolist(). "
    "The three modes alternate step by step after five warmup steps of each, so the capture is not in the samples. "
    "Ten samples, median wall from perf_counter after a sync, and the CUDA-event span of the forward. "
    "Tokens/s is batch / median wall seconds. The ceiling is decode_step_bytes at the longest prefill length "
    "over the median copy in bench/results/2026-10-31-bandwidth.json, the same denominator as the paged-decode step table. "
    "The trace is one sample of batch 8, context 512, after warmup, and it includes the harness sync that brackets the sample. "
    "Kernel events are the launches the profiler sees, including the kernels inside one graph replay. "
    "Syncs are CPU ops whose name contains item, tolist, or synchronize. "
    "The graph path's own device-to-host copy is the one pageable transfer from the token list; "
    "the device sync and the event sync are the timer. "
    "A batch that is not 1, 2, 4, 8, 16, or 32 does not take this path. Host is WSL2."
)


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
    trace = None
    with torch.no_grad():
        warm = torch.empty(4096, 4096, device="cuda", dtype=DTYPE)
        for _ in range(3):
            warm @ warm
        torch.cuda.synchronize()
        del warm
        for lengths in CASES:
            case = _case(weights, cfg, rope, lengths, bandwidth, clocks)
            cases.append(case)
            row = case["modes"]
            print(
                f"b={case['batch']} ctx={case['context']}  "
                f"item {row['eager_item']['wall_ms']:.2f} ms  "
                f"once {row['eager_once']['wall_ms']:.2f} ms  "
                f"graph {row['graph']['wall_ms']:.2f} ms  "
                f"{row['eager_item']['tokens_per_s']:.1f} -> {row['graph']['tokens_per_s']:.1f} tok/s  "
                f"{case['graph_fraction_of_ceiling']:.3f} of ceiling",
                flush=True,
            )
            if lengths == PROFILE_CASE:
                try:
                    trace = _trace(weights, cfg, rope, lengths)
                except Exception as exc:
                    trace = {"error": f"{type(exc).__name__}: {exc}"}
                    print(f"  trace failed: {trace['error']}", flush=True)
                else:
                    for label in ("eager_item", "graph"):
                        row = trace.get(label, {})
                        print(
                            f"  {label} launches {row.get('kernel_launches')}  syncs {row.get('syncs')}",
                            flush=True,
                        )
    report = {
        "part": "decode-graph",
        "date": date.today().isoformat(),
        "host": "WSL2",
        "command": "make bench-graph",
        "methodology": METHODOLOGY,
        "model": REPO,
        "dtype": "bfloat16",
        "block_size": BLOCK_SIZE,
        "bandwidth_source": BANDWIDTH_FILE,
        "bandwidth_median_bytes_per_s": bandwidth,
        "environment": before,
        "nvidia_smi_under_load": clocks["under_load"],
        "nvidia_smi_after": nvidia_smi(),
        "cases": cases,
        "trace": trace,
    }
    out = Path(__file__).resolve().parents[1] / "bench" / "results" / f"{report['date']}-graph.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out)


def _case(weights, cfg, rope, lengths, bandwidth, clocks) -> dict:
    seqs = _prepare(weights, cfg, rope, lengths)
    if clocks["under_load"] is None and lengths == PROFILE_CASE:
        loaded = DecodeGraphs()
        for _ in range(4):
            _one_step(weights, cfg, rope, seqs, lengths, "graph", loaded, *_events())
        clocks["under_load"] = nvidia_smi()
        torch.cuda.synchronize()
    modes = _time(weights, cfg, rope, seqs, lengths)
    batch = len(lengths)
    context = max(lengths)
    ceiling = decode_ceiling_tokens_per_s(bandwidth, decode_step_bytes(cfg, 2, batch, context), batch)
    del seqs
    torch.cuda.empty_cache()
    return {
        "batch": batch,
        "context": context,
        "ceiling_tokens_per_s": ceiling,
        "modes": modes,
        "graph_fraction_of_ceiling": modes["graph"]["tokens_per_s"] / ceiling,
        "eager_item_fraction_of_ceiling": modes["eager_item"]["tokens_per_s"] / ceiling,
        "wall_ratio_item_over_graph": modes["eager_item"]["wall_ms"] / modes["graph"]["wall_ms"],
    }


def _time(weights, cfg, rope, seqs, lengths) -> dict:
    start, end = _events()
    runners = {name: DecodeGraphs() for name in MODES}
    for name in MODES:
        for _ in range(WARMUP):
            _one_step(weights, cfg, rope, seqs, lengths, name, runners[name], start, end)
    walls = {name: [] for name in MODES}
    spans = {name: [] for name in MODES}
    for _ in range(SAMPLES):
        for name in MODES:
            wall, span = _one_step(weights, cfg, rope, seqs, lengths, name, runners[name], start, end)
            walls[name].append(wall)
            spans[name].append(span)
    out = {}
    for name in MODES:
        wall = statistics.median(walls[name])
        out[name] = {
            "wall_ms": wall,
            "wall_ms_min": min(walls[name]),
            "wall_ms_max": max(walls[name]),
            "gpu_span_ms": statistics.median(spans[name]),
            "samples": SAMPLES,
            "tokens_per_s": len(lengths) / (wall / 1e3),
        }
    return out


def _one_step(weights, cfg, rope, seqs, lengths, mode, runner, start, end):
    graphs = runner if mode == "graph" else None
    torch.cuda.synchronize()
    host0 = time.perf_counter()
    start.record()
    logits, _ = forward_paged_batch(
        weights, cfg, rope, torch.randint(0, cfg.vocab_size, (len(seqs), 1), device="cuda"), seqs, graphs=graphs,
    )
    end.record()
    if mode == "eager_item":
        chosen = torch.argmax(logits, dim=-1)
        for i in range(chosen.shape[0]):
            chosen[i].item()
    else:
        greedy_tokens(logits)
    _rewind(seqs, lengths)
    end.synchronize()
    return (time.perf_counter() - host0) * 1e3, start.elapsed_time(end)


def _trace(weights, cfg, rope, lengths) -> dict:
    seqs = _prepare(weights, cfg, rope, lengths)
    runners = {name: DecodeGraphs() for name in ("eager_item", "graph")}
    start, end = _events()
    for name in runners:
        for _ in range(2):
            _one_step(weights, cfg, rope, seqs, lengths, name, runners[name], start, end)
    out = {}
    for mode in runners:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            _one_step(weights, cfg, rope, seqs, lengths, mode, runners[mode], start, end)
        syncs = Counter()
        for event in prof.key_averages():
            label = str(event.key)
            if _is_sync(label):
                syncs[label] += event.count
        raw = _chrome(prof)
        kernel_names = Counter()
        copies = Counter()
        cats = Counter()
        for event in raw:
            if not isinstance(event, dict) or event.get("ph") != "X":
                continue
            label = str(event.get("name", ""))
            cat = str(event.get("cat", ""))
            cats[cat] += 1
            if cat == "kernel":
                kernel_names[label] += 1
            if "DtoH" in label or "dtoh" in label.lower():
                copies[label] += 1
        out[mode] = {
            "kernel_launches": sum(kernel_names.values()),
            "syncs": {key: syncs[key] for key in sorted(syncs)},
            "device_to_host": {key: copies[key] for key in sorted(copies)},
            "categories": {key: cats[key] for key in sorted(cats)},
        }
    del seqs
    torch.cuda.empty_cache()
    return out


def _chrome(prof):
    path = Path("/tmp/decode-graph-trace.json")
    prof.export_chrome_trace(str(path))
    try:
        data = json.loads(path.read_text())
    finally:
        path.unlink(missing_ok=True)
    return data["traceEvents"] if isinstance(data, dict) else data


def _is_sync(name: str) -> bool:
    lowered = name.lower()
    return "item" in lowered or "tolist" in lowered or "synchronize" in lowered


def _prepare(weights, cfg, rope, lengths):
    pages = sum((n + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE for n in lengths)
    pool = new_page_pool(cfg, n_pages=pages, block_size=BLOCK_SIZE, device="cuda", dtype=weights.embed.dtype)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    seqs = []
    for n in lengths:
        seq = new_sequence(pool)
        ids = torch.randint(0, cfg.vocab_size, (1, n), device="cuda", generator=generator)
        forward_paged(weights, cfg, rope, ids, seq)
        seqs.append(seq)
    return seqs


def _rewind(seqs, lengths) -> None:
    for seq, n in zip(seqs, lengths):
        seq.seq_len = n


def _events():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _copy_bandwidth() -> float:
    path = Path(__file__).resolve().parents[1] / BANDWIDTH_FILE
    return json.loads(path.read_text())["bandwidth"]["median_bytes_per_s"]


if __name__ == "__main__":
    main()
