# Hugging Face generate, eager and sdpa, at the shapes the M2 profile measured.
# make bench-baseline
#
# The profile times one decode step of this engine with the cache already full.
# generate always pays for the prompt first, so this script times a prompt
# forward on its own, and times each new token from inside generate.

import sys
from pathlib import Path

# Running this file as a script puts bench/ first on sys.path.
# bench/profile.py then shadows the stdlib profile module that cProfile needs.
_ROOT = Path(__file__).resolve().parents[1]
_BENCH = _ROOT / "bench"
sys.path[:] = [str(_ROOT)] + [entry for entry in sys.path if Path(entry).resolve() != _BENCH]

import json
import statistics
import threading
import time
from datetime import date

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

from bench.env import environment, nvidia_smi, snapshot_revision
from bench.roofline import decode_ceiling_tokens_per_s, decode_step_bytes
from engine.config import ModelConfig


MODELS = ("HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-1.7B")
ATTN = ("eager", "sdpa")
# Prompt lengths match the profile's "context" column: the cache is already that long.
DECODE = ((1, 128), (8, 128), (1, 512), (8, 512))
PREFILL = ((1, 128), (1, 512), (1, 2048))
NEW_TOKENS = 32
WARMUP = 1
REPS = 5
DTYPE = torch.bfloat16
BANDWIDTH_RESULT = Path(__file__).resolve().parent / "results" / "2026-10-31-bandwidth.json"

METHODOLOGY = (
    "Hugging Face AutoModelForCausalLM.generate on CUDA bf16, greedy, EOS disabled, compile disabled, default dynamic cache. "
    "Eager and sdpa are separate loads. Prompts are synthetic token ids, equal length within a batch, seed 0. "
    "Prefill is one forward of the prompt with use_cache, timed by CUDA events after the device is idle; tokens/s is batch * prompt_len / that time. "
    "Decode is generate of 32 new tokens. A stopping criterion synchronizes and stamps the host clock once per new token. "
    "That sync is the same device wait generate already performs on the next line, when it reads the unfinished-sequence flag; the stamp is not an extra stall. "
    "TTFT is the first stamp minus the start. Inter-token latency is the gap between later stamps, so it excludes the prompt. "
    "Decode tokens/s is batch / median inter-token latency, pooled over five repetitions after one discarded warmup. "
    "The ceiling uses the median copy bandwidth in bench/results/2026-10-31-bandwidth.json and step_bytes at the prompt length, which is the context column in docs/decode-profile.md. "
    "During the 32 new tokens the live cache grows from prompt_len+1 to prompt_len+32; the weight term dominates that difference on these shapes. "
    "p99 is over the pooled gaps (31 per repetition, 155 total). Clocks are not locked. Host is WSL2. "
    "fp32 1.7B does not fit beside a CUDA context on this 8 GB card, so the baseline is bf16. "
    "Batch sizes past 8, and a prefill batch above 1, are left for the comparison that needs them."
)


class StepClock(StoppingCriteria):
    def __init__(self):
        self.times: list[float] = []
        self._stop = None

    def __call__(self, input_ids, scores=None, **kwargs):
        torch.cuda.synchronize()
        self.times.append(time.perf_counter())
        n = input_ids.shape[0]
        if self._stop is None or self._stop.shape[0] != n:
            self._stop = torch.zeros(n, dtype=torch.bool, device=input_ids.device)
        return self._stop


def _spread(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _ids(vocab: int, batch: int, prompt_len: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    return torch.randint(0, vocab, (batch, prompt_len), device="cuda", generator=generator)


def _load_bandwidth() -> float:
    report = json.loads(BANDWIDTH_RESULT.read_text())
    median = report["bandwidth"]["median_bytes_per_s"]
    if median <= 0:
        raise SystemExit(f"bandwidth median in {BANDWIDTH_RESULT} is {median}")
    return median


def load_model(repo: str, attn: str):
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        dtype=DTYPE,
        attn_implementation=attn,
        local_files_only=True,
    ).to("cuda").eval()
    # None so a sampled EOS cannot end a row early and shrink the timed token count.
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = 0
    model.generation_config.disable_compile = True
    return model


def measure_prefill(model, ids: torch.Tensor) -> dict:
    batch, prompt_len = ids.shape
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    walls = []
    peaks = []

    def once():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start.record()
        out = model(ids, use_cache=True)
        end.record()
        end.synchronize()
        del out
        walls.append(start.elapsed_time(end))
        peaks.append(torch.cuda.max_memory_allocated())

    for _ in range(WARMUP):
        once()
    walls.clear()
    peaks.clear()
    for _ in range(REPS):
        once()
    median_ms = statistics.median(walls)
    return {
        "batch": batch,
        "prompt_len": prompt_len,
        "status": "ok",
        "warmup": WARMUP,
        "repetitions": REPS,
        "wall_ms": _spread(walls),
        "tokens_per_s": batch * prompt_len / (median_ms / 1e3),
        "peak_allocated_bytes": _spread(peaks),
    }


def measure_decode(model, ids: torch.Tensor, bandwidth: float, cfg: ModelConfig, probe_clocks: dict) -> dict:
    batch, prompt_len = ids.shape
    clock = StepClock()
    criteria = StoppingCriteriaList([clock])
    ttft = []
    walls = []
    gaps = []
    peaks = []

    def once(sample_clocks: bool):
        clock.times.clear()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        holder: dict = {}
        thread = None
        if sample_clocks:
            def sample():
                # generate has to be on the device before the sample, or nvidia-smi reads the idle clock.
                time.sleep(0.3)
                holder["smi"] = nvidia_smi()

            thread = threading.Thread(target=sample)
            thread.start()
        t0 = time.perf_counter()
        try:
            out = model.generate(
                ids,
                max_new_tokens=NEW_TOKENS,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=0,
                use_cache=True,
                disable_compile=True,
                stopping_criteria=criteria,
            )
            torch.cuda.synchronize()
            wall_s = time.perf_counter() - t0
        finally:
            if thread is not None:
                thread.join()
                probe_clocks["under_load"] = holder.get("smi")
        if out.shape != (batch, prompt_len + NEW_TOKENS):
            raise RuntimeError(f"generate returned {tuple(out.shape)}, expected {(batch, prompt_len + NEW_TOKENS)}")
        if len(clock.times) != NEW_TOKENS:
            raise RuntimeError(f"{len(clock.times)} token stamps, expected {NEW_TOKENS}")
        if clock.times[0] < t0:
            raise RuntimeError("token stamp precedes generate start")
        ttft.append((clock.times[0] - t0) * 1e3)
        walls.append(wall_s * 1e3)
        gaps.extend((b - a) * 1e3 for a, b in zip(clock.times, clock.times[1:]))
        peaks.append(torch.cuda.max_memory_allocated())
        del out

    # 135M steps finish before nvidia-smi starts. The 1.7B is the shape that holds the clocks up.
    heavy = probe_clocks["under_load"] is None and cfg.n_kv_heads >= 32 and batch >= 8 and prompt_len >= 512
    for _ in range(WARMUP):
        once(sample_clocks=heavy)
    ttft.clear()
    walls.clear()
    gaps.clear()
    peaks.clear()
    for _ in range(REPS):
        once(sample_clocks=False)
    itl_ms = statistics.median(gaps)
    step_bytes = decode_step_bytes(cfg, DTYPE.itemsize, batch, prompt_len)
    ceiling = decode_ceiling_tokens_per_s(bandwidth, step_bytes, batch)
    decode_tps = batch / (itl_ms / 1e3)
    return {
        "batch": batch,
        "prompt_len": prompt_len,
        "new_tokens": NEW_TOKENS,
        "status": "ok",
        "warmup": WARMUP,
        "repetitions": REPS,
        "itl_samples": len(gaps),
        "wall_ms": _spread(walls),
        "ttft_ms": _spread(ttft),
        "itl_ms": {
            "p50": itl_ms,
            "p99": _percentile(gaps, 99),
            "min": min(gaps),
            "max": max(gaps),
        },
        "decode_tokens_per_s": decode_tps,
        "e2e_tokens_per_s": batch * NEW_TOKENS / (statistics.median(walls) / 1e3),
        "ceiling_context": prompt_len,
        "ceiling_step_bytes": step_bytes,
        "ceiling_tokens_per_s": ceiling,
        "fraction_of_ceiling": decode_tps / ceiling,
        "peak_allocated_bytes": _spread(peaks),
    }


def _skipped(kind: str, batch: int, prompt_len: int, exc: BaseException) -> dict:
    row = {
        "batch": batch,
        "prompt_len": prompt_len,
        "status": "oom",
        "error": str(exc).splitlines()[0],
    }
    if kind == "decode":
        row["new_tokens"] = NEW_TOKENS
    return row


def run_model(repo: str, attn: str, bandwidth: float, cfg: ModelConfig, probe_clocks: dict) -> dict:
    model = load_model(repo, attn)
    prefill = []
    decode = []
    try:
        with torch.inference_mode():
            for batch, prompt_len in PREFILL:
                ids = _ids(cfg.vocab_size, batch, prompt_len)
                try:
                    row = measure_prefill(model, ids)
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    row = _skipped("prefill", batch, prompt_len, exc)
                prefill.append(row)
                _log(repo, attn, "prefill", row)
            for batch, prompt_len in DECODE:
                ids = _ids(cfg.vocab_size, batch, prompt_len)
                try:
                    row = measure_decode(model, ids, bandwidth, cfg, probe_clocks)
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    row = _skipped("decode", batch, prompt_len, exc)
                decode.append(row)
                _log(repo, attn, "decode", row)
    finally:
        cache = model.generation_config.cache_implementation
        attn_impl = model.config._attn_implementation
        del model
        torch.cuda.empty_cache()
    return {
        "cache_implementation": cache,
        "attn_implementation": attn_impl,
        "prefill": prefill,
        "decode": decode,
    }


def _log(repo: str, attn: str, kind: str, row: dict) -> None:
    name = repo.rsplit("/", 1)[-1]
    if row["status"] != "ok":
        print(f"{name} {attn} {kind} b={row['batch']} n={row['prompt_len']}  {row['status']}", flush=True)
        return
    if kind == "prefill":
        print(
            f"{name} {attn} prefill b={row['batch']} n={row['prompt_len']}  "
            f"{row['wall_ms']['median']:.2f} ms  {row['tokens_per_s']:.1f} tok/s",
            flush=True,
        )
        return
    print(
        f"{name} {attn} decode b={row['batch']} n={row['prompt_len']}  "
        f"itl p50 {row['itl_ms']['p50']:.2f} ms  {row['decode_tokens_per_s']:.1f} tok/s  "
        f"{100 * row['fraction_of_ceiling']:.1f}% of ceiling",
        flush=True,
    )


def main():
    if not torch.cuda.is_available():
        raise SystemExit("cuda is not available")
    bandwidth = _load_bandwidth()
    before = environment()
    probe_clocks = {"under_load": None}
    models = {}
    for repo in MODELS:
        cfg = ModelConfig.from_json(hf_hub_download(repo, "config.json", local_files_only=True))
        attn_rows = {}
        for attn in ATTN:
            attn_rows[attn] = run_model(repo, attn, bandwidth, cfg, probe_clocks)
        models[repo] = {
            "revision": snapshot_revision(repo),
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "n_kv_heads": cfg.n_kv_heads,
            "head_dim": cfg.head_dim,
            "dtype": "bf16",
            "attn": attn_rows,
        }
    report = {
        "milestone": "M2",
        "part": "hf-generate-baseline",
        "date": date.today().isoformat(),
        "host": "WSL2",
        "command": "make bench-baseline",
        "methodology": METHODOLOGY,
        "environment": before,
        "nvidia_smi_under_load": probe_clocks["under_load"],
        "nvidia_smi_after": nvidia_smi(),
        "bandwidth_bytes_per_s": bandwidth,
        "bandwidth_source": str(BANDWIDTH_RESULT.relative_to(Path(__file__).resolve().parents[1])),
        "models": models,
    }
    out = Path(__file__).resolve().parent / "results" / f"{report['date']}-baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
