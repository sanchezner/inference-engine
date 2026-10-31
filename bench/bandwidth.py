# achievable device-to-device bandwidth, and the decode ceiling it implies.
# make bench-bandwidth

import json
import statistics
from datetime import date
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from bench.env import environment, nvidia_smi, snapshot_revision
from bench.roofline import (
    decode_ceiling_tokens_per_s,
    decode_step_bytes,
    kv_bytes_per_token,
    parameter_elements,
    streamed_weight_bytes,
    streamed_weight_elements,
)
from engine.config import ModelConfig


MODELS = ("HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-1.7B")
DTYPES = {"bf16": 2, "fp32": 4}
BATCHES = (1, 2, 4, 8, 16, 32)
CONTEXTS = (128, 512, 2048)
BUFFER_BYTES = 1 << 30
COPIES_PER_SAMPLE = 100
WARMUP_SAMPLES = 1
REPS = 10
MIN_MEDIAN_GBPS = 100

METHODOLOGY = (
    "Device-to-device copy of a 1 GiB buffer, timed with CUDA events. Bytes moved per copy are 2 * buffer (one read and one write), which is what the spec-sheet GB/s number counts. Each sample times 100 back-to-back copies, because a single copy is short enough for the memory clock to drop and this RTX 3070 is also driving the display. One untimed sample warms up, then ten timed samples; the ceiling uses the median. Clocks are read while a copy series is queued, and once more immediately after the last sample. That second read can still see boost; the idle clock is the sample taken before any copy. Clocks are not locked. The decode ceiling is batch * bandwidth / step_bytes, where a step reads every streamed weight once, one embedding row per sequence, and the live KV of every sequence (K and V, context includes the new token). Activations are not in the byte count. The 1.7B fp32 ceiling is arithmetic only: those weights are about 6.8 GB and do not fit beside a CUDA context on this 8 GB card. Host is WSL2."
)


def _time_copies(src, dst, copies, start, end) -> float:
    start.record()
    for _ in range(copies):
        dst.copy_(src)
    end.record()
    end.synchronize()
    seconds = start.elapsed_time(end) / 1e3
    return (2 * src.numel() * copies) / seconds


def clocks_under_load(src, dst, copies) -> dict:
    # queue the work first, then nvidia-smi samples while GPU is still draining the series, instead of after it falls back to the idle clock
    for _ in range(copies):
        dst.copy_(src)
    sample = nvidia_smi()
    torch.cuda.synchronize()
    return sample


def measure_copy(
    nbytes: int = BUFFER_BYTES,
    copies: int = COPIES_PER_SAMPLE,
    warmup: int = WARMUP_SAMPLES,
    reps: int = REPS,
) -> dict:
    if not torch.cuda.is_available():
        raise SystemExit("cuda is not available")
    src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    src.fill_(1)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        _time_copies(src, dst, copies, start, end)
    under_load = clocks_under_load(src, dst, copies)

    samples_bps = [_time_copies(src, dst, copies, start, end) for _ in range(reps)]
    median = statistics.median(samples_bps)
    return {
        "method": "device-to-device copy; bytes = 2 * buffer * copies (read + write)",
        "buffer_bytes": nbytes,
        "copies_per_sample": copies,
        "warmup_samples": warmup,
        "repetitions": reps,
        "nvidia_smi_under_load": under_load,
        "samples_bytes_per_s": samples_bps,
        "median_bytes_per_s": median,
        "min_bytes_per_s": min(samples_bps),
        "max_bytes_per_s": max(samples_bps),
        "median_gbps": median / 1e9,
        "min_gbps": min(samples_bps) / 1e9,
        "max_gbps": max(samples_bps) / 1e9,
        "fraction_of_spec": (median / 1e9) / 448,
        "spec_gbps_stock_3070": 448,
        "spec_note": "Stock RTX 3070 is ~448 GB/s. A 3070 Ti is ~608 GB/s; the device name above decides which.",
    }


def _ceilings(cfg: ModelConfig, dtype_bytes: int, bandwidth: float) -> list[dict]:
    rows = []
    for batch in BATCHES:
        for context in CONTEXTS:
            step = decode_step_bytes(cfg, dtype_bytes, batch, context)
            rows.append(
                {
                    "batch": batch,
                    "context": context,
                    "step_bytes": step,
                    "tokens_per_s": decode_ceiling_tokens_per_s(bandwidth, step, batch),
                }
            )
    return rows


def model_report(repo: str, bandwidth: float) -> dict:
    cfg = ModelConfig.from_json(hf_hub_download(repo, "config.json", local_files_only=True))
    dtypes = {}
    for name, width in DTYPES.items():
        weight_bytes = streamed_weight_bytes(cfg, width)
        dtypes[name] = {
            "dtype_bytes": width,
            "streamed_weight_bytes": weight_bytes,
            "streamed_weight_gib": weight_bytes / (1024**3),
            "kv_bytes_per_token": kv_bytes_per_token(cfg, width),
            "ceilings": _ceilings(cfg, width, bandwidth),
        }
    return {
        "revision": snapshot_revision(repo),
        "parameter_elements": parameter_elements(cfg),
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "dtypes": dtypes,
    }


def main():
    before = environment()
    measured = measure_copy()
    after = nvidia_smi()
    if measured["median_gbps"] < MIN_MEDIAN_GBPS:
        raise SystemExit(
            f"median copy bandwidth {measured['median_gbps']:.1f} GB/s is below {MIN_MEDIAN_GBPS}; the discrete GPU did not take the copy"
        )
    report = {
        "milestone": "M2",
        "part": "bandwidth-and-roofline",
        "date": date.today().isoformat(),
        "host": "WSL2",
        "command": "make bench-bandwidth",
        "methodology": METHODOLOGY,
        "environment": before,
        "nvidia_smi_after": after,
        "bandwidth": measured,
        "roofline": {
            "formula": "tokens_per_s = batch * median_bytes_per_s / step_bytes",
            "step_bytes": "streamed weights + batch * hidden * dtype_bytes + batch * context * kv_bytes_per_token",
            "kv_bytes_per_token": "2 * n_layers * n_kv_heads * head_dim * dtype_bytes",
            "activations_included": False,
            "models": {repo: model_report(repo, measured["median_bytes_per_s"]) for repo in MODELS},
        },
    }
    out = Path(__file__).resolve().parents[1] / "bench" / "results" / f"{report['date']}-m2-bandwidth.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out)
    print(f"median {measured['median_gbps']:.1f} GB/s  spread {measured['min_gbps']:.1f}..{measured['max_gbps']:.1f}")


if __name__ == "__main__":
    main()
