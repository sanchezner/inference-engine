import tomllib
from pathlib import Path

import torch


PROMPTS = (
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Spain is",
    "The capital of Italy is",
    "The cat",
    "Once upon a time there was a",
)
NEW_TOKENS = 10


def kl(oracle_logits, engine_logits) -> torch.Tensor:
    # KL(oracle || engine) in fp32 on CPU.
    log_p = torch.log_softmax(oracle_logits.detach().float().cpu(), dim=-1)
    log_q = torch.log_softmax(engine_logits.detach().float().cpu(), dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1).mean()


def mean_kl_and_top1(tok, oracle, start, step):
    device = next(oracle.parameters()).device
    kls = []
    hits = 0
    total = 0
    eos = oracle.config.eos_token_id
    for prompt in PROMPTS:
        prompt_ids = tok(prompt, return_tensors="pt").input_ids
        ref_ids = oracle.generate(
            prompt_ids.to(device),
            max_new_tokens=NEW_TOKENS,
            do_sample=False,
            pad_token_id=eos,
        ).cpu()
        state = start()
        logits, state = step(state, prompt_ids)
        ref_logits = oracle(prompt_ids.to(device)).logits[:, -1]
        kls.append(float(kl(ref_logits, logits)))
        prompt_len = prompt_ids.shape[1]
        for t in range(ref_ids.shape[1] - prompt_len):
            pred = int(torch.argmax(logits, dim=-1).item())
            hits += pred == int(ref_ids[0, prompt_len + t].item())
            total += 1
            logits, state = step(state, ref_ids[:, : prompt_len + t + 1])
    return sum(kls) / len(kls), hits / total


def assert_bf16(kl_value, top1, section="cuda-bf16", label=""):
    path = Path(__file__).resolve().parents[1] / "tests" / "thresholds.toml"
    with open(path, "rb") as f:
        limits = tomllib.load(f)[section]
    prefix = f"{label}: " if label else ""
    assert kl_value <= limits["mean_kl_max"], (
        f"{prefix}mean KL {kl_value:.6f} > {limits['mean_kl_max']}"
    )
    assert top1 >= limits["top1_min"], (
        f"{prefix}top-1 agreement {top1:.4f} < {limits['top1_min']}"
    )


def fixed_greedy_trace(tok, oracle, prompts, new_tokens):
    # Oracle ids and the logits that chose them. EOS is not a stop, so every prompt contributes new_tokens rows.
    trace = []
    for prompt in prompts:
        prompt_ids = tok(prompt, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]
        out = oracle.generate(
            prompt_ids,
            max_new_tokens=new_tokens,
            do_sample=False,
            eos_token_id=None,
            return_dict_in_generate=True,
            output_scores=True,
        )
        ids = out.sequences.cpu()
        if ids.shape[1] != prompt_len + new_tokens:
            raise AssertionError(
                f"{prompt!r} produced {ids.shape[1] - prompt_len} tokens, wanted {new_tokens}"
            )
        scores = torch.stack([row.detach().float().cpu() for row in out.scores])
        trace.append((prompt_len, ids, scores))
    return trace


def score_trace(trace, start, step):
    # Teacher-forced KL and top-1. The engine sees the oracle prefix at every position.
    kls = []
    hits = 0
    for prompt_len, ids, scores in trace:
        state = start()
        logits, state = step(state, ids[:, :prompt_len])
        n = scores.shape[0]
        for t in range(n):
            kls.append(float(kl(scores[t], logits)))
            pred = int(torch.argmax(logits, dim=-1).item())
            hits += pred == int(ids[0, prompt_len + t].item())
            if t + 1 < n:
                logits, state = step(state, ids[:, : prompt_len + t + 1])
    total = len(kls)
    return sum(kls) / total, hits / total, total
