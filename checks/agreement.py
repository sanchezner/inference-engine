import tomllib
from pathlib import Path

import torch


# The prompts already used by the logit and batch checks. Agreement is
# teacher-forced: each position is scored on the oracle prefix, so one
# near-tie does not sink every token after it. Token identity is the fp32 test.
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
    # KL(oracle || engine), mean over the batch. Both sides are promoted to
    # fp32 on CPU so a bf16 engine can be scored against an fp32 oracle.
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


def bf16_limits():
    path = Path(__file__).resolve().parents[1] / "tests" / "thresholds.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)["cuda-bf16"]


def assert_bf16(kl_value, top1):
    limits = bf16_limits()
    assert kl_value <= limits["mean_kl_max"], f"mean KL {kl_value:.6f} > {limits['mean_kl_max']}"
    assert top1 >= limits["top1_min"], f"top-1 agreement {top1:.4f} < {limits['top1_min']}"
