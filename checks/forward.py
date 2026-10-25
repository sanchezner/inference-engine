import torch
from functools import partial

from checks.agreement import assert_bf16, mean_kl_and_top1
from checks.oracle import load_engine, load_gpt2, load_smollm
from checks.tolerances import GPT2_LOGIT_ATOL, LLAMA_LOGIT_ATOL, assert_logits
from engine.generate import generate
from engine.legacy.gpt2 import forward
from engine.model import forward as forward_llama


def main():
    torch.manual_seed(0)
    _gpt2()
    _llama_fp32("cpu")
    if not torch.cuda.is_available():
        raise SystemExit("cuda is required: M1 checks the engine on the GPU")
    _llama_fp32("cuda")
    _llama_bf16()


def _gpt2():
    prompt = "The capital of France is"
    tok, model = load_gpt2()
    gpt2_fwd = partial(forward, model)
    ids = tok(prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        logits = model(ids).logits[0, -1]
        my_logits = gpt2_fwd(ids)
        mine = generate(gpt2_fwd, ids, max_new_tokens=10, eos_id=model.config.eos_token_id)
        ref = model.generate(ids, max_new_tokens=10, do_sample=False, pad_token_id=model.config.eos_token_id)

    assert_logits("gpt2", my_logits, logits, GPT2_LOGIT_ATOL)
    assert torch.equal(mine, ref)


def _llama_fp32(device):
    prompt = "The capital of France is"
    tok, model = load_smollm(device=device, dtype=torch.float32)
    cfg, weights, rope = load_engine(device, torch.float32)
    assert rope.cos.dtype == torch.float32 and rope.sin.dtype == torch.float32
    assert rope.cos.device == weights.embed.device
    uncached = partial(forward_llama, weights, cfg, rope=rope)
    ids = tok(prompt, return_tensors="pt").input_ids
    eos = model.config.eos_token_id
    assert cfg.eos_token_ids[0] == eos

    with torch.no_grad():
        ref_logits = model(ids.to(device)).logits[:, -1]
        my_logits = uncached(ids)
        two = uncached(ids.repeat(2, 1))
        mine = generate(uncached, ids, max_new_tokens=10, eos_id=eos)
        ref_ids = model.generate(ids.to(device), max_new_tokens=10, do_sample=False, pad_token_id=eos)

    # x[:, -1] keeps every row. GEMMs are not bitwise batch-invariant, so the
    # recorded logit atol is the check, not torch.equal between the two rows.
    assert my_logits.shape == (1, cfg.vocab_size)
    assert my_logits.device.type == torch.device(device).type
    assert my_logits.dtype == torch.float32
    assert two.shape == (2, cfg.vocab_size)
    assert_logits(f"llama_batch_row0_{device}", two[0], my_logits[0], LLAMA_LOGIT_ATOL)
    assert_logits(f"llama_batch_row1_{device}", two[1], my_logits[0], LLAMA_LOGIT_ATOL)
    assert_logits(f"llama_{device}", my_logits, ref_logits, LLAMA_LOGIT_ATOL)
    assert torch.equal(mine.cpu(), ref_ids.cpu())


def _llama_bf16():
    # Plumbing: the forward allocates in the weight dtype. Quality is the
    # KL / top-1 check below, not an atol against fp32.
    cfg, weights, rope = load_engine("cuda", torch.bfloat16)
    tok, oracle = load_smollm(device="cuda", dtype=torch.float32)
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        logits = forward_llama(weights, cfg, ids, rope)
    assert logits.dtype == torch.bfloat16
    assert logits.shape == (1, cfg.vocab_size)
    assert logits.device.type == "cuda"

    def start():
        return None

    def step(_state, prefix):
        return forward_llama(weights, cfg, prefix, rope), None

    with torch.no_grad():
        kl_value, top1 = mean_kl_and_top1(tok, oracle, start, step)
    assert_bf16(kl_value, top1)


if __name__ == "__main__":
    main()
