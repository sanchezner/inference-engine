# fp32 CPU, WSL2, torch 2.14.1+cu132, 2026-10-20.
# Observed max-abs logit error: gpt2 2.14e-4, SmolLM2 at most 2.67e-5.
# A real attention or norm bug lands near 1e-1 or worse, so these sit
# a few times above the measured noise and well below a broken forward.
GPT2_LOGIT_ATOL = 1e-3
LLAMA_LOGIT_ATOL = 1e-4


def assert_logits(name, got, ref, atol):
    err = (got - ref).abs().max().item()
    assert err <= atol, f"{name}: max abs err {err} > {atol}"
