# inference engine

## what this is

A from-scratch inference runtime I'm building to get a deeper understanding of inference engines.

## log:

**<ins>2026-10-24</ins> ~ Off the Hugging Face module**

The Llama runtime loads its own config and safetensors and no longer reads a Hugging Face module. The uncached forward and the page pool run on CPU fp32, CUDA fp32, and CUDA bf16 on the RTX 3070. fp32 greedy output matches Hugging Face, including the out-of-order block table `[4, 5, 1]`. bf16 is scored by KL and teacher-forced top-1, and the first measured limits are frozen in `tests/thresholds.toml`. Page state is a `PagePool` and a `Sequence`. The dense cache stays in `engine/legacy/` as the CPU ablation.

**<ins>2026-10-20</ins> ~ Checks can fail + refactor**

The printed comparisons are assertions now. `make check` runs the GPT-2 forward, the dense cache, static and continuous batching, and the page pool, and it fails if a logit drifts past the recorded tolerance or if a retired sequence doesn't return its pages. GPT-2 and the dense-cache batcher were moved to `engine/legacy/` and stay there as the ablation baseline.

**<ins>2026-10-15</ins> ~ A pool of pages** [[blog post](https://sanchezner.com/2026/10/15/a-pool-of-pages)]

Keys and values now live in a shared page pool instead of one dense rectangle per batch. One sequence matches the dense cached generate, two sequences share the pool, and a finished sequence returns its pages while the survivor keeps its block ids, including the out-of-order list `[4, 5, 1]`. The continuous batcher admits and retires on that pool without shifting anyone else’s pages.

**<ins>2026-10-12</ins> ~ A cache, then a batch** [[blog post](https://sanchezner.com/2026/10/12/a-cache-then-a-batch)]

The GPT-2 forward now sits in a greedy loop that rescores the whole sequence every token, matching Hugging Face for 10 tokens, and the same prompt runs on SmolLM2-135M with RMSNorm, RoPE, grouped-query attention, and a SwiGLU MLP. A KV cache scores only the new token. Four prompts share that forward, mixed lengths are left-padded, and a continuous batch drops a finished row and admits the next request into the free slot.

**<ins>2026-10-03</ins> ~ Forward pass matches the oracle** [[blog post](https://sanchezner.com/2026/10/03/a-naive-forward-pass)]

Wrote the GPT-2-small forward pass and checked it against the Hugging Face `gpt2` checkpoint on "The capital of France is". Token and position embeddings feed 12 pre-norm blocks, each doing causal attention and a tanh-GELU MLP, and the last position is multiplied by the tied token embedding to score the next token.