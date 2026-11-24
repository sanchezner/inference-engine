# inference engine

This is a single-GPU inference engine for Llama-architecture models. It seats waiting requests when the page pool can hold the request's prompt and its token budget, and computes every admitted request's next token in one batched forward pass.

The engine is a library, not a server. There are no threads or networking; a caller submits requests, calls `step()` to advance running requests by one token, and decides what to do with the tokens that return. Requests arrive as token ids, so the caller also handles tokenization.

Python (on the CPU) handles scheduling and memory bookkeeping, and the model math runs on the GPU in PyTorch and Triton. The two halves communicate through the page tables.

For the sake of learning, every path is checked against Hugging Face as a reference and benchmarked against it on the same GPU.

## Scope

**Models**: Llama-architecture decoder-only models; the engine runs SmolLM2-135M and SmolLM2-1.7B. The 135M is small enough to check on CPU and the 1.7B is large enough to show how ~~bad~~ good my kernels are. It also uses full multi-head attention, so its KV cache is big per token.

**Hardware**: a single RTX 3070 (Ampere, 8GB) under WSL2, so bf16 tensor cores and no FP8 :(

**Precision**: bf16 for serving, and fp32 wherever bf16 would round the answer (so attention softmax, RoPE, RMSNorm) and for matching HF token for token.

**What's built**: model forward pass, paged KV cache and prefix cache, scheduler behind `EngineCore`, and two Triton kernels (decode attention that reads K and V directly from pages and a cache write that scatters new K and V into pages in one launch).

**"from scratch"**: you won't find a single `import transformers` in the engine, I promise. The config is parsed from `config.json` and weights load from safetensors. You may or may not find tensors and matmuls to come from PyTorch and cuBLAS... and that prefill attention uses PyTorch's dot product attention, but that's it... *for now*.

## Non-goals

- **Serving Infrastructure**: HTTP, networking, status codes, metrics, routing, etc. all belong to a separate server project that builds on `EngineCore`, [inference-router](https://github.com/sanchezner/inference-router).
- **Beyond hardware**: multi-GPU and FP8... nope.
- **Extraneous workloads**: training, fine-tuning, beam search, and other architectures outside the Llama family.
- **Parity for its own sake**: not gonna be matching vLLM feature for feature, or a matmul kernel that competes with cuBLAS.
