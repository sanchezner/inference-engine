from engine.config import ModelConfig


def kv_bytes_per_token(cfg: ModelConfig, dtype_bytes: int) -> int:
    # K and V, one vector per KV head per layer, GQA stores the KV heads
    return 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * dtype_bytes


def streamed_weight_elements(cfg: ModelConfig) -> int:
    # parameter elements read in full on one decode step.
    hidden = cfg.hidden_size
    q = cfg.n_heads * cfg.head_dim * hidden
    kv = cfg.n_kv_heads * cfg.head_dim * hidden
    o = hidden * cfg.n_heads * cfg.head_dim
    layer = q + 2 * kv + o + 3 * cfg.intermediate_size * hidden + 2 * hidden
    return cfg.n_layers * layer + hidden + cfg.vocab_size * hidden


def streamed_weight_bytes(cfg: ModelConfig, dtype_bytes: int) -> int:
    return streamed_weight_elements(cfg) * dtype_bytes


def parameter_elements(cfg: ModelConfig) -> int:
    # storage; untied models keep a second copy of the table
    streamed = streamed_weight_elements(cfg)
    if cfg.tie_word_embeddings:
        return streamed
    return streamed + cfg.vocab_size * cfg.hidden_size


def embedding_lookup_bytes(cfg: ModelConfig, dtype_bytes: int, batch: int) -> int:
    return batch * cfg.hidden_size * dtype_bytes


def decode_step_bytes(cfg: ModelConfig, dtype_bytes: int, batch: int, context: int) -> int:
    # bytes moved to produce batch tokens that each attend over context KV tokens.
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    if context < 0:
        raise ValueError(f"context must be >= 0, got {context}")
    if dtype_bytes < 1:
        raise ValueError(f"dtype_bytes must be >= 1, got {dtype_bytes}")
    return (
        streamed_weight_bytes(cfg, dtype_bytes)
        + embedding_lookup_bytes(cfg, dtype_bytes, batch)
        + batch * context * kv_bytes_per_token(cfg, dtype_bytes)
    )


def decode_ceiling_tokens_per_s(bandwidth_bytes_per_s: float, step_bytes: int, batch: int) -> float:
    if bandwidth_bytes_per_s <= 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth_bytes_per_s}")
    if step_bytes <= 0:
        raise ValueError(f"step_bytes must be positive, got {step_bytes}")
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    return batch * bandwidth_bytes_per_s / step_bytes
