from huggingface_hub import hf_hub_download

from bench.roofline import (
    decode_step_bytes,
    kv_bytes_per_token,
    parameter_elements,
    streamed_weight_elements,
)
from engine.config import ModelConfig


SMALL = "HuggingFaceTB/SmolLM2-135M"
LARGE = "HuggingFaceTB/SmolLM2-1.7B"


def _synthetic():
    cfg = ModelConfig(
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        hidden_size=32,
        intermediate_size=48,
        rms_norm_eps=1e-5,
        vocab_size=50,
        max_position=128,
        rope_theta=10_000.0,
        rope_type="default",
        tie_word_embeddings=False,
        eos_token_ids=(0,),
        bos_token_id=1,
    )
    q = 4 * 16 * 32
    kv = 2 * 16 * 32
    o = 32 * (4 * 16)
    layer = q + 2 * kv + o + 3 * 48 * 32 + 2 * 32
    streamed = layer + 32 + 50 * 32
    assert streamed_weight_elements(cfg) == streamed
    # untied storage keeps the embedding table and head.
    assert parameter_elements(cfg) == streamed + 50 * 32
    tied = ModelConfig(
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        rms_norm_eps=cfg.rms_norm_eps,
        vocab_size=cfg.vocab_size,
        max_position=cfg.max_position,
        rope_theta=cfg.rope_theta,
        rope_type=cfg.rope_type,
        tie_word_embeddings=True,
        eos_token_ids=cfg.eos_token_ids,
        bos_token_id=cfg.bos_token_id,
    )
    assert streamed_weight_elements(tied) == streamed
    assert parameter_elements(tied) == streamed
    assert kv_bytes_per_token(cfg, 2) == 2 * 1 * 2 * 16 * 2

    # a second sequence adds its KV and one embedding row, which is the whole point of dividing the ceiling by the step rather than charging every sequence for the full model.
    one = decode_step_bytes(cfg, 2, 1, 128)
    two = decode_step_bytes(cfg, 2, 2, 128)
    assert two - one == 32 * 2 + 128 * kv_bytes_per_token(cfg, 2)
    assert one == streamed * 2 + 32 * 2 + 128 * kv_bytes_per_token(cfg, 2)


def _known(repo, shape, streamed, kv_bf16):
    cfg = ModelConfig.from_json(hf_hub_download(repo, "config.json", local_files_only=True))
    assert (
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.vocab_size,
    ) == shape
    assert cfg.tie_word_embeddings is True
    assert streamed_weight_elements(cfg) == streamed
    assert parameter_elements(cfg) == streamed
    assert kv_bytes_per_token(cfg, 2) == kv_bf16
    assert kv_bytes_per_token(cfg, 4) == kv_bf16 * 2
    batch, context = 4, 128
    assert decode_step_bytes(cfg, 2, batch, context) == (
        streamed * 2 + batch * cfg.hidden_size * 2 + batch * context * kv_bf16
    )


def main():
    _synthetic()
    _known(SMALL, (30, 9, 3, 64, 576, 1536, 49152), 134_515_008, 23_040)
    _known(LARGE, (24, 32, 32, 64, 2048, 8192, 49152), 1_711_376_384, 196_608)


if __name__ == "__main__":
    main()
