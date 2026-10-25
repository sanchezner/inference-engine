import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoConfig

from engine.config import ModelConfig


MODEL = "HuggingFaceTB/SmolLM2-135M"


def _fails(raw, text):
    try:
        ModelConfig.from_dict(raw)
    except ValueError as exc:
        assert text in str(exc), exc
        return
    raise AssertionError(f"expected ValueError containing {text!r}")


def main():
    path = hf_hub_download(MODEL, "config.json", local_files_only=True)
    cfg = ModelConfig.from_json(path)
    hf = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    rope = hf.rope_parameters

    assert cfg.n_layers == hf.num_hidden_layers
    assert cfg.n_heads == hf.num_attention_heads
    assert cfg.n_kv_heads == hf.num_key_value_heads
    assert cfg.head_dim == hf.head_dim
    assert cfg.hidden_size == hf.hidden_size
    assert cfg.intermediate_size == hf.intermediate_size
    assert cfg.rms_norm_eps == hf.rms_norm_eps
    assert cfg.vocab_size == hf.vocab_size
    assert cfg.max_position == hf.max_position_embeddings
    assert cfg.rope_theta == rope["rope_theta"]
    assert cfg.rope_type == rope["rope_type"]
    assert cfg.tie_word_embeddings == hf.tie_word_embeddings
    hf_eos = hf.eos_token_id
    assert cfg.eos_token_ids == ((hf_eos,) if isinstance(hf_eos, int) else tuple(hf_eos))
    assert cfg.bos_token_id == hf.bos_token_id

    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    # The checkpoint omits head_dim. An explicit value must win over hidden // heads.
    explicit = dict(raw)
    explicit["head_dim"] = 128
    assert ModelConfig.from_dict(explicit).head_dim == 128

    # Missing kv heads means MHA.
    mha = dict(raw)
    del mha["num_key_value_heads"]
    assert ModelConfig.from_dict(mha).n_kv_heads == raw["num_attention_heads"]

    # Missing rope_theta is the original Llama default, not SmolLM's 100000.
    default_theta = dict(raw)
    del default_theta["rope_theta"]
    assert ModelConfig.from_dict(default_theta).rope_theta == 10_000.0

    # Newer files store the standardized dict and drop the legacy keys.
    modern = dict(raw)
    del modern["rope_theta"]
    del modern["rope_scaling"]
    modern["rope_parameters"] = {"rope_type": "default", "rope_theta": 100000}
    modern_cfg = ModelConfig.from_dict(modern)
    assert modern_cfg.rope_type == "default"
    assert modern_cfg.rope_theta == 100000

    scaled = dict(raw)
    scaled["rope_scaling"] = {"rope_type": "llama3", "factor": 8.0, "rope_theta": 500000.0}
    _fails(scaled, "llama3")

    bias = dict(raw)
    bias["attention_bias"] = True
    _fails(bias, "attention_bias")

    interleaved = dict(raw)
    interleaved["rope_interleaved"] = True
    _fails(interleaved, "rope_interleaved")

    partial = dict(raw)
    partial["partial_rotary_factor"] = 0.5
    _fails(partial, "partial_rotary_factor")

    several_eos = dict(raw)
    several_eos["eos_token_id"] = [128001, 128009]
    assert ModelConfig.from_dict(several_eos).eos_token_ids == (128001, 128009)


if __name__ == "__main__":
    main()
