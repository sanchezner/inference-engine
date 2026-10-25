import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from engine.config import ModelConfig
from engine.weights import Weights


MODEL = "HuggingFaceTB/SmolLM2-135M"


def _load_fails(cfg, blobs, text):
    try:
        _load(cfg, blobs)
    except ValueError as exc:
        assert text in str(exc), exc
        return
    raise AssertionError(f"expected ValueError containing {text!r}")


def _tiny_cfg(**overrides):
    raw = {
        "model_type": "llama",
        "hidden_act": "silu",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rms_norm_eps": 1e-5,
        "vocab_size": 4,
        "max_position_embeddings": 32,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "eos_token_id": 0,
        "bos_token_id": 0,
    }
    raw.update(overrides)
    return ModelConfig.from_dict(raw)


def _blobs(cfg, *, head="omit", skip=(), extra=None):
    hidden = cfg.hidden_size
    intermediate = cfg.intermediate_size
    q_out = cfg.n_heads * cfg.head_dim
    kv_out = cfg.n_kv_heads * cfg.head_dim
    generator = torch.Generator().manual_seed(0)
    blobs = {}

    _put(blobs, generator, "model.embed_tokens.weight", (cfg.vocab_size, hidden))
    for i in range(cfg.n_layers):
        prefix = f"model.layers.{i}."
        _put(blobs, generator, prefix + "self_attn.q_proj.weight", (q_out, hidden))
        _put(blobs, generator, prefix + "self_attn.k_proj.weight", (kv_out, hidden))
        _put(blobs, generator, prefix + "self_attn.v_proj.weight", (kv_out, hidden))
        _put(blobs, generator, prefix + "self_attn.o_proj.weight", (hidden, q_out))
        _put(blobs, generator, prefix + "mlp.gate_proj.weight", (intermediate, hidden))
        _put(blobs, generator, prefix + "mlp.up_proj.weight", (intermediate, hidden))
        _put(blobs, generator, prefix + "mlp.down_proj.weight", (hidden, intermediate))
        _put(blobs, generator, prefix + "input_layernorm.weight", (hidden,))
        _put(blobs, generator, prefix + "post_attention_layernorm.weight", (hidden,))
    _put(blobs, generator, "model.norm.weight", (hidden,))
    if head == "copy":
        blobs["lm_head.weight"] = blobs["model.embed_tokens.weight"].clone()
    elif head == "other":
        blobs["lm_head.weight"] = torch.randn(cfg.vocab_size, hidden, generator=generator)
    elif head != "omit":
        raise ValueError(head)
    for key in skip:
        del blobs[key]
    if extra:
        blobs.update(extra)
    return blobs


def _put(blobs, generator, key, shape):
    blobs[key] = torch.randn(*shape, generator=generator)


def _load(cfg, blobs):
    directory = tempfile.mkdtemp()
    try:
        path = Path(directory) / "model.safetensors"
        save_file(blobs, path)
        return Weights.from_safetensors(path, cfg, device="cpu", dtype=torch.float32)
    finally:
        shutil.rmtree(directory)


def _same(name, got, ref):
    err = (got - ref).abs().max().item() if got.shape == ref.shape else None
    assert got.shape == ref.shape and got.dtype == ref.dtype and err == 0, (
        f"{name}: shape {tuple(got.shape)} vs {tuple(ref.shape)}, "
        f"dtype {got.dtype} vs {ref.dtype}, max abs {err}"
    )


def _synthetic():
    cfg = _tiny_cfg()
    tied = _load(cfg, _blobs(cfg, head="omit"))
    assert tied.lm_head.data_ptr() == tied.embed.data_ptr()

    copied = _load(cfg, _blobs(cfg, head="copy"))
    assert copied.lm_head.data_ptr() == copied.embed.data_ptr()
    assert torch.equal(copied.lm_head, copied.embed)

    _load_fails(cfg, _blobs(cfg, head="other"), "differs")

    untied_cfg = _tiny_cfg(tie_word_embeddings=False)
    untied = _load(untied_cfg, _blobs(untied_cfg, head="other"))
    assert untied.lm_head.data_ptr() != untied.embed.data_ptr()
    _load_fails(untied_cfg, _blobs(untied_cfg, head="omit"), "lm_head.weight")
    _load_fails(cfg, _blobs(cfg, skip=("model.layers.0.self_attn.k_proj.weight",)), "k_proj.weight")
    _load_fails(
        cfg,
        _blobs(cfg, extra={"model.layers.0.self_attn.q_proj.bias": torch.zeros(8)}),
        "unexpected tensors",
    )

    # Explicit head_dim, not hidden // n_heads. q is [n_heads * head_dim, hidden].
    wide = _tiny_cfg(head_dim=6)
    wide_w = _load(wide, _blobs(wide))
    assert wide_w.layers[0].q.shape == (12, 8)
    assert wide_w.layers[0].o.shape == (8, 12)
    assert wide_w.layers[0].k.shape == (6, 8)


def _checkpoint():
    path = hf_hub_download(MODEL, "model.safetensors", local_files_only=True)
    cfg_path = hf_hub_download(MODEL, "config.json", local_files_only=True)
    cfg = ModelConfig.from_json(cfg_path)
    weights = Weights.from_safetensors(path, cfg, device="cpu", dtype=torch.float32)
    assert weights.embed.dtype == torch.float32
    assert weights.embed.device.type == "cpu"
    assert weights.lm_head.data_ptr() == weights.embed.data_ptr()
    assert weights.layers[0].q.shape == (576, 576)
    assert weights.layers[0].k.shape == (192, 576)

    # Same snapshot the loader read. local_files_only keeps the oracle off the hub.
    llama = AutoModelForCausalLM.from_pretrained(
        Path(path).parent, dtype=torch.float32, local_files_only=True
    ).eval()
    sd = llama.state_dict()
    _same("embed", weights.embed, sd["model.embed_tokens.weight"])
    _same("lm_head", weights.lm_head, sd["lm_head.weight"])
    _same("norm", weights.norm, sd["model.norm.weight"])
    for i, layer in enumerate(weights.layers):
        prefix = f"model.layers.{i}."
        _same(f"q{i}", layer.q, sd[prefix + "self_attn.q_proj.weight"])
        _same(f"k{i}", layer.k, sd[prefix + "self_attn.k_proj.weight"])
        _same(f"v{i}", layer.v, sd[prefix + "self_attn.v_proj.weight"])
        _same(f"o{i}", layer.o, sd[prefix + "self_attn.o_proj.weight"])
        _same(f"gate{i}", layer.gate, sd[prefix + "mlp.gate_proj.weight"])
        _same(f"up{i}", layer.up, sd[prefix + "mlp.up_proj.weight"])
        _same(f"down{i}", layer.down, sd[prefix + "mlp.down_proj.weight"])
        _same(f"input_norm{i}", layer.input_norm, sd[prefix + "input_layernorm.weight"])
        _same(f"post_norm{i}", layer.post_norm, sd[prefix + "post_attention_layernorm.weight"])


def main():
    _synthetic()
    _checkpoint()


if __name__ == "__main__":
    main()
