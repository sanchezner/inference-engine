from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from engine.config import ModelConfig


@dataclass(frozen=True)
class LayerWeights:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    input_norm: torch.Tensor
    post_norm: torch.Tensor


@dataclass(frozen=True)
class Weights:
    embed: torch.Tensor
    layers: tuple[LayerWeights, ...]
    norm: torch.Tensor
    lm_head: torch.Tensor

    @classmethod
    def from_safetensors(cls, path: str | Path, cfg: ModelConfig, *, device, dtype) -> "Weights":
        if not dtype.is_floating_point:
            raise ValueError(f"dtype must be floating point, got {dtype}")
        device = torch.device(device)
        blobs = load_file(path)
        used: set[str] = set()

        embed = _take(blobs, used, "model.embed_tokens.weight", device, dtype)
        layers = tuple(_layer(blobs, used, i, device, dtype) for i in range(cfg.n_layers))
        norm = _take(blobs, used, "model.norm.weight", device, dtype)
        lm_head = _lm_head(blobs, used, cfg, embed)

        unused = set(blobs) - used
        if unused:
            listed = ", ".join(sorted(unused))
            raise ValueError(f"unexpected tensors: {listed}")

        weights = cls(embed=embed, layers=layers, norm=norm, lm_head=lm_head)
        _check_shapes(cfg, weights)
        return weights


def _take(blobs, used, key, device, dtype) -> torch.Tensor:
    if key not in blobs:
        raise ValueError(f"safetensors is missing {key}")
    used.add(key)
    return blobs[key].to(device=device, dtype=dtype)


def _layer(blobs, used, index, device, dtype) -> LayerWeights:
    prefix = f"model.layers.{index}."
    return LayerWeights(
        q=_take(blobs, used, prefix + "self_attn.q_proj.weight", device, dtype),
        k=_take(blobs, used, prefix + "self_attn.k_proj.weight", device, dtype),
        v=_take(blobs, used, prefix + "self_attn.v_proj.weight", device, dtype),
        o=_take(blobs, used, prefix + "self_attn.o_proj.weight", device, dtype),
        gate=_take(blobs, used, prefix + "mlp.gate_proj.weight", device, dtype),
        up=_take(blobs, used, prefix + "mlp.up_proj.weight", device, dtype),
        down=_take(blobs, used, prefix + "mlp.down_proj.weight", device, dtype),
        input_norm=_take(blobs, used, prefix + "input_layernorm.weight", device, dtype),
        post_norm=_take(blobs, used, prefix + "post_attention_layernorm.weight", device, dtype),
    )


def _lm_head(blobs, used, cfg: ModelConfig, embed: torch.Tensor) -> torch.Tensor:
    key = "lm_head.weight"
    if not cfg.tie_word_embeddings:
        if key not in blobs:
            raise ValueError(f"safetensors is missing {key}")
        used.add(key)
        return blobs[key].to(device=embed.device, dtype=embed.dtype)

    # Compare in the file dtype, then drop the duplicate. The live head is the
    # embedding after the cast, so the two cannot become separate copies.
    if key in blobs:
        used.add(key)
        raw_head = blobs[key]
        raw_embed = blobs["model.embed_tokens.weight"]
        if raw_head.shape != raw_embed.shape or not torch.equal(raw_head, raw_embed):
            raise ValueError(
                "tie_word_embeddings is set, but lm_head.weight differs from model.embed_tokens.weight"
            )
    return embed


def _check_shapes(cfg: ModelConfig, weights: Weights) -> None:
    hidden = cfg.hidden_size
    intermediate = cfg.intermediate_size
    q_out = cfg.n_heads * cfg.head_dim
    kv_out = cfg.n_kv_heads * cfg.head_dim
    vocab = cfg.vocab_size
    dtype = weights.embed.dtype
    device = weights.embed.device

    _expect(weights.embed, "embed", (vocab, hidden), dtype, device)
    _expect(weights.lm_head, "lm_head", (vocab, hidden), dtype, device)
    _expect(weights.norm, "norm", (hidden,), dtype, device)
    if len(weights.layers) != cfg.n_layers:
        raise ValueError(f"loaded {len(weights.layers)} layers, config has {cfg.n_layers}")
    for i, layer in enumerate(weights.layers):
        _expect(layer.q, f"layers[{i}].q", (q_out, hidden), dtype, device)
        _expect(layer.k, f"layers[{i}].k", (kv_out, hidden), dtype, device)
        _expect(layer.v, f"layers[{i}].v", (kv_out, hidden), dtype, device)
        _expect(layer.o, f"layers[{i}].o", (hidden, q_out), dtype, device)
        _expect(layer.gate, f"layers[{i}].gate", (intermediate, hidden), dtype, device)
        _expect(layer.up, f"layers[{i}].up", (intermediate, hidden), dtype, device)
        _expect(layer.down, f"layers[{i}].down", (hidden, intermediate), dtype, device)
        _expect(layer.input_norm, f"layers[{i}].input_norm", (hidden,), dtype, device)
        _expect(layer.post_norm, f"layers[{i}].post_norm", (hidden,), dtype, device)


def _expect(tensor, name, shape, dtype, device):
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
    if tensor.dtype != dtype or tensor.device != device:
        raise ValueError(f"{name} is {tensor.dtype} on {tensor.device}, expected {dtype} on {device}")
