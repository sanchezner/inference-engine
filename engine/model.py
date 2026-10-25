from dataclasses import dataclass

import torch

from engine.config import ModelConfig
from engine.weights import LayerWeights, Weights


# Cos/sin for every position, built once. Recomputing them per token would be
# a kernel launch and a sync once this runs on GPU, and the values never change.
@dataclass(frozen=True)
class RopeTables:
    cos: torch.Tensor
    sin: torch.Tensor

    @classmethod
    def build(cls, cfg: ModelConfig, device) -> "RopeTables":
        device = torch.device(device)
        head_dim = cfg.head_dim
        inv_freq = 1.0 / (
            cfg.rope_theta
            ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
        )
        pos = torch.arange(cfg.max_position, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return cls(cos=emb.cos(), sin=emb.sin())

    def gather(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.device != self.cos.device:
            raise ValueError(
                f"positions are on {position_ids.device}, RoPE tables are on {self.cos.device}"
            )
        return self.cos[position_ids], self.sin[position_ids]


def forward(weights: Weights, cfg: ModelConfig, ids: torch.Tensor, rope: RopeTables) -> torch.Tensor:
    # Uncached prefill. Every row is a full prompt with positions 0..T-1, so a
    # left-padded batch does not belong here. Logits are the last column of
    # every row: x[0, -1] used to drop the rest of the batch.
    if ids.ndim != 2:
        raise ValueError(f"ids must have shape [batch, time], got {tuple(ids.shape)}")
    device = weights.embed.device
    ids = ids.to(device=device)
    x = weights.embed[ids]
    positions = torch.arange(ids.shape[1], device=device).unsqueeze(0)
    cos, sin = rope.gather(positions)

    for layer in weights.layers:
        h = rms_norm(x, layer.input_norm, cfg.rms_norm_eps)
        x = x + attention(h, layer, cfg, cos, sin)
        h = rms_norm(x, layer.post_norm, cfg.rms_norm_eps)
        x = x + mlp(h, layer)

    x = rms_norm(x, weights.norm, cfg.rms_norm_eps)
    return x[:, -1] @ weights.lm_head.T


def rms_norm(x, weight, eps):
    # Statistics in fp32. Cast back so a bf16 weight does not promote the residual.
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normed = x.float() * torch.rsqrt(variance + eps)
    return (normed * weight.float()).to(dtype=x.dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q, k, cos, sin):
    # Tables stay fp32. The rotation is done in fp32 and cast back so the
    # following matmul stays in the activation dtype.
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return _rotate(q, cos, sin), _rotate(k, cos, sin)


def _rotate(t, cos, sin):
    rotated = t.float() * cos + rotate_half(t.float()) * sin
    return rotated.to(dtype=t.dtype)


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, n_kv, T, D = x.shape
    x = x[:, :, None, :, :].expand(B, n_kv, n_rep, T, D)
    return x.reshape(B, n_kv * n_rep, T, D)


def attention(x, layer: LayerWeights, cfg: ModelConfig, cos, sin):
    # Prefill is the cached op with an empty cache. One mask, so a prefill and
    # the first paged step cannot drift apart by construction.
    delta, _, _ = attention_cached(x, layer, cfg, cos, sin, None, None)
    return delta


def attention_cached(x, layer: LayerWeights, cfg: ModelConfig, cos, sin, past_k, past_v, pad_lens=None):
    # Returns the attention residual plus the new K/V only. The caller stores
    # those; concatenating them back into the next past is the caller's job.
    # A square triu mask is wrong here: the keys are longer than the queries
    # once a cache exists, and a left-padded batch has holes at the front.
    B, T, _ = x.shape
    n_heads = cfg.n_heads
    n_kv = cfg.n_kv_heads
    head_dim = cfg.head_dim
    q = (x @ layer.q.T).view(B, T, n_heads, head_dim).transpose(1, 2)
    k = (x @ layer.k.T).view(B, T, n_kv, head_dim).transpose(1, 2)
    v = (x @ layer.v.T).view(B, T, n_kv, head_dim).transpose(1, 2)
    q, k = apply_rope(q, k, cos, sin)
    k_new, v_new = k, v

    if past_k is not None:
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)

    n_rep = n_heads // n_kv
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)
    # Softmax accumulator is fp32 even when q and k are bf16.
    scores = (q.float() @ k.float().transpose(-1, -2)) * (head_dim ** -0.5)
    scores = scores.masked_fill(_causal_mask(T, k.shape[2], pad_lens, x.device), torch.finfo(torch.float32).min)
    attn = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
    merged = (attn @ v).transpose(1, 2).contiguous().view(B, T, n_heads * head_dim)
    return merged @ layer.o.T, k_new, v_new


def _causal_mask(query_len, key_len, pad_lens, device):
    q_idx = torch.arange(query_len, device=device)[:, None]
    k_idx = torch.arange(key_len, device=device)[None, :]
    # Query t may attend through the cached prefix plus its own position.
    causal = k_idx > (key_len - query_len) + q_idx
    if pad_lens is None:
        return causal
    is_pad = k_idx < pad_lens.to(device=device)[:, None]
    blocked = causal[None, :, :] | is_pad[:, None, :]
    return blocked[:, None, :, :]


def mlp(x, layer: LayerWeights):
    hidden = torch.nn.functional.silu(x @ layer.gate.T) * (x @ layer.up.T)
    return hidden @ layer.down.T
