import torch


# Frozen dense KV cache. The live path is the page pool. These helpers stay
# here, with the original numerics, so the ablation does not import engine.llama.


def rms_norm(x, weight, eps):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return weight * x * torch.rsqrt(variance + eps)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_cos_sin(position_ids, head_dim, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = position_ids.float()[..., None] * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, n_kv, T, D = x.shape
    x = x[:, :, None, :, :].expand(B, n_kv, n_rep, T, D)
    return x.reshape(B, n_kv * n_rep, T, D)


def llama_mlp(x, gate_w, up_w, down_w):
    hidden = torch.nn.functional.silu(x @ gate_w.T) * (x @ up_w.T)
    return hidden @ down_w.T


def llama_attention_cached(x, q_w, k_w, v_w, o_w, n_heads, n_kv_heads, cos, sin, past_k, past_v, pad_lens=None):
    B, T, _ = x.shape
    head_dim = q_w.shape[0] // n_heads
    q = (x @ q_w.T).view(B, T, n_heads, head_dim).transpose(1, 2)
    k = (x @ k_w.T).view(B, T, n_kv_heads, head_dim).transpose(1, 2)
    v = (x @ v_w.T).view(B, T, n_kv_heads, head_dim).transpose(1, 2)

    q, k = apply_rope(q, k, cos, sin)

    if past_k is not None:
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)

    k_cache, v_cache = k, v
    k = repeat_kv(k, n_heads // n_kv_heads)
    v = repeat_kv(v, n_heads // n_kv_heads)

    scores = (q @ k.transpose(-1, -2)) * (head_dim ** -0.5)
    T_k = k.shape[2]
    q_idx = torch.arange(T)[:, None]
    k_idx = torch.arange(T_k)[None, :]
    causal = k_idx > (T_k - T) + q_idx

    if pad_lens is None:
        blocked = causal
    else:
        is_pad = k_idx < pad_lens[:, None]
        blocked = causal[None, :, :] | is_pad[:, None, :]
        blocked = blocked[:, None, :, :]
    scores = scores.masked_fill(blocked, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    merged = (weights @ v).transpose(1, 2).contiguous().view(B, T, -1)

    return merged @ o_w.T, k_cache, v_cache


def forward_llama_cached(llama, ids, cache, pad_lens=None):
    cfg = llama.config
    past = 0 if len(cache) == 0 else cache[0][0].shape[2]
    new_ids = ids[:, past:]
    x = llama.model.embed_tokens(new_ids)
    T_new = new_ids.shape[1]

    if pad_lens is None:
        pos = torch.arange(past, past + new_ids.shape[1]).unsqueeze(0)
    else:
        cols = torch.arange(past, past + T_new)
        pos = (cols.unsqueeze(0) - pad_lens.unsqueeze(1)).clamp(min=0)

    cos, sin = rope_cos_sin(pos, cfg.head_dim, cfg.rope_parameters["rope_theta"])

    new_cache = []
    for i, layer in enumerate(llama.model.layers):
        h = rms_norm(x, layer.input_layernorm.weight, cfg.rms_norm_eps)
        past_k, past_v = (None, None) if len(cache) == 0 else cache[i]
        delta, k, v = llama_attention_cached(
            h,
            layer.self_attn.q_proj.weight,
            layer.self_attn.k_proj.weight,
            layer.self_attn.v_proj.weight,
            layer.self_attn.o_proj.weight,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cos,
            sin,
            past_k,
            past_v,
            pad_lens,
        )
        new_cache.append((k, v))
        x += delta
        h = rms_norm(x, layer.post_attention_layernorm.weight, cfg.rms_norm_eps)
        x += llama_mlp(h, layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight, layer.mlp.down_proj.weight)

    x = rms_norm(x, llama.model.norm.weight, cfg.rms_norm_eps)
    return x[:, -1] @ llama.lm_head.weight.T, new_cache
