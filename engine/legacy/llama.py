import torch
from engine.llama import llama_attention_cached, llama_mlp, rms_norm, rope_cos_sin


# dense KV cache; one rectangular tensor per sequence, grown by concatenation
# also frozen... live path is the page pool in engine/paged.py
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
