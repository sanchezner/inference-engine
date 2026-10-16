import torch
from engine.llama import rms_norm, rope_cos_sin, llama_attention_cached, llama_mlp


# allocates the shared storage
def new_page_pool(llama, n_pages, block_size):
    cfg = llama.config
    n_layers = cfg.num_hidden_layers
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    k = torch.zeros(n_layers, n_pages, n_kv, block_size, head_dim)
    v = torch.zeros(n_layers, n_pages, n_kv, block_size, head_dim)
    return {
        "k": k,
        "v": v,
        "free": list(range(n_pages)),
        "block_size": block_size,
    }


# one sequence in a pool; stores blocks and seq_len
def new_sequence(pool):
    return {"pool": pool, "blocks": [], "seq_len": 0}


# builds a fresh pool and one sequence on it; cache object a single generate takes
def new_page_cache(llama, n_pages, block_size):
    return new_sequence(new_page_pool(llama, n_pages, block_size))


# pops from pool's free list until blocks covers ceil(total_len / block_size). called once per forward, before the layer loop
def reserve_pages(cache, total_len):
    pool = cache["pool"]
    bs = pool["block_size"]
    need = (total_len + bs - 1) // bs
    while len(cache["blocks"]) < need:
        cache["blocks"].append(pool["free"].pop(0))


# put's the sequence's page ids back on free and sorts the list so the next pop is the lowest id
def release_pages(seq):
    pool = seq["pool"]
    pool["free"].extend(seq["blocks"])
    pool["free"].sort()
    seq["blocks"] = []


# writes new tokens into the pool, stores new tail only
def page_write(cache, layer, k_new, v_new, start):
    pool = cache["pool"]
    bs = pool["block_size"]
    T_new = k_new.shape[2]
    for t in range(T_new):
        p = start + t
        page = cache["blocks"][p // bs]
        slot = p % bs
        pool["k"][layer, page, :, slot] = k_new[0, :, t]
        pool["v"][layer, page, :, slot] = v_new[0, :, t]


# reads this sequence's pages back out in blocks order
def page_gather(cache, layer, n):
    pool = cache["pool"]
    blocks = cache["blocks"]
    bs = pool["block_size"]
    n_blocks = len(blocks)
    k_pages = pool["k"][layer, blocks]
    v_pages = pool["v"][layer, blocks]
    k = k_pages.transpose(0, 1).reshape(k_pages.shape[1], n_blocks * bs, k_pages.shape[-1])[:, :n].unsqueeze(0)
    v = v_pages.transpose(0, 1).reshape(v_pages.shape[1], n_blocks * bs, v_pages.shape[-1])[:, :n].unsqueeze(0)
    return k, v


def forward_llama_paged(llama, ids, cache):
    cfg = llama.config
    past = cache["seq_len"]
    new_ids = ids[:, past:]
    x = llama.model.embed_tokens(new_ids)
    T_new = new_ids.shape[1]
    pos = torch.arange(past, past + T_new).unsqueeze(0)
    cos, sin = rope_cos_sin(pos, cfg.head_dim, cfg.rope_parameters["rope_theta"])

    reserve_pages(cache, past + T_new)

    for i, layer in enumerate(llama.model.layers):
        h = rms_norm(x, layer.input_layernorm.weight, cfg.rms_norm_eps)
        past_k, past_v = (None, None) if past == 0 else page_gather(cache, i, past)
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
        )
        page_write(cache, i, k[:, :, past:], v[:, :, past:], past)
        x += delta
        h = rms_norm(x, layer.post_attention_layernorm.weight, cfg.rms_norm_eps)
        x += llama_mlp(h, layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight, layer.mlp.down_proj.weight)

    cache["seq_len"] = past + T_new
    x = rms_norm(x, llama.model.norm.weight, cfg.rms_norm_eps)
    return x[:, -1] @ llama.lm_head.weight.T, cache


def gather_padded(seqs, layer, lens, width):
    ks = []
    vs = []
    for seq, n in zip(seqs, lens):
        k, v = page_gather(seq, layer, n)
        n_pad = width - n
        if n_pad:
            zk = k.new_zeros(1, k.shape[1], n_pad, k.shape[3])
            zv = v.new_zeros(1, v.shape[1], n_pad, v.shape[3])
            k = torch.cat([zk, k], dim=2)
            v = torch.cat([zv, v], dim=2)
        ks.append(k)
        vs.append(v)
    return torch.cat(ks, dim=0), torch.cat(vs, dim=0)


def forward_llama_paged_batch(llama, ids, seqs):
    cfg = llama.config
    lens = [s["seq_len"] for s in seqs]
    width = max(lens)
    T_new = ids.shape[1]
    pad_lens = torch.tensor([width - n for n in lens])
    x = llama.model.embed_tokens(ids)
    pos = torch.tensor(lens).unsqueeze(1) + torch.arange(T_new)
    cos, sin = rope_cos_sin(pos, cfg.head_dim, cfg.rope_parameters["rope_theta"])

    for seq, n in zip(seqs, lens):
        reserve_pages(seq, n + T_new)

    for i, layer in enumerate(llama.model.layers):
        h = rms_norm(x, layer.input_layernorm.weight, cfg.rms_norm_eps)
        past_k, past_v = gather_padded(seqs, i, lens, width)
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
        for row, seq in enumerate(seqs):
            page_write(seq, i, k[row:row + 1, :, -T_new:], v[row:row + 1, :, -T_new:], lens[row])
        x += delta
        h = rms_norm(x, layer.post_attention_layernorm.weight, cfg.rms_norm_eps)
        x += llama_mlp(h, layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight, layer.mlp.down_proj.weight)

    for seq, n in zip(seqs, lens):
        seq["seq_len"] = n + T_new

    x = rms_norm(x, llama.model.norm.weight, cfg.rms_norm_eps)
    return x[:, -1] @ llama.lm_head.weight.T, seqs