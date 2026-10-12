import torch
import math
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(0)
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
model.config._attn_implementation = "eager"

prompt = "The capital of France is"
ids = tok(prompt, return_tensors="pt").input_ids

with torch.no_grad():
    logits = model(ids).logits[0, -1]


wte = model.transformer.wte.weight
wpe = model.transformer.wpe.weight


# standard GPT-2 layer normalization: subtract mean across the hidden dimension (C) and divide by stdev, then apply affine scale (weight) and shift (bias)
# this keeps activation values from vanishing as vectors pass through transformer blocks
def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    
    return (x - mean) * torch.rsqrt(var + eps) * weight + bias


# standard Multi-Head Attention for GPT-2 with 1D causal masking
def attention(x, c_attn_w, c_attn_b, c_proj_w, c_proj_b, n_heads=12):
    B, T, C = x.shape
    qkv = x @ c_attn_w + c_attn_b
    q, k, v = qkv.split(C, dim=-1)
    head_dim = C // n_heads

    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, n_heads, head_dim).transpose(1, 2)

    scores = (q @ k.transpose(-1, -2)) * (head_dim ** -0.5)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    merged = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
    
    return merged @ c_proj_w + c_proj_b, weights


# gaussian error linear unit activation function (used by GPT-2)
def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def mlp(x, c_fc_w, c_fc_b, c_proj_w, c_proj_b):
    hidden = gelu_new(x @ c_fc_w + c_fc_b)
    return hidden @ c_proj_w + c_proj_b


def forward(ids):
    pos = torch.arange(ids.shape[1])
    x = wte[ids] + wpe[pos]
    
    for block in model.transformer.h:
        x_hat = layer_norm(x, block.ln_1.weight, block.ln_1.bias)
        delta, _ = attention(x_hat, block.attn.c_attn.weight, block.attn.c_attn.bias, block.attn.c_proj.weight, block.attn.c_proj.bias)
        x += delta
        h = layer_norm(x, block.ln_2.weight, block.ln_2.bias)
        mlp_delta = mlp(h, block.mlp.c_fc.weight, block.mlp.c_fc.bias, block.mlp.c_proj.weight, block.mlp.c_proj.bias)
        x += mlp_delta
    
    x = layer_norm(x, model.transformer.ln_f.weight, model.transformer.ln_f.bias)

    return x[0, -1] @ wte.T


# root mean square normalization: only scaled by root mean square of the activations
# eliminating the mean calculation saves mem bandwidth and computation while maintaining training stability
def rms_norm(x, weight, eps):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return weight * x * torch.rsqrt(variance + eps)


# splits vector into two halves to implement complex number rotation in 2D vector space
def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


# calculates geometric rotation frequencies for each token pos based on index and base frequency theta
def rope_cos_sin(position_ids, head_dim, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = position_ids.float()[..., None] * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)

    return emb.cos(), emb.sin()


# rotates query q and key k vectors in 2D slices according to their pos in the sequence
# (RoPE) injects relative distance directly into attention score
def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin

    return q, k


# repeats key and value head tensors to match number of query heads
# implements Grouped-Query Attention (GQA), reducing mem footprint of the KV cache with almost no loss in output quality
def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x

    B, n_kv, T, D = x.shape
    x = x[:, :, None, :, :].expand(B, n_kv, n_rep, T, D)
    return x.reshape(B, n_kv * n_rep, T, D)


# modern GQA attention with RoPE applied to queries and keys before the causal scaled dot-product attention calculation
def  llama_attention(x, q_w, k_w, v_w, o_w, n_heads, n_kv_heads, cos, sin):
    B, T, _ = x.shape
    head_dim = q_w.shape[0] // n_heads
    q = (x @ q_w.T).view(B, T, n_heads, head_dim).transpose(1, 2)
    k = (x @ k_w.T).view(B, T, n_kv_heads, head_dim).transpose(1, 2)
    v = (x @ v_w.T).view(B, T, n_kv_heads, head_dim).transpose(1, 2)

    q, k = apply_rope(q, k, cos, sin)

    k = repeat_kv(k, n_heads // n_kv_heads)
    v = repeat_kv(v, n_heads // n_kv_heads)

    scores = (q @ k.transpose(-1, -2)) * (head_dim ** -0.5)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    merged = (weights @ v).transpose(1, 2).contiguous().view(B, T, -1)

    return merged @ o_w.T


# (SwiGLU) replaces standard 2-layer MLPs
# this performs a computation to give the model higher representational capacity per param
def llama_mlp(x, gate_w, up_w, down_w):
    hidden = torch.nn.functional.silu(x @ gate_w.T) * (x @ up_w.T)
    return hidden @ down_w.T

    
def forward_llama(ids):
    cfg = llama.config
    x = llama.model.embed_tokens(ids)
    pos = torch.arange(ids.shape[1]).unsqueeze(0)
    cos, sin = rope_cos_sin(pos, cfg.head_dim, cfg.rope_parameters["rope_theta"])

    for layer in llama.model.layers:
        h = rms_norm(x, layer.input_layernorm.weight, cfg.rms_norm_eps)
        llama_attn_delta = llama_attention(h, layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight, layer.self_attn.v_proj.weight, layer.self_attn.o_proj.weight, cfg.num_attention_heads, cfg.num_key_value_heads, cos, sin)
        x += llama_attn_delta
        h = rms_norm(x, layer.post_attention_layernorm.weight, cfg.rms_norm_eps)
        x += llama_mlp(h, layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight, layer.mlp.down_proj.weight)
    
    x = rms_norm(x, llama.model.norm.weight, cfg.rms_norm_eps)

    return x[0, -1] @ llama.lm_head.weight.T
        

# acceptss prev past_k and past_v from past steps, computes Q, K, V for only new tokens, concats new K, V onto existing cache, and runs attention across the combined sequence history
# dynamic mask offset construction so a single new query token can attend to all prior cached tokens while preserving causal blocking during batch prefill
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

    scores = (q @k.transpose(-1, -2)) * (head_dim ** -0.5)
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


# slices ids so that only uncomputed tokens pass through token embedding, RoPE, RMSNorm, and MLP layers, returning the updated cache list across all transformer layers
def forward_llama_cached(ids, cache, pad_lens=None):
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
        delta, k, v = llama_attention_cached(h, layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight, layer.self_attn.v_proj.weight, layer.self_attn.o_proj.weight, cfg.num_attention_heads, cfg.num_key_value_heads, cos, sin, past_k, past_v, pad_lens)
        new_cache.append((k, v))
        x += delta
        h = rms_norm(x, layer.post_attention_layernorm.weight, cfg.rms_norm_eps)
        x += llama_mlp(h, layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight, layer.mlp.down_proj.weight)

    x = rms_norm(x, llama.model.norm.weight, cfg.rms_norm_eps)
    return x[:, -1] @ llama.lm_head.weight.T, new_cache


# generates tokens sequentially for a single prompt w/ greedy argmax decoding until max_new_tokens or eos
def generate(forward_fn, ids, max_new_tokens, eos_id, cache=None):
    for _ in range(max_new_tokens):
        if cache is None:
            logits = forward_fn(ids)
        else:
            logits, cache = forward_fn(ids, cache)
    
        next_id = torch.argmax(logits)
        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
        if next_id.item() == eos_id:
            break

    return ids


# runs multiple prompts in lockstep
# if one prompt needs 2 tokens and another needs 100, the first must wait while empty padding tokens are processed until the second one finishes
def generate_batch(forward_fn, ids, max_new_tokens, eos_id, cache, pad_lens=None):
    finished = torch.zeros(ids.shape[0], dtype=torch.bool)

    for _ in range(max_new_tokens):
        logits, cache = forward_fn(ids, cache, pad_lens)
        next_id = torch.argmax(logits, dim=-1)
        next_id = torch.where(finished, torch.full_like(next_id, eos_id), next_id)
        ids = torch.cat([ids, next_id[:, None]], dim=1)
        finished = finished | (next_id == eos_id)
        
        if bool(finished.all()):
            break

    return ids


# helper functino that pads input sequences with pad_id on the left instead of the right
# autoregressive generation appens new tokesn to the right; if right padding was used, new tkoens would sit in the middle, breaking position indices
# left padding keeps active generation heads aligned at the very last column index (-1)
def _left_pad(rows, pad_id):
    width = max(r.shape[0] for r in rows)
    pad_lens = []
    padded = []
    
    for r in rows:
        n = width - r.shape[0]
        pad_lens.append(n)
        
        if n:
            r = torch.cat([torch.full((n,), pad_id, dtype=r.dtype), r])
        
        padded.append(r)
    
    return torch.stack(padded), torch.tensor(pad_lens)


# extracts KV cache of a single sequence out of a batched KV cache tensor
def cache_row(cache, index):
    return [(k[index:index + 1], v[index:index + 1]) for k, v in cache]


# stacks independent request caches along batch dimension so they can be processed in a single batched GEMM kernel
def cache_cat(caches):
    merged = []
    n_layers = len(caches[0])
    
    for layer in range(n_layers):
        k = torch.cat([c[layer][0] for c in caches], dim=0)
        v = torch.cat([c[layer][1] for c in caches], dim=0)
        merged.append((k, v))

    return merged


# prepends zeros to time dimension T of a sequence's KV cache
# when two requests have different prompt lengths or cache sizes, their KV caches must have matching sequence lengths to be batched together
# left padding the shorter cache with dummy vectors aligns their dims
def cache_pad_left(cache, n_pad):
    if n_pad == 0:
        return cache

    padded = []

    for k, v in cache:
        zk = k.new_zeros(k.shape[0], k.shape[1], n_pad, k.shape[3])
        zv = v.new_zeros(v.shape[0], v.shape[1], n_pad, v.shape[3])
        padded.append((torch.cat([zk, k], dim=2), torch.cat([zv, v], dim=2)))

    return padded


# continuous batching
def admit_into_batch(forward_fn, slots, incoming, outputs, eos_id):
    prompts = [req["prompt"] for req in incoming]
    padded, group_pad = _left_pad(prompts, eos_id)
    logits, cache = forward_fn(padded, [], group_pad)
    next_id = torch.argmax(logits, dim=-1)

    width = padded.shape[1]
    resident_t = slots[0]["cache"][0][0].shape[2]
    
    if width > resident_t:
        extra = width - resident_t
        align = 0
        
        for slot in slots:
            slot["cache"] = cache_pad_left(slot["cache"], extra)
            slot["pad_len"] += extra
            filler = torch.full((extra,), eos_id, dtype=slot["ids"].dtype)
            slot["ids"] = torch.cat([filler, slot["ids"]])
    else:
        align = resident_t - width
        cache = cache_pad_left(cache, align)

    accepted = []
    
    for i, req in enumerate(incoming):
        tok = next_id[i]
        prompt = req["prompt"]
        
        if tok.item() == eos_id or req["budget"] == 1:
            outputs[req["index"]] = torch.cat([prompt, tok.view(1)])
            continue

        n_left = (width - prompt.shape[0]) + align
        filler = torch.full((n_left,), eos_id, dtype=prompt.dtype)
        accepted.append({
            "index": req["index"],
            "ids": torch.cat([filler, prompt, tok.view(1)]),
            "pad_len": n_left,
            "cache": cache_row(cache, i),
            "generated": 1,
            "budget": req["budget"],
        })

    return slots + accepted


def generate_continuous(forward_fn, prompts, budgets, eos_id, max_batch):
    waiting = [{"prompt": p, "budget": b, "index": i} for i, (p, b) in enumerate(zip(prompts, budgets))]
    slots = []
    outputs = [None] * len(prompts)

    while waiting or slots:
        free = max_batch - len(slots)
        if free > 0 and waiting:
            incoming = waiting[:free]
            del waiting[:free]
            
            if not slots:
                padded, pad_lens = _left_pad([req["prompt"] for req in incoming], eos_id)
                for row, req, n_pad in zip(padded, incoming, pad_lens.tolist()):
                    slots.append({
                        "index": req["index"],
                        "ids": row.clone(),
                        "pad_len": n_pad,
                        "cache": None,
                        "generated": 0,
                        "budget": req["budget"],
                    })
            else:
                slots = admit_into_batch(forward_fn, slots, incoming, outputs, eos_id)

        if not slots:
            continue

        print("batch", [s["index"] for s in slots], "gen", [s["generated"] for s in slots], "pad", [s["pad_len"] for s in slots])
        ids = torch.stack([s["ids"] for s in slots])
        pad_lens = torch.tensor([s["pad_len"] for s in slots])

        if slots[0]["cache"] is None:
            logits, cache = forward_fn(ids, [], pad_lens)
        else:
            logits, cache = forward_fn(ids, cache_cat([s["cache"] for s in slots]), pad_lens)

        next_id = torch.argmax(logits, dim=-1)
        alive = []

        for i, slot in enumerate(slots):
            slot["cache"] = cache_row(cache, i )
            tok = next_id[i]
            slot["ids"] = torch.cat([slot["ids"], tok.view(1)])
            slot["generated"] += 1

            if tok.item() == eos_id or slot["generated"] == slot["budget"]:
                outputs[slot["index"]] = slot["ids"][slot["pad_len"]:]
            else:
                alive.append(slot)

        slots = alive

    return outputs


with torch.no_grad():
    my_logits = forward(ids)
    mine = generate(forward, ids, max_new_tokens=10, eos_id=model.config.eos_token_id)
    ref = model.generate(ids, max_new_tokens=10, do_sample=False, pad_token_id=model.config.eos_token_id)

print("logit_max_abs_err", (my_logits - logits).abs().max().item())
print(torch.equal(mine, ref))
print(tok.decode(mine[0]))

llama_name = "HuggingFaceTB/SmolLM2-135M"
llama_tok = AutoTokenizer.from_pretrained(llama_name)
llama = AutoModelForCausalLM.from_pretrained(llama_name, dtype=torch.float32).eval()
llama.config._attn_implementation = "eager"

llama_ids = llama_tok(prompt, return_tensors="pt").input_ids

with torch.no_grad():
    ref_logits = llama(llama_ids).logits[0, -1]
    my_llama_logits = forward_llama(llama_ids)
    mine = generate(forward_llama, llama_ids, max_new_tokens=10, eos_id=llama.config.eos_token_id)
    ref_ids = llama.generate(llama_ids, max_new_tokens=10, do_sample=False, pad_token_id=llama.config.eos_token_id)

print("llama_logit_max_abs_err", (my_llama_logits - ref_logits).abs().max().item())
print(torch.equal(mine, ref_ids))
print(llama_tok.decode(mine[0]))

eos = llama.config.eos_token_id
with torch.no_grad():
    prefill_logits, prefill_cache = forward_llama_cached(llama_ids, [])
    print("prefill_logit_max_abs_err", (prefill_logits - my_llama_logits).abs().max().item())
    print("cache_k_shape", tuple(prefill_cache[0][0].shape), "layers", len(prefill_cache))

    generate(forward_llama_cached, llama_ids, max_new_tokens=10, eos_id=eos, cache=[])

    t0 = time.perf_counter()
    uncached_ids = generate(forward_llama, llama_ids, max_new_tokens=10, eos_id=eos)
    uncached_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    cached_ids = generate(forward_llama_cached, llama_ids, max_new_tokens=10, eos_id=eos, cache=[])
    cached_s = time.perf_counter() - t0

print(torch.equal(cached_ids, ref_ids))
print(torch.equal(cached_ids, uncached_ids))
print("uncached_s", uncached_s, "cached_s", cached_s)

prompts = [
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Spain is",
    "The capital of Italy is",
]

batch_ids = torch.cat([llama_tok(p, return_tensors="pt").input_ids for p in prompts], dim=0)

with torch.no_grad():
    prefill_b, cache_b = forward_llama_cached(batch_ids, [])
    print(tuple(prefill_b.shape), tuple(cache_b[0][0].shape))
    print("row0_max_abs_err", (prefill_b[0] - my_llama_logits).abs().max().item())

    t0 = time.perf_counter()
    serial_ids = [generate(forward_llama_cached, row.view(1, -1), max_new_tokens=10, eos_id=eos, cache=[]) for row in batch_ids]
    serial_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    batched_ids = generate_batch(forward_llama_cached, batch_ids, max_new_tokens=10, eos_id=eos, cache=[])
    batched_s = time.perf_counter() - t0

for i, serial in enumerate(serial_ids):
    stop = serial.shape[1]
    print(torch.equal(batched_ids[i, :stop], serial[0]), llama_tok.decode(serial[0]))

print("serial_s", serial_s, "batched_s", batched_s)

mixed_prompts = [
    "The capital of France is",
    "The capital of Germany is",
    "The cat",
]

mixed_rows = [llama_tok(p, return_tensors="pt").input_ids[0] for p in mixed_prompts]
mixed_ids, pad_lens = _left_pad(mixed_rows, eos)

with torch.no_grad():
    prefill_m, _ = forward_llama_cached(mixed_ids, [], pad_lens)
    cat_logits = forward_llama(mixed_rows[2].view(1, -1))
    print("france_pad_err", (prefill_m[0] - my_llama_logits).abs().max().item())
    print("cat_pad_err", (prefill_m[2] - cat_logits).abs().max().item())

    serial_m = [generate(forward_llama_cached, row.view(1, -1), max_new_tokens=10, eos_id=eos, cache=[]) for row in mixed_rows]
    batched_m = generate_batch(forward_llama_cached, mixed_ids, max_new_tokens=10, eos_id=eos, cache=[], pad_lens=pad_lens)

for i, serial in enumerate(serial_m):
    start = int(pad_lens[i])
    stop = serial.shape[1]
    print(torch.equal(batched_m[i, start:start + stop], serial[0]), llama_tok.decode(serial[0]))


def run_continuous(texts, budgets):
    rows = [llama_tok(p, return_tensors="pt").input_ids[0] for p in texts]

    with torch.no_grad():
        t0 = time.perf_counter()
        serial = [generate(forward_llama_cached, row.view(1, -1), max_new_tokens=b, eos_id=eos, cache=[]) for row, b in zip(rows, budgets)]
        serial_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        cont = generate_continuous(forward_llama_cached, rows, budgets, eos, max_batch=2)
        cont_s = time.perf_counter() - t0
    
    for i, s in enumerate(serial):
        print(torch.equal(cont[i], s[0]), llama_tok.decode(s[0]))

    print("serial_c_s", serial_s, "cont_s", cont_s)


run_continuous(["The capital of France is", "The cat", "The capital of Germany is"], [6, 2, 4])
run_continuous(["The capital of France is", "The cat", "Once upon a time there was a"], [6, 2, 4])

