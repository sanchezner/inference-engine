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

values, indices = torch.topk(logits, k=5)
for v, i in zip(values.tolist(), indices.tolist()):
    print(f"{v: .4f} {tok.decode([i])!r}")

print(ids.tolist(), ids.shape, logits.shape)

for name, p in model.named_parameters():
    print(f"{name:40s} {tuple(p.shape)}")

wte = model.transformer.wte.weight
wpe = model.transformer.wpe.weight
same = model.lm_head.weight.data_ptr() == wte.data_ptr()
unique = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
print("tied", same, "unique_params", unique)

out = model.transformer(ids, output_hidden_states=True)
pos = torch.arange(ids.shape[1])
x = wte[ids] + wpe[pos]
err = (x - out.hidden_states[0]).abs().max()
print("embed_max_abs_err", err.item())


def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    
    return (x - mean) * torch.rsqrt(var + eps) * weight + bias


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


block = model.transformer.h[0]
x_hat = layer_norm(x, block.ln_1.weight, block.ln_1.bias)
delta, weights = attention(x_hat, block.attn.c_attn.weight, block.attn.c_attn.bias, block.attn.c_proj.weight, block.attn.c_proj.bias)

T = x.shape[1]
future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
mask = torch.zeros(1, 1, T, T).masked_fill(future, torch.finfo(x.dtype).min)
ref, ref_weights = block.attn(block.ln_1(x), attention_mask=mask)

print("ln_max_abs_err", (x_hat - block.ln_1(x)).abs().max().item())
print("attn_max_abs_err", (delta - ref).abs().max().item())
print("row0", [round(v, 4) for v in ref_weights[0, 0, 0].tolist()])
print("row_sum_err", (ref_weights[0, 0].sum(-1) - 1).abs().max().item())


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
        

def llama_attention_cached(x, q_w, k_w, v_w, o_w, n_heads, n_kv_heads, cos, sin, past_k, past_v):
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
    future = k_idx > (T_k - T) + q_idx
    scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    merged = (weights @ v).transpose(1, 2).contiguous().view(B, T, -1)

    return merged @ o_w.T, k_cache, v_cache


def forward_llama_cached(ids, cache):
    cfg = llama.config
    past = 0 if len(cache) == 0 else cache[0][0].shape[2]
    new_ids = ids[:, past:]
    x = llama.model.embed_tokens(new_ids)
    pos = torch.arange(past, past + new_ids.shape[1]).unsqueeze(0)
    cos, sin = rope_cos_sin(pos, cfg.head_dim, cfg.rope_parameters["rope_theta"])

    new_cache = []
    for i, layer in enumerate(llama.model.layers):
        h = rms_norm(x, layer.input_layernorm.weight, cfg.rms_norm_eps)
        past_k, past_v = (None, None) if len(cache) == 0 else cache[i]
        delta, k, v = llama_attention_cached(h, layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight, layer.self_attn.v_proj.weight, layer.self_attn.o_proj.weight, cfg.num_attention_heads, cfg.num_key_value_heads, cos, sin, past_k, past_v)
        new_cache.append((k, v))
        x += delta
        h = rms_norm(x, layer.post_attention_layernorm.weight, cfg.rms_norm_eps)
        x += llama_mlp(h, layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight, layer.mlp.down_proj.weight)

    x = rms_norm(x, llama.model.norm.weight, cfg.rms_norm_eps)
    return x[0, -1] @ llama.lm_head.weight.T, new_cache


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
print(llama.config.rope_parameters)

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