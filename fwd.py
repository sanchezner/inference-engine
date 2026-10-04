import torch
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer

torch.manual_seed(0)
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

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


model.config._attn_implementation = "eager"
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


x += delta
h = layer_norm(x, block.ln_2.weight, block.ln_2.bias)
mlp_delta = mlp(h, block.mlp.c_fc.weight, block.mlp.c_fc.bias, block.mlp.c_proj.weight, block.mlp.c_proj.bias)
x += mlp_delta
print("block0_max_abs_err", (x - out.hidden_states[1]).abs().max().item()) 

for block in model.transformer.h[1:]:
    x_hat = layer_norm(x, block.ln_1.weight, block.ln_1.bias)
    delta, _ = attention(x_hat, block.attn.c_attn.weight, block.attn.c_attn.bias, block.attn.c_proj.weight, block.attn.c_proj.bias)
    x += delta
    h = layer_norm(x, block.ln_2.weight, block.ln_2.bias)
    x += mlp(h, block.mlp.c_fc.weight, block.mlp.c_fc.bias, block.mlp.c_proj.weight, block.mlp.c_proj.bias)

x = layer_norm(x, model.transformer.ln_f.weight, model.transformer.ln_f.bias)
print("final_ln_max_abs_err", (x - out.hidden_states[-1]).abs().max().item())

my_logits = x[0, -1] @ wte.T
print("logit_max_abs_err", (my_logits - logits).abs().max().item())
values, indices = torch.topk(my_logits, k=5)
for v, i in zip(values.tolist(), indices.tolist()):
    print(f"{v: .4f} {tok.decode([i])!r}")