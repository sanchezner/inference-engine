import torch
import math


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


def forward(model, ids):
    wte = model.transformer.wte.weight
    wpe = model.transformer.wpe.weight
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