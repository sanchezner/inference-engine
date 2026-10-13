import torch
from functools import partial
from engine.load import load_gpt2, load_smollm
from engine.gpt2 import forward
from engine.llama import forward_llama
from engine.generate import generate

torch.manual_seed(0)
prompt = "The capital of France is"

tok, model = load_gpt2()
gpt2_fwd = partial(forward, model)
ids = tok(prompt, return_tensors="pt").input_ids

with torch.no_grad():
    logits = model(ids).logits[0, -1]
    my_logits = gpt2_fwd(ids)
    mine = generate(gpt2_fwd, ids, max_new_tokens=10, eos_id=model.config.eos_token_id)
    ref = model.generate(ids, max_new_tokens=10, do_sample=False, pad_token_id=model.config.eos_token_id)

print("logit_max_abs_err", (my_logits - logits).abs().max().item())
print(torch.equal(mine, ref))
print(tok.decode(mine[0]))

llama_tok, llama = load_smollm()
uncached = partial(forward_llama, llama)
llama_ids = llama_tok(prompt, return_tensors="pt").input_ids

with torch.no_grad():
    ref_logits = llama(llama_ids).logits[0, -1]
    my_llama_logits = uncached(llama_ids)
    mine = generate(uncached, llama_ids, max_new_tokens=10, eos_id=llama.config.eos_token_id)
    ref_ids = llama.generate(llama_ids, max_new_tokens=10, do_sample=False, pad_token_id=llama.config.eos_token_id)

print("llama_logit_max_abs_err", (my_llama_logits - ref_logits).abs().max().item())
print(torch.equal(mine, ref_ids))
print(llama_tok.decode(mine[0]))