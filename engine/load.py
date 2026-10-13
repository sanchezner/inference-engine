import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM


# GPT-2 tokenizer and modell in eval mode w/ eager attention set before any oracle fwd
def load_gpt2():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    model.config._attn_implementation = "eager"
    return tok, model


def load_smollm(name="HuggingFaceTB/SmolLM2-135M"):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    model.config._attn_implementation = "eager"
    return tok, model