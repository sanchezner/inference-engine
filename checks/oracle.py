import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

from engine.config import ModelConfig
from engine.model import RopeTables
from engine.weights import Weights


MODEL = "HuggingFaceTB/SmolLM2-135M"


def load_gpt2():
    tok = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True).eval()
    model.config._attn_implementation = "eager"
    return tok, model


def load_smollm(name=MODEL, device="cpu", dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, local_files_only=True).to(device).eval()
    model.config._attn_implementation = "eager"
    return tok, model


def load_engine(device, dtype, name=MODEL):
    cfg = ModelConfig.from_json(hf_hub_download(name, "config.json", local_files_only=True))
    weights = Weights.from_safetensors(
        hf_hub_download(name, "model.safetensors", local_files_only=True),
        cfg,
        device=device,
        dtype=dtype,
    )
    rope = RopeTables.build(cfg, device=weights.embed.device)
    return cfg, weights, rope
