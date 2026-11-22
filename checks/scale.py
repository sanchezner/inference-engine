import gc
from functools import partial

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from checks.agreement import PROMPTS, assert_bf16, fixed_greedy_trace, score_trace
from checks.oracle import MODEL as SMALL, load_engine
from checks.tolerances import LLAMA_LOGIT_ATOL, assert_logits
from engine.config import ModelConfig
from engine.generate import generate
from engine.model import forward as forward_llama
from engine.paged import forward_paged, new_page_cache


# SmolLM2-1.7B. fp32 is ~6.8 GB, so the oracle stays on CPU.
LARGE = "HuggingFaceTB/SmolLM2-1.7B"
PROMPT = "The capital of France is"
NEW_TOKENS = 10


def _config():
    path = hf_hub_download(LARGE, "config.json", local_files_only=True)
    cfg = ModelConfig.from_json(path)
    hf = AutoConfig.from_pretrained(LARGE, local_files_only=True)
    rope = hf.rope_parameters

    assert cfg.n_layers == hf.num_hidden_layers == 24
    assert cfg.n_heads == hf.num_attention_heads == 32
    # Full MHA (32/32). The 135M is GQA (9 query, 3 key).
    assert cfg.n_kv_heads == hf.num_key_value_heads == cfg.n_heads
    assert cfg.head_dim == hf.head_dim == 64
    assert cfg.hidden_size == hf.hidden_size == 2048
    assert cfg.intermediate_size == hf.intermediate_size == 8192
    assert cfg.rms_norm_eps == hf.rms_norm_eps
    assert cfg.vocab_size == hf.vocab_size == 49152
    assert cfg.max_position == hf.max_position_embeddings == 8192
    # 135M rope base is 100000.
    assert cfg.rope_theta == rope["rope_theta"] == 130_000
    assert cfg.rope_type == rope["rope_type"] == "default"
    assert cfg.tie_word_embeddings is True
    assert cfg.eos_token_ids == (hf.eos_token_id,)
    assert cfg.bos_token_id == hf.bos_token_id == 0


def _tokenizer_family():
    # Draft and target must encode the same ids.
    small = AutoTokenizer.from_pretrained(SMALL, local_files_only=True)
    large = AutoTokenizer.from_pretrained(LARGE, local_files_only=True)
    assert small.vocab_size == large.vocab_size == 49152
    for text in PROMPTS:
        assert small(text, add_special_tokens=False).input_ids == large(
            text, add_special_tokens=False
        ).input_ids, text


def _fp32_greedy():
    tok = AutoTokenizer.from_pretrained(LARGE, local_files_only=True)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    cfg, weights, rope = load_engine("cpu", torch.float32, name=LARGE)
    assert weights.embed.shape == (49152, 2048)
    assert weights.lm_head.data_ptr() == weights.embed.data_ptr()
    # q and k are both [n_heads * head_dim, hidden].
    assert weights.layers[0].q.shape == (2048, 2048)
    assert weights.layers[0].k.shape == (2048, 2048)
    eos = cfg.eos_token_ids[0]

    paged = partial(forward_paged, weights, cfg, rope)
    with torch.no_grad():
        # Fresh cache: forward_paged rejects a call with no new tokens.
        logits, _prefill = paged(
            ids, new_page_cache(cfg, n_pages=8, block_size=4, device="cpu", dtype=torch.float32)
        )
        mine = generate(
            paged,
            ids,
            max_new_tokens=NEW_TOKENS,
            eos_id=eos,
            cache=new_page_cache(cfg, n_pages=8, block_size=4, device="cpu", dtype=torch.float32),
        )
    logits = logits.detach().cpu()
    mine = mine.detach().cpu()
    del weights, rope, paged, cfg, _prefill
    gc.collect()

    model = AutoModelForCausalLM.from_pretrained(
        LARGE, dtype=torch.float32, local_files_only=True
    ).eval()
    model.config._attn_implementation = "eager"
    with torch.no_grad():
        ref_logits = model(ids).logits[:, -1]
        ref = model.generate(ids, max_new_tokens=NEW_TOKENS, do_sample=False, pad_token_id=eos)
    assert_logits("smol_1_7b_cpu", logits, ref_logits, LLAMA_LOGIT_ATOL)
    assert torch.equal(mine, ref.cpu()), (mine.tolist(), ref.cpu().tolist())


# 20 prompts x 20 tokens = 400 positions. The 135M cap was frozen on 60.
LARGE_PROMPTS = PROMPTS + (
    "Hello",
    "def add(a, b):\n    return",
    "The weather today is",
    "Q: What is two plus two?\nA:",
    "She opened the door and",
    "The largest planet in the solar system is",
    "1, 1, 2, 3, 5, 8,",
    "The opposite of cold is",
    "Once the rain stopped,",
    "Translate to French: good morning\n",
    "A triangle has three",
    "User: How do I boil an egg?\nAssistant:",
    "In 1969, astronauts",
    "The word for a baby cat is",
)
LARGE_NEW_TOKENS = 20


def _bf16_agreement():
    # Oracle is CPU fp32. Engine is CUDA bf16. Drop the oracle before loading weights.
    tok = AutoTokenizer.from_pretrained(LARGE, local_files_only=True)
    oracle = AutoModelForCausalLM.from_pretrained(
        LARGE, dtype=torch.float32, local_files_only=True
    ).eval()
    oracle.config._attn_implementation = "eager"
    with torch.inference_mode():
        trace = fixed_greedy_trace(tok, oracle, LARGE_PROMPTS, LARGE_NEW_TOKENS)
    del oracle
    gc.collect()

    cfg, weights, rope = load_engine("cuda", torch.bfloat16, name=LARGE)
    # name=LARGE. The default load is the 135M, and that KL looks like a broken kernel.
    assert cfg.n_layers == 24 and weights.embed.shape == (49152, 2048)

    def start_uncached():
        return None

    def step_uncached(_state, prefix):
        return forward_llama(weights, cfg, prefix, rope), None

    def start_paged():
        # block_size 4 leaves a partial last page. 64 pages = 256 slots.
        return new_page_cache(cfg, n_pages=64, block_size=4, device="cuda", dtype=torch.bfloat16)

    def step_paged(seq, prefix):
        return forward_paged(weights, cfg, rope, prefix, seq)

    with torch.inference_mode():
        uncached = score_trace(trace, start_uncached, step_uncached)
        paged = score_trace(trace, start_paged, step_paged)
    for label, (kl_value, top1, total) in (("uncached", uncached), ("paged", paged)):
        assert total == len(LARGE_PROMPTS) * LARGE_NEW_TOKENS, (label, total)
        assert_bf16(kl_value, top1, section="cuda-bf16-1_7b", label=label)
    return uncached, paged


def main():
    _config()
    _tokenizer_family()
    _fp32_greedy()
    if not torch.cuda.is_available():
        raise SystemExit("cuda is required: the 1.7B bf16 agreement runs on the GPU")
    _bf16_agreement()


if __name__ == "__main__":
    main()
