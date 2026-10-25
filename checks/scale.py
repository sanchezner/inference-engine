import gc
from functools import partial

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from checks.agreement import PROMPTS
from checks.oracle import MODEL as SMALL, load_engine
from checks.tolerances import LLAMA_LOGIT_ATOL, assert_logits
from engine.config import ModelConfig
from engine.generate import generate
from engine.paged import forward_paged, new_page_cache


# Same layout as the 135M, sized so a benchmark is about the model and not
# about Python. fp32 weights are about 6.8 GB, which does not fit beside a
# CUDA context on an 8 GB 3070, so this oracle stays on CPU. It is in
# `make check` anyway: a full pass is about 15 seconds once the snapshot
# is cached.
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
    # Full MHA: every query head has its own key. The 135M is GQA (9 query
    # heads, 3 key heads). KV per token is large, which is why paging matters
    # on this checkpoint.
    assert cfg.n_kv_heads == hf.num_key_value_heads == cfg.n_heads
    assert cfg.head_dim == hf.head_dim == 64
    assert cfg.hidden_size == hf.hidden_size == 2048
    assert cfg.intermediate_size == hf.intermediate_size == 8192
    assert cfg.rms_norm_eps == hf.rms_norm_eps
    assert cfg.vocab_size == hf.vocab_size == 49152
    assert cfg.max_position == hf.max_position_embeddings == 8192
    # 135M uses 100000. Sharing that constant would rotate this model wrong.
    assert cfg.rope_theta == rope["rope_theta"] == 130_000
    assert cfg.rope_type == rope["rope_type"] == "default"
    assert cfg.tie_word_embeddings is True
    assert cfg.eos_token_ids == (hf.eos_token_id,)
    assert cfg.bos_token_id == hf.bos_token_id == 0


def _tokenizer_family():
    # Speculative decoding (M10) needs the draft and the target to name the
    # same ids. Same vocab size is not enough; the encodings have to match.
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
    # MHA projection: q and k are both [n_heads * head_dim, hidden].
    assert weights.layers[0].q.shape == (2048, 2048)
    assert weights.layers[0].k.shape == (2048, 2048)
    eos = cfg.eos_token_ids[0]

    paged = partial(forward_paged, weights, cfg, rope)
    with torch.no_grad():
        # Logits come from one prefill. Generate starts from an empty cache:
        # forward_paged refuses a call that brings no new tokens, so the
        # prefilled page cannot be reused as the generate cache.
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


def main():
    _config()
    _tokenizer_family()
    _fp32_greedy()


if __name__ == "__main__":
    main()
