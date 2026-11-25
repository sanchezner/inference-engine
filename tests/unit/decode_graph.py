import torch

from engine.config import ModelConfig
from engine.decode_graph import DecodeGraphs, greedy_tokens
from engine.model import RopeTables
from engine.paged import forward_paged, forward_paged_batch, new_page_pool, new_sequence
from engine.weights import LayerWeights, Weights


def main():
    if not torch.cuda.is_available():
        raise SystemExit("cuda is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    _a_bucket_matches_eager()
    _a_second_step_reuses_the_graph()
    _one_sequence_matches_eager()
    _an_odd_batch_stays_eager()
    _greedy_tokens_are_one_list()


def _a_bucket_matches_eager():
    for dtype, atol in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2)):
        cfg, weights, rope = _model(dtype, seed=1)
        seqs = _prefill(cfg, weights, rope, (20, 7), seed=2)
        snap = _snapshot(seqs)
        eager, eager_state = _step(weights, cfg, rope, seqs, None)
        _restore(seqs, snap)
        runner = DecodeGraphs()
        got, got_state = _step(weights, cfg, rope, seqs, runner)
        _close(got, eager, atol, f"logits {dtype}")
        _close(got_state["k"], eager_state["k"], atol, f"k {dtype}")
        _close(got_state["v"], eager_state["v"], atol, f"v {dtype}")
        assert got_state["seqs"] == eager_state["seqs"]
        assert len(runner._graphs) == 1


def _a_second_step_reuses_the_graph():
    cfg, weights, rope = _model(torch.float32, seed=3)
    seqs = _prefill(cfg, weights, rope, (20, 7), seed=4)
    runner = DecodeGraphs()
    _step(weights, cfg, rope, seqs, runner)
    assert len(runner._graphs) == 1
    snap = _snapshot(seqs)
    eager, eager_state = _step(weights, cfg, rope, seqs, None)
    _restore(seqs, snap)
    got, got_state = _step(weights, cfg, rope, seqs, runner)
    assert len(runner._graphs) == 1
    _close(got, eager, 1e-5, "replay logits")
    _close(got_state["k"], eager_state["k"], 1e-5, "replay k")
    assert greedy_tokens(got) == greedy_tokens(eager)


def _one_sequence_matches_eager():
    cfg, weights, rope = _model(torch.float32, seed=5)
    seqs = _prefill(cfg, weights, rope, (18,), seed=6)
    seq = seqs[0]
    prompt = torch.randint(0, cfg.vocab_size, (1, seq.seq_len), device="cuda", generator=_gen(6))
    # The prefill already wrote those positions. The next call only scores the new column.
    new_id = torch.randint(0, cfg.vocab_size, (1, 1), device="cuda", generator=_gen(7))
    full = torch.cat([prompt, new_id], dim=1)
    snap = _snapshot(seqs)
    eager, seq = forward_paged(weights, cfg, rope, full, seq)
    eager_k = seq.pool.k.clone()
    _restore(seqs, snap)
    runner = DecodeGraphs()
    got, seq = forward_paged(weights, cfg, rope, full, seq, graphs=runner)
    _close(got, eager, 1e-5, "single sequence")
    _close(seq.pool.k, eager_k, 1e-5, "single sequence k")
    assert len(runner._graphs) == 1


def _an_odd_batch_stays_eager():
    cfg, weights, rope = _model(torch.float32, seed=8)
    seqs = _prefill(cfg, weights, rope, (8, 8, 8), seed=9)
    snap = _snapshot(seqs)
    eager, _state = _step(weights, cfg, rope, seqs, None)
    _restore(seqs, snap)
    runner = DecodeGraphs()
    got, _state = _step(weights, cfg, rope, seqs, runner)
    assert runner._graphs == {}
    assert torch.equal(got, eager)


def _greedy_tokens_are_one_list():
    logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, 0.0, 0.5]])
    assert greedy_tokens(logits) == [1, 0]


def _step(weights, cfg, rope, seqs, graphs):
    ids = torch.randint(0, cfg.vocab_size, (len(seqs), 1), device="cuda", generator=_gen(11))
    logits, seqs = forward_paged_batch(weights, cfg, rope, ids, seqs, graphs=graphs)
    return logits, _snapshot(seqs)


def _prefill(cfg, weights, rope, lengths, seed):
    pages = sum((n + 16) // 16 for n in lengths) + 4
    pool = new_page_pool(cfg, n_pages=pages, block_size=16, device="cuda", dtype=weights.embed.dtype)
    generator = _gen(seed)
    seqs = []
    for n in lengths:
        seq = new_sequence(pool)
        ids = torch.randint(0, cfg.vocab_size, (1, n), device="cuda", generator=generator)
        forward_paged(weights, cfg, rope, ids, seq)
        seqs.append(seq)
    return seqs


def _snapshot(seqs):
    pool = seqs[0].pool
    return {
        "k": pool.k.clone(),
        "v": pool.v.clone(),
        "free": list(pool.free),
        "refcount": list(pool.refcount),
        "seqs": [(list(seq.blocks), seq.seq_len) for seq in seqs],
    }


def _restore(seqs, snap) -> None:
    pool = seqs[0].pool
    pool.k.copy_(snap["k"])
    pool.v.copy_(snap["v"])
    pool.free[:] = snap["free"]
    pool.refcount[:] = snap["refcount"]
    for seq, (blocks, n) in zip(seqs, snap["seqs"]):
        seq.blocks = list(blocks)
        seq.seq_len = n


def _model(dtype, seed):
    g = _gen(seed)
    n_heads, n_kv, head_dim = 4, 4, 64
    hidden = n_heads * head_dim
    intermediate = 128
    cfg = ModelConfig(
        n_layers=2,
        n_heads=n_heads,
        n_kv_heads=n_kv,
        head_dim=head_dim,
        hidden_size=hidden,
        intermediate_size=intermediate,
        rms_norm_eps=1e-5,
        vocab_size=32,
        max_position=64,
        rope_theta=10_000.0,
        rope_type="default",
        tie_word_embeddings=True,
        eos_token_ids=(0,),
        bos_token_id=None,
    )

    def draw(*shape):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=g) * 0.02

    layers = tuple(
        LayerWeights(
            q=draw(n_heads * head_dim, hidden),
            k=draw(n_kv * head_dim, hidden),
            v=draw(n_kv * head_dim, hidden),
            o=draw(hidden, n_heads * head_dim),
            gate=draw(intermediate, hidden),
            up=draw(intermediate, hidden),
            down=draw(hidden, intermediate),
            input_norm=torch.ones(hidden, device="cuda", dtype=dtype),
            post_norm=torch.ones(hidden, device="cuda", dtype=dtype),
        )
        for _ in range(cfg.n_layers)
    )
    embed = draw(cfg.vocab_size, hidden)
    weights = Weights(
        embed=embed,
        layers=layers,
        norm=torch.ones(hidden, device="cuda", dtype=dtype),
        lm_head=embed,
    )
    return cfg, weights, RopeTables.build(cfg, "cuda")


def _gen(seed):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    return g


def _close(got, ref, atol, name) -> None:
    err = (got - ref).abs().max().item()
    assert err <= atol, f"{name}: max abs err {err} > {atol}"


if __name__ == "__main__":
    main()
