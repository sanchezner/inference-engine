import torch

from functools import partial

from engine.config import ModelConfig
from engine.decode_graph import DecodeGraphs, _table_bucket, greedy_tokens
from engine.model import RopeTables
from engine.paged import forward_paged, forward_paged_batch, new_page_pool, new_sequence
from engine.sched.engine_core import EngineCore
from engine.sched.request import Request
from engine.weights import LayerWeights, Weights


def main():
    if not torch.cuda.is_available():
        raise SystemExit("cuda is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    _table_buckets_stop_at_the_pool()
    _a_bucket_matches_eager()
    _a_second_step_reuses_the_graph()
    _one_sequence_matches_eager()
    _an_odd_batch_stays_eager()
    _crossing_pages_uses_one_graph_per_bucket()
    _greedy_tokens_are_one_list()
    _the_engine_matches_eager()
    _a_stable_batch_reuses_the_booked_table()
    _a_free_seat_is_booked_before_the_token_read()
    _a_finish_does_not_free_its_seat_early()


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


def _table_buckets_stop_at_the_pool():
    assert _table_bucket(1, 10) == 1
    assert _table_bucket(2, 10) == 2
    assert _table_bucket(3, 10) == 4
    assert _table_bucket(5, 10) == 8
    assert _table_bucket(9, 10) == 10
    assert _table_bucket(8, 8) == 8
    assert _table_bucket(1, 1) == 1


def _crossing_pages_uses_one_graph_per_bucket():
    # Block 16. Lengths 16, 32, 48, and 64 are four pages and three width buckets: 1, 2, and 4.
    cfg, weights, rope = _model(torch.float32, seed=12)
    _walk_across_pages(cfg, weights, rope, (15,), 64, {1, 2, 4}, seed=13)
    _walk_across_pages(cfg, weights, rope, (30, 4), 64, {2, 4}, seed=14)


def _walk_across_pages(cfg, weights, rope, lengths, stop, buckets, seed):
    seqs = _prefill(cfg, weights, rope, lengths, seed)
    runner = DecodeGraphs()
    seen = None
    while seqs[0].seq_len < stop:
        snap = _snapshot(seqs)
        eager, eager_state = _step(weights, cfg, rope, seqs, None)
        _restore(seqs, snap)
        got, got_state = _step(weights, cfg, rope, seqs, runner)
        _close(got, eager, 1e-5, f"crossing {lengths} at {seqs[0].seq_len}")
        _close(got_state["k"], eager_state["k"], 1e-5, f"crossing k {lengths}")
        widths = {key[3] for key in runner._graphs}
        assert widths <= buckets
        assert len(runner._graphs) == len(widths)
        if max(len(seq.blocks) for seq in seqs) >= 3 and seen is None:
            seen = len(runner._graphs)
        elif seen is not None:
            assert len(runner._graphs) == seen
    assert {key[3] for key in runner._graphs} == buckets
    assert seen is not None


def _a_stable_batch_reuses_the_booked_table():
    cfg, weights, rope = _model(torch.float32, seed=30)
    prompts = ((1, 2, 3, 4, 5), (6, 7, 8))
    # Three joint decodes after the solo step: capture, book, then reuse.
    off, _ = _engine_tokens(cfg, weights, rope, prompts, graphs=False, max_batch=2, max_new=6)
    on, runner = _engine_tokens(cfg, weights, rope, prompts, graphs=True, max_batch=2, max_new=6)
    assert on == off
    # Capture, then one replay books the table, then the next replay uses it.
    assert runner.reused >= 1


def _a_free_seat_is_booked_before_the_token_read():
    cfg, weights, rope = _model(torch.float32, seed=31)
    notes = []
    engine = _booked_engine(cfg, weights, rope, 2, notes)
    engine.add_request(Request("a", (1, 2, 3, 4), max_new_tokens=4))
    engine.step()
    engine.step()
    assert notes == []
    engine.add_request(Request("b", (5, 6, 7), max_new_tokens=2))
    outs = engine.step()
    assert len(notes) == 1
    before, after = notes[0]
    assert before["waiting"] == ["b"]
    assert [row[0] for row in before["running"]] == ["a"]
    assert after["waiting"] == []
    booked = [row for row in after["running"] if row[0] == "b"]
    assert booked == [("b", None, 0)]
    assert [row for row in after["running"] if row[0] == "a"] == before["running"]
    assert any(out.request_id == "b" and out.token_id is not None for out in outs)
    _idle(engine)


def _a_finish_does_not_free_its_seat_early():
    cfg, weights, rope = _model(torch.float32, seed=32)
    notes = []
    engine = _booked_engine(cfg, weights, rope, 1, notes)
    engine.add_request(Request("a", (1, 2, 3, 4), max_new_tokens=3))
    engine.step()
    engine.step()
    assert notes == []
    engine.add_request(Request("b", (8, 9), max_new_tokens=2))
    outs = engine.step()
    assert len(notes) == 1
    before, after = notes[0]
    assert before["waiting"] == ["b"] and after["waiting"] == ["b"]
    assert [row[0] for row in after["running"]] == ["a"]
    assert before["running"][0][1] == after["running"][0][1]
    assert any(out.request_id == "a" and out.finish_reason == "length" for out in outs)
    assert any(out.request_id == "b" and out.token_id is not None for out in outs)
    _idle(engine)


def _booked_engine(cfg, weights, rope, max_batch, notes):
    pool = new_page_pool(cfg, n_pages=32, block_size=16, device="cuda", dtype=weights.embed.dtype)
    base = partial(forward_paged_batch, weights, cfg, rope)
    engine = None

    def batch(ids, seqs, graphs=None, on_replay=None):
        def wrapped():
            before = _snap(engine)
            on_replay()
            notes.append((before, _snap(engine)))

        return base(ids, seqs, graphs=graphs, on_replay=wrapped if on_replay is not None else None)

    engine = EngineCore(
        partial(forward_paged, weights, cfg, rope),
        batch,
        pool,
        max_batch,
        graphs=True,
    )
    return engine


def _snap(engine):
    return {
        "waiting": [item.request.request_id for item in engine._waiting],
        "running": [
            (slot.request_id, slot.last_token, slot.seq.seq_len)
            for slot in engine._running
        ],
    }


def _idle(engine):
    for _ in range(16):
        engine.step()
        stats = engine.stats()
        if stats.requests_waiting == 0 and stats.requests_running == 0:
            assert stats.blocks_free == stats.blocks_total
            return
    raise AssertionError("engine did not finish")


def _greedy_tokens_are_one_list():
    logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, 0.0, 0.5]])
    assert greedy_tokens(logits) == [1, 0]


def _the_engine_matches_eager():
    # fp32 greedy tokens are exact, so a replayed decode has to match the eager engine.
    cfg, weights, rope = _model(torch.float32, seed=21)
    pair = ((1, 2, 3, 4, 5), (6, 7, 8))
    off, off_runner = _engine_tokens(cfg, weights, rope, pair, graphs=False, max_batch=2)
    on, on_runner = _engine_tokens(cfg, weights, rope, pair, graphs=True, max_batch=2)
    assert on == off
    assert off_runner is None
    # The first request decodes alone, then the pair. Both sizes are captured.
    assert {key[2] for key in on_runner._graphs} == {1, 2}
    three = pair + ((9, 10, 11),)
    odd, odd_runner = _engine_tokens(cfg, weights, rope, three, graphs=True, max_batch=3)
    eager, _eager_runner = _engine_tokens(cfg, weights, rope, three, graphs=False, max_batch=3)
    assert odd == eager
    # Batch 3 is not a capture size. The ramp still captures 1 and 2 on the way there.
    assert {key[2] for key in odd_runner._graphs} == {1, 2}


def _engine_tokens(cfg, weights, rope, prompts, graphs, max_batch, max_new=4):
    pool = new_page_pool(cfg, n_pages=16, block_size=16, device="cuda", dtype=weights.embed.dtype)
    engine = EngineCore(
        partial(forward_paged, weights, cfg, rope),
        partial(forward_paged_batch, weights, cfg, rope),
        pool,
        max_batch,
        graphs=graphs,
    )
    got = {}
    for i, prompt in enumerate(prompts):
        engine.add_request(Request(str(i), prompt, max_new_tokens=max_new))
        got[str(i)] = []
    for _ in range(32):
        outs = engine.step()
        for out in outs:
            if out.token_id is not None:
                got[out.request_id].append(out.token_id)
        stats = engine.stats()
        if stats.requests_waiting == 0 and stats.requests_running == 0:
            assert stats.blocks_free == stats.blocks_total
            return got, engine._graphs
    raise AssertionError("engine did not finish")


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
