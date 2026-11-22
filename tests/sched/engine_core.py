import random

import torch

from engine.paged import PagePool, reserve_pages
from engine.schedule import generate_continuous_paged
from engine.sched.engine_core import EngineCore
from engine.sched.request import QueueFull, Request, SamplingParams

VOCAB = 32


def main():
    _rejects_bad_requests()
    _length_and_stop_match_the_recurrence()
    _batch_driver_matches_the_recurrence()
    _max_batch_leaves_the_rest_waiting()
    _budget_one_frees_pages()
    _abort_waiting_running_and_unknown()
    _duplicate_id_is_rejected_until_the_output_is_delivered()
    _never_fits_is_rejected()
    _reserve_max_keeps_growth_pages()
    _oldest_waiter_blocks_a_smaller_one()
    _timestamps_are_ordered()
    _idle_stats_echo_the_pool()
    _queue_cap_refuses_the_next_waiter()
    _interleaving_conserves_pages()


def _rejects_bad_requests():
    try:
        SamplingParams(temperature=0.7)
    except ValueError as exc:
        assert "greedy" in str(exc)
    else:
        raise AssertionError("non-zero temperature was accepted")

    try:
        Request("", (1,), 1)
    except ValueError as exc:
        assert "request_id" in str(exc)
    else:
        raise AssertionError("empty request id was accepted")

    try:
        Request("a", (), 1)
    except ValueError as exc:
        assert "empty prompt" in str(exc)
    else:
        raise AssertionError("empty prompt was accepted")

    try:
        Request("a", (1,), 0)
    except ValueError as exc:
        assert "max_new_tokens" in str(exc)
    else:
        raise AssertionError("zero budget was accepted")

    engine = _engine(_pool(4, 4), max_batch=1)
    try:
        engine.add_request(("not", "a", "request"))
    except TypeError as exc:
        assert "Request" in str(exc)
    else:
        raise AssertionError("a non-Request was accepted")


def _length_and_stop_match_the_recurrence():
    pool = _pool(16, 4)
    engine = _engine(pool, max_batch=2)
    engine.add_request(Request("paris", (1, 2), max_new_tokens=4))
    engine.add_request(Request("stop-me", (7,), max_new_tokens=10, stop_token_ids=(9,)))
    engine.add_request(Request("both", (4,), max_new_tokens=1, stop_token_ids=(5,)))

    got = _drain(engine)
    assert got["paris"] == [(3, None), (4, None), (5, None), (6, "length")]
    assert got["stop-me"] == [(8, None), (9, "stop")]
    assert got["both"] == [(5, "stop")]
    assert engine.stats().counters.requests_finished == 3
    assert engine.stats().counters.tokens_generated == 7
    assert engine.stats().blocks_free == engine.stats().blocks_total
    _conserved(pool, engine)

    again = _engine(_pool(16, 4), max_batch=2)
    again.add_request(Request("paris", (1, 2), max_new_tokens=4))
    assert _drain(again)["paris"] == got["paris"]


def _batch_driver_matches_the_recurrence():
    prompts = [torch.tensor([1, 2]), torch.tensor([7]), torch.tensor([4, 4, 4])]
    budgets = [4, 2, 3]
    pool = _pool(16, 4)
    outs = generate_continuous_paged(_forward_one, _forward_batch, pool, prompts, budgets, eos_id=9, max_batch=2)
    assert len(pool.free) == 16
    for prompt, budget, got in zip(prompts, budgets, outs):
        expected = list(prompt.tolist()) + _expect(prompt.tolist(), budget, stop=(9,))
        assert got.tolist() == expected, (got.tolist(), expected)


def _max_batch_leaves_the_rest_waiting():
    pool = _pool(16, 4)
    engine = _engine(pool, max_batch=1)
    engine.add_request(Request("a", (1, 1, 1, 1), max_new_tokens=5))
    engine.add_request(Request("b", (2, 2, 2, 2, 2, 2), max_new_tokens=5))
    stats = engine.stats()
    assert stats.requests_waiting == 2
    assert stats.pending_prefill_tokens == 10

    first = engine.step()
    assert [out.request_id for out in first] == ["a", "a"]
    stats = engine.stats()
    assert stats.requests_waiting == 1
    assert stats.requests_running == 1
    assert stats.pending_prefill_tokens == 6
    assert stats.blocks_free == 16 - 2
    _conserved(pool, engine)


def _budget_one_frees_pages():
    pool = _pool(8, 4)
    engine = _engine(pool, max_batch=2)
    engine.add_request(Request("short", (1, 2, 3, 4, 5), max_new_tokens=1))
    outs = engine.step()
    assert [(out.token_id, out.finish_reason) for out in outs] == [(6, "length")]
    assert engine.stats().blocks_free == 8
    assert engine.stats().requests_running == 0
    _conserved(pool, engine)


def _abort_waiting_running_and_unknown():
    pool = _pool(8, 4)
    engine = _engine(pool, max_batch=1)
    engine.add_request(Request("live", (1, 2, 3, 4, 5), max_new_tokens=8))
    engine.add_request(Request("queued", (8, 8), max_new_tokens=8))
    engine.abort("queued")
    engine.abort("missing")
    stats = engine.stats()
    assert stats.counters.requests_aborted == 1
    assert stats.requests_waiting == 1
    assert stats.blocks_free == 8

    ran = engine.step()
    assert [out.request_id for out in ran] == ["queued", "live", "live"]
    assert ran[0].finish_reason == "abort" and ran[0].token_id is None
    assert engine.stats().counters.tokens_generated == 2
    held = engine.stats().blocks_free
    assert held < 8

    engine.abort("live")
    assert engine.stats().blocks_free == 8
    assert engine.stats().requests_running == 0
    assert engine.stats().counters.requests_aborted == 2
    assert engine.stats().counters.tokens_generated == 2
    aborted = engine.step()
    assert len(aborted) == 1 and aborted[0].finish_reason == "abort"
    assert engine.stats().counters.requests_finished == 0

    engine.abort("live")
    engine.abort("queued")
    assert engine.stats().counters.requests_aborted == 2
    assert engine.step() == []
    _conserved(pool, engine)


def _duplicate_id_is_rejected_until_the_output_is_delivered():
    engine = _engine(_pool(8, 4), max_batch=1)
    engine.add_request(Request("a", (1,), max_new_tokens=4))
    try:
        engine.add_request(Request("a", (2,), max_new_tokens=1))
    except ValueError as exc:
        assert "already" in str(exc)
    else:
        raise AssertionError("duplicate id was accepted")

    engine.abort("a")
    try:
        engine.add_request(Request("a", (2,), max_new_tokens=1))
    except ValueError as exc:
        assert "already" in str(exc)
    else:
        raise AssertionError("id was reused while its abort output was still queued")

    assert engine.step()[0].finish_reason == "abort"
    engine.add_request(Request("a", (2,), max_new_tokens=1))
    assert engine.stats().requests_waiting == 1


def _never_fits_is_rejected():
    pool = _pool(1, 4)
    engine = _engine(pool, max_batch=2)
    # 1 + 8 - 1 = 8 tokens, two pages. The prompt itself would fit in the one page.
    engine.add_request(Request("nope", (1,), max_new_tokens=8))
    try:
        engine.add_request(Request("nope", (1,), max_new_tokens=1))
    except ValueError as exc:
        assert "already" in str(exc)
    else:
        raise AssertionError("rejected id was reused before its output was delivered")
    engine.add_request(Request("ok", (2,), max_new_tokens=1))
    stats = engine.stats()
    assert stats.counters.requests_rejected == 1
    assert stats.requests_waiting == 1
    assert stats.blocks_free == 1

    outs = engine.step()
    assert [(out.request_id, out.token_id, out.finish_reason) for out in outs] == [
        ("nope", None, "rejected"),
        ("ok", 3, "length"),
    ]
    assert engine.stats().blocks_free == 1
    assert engine.stats().counters.requests_finished == 1
    assert engine.stats().counters.tokens_generated == 1
    engine.add_request(Request("nope", (3, 3, 3, 3, 3), max_new_tokens=1))
    assert engine.stats().counters.requests_rejected == 2
    assert engine.stats().requests_waiting == 0
    again = engine.step()
    assert len(again) == 1 and again[0].finish_reason == "rejected" and again[0].token_id is None
    assert engine.stats().blocks_free == 1
    _conserved(pool, engine)


def _reserve_max_keeps_growth_pages():
    # grow ends on 2 pages. hold needs 1, and that page is still free after grow's prefill.
    pool = _pool(2, 4)
    engine = _engine(pool, max_batch=2)
    engine.add_request(Request("grow", (1, 1, 1, 1), max_new_tokens=5))
    engine.add_request(Request("hold", (2,), max_new_tokens=4))
    first = engine.step()
    assert [(out.request_id, out.token_id) for out in first] == [("grow", 2), ("grow", 3)]
    assert engine.stats().requests_waiting == 1
    assert engine.stats().requests_running == 1
    assert engine.stats().blocks_free == 0
    outs = first + _collect(engine)
    assert [(out.request_id, out.token_id, out.finish_reason) for out in outs] == [
        ("grow", 2, None),
        ("grow", 3, None),
        ("grow", 4, None),
        ("grow", 5, None),
        ("grow", 6, "length"),
        ("hold", 3, None),
        ("hold", 4, None),
        ("hold", 5, None),
        ("hold", 6, "length"),
    ]
    assert engine.stats().blocks_free == 2


def _oldest_waiter_blocks_a_smaller_one():
    # After anchor's first step one page is free. thin would fit; wide, in front of it, would not.
    pool = _pool(3, 4)
    engine = _engine(pool, max_batch=2)
    engine.add_request(Request("anchor", (1, 1, 1, 1), max_new_tokens=5))
    engine.add_request(Request("wide", (2, 2, 2, 2, 2), max_new_tokens=5))
    engine.add_request(Request("thin", (3,), max_new_tokens=1))
    first = engine.step()
    assert [out.request_id for out in first] == ["anchor", "anchor"]
    stats = engine.stats()
    assert stats.requests_waiting == 2
    assert stats.requests_running == 1
    assert stats.blocks_free == 1
    outs = first + _collect(engine)
    assert [out.request_id for out in outs] == ["anchor"] * 5 + ["wide"] * 5 + ["thin"]
    assert [(out.token_id, out.finish_reason) for out in outs if out.request_id == "thin"] == [(4, "length")]
    assert engine.stats().blocks_free == 3


def _timestamps_are_ordered():
    clock = _Clock()
    engine = _engine(_pool(8, 4), max_batch=1, clock=clock)
    engine.add_request(Request("a", (3,), max_new_tokens=3))
    outs = []
    for _ in range(8):
        step_outs = engine.step()
        outs.extend(step_outs)
        if not step_outs and engine.stats().requests_running == 0:
            break
    assert [out.token_id for out in outs] == [4, 5, 6]
    assert outs[0].timestamps.arrival == 1
    emitted = [out.timestamps.emitted for out in outs]
    assert emitted == sorted(emitted) and len(set(emitted)) == len(emitted)
    assert all(out.timestamps.emitted > out.timestamps.arrival for out in outs)


def _idle_stats_echo_the_pool():
    engine = _engine(_pool(8, 4), max_batch=2)
    stats = engine.stats()
    assert stats.requests_waiting == 0 and stats.requests_running == 0
    assert stats.blocks_total == 8 and stats.blocks_free == 8 and stats.block_size == 4
    assert stats.pending_prefill_tokens == 0
    assert stats.counters.requests_finished == 0
    assert stats.counters.requests_aborted == 0
    assert stats.counters.requests_rejected == 0
    assert stats.counters.tokens_generated == 0
    assert stats.counters.prefix_cache_hit_tokens == 0
    assert stats.counters.prefix_cache_query_tokens == 0
    assert engine.step() == []


def _queue_cap_refuses_the_next_waiter():
    pool = _pool(8, 4)
    try:
        _engine(pool, max_batch=1, max_waiting=0)
    except ValueError as exc:
        assert "max_waiting" in str(exc)
    else:
        raise AssertionError("max_waiting of 0 was accepted")

    engine = _engine(pool, max_batch=1, max_waiting=1)
    engine.add_request(Request("a", (1, 1, 1, 1), max_new_tokens=6))
    # 40 tokens need 10 pages. The pool has 8, so this never waits and never takes the one seat.
    engine.add_request(Request("huge", (2,) * 40, max_new_tokens=1))
    assert engine.stats().requests_waiting == 1
    assert engine.stats().counters.requests_rejected == 1
    try:
        engine.add_request(Request("b", (3,), max_new_tokens=2))
    except QueueFull as exc:
        assert exc.waiting == 1 and exc.max_waiting == 1
    else:
        raise AssertionError("a full queue accepted another waiter")
    try:
        engine.add_request(Request("a", (3,), max_new_tokens=1))
    except ValueError as exc:
        assert "already" in str(exc)
    else:
        raise AssertionError("duplicate id on a full queue was accepted")

    engine.step()
    assert engine.stats().requests_waiting == 0
    engine.add_request(Request("b", (3,), max_new_tokens=2))
    assert engine.stats().requests_waiting == 1
    _conserved(pool, engine)


def _interleaving_conserves_pages():
    for seed in range(8):
        _one_interleaving(seed)


def _one_interleaving(seed):
    # Six pages cannot cover three fat requests at once, so some have to wait.
    rng = random.Random(seed)
    pool = _pool(6, 4)
    engine = _engine(pool, max_batch=3, max_waiting=5)
    accepted = []
    for _ in range(250):
        roll = rng.randrange(3)
        if roll == 0:
            rid = f"s{seed}-n{len(accepted)}"
            if rng.randrange(10) == 0:
                req = Request(rid, (1,) * 25, max_new_tokens=1)
            else:
                prompt = tuple(rng.randrange(VOCAB) for _ in range(rng.randint(1, 8)))
                req = Request(rid, prompt, max_new_tokens=rng.randint(1, 8))
            try:
                engine.add_request(req)
            except QueueFull as exc:
                assert exc.waiting == engine.max_waiting
                assert engine.stats().requests_waiting == engine.max_waiting
            else:
                accepted.append(rid)
        elif roll == 1:
            engine.step()
        elif accepted:
            engine.abort(rng.choice(accepted))
        _conserved(pool, engine)

    for rid in accepted:
        engine.abort(rid)
    engine.step()
    assert engine.step() == []
    stats = engine.stats()
    assert stats.requests_waiting == 0 and stats.requests_running == 0
    assert stats.blocks_free == stats.blocks_total
    counters = stats.counters
    assert (
        counters.requests_finished + counters.requests_aborted + counters.requests_rejected
        == len(accepted)
    )
    _conserved(pool, engine)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1.0
        return self.t


def _engine(pool, max_batch, clock=None, max_waiting=256):
    return EngineCore(_forward_one, _forward_batch, pool, max_batch, clock=clock, max_waiting=max_waiting)


def _pool(n_pages, block_size):
    shape = (1, n_pages, 1, block_size, 1)
    return PagePool(
        k=torch.zeros(shape),
        v=torch.zeros(shape),
        free=list(range(n_pages)),
        block_size=block_size,
    )


def _forward_one(ids, seq):
    n_new = ids.shape[1] - seq.seq_len
    if n_new <= 0:
        raise ValueError("paged forward was given no new tokens")
    row = _advance(seq, n_new, int(ids[0, -1].item()))
    return row.view(1, -1), seq


def _forward_batch(ids, seqs):
    rows = [_advance(seq, ids.shape[1], int(ids[i, -1].item())) for i, seq in enumerate(seqs)]
    return torch.stack(rows), seqs


def _advance(seq, n_new, last_id):
    reserve_pages(seq, seq.seq_len + n_new)
    seq.seq_len += n_new
    nxt = (last_id + 1) % VOCAB
    row = torch.zeros(VOCAB)
    row[nxt] = 10
    return row


def _expect(prompt, budget, stop=()):
    tokens = []
    last = prompt[-1]
    for _ in range(budget):
        last = (last + 1) % VOCAB
        tokens.append(last)
        if last in stop:
            break
    return tokens


def _collect(engine):
    outs = []
    for _ in range(32):
        step_outs = engine.step()
        outs.extend(step_outs)
        stats = engine.stats()
        _conserved(engine.pool, engine)
        if not step_outs and stats.requests_waiting == 0 and stats.requests_running == 0:
            return outs
    raise AssertionError("engine did not finish")


def _drain(engine):
    grouped = {}
    for _ in range(32):
        outs = engine.step()
        for out in outs:
            grouped.setdefault(out.request_id, []).append((out.token_id, out.finish_reason))
        stats = engine.stats()
        _conserved(engine.pool, engine)
        if stats.requests_waiting == 0 and stats.requests_running == 0:
            return grouped
    raise AssertionError("engine did not finish")


def _conserved(pool, engine):
    n_pages = pool.k.shape[1]
    held = [block for slot in engine._running for block in slot.seq.blocks]
    assert len(held) == len(set(held)), held
    assert sorted(held + pool.free) == list(range(n_pages))


if __name__ == "__main__":
    main()
