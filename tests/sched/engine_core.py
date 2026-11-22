import torch

from engine.paged import PagePool, reserve_pages
from engine.schedule import generate_continuous_paged
from engine.sched.engine_core import EngineCore
from engine.sched.request import Request, SamplingParams

VOCAB = 32


def main():
    _rejects_bad_requests()
    _length_and_stop_match_the_recurrence()
    _batch_driver_matches_the_recurrence()
    _max_batch_leaves_the_rest_waiting()
    _budget_one_frees_pages()
    _abort_waiting_running_and_unknown()
    _duplicate_id_is_rejected_until_the_output_is_delivered()
    _timestamps_are_ordered()
    _idle_stats_echo_the_pool()


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


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1.0
        return self.t


def _engine(pool, max_batch, clock=None):
    return EngineCore(_forward_one, _forward_batch, pool, max_batch, clock=clock)


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
