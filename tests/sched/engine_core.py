import random

import torch

from engine.paged import PagePool, reserve_pages
from engine.schedule import generate_continuous_paged
from engine.sched.engine_core import EngineCore
from engine.sched.request import QueueFull, Request, SamplingParams
from engine.sched.sampling import sample_token

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
    _margin_one_seats_beside_a_long_budget()
    _margin_covering_the_budget_holds_the_spare_page()
    _preemption_matches_a_roomy_run()
    _random_pressure_matches_a_roomy_run()
    _a_preempted_request_can_stand_past_the_cap()
    _a_budget_above_the_prompt_matches_the_uncapped_steps()
    _a_chunk_emits_nothing_until_the_prompt_is_scored()
    _a_step_splits_the_budget_across_waiters()
    _decode_continues_during_a_chunked_prefill()
    _chunked_prefill_matches_an_uncapped_run()
    _chunked_interleaving_conserves_pages()
    _rejects_bad_sampling()
    _a_cpu_graph_flag_stays_eager()
    _greedy_still_matches_the_recurrence()
    _a_seed_matches_its_own_stream_beside_another()
    _preemption_keeps_the_sample_stream()
    _chunked_prefill_keeps_the_sample_stream()


def _rejects_bad_requests():
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

    try:
        _engine(_pool(1, 4), max_batch=1, growth_margin=-1)
    except ValueError as exc:
        assert "growth_margin" in str(exc)
    else:
        raise AssertionError("a negative growth margin was accepted")

    try:
        _engine(_pool(1, 4), max_batch=1, growth_margin=True)
    except ValueError as exc:
        assert "growth_margin" in str(exc)
    else:
        raise AssertionError("a bool growth margin was accepted")

    try:
        _engine(_pool(1, 4), max_batch=1, graphs=1)
    except ValueError as exc:
        assert "graphs" in str(exc)
    else:
        raise AssertionError("graphs=1 was accepted")

    try:
        _engine(_pool(1, 4), max_batch=1, prefill_budget=0)
    except ValueError as exc:
        assert "prefill_budget" in str(exc)
    else:
        raise AssertionError("a prefill budget of 0 was accepted")

    try:
        _engine(_pool(1, 4), max_batch=1, prefill_budget=True)
    except ValueError as exc:
        assert "prefill_budget" in str(exc)
    else:
        raise AssertionError("a bool prefill budget was accepted")


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
    assert [out.request_id for out in first] == ["a"]
    stats = engine.stats()
    assert stats.requests_waiting == 1
    assert stats.requests_running == 1
    assert stats.pending_prefill_tokens == 6
    assert stats.blocks_free == 16 - 1
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
    assert [out.request_id for out in ran] == ["queued", "live"]
    assert ran[0].finish_reason == "abort" and ran[0].token_id is None
    assert engine.stats().counters.tokens_generated == 1
    held = engine.stats().blocks_free
    assert held < 8

    engine.abort("live")
    assert engine.stats().blocks_free == 8
    assert engine.stats().requests_running == 0
    assert engine.stats().counters.requests_aborted == 2
    assert engine.stats().counters.tokens_generated == 1
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
    assert [(out.request_id, out.token_id) for out in first] == [("grow", 2)]
    assert engine.stats().requests_waiting == 1
    assert engine.stats().requests_running == 1
    # grow still needs its second page, and that page is the margin, so hold waits.
    assert engine.stats().blocks_free == 1
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
    assert [out.request_id for out in first] == ["anchor"]
    stats = engine.stats()
    assert stats.requests_waiting == 2
    assert stats.requests_running == 1
    assert stats.blocks_free == 2
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


def _one_interleaving(seed, prefill_budget=None):
    # Six pages cannot cover three fat requests at once, so some have to wait.
    rng = random.Random(seed)
    pool = _pool(6, 4)
    engine = _engine(pool, max_batch=3, max_waiting=5, prefill_budget=prefill_budget)
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
                # A preempted request already has a seat, so the line can stand past the cap.
                assert exc.waiting >= engine.max_waiting
                assert engine.stats().requests_waiting == exc.waiting
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


def _margin_one_seats_beside_a_long_budget():
    # grow ends on 3 pages but holds 1 after its prefill. One spare page seats hold now.
    reqs = [
        Request("grow", (1,), max_new_tokens=12),
        Request("hold", (2,), max_new_tokens=1),
    ]
    roomy = _run(reqs, pages=8, margin=1)
    tight = _engine(_pool(4, 4), max_batch=2, growth_margin=1)
    for req in reqs:
        tight.add_request(req)
    first = tight.step()
    assert [out.request_id for out in first] == ["grow", "hold"]
    assert first[1].finish_reason == "length"
    got = _group(first)
    for rid, pairs in _drain(tight).items():
        got.setdefault(rid, []).extend(pairs)
    assert got == roomy
    assert tight.stats().blocks_free == tight.stats().blocks_total


def _margin_covering_the_budget_holds_the_spare_page():
    # Same pair. A margin that covers grow's whole budget keeps hold waiting while a page is free.
    pool = _pool(3, 4)
    engine = _engine(pool, max_batch=2, growth_margin=8)
    engine.add_request(Request("grow", (1,), max_new_tokens=12))
    engine.add_request(Request("hold", (2,), max_new_tokens=1))
    first = engine.step()
    assert [out.request_id for out in first] == ["grow"]
    stats = engine.stats()
    assert stats.requests_waiting == 1 and stats.requests_running == 1
    assert stats.blocks_free >= 1
    assert _drain(engine)["hold"] == [(3, "length")]
    assert engine.stats().blocks_free == 3
    _conserved(pool, engine)


def _preemption_matches_a_roomy_run():
    reqs = [
        Request("old", (1,), max_new_tokens=8),
        Request("new", (2,), max_new_tokens=5),
    ]
    roomy = _run(reqs, pages=8, margin=0)
    tight = _engine(_pool(2, 4), max_batch=2, growth_margin=0)
    for req in reqs:
        tight.add_request(req)
    saw_new_waiting = False
    grouped = {}
    for _ in range(32):
        if grouped.get("new") and any(item.request.request_id == "new" for item in tight._waiting):
            assert any(slot.request_id == "old" for slot in tight._running)
            saw_new_waiting = True
        outs = tight.step()
        for out in outs:
            grouped.setdefault(out.request_id, []).append((out.token_id, out.finish_reason))
        _conserved(tight.pool, tight)
        stats = tight.stats()
        if stats.requests_waiting == 0 and stats.requests_running == 0:
            break
    else:
        raise AssertionError("tight engine did not finish")
    assert saw_new_waiting
    assert grouped == roomy
    assert tight.stats().blocks_free == tight.stats().blocks_total

    # Aborting the preempted request still gives its pages back, and old's tokens stay the same.
    again = _engine(_pool(2, 4), max_batch=2, growth_margin=0)
    for req in reqs:
        again.add_request(req)
    aborted = False
    old_tokens = []
    for _ in range(32):
        if any(item.request.request_id == "new" for item in again._waiting) and old_tokens:
            again.abort("new")
            aborted = True
        outs = again.step()
        for out in outs:
            if out.request_id == "old" and out.token_id is not None:
                old_tokens.append(out.token_id)
        _conserved(again.pool, again)
        stats = again.stats()
        if aborted and stats.requests_waiting == 0 and stats.requests_running == 0:
            break
    else:
        raise AssertionError("aborted engine did not finish")
    assert aborted
    assert [token for token, _reason in roomy["old"]] == old_tokens
    assert again.stats().counters.requests_aborted == 1
    assert again.stats().blocks_free == again.stats().blocks_total


def _random_pressure_matches_a_roomy_run():
    # Each request fits alone. Together they do not, so a tight pool has to preempt.
    rng = random.Random(0)
    for trial in range(12):
        block = rng.choice((4, 8))
        n = rng.randint(2, 3)
        budget = None if rng.randrange(2) == 0 else rng.choice((1, 3))
        reqs = []
        needs = []
        for i in range(n):
            prompt_len = rng.randint(1, block)
            max_new = block - prompt_len + 2 + rng.randint(0, block)
            prompt = tuple(rng.randrange(VOCAB) for _ in range(prompt_len))
            reqs.append(Request(f"p{trial}-{i}", prompt, max_new_tokens=max_new))
            needs.append(_pages_needed(block, prompt_len, max_new))
        tight = max(needs)
        assert sum(needs) > tight
        order = list(reqs)
        rng.shuffle(order)
        got, saw = _pressure(order, tight, block, budget)
        assert saw, trial
        assert got == _run(reqs, pages=sum(needs) + n, margin=0, prefill_budget=budget)
        for pairs in got.values():
            assert pairs[-1][1] == "length"
            assert all(reason is None for _token, reason in pairs[:-1])

    # One request cannot fit the tight pool. It is rejected; the others still match a roomy run.
    fitting = [
        Request("fit-a", (1, 1), max_new_tokens=6),
        Request("fit-b", (2,), max_new_tokens=6),
    ]
    huge = Request("huge", (3,) * 5, max_new_tokens=8)
    roomy = _run(fitting, pages=16, margin=0)
    got, saw = _pressure(fitting + [huge], tight=2, block=4, budget=None)
    assert saw
    assert got["huge"] == [(None, "rejected")]
    for req in fitting:
        assert got[req.request_id] == roomy[req.request_id]


def _pressure(reqs, tight, block, budget):
    pool = _pool(tight, block)
    engine = _engine(pool, max_batch=len(reqs), growth_margin=0, prefill_budget=budget)
    for req in reqs:
        engine.add_request(req)
    grouped = {}
    saw = False
    limit = 8 * sum(len(req.prompt_token_ids) + req.max_new_tokens for req in reqs) + 20
    for _ in range(limit):
        # A victim re-admitted in the same step is running again, with a shorter cache.
        before = {slot.request_id: slot.seq.seq_len for slot in engine._running}
        outs = engine.step()
        waiting = {item.request.request_id for item in engine._waiting}
        after = {slot.request_id: slot.seq.seq_len for slot in engine._running}
        for rid, n in before.items():
            if rid in waiting or after.get(rid, n) < n:
                saw = True
        for out in outs:
            grouped.setdefault(out.request_id, []).append((out.token_id, out.finish_reason))
        _conserved(pool, engine)
        stats = engine.stats()
        if stats.requests_waiting == 0 and stats.requests_running == 0:
            finished = stats.counters.requests_finished
            rejected = stats.counters.requests_rejected
            assert finished + rejected == len(reqs)
            assert stats.blocks_free == stats.blocks_total
            return grouped, saw
    raise AssertionError("pressured engine did not finish")


def _pages_needed(block, prompt_len, max_new):
    total = prompt_len + max_new - 1
    return (total + block - 1) // block


def _a_preempted_request_can_stand_past_the_cap():
    pool = _pool(2, 4)
    engine = _engine(pool, max_batch=2, max_waiting=1, growth_margin=0)
    engine.add_request(Request("old", (1,), max_new_tokens=8))
    engine.step()
    engine.add_request(Request("new", (2,), max_new_tokens=5))
    engine.step()
    assert engine.stats().requests_running == 2
    engine.add_request(Request("third", (3,), max_new_tokens=1))
    assert engine.stats().requests_waiting == 1
    for _ in range(16):
        engine.step()
        _conserved(pool, engine)
        if engine.stats().requests_waiting > engine.max_waiting:
            break
    else:
        raise AssertionError("preemption did not stand past the waiting cap")
    try:
        engine.add_request(Request("nope", (4,), max_new_tokens=1))
    except QueueFull as exc:
        assert exc.waiting > exc.max_waiting
    else:
        raise AssertionError("a line past the cap accepted another waiter")
    for rid in ("old", "new", "third"):
        engine.abort(rid)
    assert engine.step()
    assert engine.stats().requests_waiting == 0 and engine.stats().requests_running == 0
    assert engine.stats().blocks_free == 2
    _conserved(pool, engine)


def _a_budget_above_the_prompt_matches_the_uncapped_steps():
    reqs = [
        Request("grow", (1,), max_new_tokens=4),
        Request("hold", (2, 2), max_new_tokens=2),
    ]
    bare = _steps(_engine(_pool(8, 4), max_batch=2), reqs)
    capped = _steps(_engine(_pool(8, 4), max_batch=2, prefill_budget=50), reqs)
    assert bare == capped


def _a_chunk_emits_nothing_until_the_prompt_is_scored():
    # 10 prompt tokens, 4 per step. The first generated token waits until the third step.
    pool = _pool(16, 4)
    engine = _engine(pool, max_batch=1, prefill_budget=4)
    prompt = tuple(range(1, 11))
    engine.add_request(Request("p", prompt, max_new_tokens=1))
    assert engine.stats().pending_prefill_tokens == 10
    assert engine.step() == []
    assert engine.stats().requests_running == 1
    assert engine.stats().pending_prefill_tokens == 6
    assert engine.stats().counters.tokens_generated == 0
    assert engine.step() == []
    assert engine.stats().pending_prefill_tokens == 2
    third = engine.step()
    assert [(out.token_id, out.finish_reason, out.cached_tokens) for out in third] == [
        ((prompt[-1] + 1) % VOCAB, "length", 0),
    ]
    assert engine.stats().requests_running == 0
    assert engine.stats().pending_prefill_tokens == 0
    assert engine.stats().blocks_free == 16
    _conserved(pool, engine)


def _a_step_splits_the_budget_across_waiters():
    pool = _pool(16, 4)
    engine = _engine(pool, max_batch=2, prefill_budget=6)
    engine.add_request(Request("a", (1, 1, 1, 1), max_new_tokens=1))
    engine.add_request(Request("b", (2, 2, 2, 2), max_new_tokens=1))
    first = engine.step()
    assert [(out.request_id, out.finish_reason) for out in first] == [("a", "length")]
    assert engine.stats().requests_running == 1
    assert engine.stats().pending_prefill_tokens == 2
    second = engine.step()
    assert [(out.request_id, out.token_id, out.finish_reason) for out in second] == [(
        "b", (2 + 1) % VOCAB, "length",
    )]
    assert engine.stats().blocks_free == 16
    _conserved(pool, engine)


def _decode_continues_during_a_chunked_prefill():
    # chat is already generating. long is scored 3 tokens at a time and must not skip chat's turns.
    pool = _pool(16, 4)
    engine = _engine(pool, max_batch=2, prefill_budget=3)
    engine.add_request(Request("chat", (1,), max_new_tokens=8))
    first = engine.step()
    assert [out.token_id for out in first] == [2]
    long_prompt = tuple(range(2, 12))
    engine.add_request(Request("long", long_prompt, max_new_tokens=2))
    chat = [2]
    long_tokens = []
    saw_long_waiting_on_its_prompt = False
    for _ in range(24):
        outs = engine.step()
        long_scoring = any(slot.request_id == "long" and slot.last_token is None for slot in engine._running)
        if long_scoring:
            assert [out.request_id for out in outs] == ["chat"]
            saw_long_waiting_on_its_prompt = True
        for out in outs:
            if out.request_id == "chat":
                chat.append(out.token_id)
            else:
                long_tokens.append(out.token_id)
        _conserved(pool, engine)
        stats = engine.stats()
        if stats.requests_waiting == 0 and stats.requests_running == 0:
            break
    else:
        raise AssertionError("chunked run did not finish")
    assert saw_long_waiting_on_its_prompt
    assert chat == _expect((1,), 8)
    assert long_tokens == _expect(long_prompt, 2)
    assert engine.stats().blocks_free == 16


def _chunked_prefill_matches_an_uncapped_run():
    reqs = [
        Request("long", tuple(range(10)), max_new_tokens=4),
        Request("short", (7, 7), max_new_tokens=3),
    ]
    assert _run(reqs, pages=16, margin=1, prefill_budget=3) == _run(reqs, pages=16, margin=1)
    assert _run(reqs, pages=16, margin=1, prefill_budget=1) == _run(reqs, pages=16, margin=1)


def _chunked_interleaving_conserves_pages():
    for seed in range(4):
        _one_interleaving(seed, prefill_budget=3)


def _rejects_bad_sampling():
    SamplingParams(temperature=0.7, top_k=4, top_p=0.9, seed=1)
    bad = (
        {"temperature": -0.1},
        {"temperature": True},
        {"temperature": float("nan")},
        {"top_k": -1},
        {"top_k": True},
        {"top_k": 1.5},
        {"top_p": 0},
        {"top_p": 1.1},
        {"top_p": True},
        {"seed": -1},
        {"seed": True},
        {"seed": 2**64},
    )
    for kwargs in bad:
        try:
            SamplingParams(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"accepted sampling params {kwargs}")


def _a_cpu_graph_flag_stays_eager():
    # The flag builds a runner, but a CPU pool never hands it to the stub forward.
    reqs = [
        Request("a", (1, 2, 3), max_new_tokens=4),
        Request("b", (4, 5), max_new_tokens=3, stop_token_ids=(6,)),
    ]
    off = _run(reqs, pages=8, margin=1)
    pool = _pool(8, 4)
    engine = _engine(pool, max_batch=4, graphs=True)
    assert engine._graphs is not None
    for req in reqs:
        engine.add_request(req)
    assert _drain(engine) == off
    assert engine._graphs._graphs == {}
    _conserved(pool, engine)


def _greedy_still_matches_the_recurrence():
    sampling = SamplingParams(temperature=0, top_k=2, top_p=0.2, seed=3)
    pool = _pool(16, 4)
    engine = _engine(pool, max_batch=2)
    engine.add_request(Request("paris", (1, 2), max_new_tokens=4, sampling=sampling))
    first = engine.step()
    assert [out.token_id for out in first] == [3]
    assert engine._running[0].generator is None
    assert _drain(engine)["paris"] == [(4, None), (5, None), (6, "length")]
    _conserved(pool, engine)


def _a_seed_matches_its_own_stream_beside_another():
    left = SamplingParams(temperature=1.0, seed=1)
    right = SamplingParams(temperature=1.0, seed=2)
    pool = _pool(16, 4)
    engine = _open_engine(pool, max_batch=2)
    engine.add_request(Request("left", (1, 2), max_new_tokens=6, sampling=left))
    engine.add_request(Request("right", (3,), max_new_tokens=5, sampling=right))
    got = _drain(engine)
    assert _ids(got["left"]) == _reference_tokens(left, 6)
    assert _ids(got["right"]) == _reference_tokens(right, 5)
    assert engine._running == []
    assert got["left"][-1][1] == "length" and got["right"][-1][1] == "length"
    _conserved(pool, engine)

    # top-k of 1 is argmax, so the seed never gets a choice.
    forced = SamplingParams(temperature=1.0, top_k=1, seed=9)
    peaked = int(torch.argmax(_LOGITS).item())
    alone = _open_engine(_pool(8, 4), max_batch=1)
    alone.add_request(Request("peak", (4,), max_new_tokens=4, sampling=forced))
    assert _ids(_drain(alone)["peak"]) == [peaked, peaked, peaked, peaked]


def _preemption_keeps_the_sample_stream():
    left = SamplingParams(temperature=1.0, seed=11)
    right = SamplingParams(temperature=1.0, seed=12)
    reqs = [
        Request("old", (1,), max_new_tokens=8, sampling=left),
        Request("new", (2,), max_new_tokens=5, sampling=right),
    ]
    roomy = _drain_open(_pool(8, 4), reqs, growth_margin=0)
    tight = _open_engine(_pool(2, 4), max_batch=2, growth_margin=0)
    for req in reqs:
        tight.add_request(req)
    saw_parked_generator = False
    grouped = {}
    for _ in range(32):
        parked = [item for item in tight._waiting if item.request.request_id == "new"]
        if parked and parked[0].generator is not None:
            assert any(slot.request_id == "old" for slot in tight._running)
            saw_parked_generator = True
        outs = tight.step()
        for out in outs:
            grouped.setdefault(out.request_id, []).append((out.token_id, out.finish_reason))
        _conserved(tight.pool, tight)
        stats = tight.stats()
        if stats.requests_waiting == 0 and stats.requests_running == 0:
            break
    else:
        raise AssertionError("tight sampled engine did not finish")
    assert saw_parked_generator
    assert grouped == roomy
    assert _ids(grouped["old"]) == _reference_tokens(left, 8)
    assert _ids(grouped["new"]) == _reference_tokens(right, 5)
    assert tight.stats().blocks_free == tight.stats().blocks_total


def _chunked_prefill_keeps_the_sample_stream():
    sampling = SamplingParams(temperature=1.0, top_p=0.9, seed=4)
    req = Request("long", tuple(range(1, 11)), max_new_tokens=4, sampling=sampling)
    whole = _drain_open(_pool(16, 4), [req])
    chunked = _drain_open(_pool(16, 4), [req], prefill_budget=3)
    assert whole == chunked
    assert _ids(whole["long"]) == _reference_tokens(sampling, 4)


def _steps(engine, reqs=None):
    if reqs is not None:
        for req in reqs:
            engine.add_request(req)
    got = []
    for _ in range(32):
        outs = engine.step()
        got.append([(out.request_id, out.token_id, out.finish_reason) for out in outs])
        stats = engine.stats()
        _conserved(engine.pool, engine)
        if stats.requests_waiting == 0 and stats.requests_running == 0:
            return got
    raise AssertionError("engine did not finish")


def _run(reqs, pages, margin, prefill_budget=None):
    engine = _engine(_pool(pages, 4), max_batch=4, growth_margin=margin, prefill_budget=prefill_budget)
    for req in reqs:
        engine.add_request(req)
    return _drain(engine)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1.0
        return self.t


def _engine(pool, max_batch, clock=None, max_waiting=256, growth_margin=1, prefill_budget=None, graphs=False):
    return EngineCore(
        _forward_one, _forward_batch, pool, max_batch,
        clock=clock, max_waiting=max_waiting, growth_margin=growth_margin,
        prefill_budget=prefill_budget, graphs=graphs,
    )


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


_LOGITS = torch.tensor(
    [0.4, 1.7, 0.1, 1.2, 0.8, 0.0, 1.5, 0.3] + [0.05] * (VOCAB - 8),
    dtype=torch.float32,
)


def _open_one(ids, seq):
    n_new = ids.shape[1] - seq.seq_len
    if n_new <= 0:
        raise ValueError("paged forward was given no new tokens")
    reserve_pages(seq, seq.seq_len + n_new)
    seq.seq_len += n_new
    return _LOGITS.view(1, -1), seq


def _open_batch(ids, seqs):
    rows = []
    for seq in seqs:
        reserve_pages(seq, seq.seq_len + ids.shape[1])
        seq.seq_len += ids.shape[1]
        rows.append(_LOGITS)
    return torch.stack(rows), seqs


def _open_engine(pool, max_batch, **kwargs):
    return EngineCore(_open_one, _open_batch, pool, max_batch, **kwargs)


def _drain_open(pool, reqs, **kwargs):
    engine = _open_engine(pool, max_batch=2, **kwargs)
    for req in reqs:
        engine.add_request(req)
    got = _drain(engine)
    _conserved(pool, engine)
    return got


def _reference_tokens(params, n):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(params.seed)
    return [sample_token(_LOGITS, params, generator) for _ in range(n)]


def _ids(pairs):
    return [token for token, _reason in pairs]


def _expect(prompt, budget, stop=()):
    tokens = []
    last = prompt[-1]
    for _ in range(budget):
        last = (last + 1) % VOCAB
        tokens.append(last)
        if last in stop:
            break
    return tokens


def _group(outs):
    grouped = {}
    for out in outs:
        grouped.setdefault(out.request_id, []).append((out.token_id, out.finish_reason))
    return grouped


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
