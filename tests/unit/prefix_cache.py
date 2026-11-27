import torch

from engine.kv.prefix_cache import PrefixCache, attach_prefix_cache, block_hashes
from engine.paged import PagePool, new_sequence, pin_page, release_pages, reserve_pages, take_page
from engine.sched.engine_core import EngineCore
from engine.sched.request import Request

VOCAB = 32


def main():
    _salt_and_chain_change_the_hash()
    _only_full_blocks_are_shared()
    _lowest_free_id_is_still_reused_next()
    _a_shared_page_waits_for_the_last_reader()
    _free_pages_are_taken_before_cached_ones()
    _eviction_spares_a_live_page()
    _a_hit_admits_on_fewer_fresh_pages()
    _a_partial_tail_is_not_shared()
    _hit_tokens_land_on_the_first_output()
    _an_empty_pool_refuses()
    _a_preempted_full_block_is_reused()
    _a_preempted_chunk_keeps_its_full_block()


def _salt_and_chain_change_the_hash():
    tokens = tuple(range(8))
    assert block_hashes("m", 4, tokens) != block_hashes("other", 4, tokens)
    assert block_hashes("m", 4, tokens) != block_hashes("m", 8, tokens)
    head, tail = block_hashes("m", 4, tokens)
    swapped = (9, 9, 9, 9) + tokens[4:]
    assert block_hashes("m", 4, swapped)[0] != head
    assert len(block_hashes("m", 4, swapped)) == 2
    assert block_hashes("m", 4, tokens[:5]) == [head]
    assert block_hashes("m", 4, tokens[:3]) == []

    cache = PrefixCache("m", 4)
    cache.insert(3, head)
    cache.insert(1, tail)
    assert cache.lookup(tokens) == ([3, 1], 8)
    assert cache.lookup(swapped) == ([], 0)
    cache.insert(9, head)
    assert cache.by_hash[head] == 3


def _only_full_blocks_are_shared():
    pool = _pool(4, 4)
    attach_prefix_cache(pool, "m")
    seq = new_sequence(pool)
    reserve_pages(seq, 5)
    seq.seq_len = 5
    pool.cache.insert(seq.blocks[0], block_hashes("m", 4, (1, 2, 3, 4, 5))[0])
    assert not pool.cache.holds(seq.blocks[1])
    release_pages(seq)
    assert seq.blocks == []
    assert pool.refcount[0] == 0
    assert 0 not in pool.free
    assert pool.free == [1, 2, 3]
    assert pool.cache.evict() == 0
    assert pool.free == [1, 2, 3]


def _lowest_free_id_is_still_reused_next():
    pool = _pool(8, 4)
    attach_prefix_cache(pool, "m")
    france = new_sequence(pool)
    germany = new_sequence(pool)
    reserve_pages(france, 14)
    reserve_pages(germany, 8)
    assert france.blocks == [0, 1, 2, 3]
    assert germany.blocks == [4, 5]
    release_pages(france)
    assert pool.free == [0, 1, 2, 3, 6, 7]
    cat = new_sequence(pool)
    reserve_pages(cat, 1)
    reserve_pages(germany, 9)
    assert cat.blocks == [0]
    assert germany.blocks == [4, 5, 1]
    _pages(pool, [france, germany, cat])


def _a_shared_page_waits_for_the_last_reader():
    pool = _pool(4, 4)
    attach_prefix_cache(pool, "m")
    first = new_sequence(pool)
    reserve_pages(first, 4)
    first.seq_len = 4
    digest = block_hashes("m", 4, (1, 2, 3, 4))[0]
    pool.cache.insert(first.blocks[0], digest)
    second = new_sequence(pool)
    page = first.blocks[0]
    pin_page(pool, page)
    second.blocks = [page]
    second.seq_len = 4
    assert pool.refcount[page] == 2
    release_pages(first)
    assert page not in pool.free
    assert pool.refcount[page] == 1
    release_pages(second)
    assert page not in pool.free
    assert page in pool.cache.lru
    assert [take_page(pool) for _ in range(3)] == [1, 2, 3]
    assert take_page(pool) == page
    assert pool.refcount[page] == 1
    assert not pool.cache.holds(page)


def _free_pages_are_taken_before_cached_ones():
    pool = _pool(2, 4)
    attach_prefix_cache(pool, "m")
    seq = new_sequence(pool)
    reserve_pages(seq, 4)
    seq.seq_len = 4
    pool.cache.insert(0, block_hashes("m", 4, (1, 2, 3, 4))[0])
    release_pages(seq)
    assert pool.free == [1]
    assert take_page(pool) == 1
    assert pool.cache.holds(0)
    assert take_page(pool) == 0
    assert not pool.cache.holds(0)


def _eviction_spares_a_live_page():
    record = {}
    pool = _pool(4, 4)
    attach_prefix_cache(pool, "m")
    engine = _engine(pool, _checking(record))
    engine.add_request(Request("live", (1, 2, 3, 4), max_new_tokens=8))
    engine.step()
    live = engine._running[0]
    held = list(live.seq.blocks)
    engine.add_request(Request("old", (10, 11, 12, 13), max_new_tokens=1))
    engine.add_request(Request("new", (20, 21, 22, 23), max_new_tokens=1))
    # old samples, so new waits for the next step. Each leaves a cached block behind.
    engine.step()
    engine.step()
    assert engine.stats().requests_running == 1
    assert engine.stats().requests_waiting == 0
    assert pool.free == []
    assert set(pool.cache.lru) == {2, 3}
    engine.add_request(Request("bump", (30, 31, 32, 33), max_new_tokens=1))
    engine.step()
    assert engine._running[0].seq.blocks[:len(held)] == held
    assert 2 not in engine._running[0].seq.blocks
    for _ in range(6):
        engine.step()
        _pages_engine(pool, engine)
    assert engine.stats().requests_running == 0
    assert engine.stats().blocks_free == engine.stats().blocks_total


def _a_hit_admits_on_fewer_fresh_pages():
    prefix = (1, 2, 3, 4, 5, 6, 7, 8)
    longer = prefix + (9, 10, 11, 12)
    cold = _engine(_pool(4, 4))
    cold.add_request(Request("a", prefix, max_new_tokens=4))
    cold.step()
    cold.add_request(Request("b", longer, max_new_tokens=1))
    held_back = cold.step()
    assert [out.request_id for out in held_back] == ["a"]
    assert cold.stats().requests_waiting == 1

    warm = _engine(_pool(4, 4), cache=True)
    warm.add_request(Request("a", prefix, max_new_tokens=4))
    warm.step()
    warm.add_request(Request("b", longer, max_new_tokens=1))
    both = warm.step()
    assert [out.request_id for out in both] == ["a", "b"]
    assert both[1].cached_tokens == 8
    assert both[1].finish_reason == "length"
    assert warm.stats().requests_waiting == 0
    _pages_engine(warm.pool, warm)


def _a_partial_tail_is_not_shared():
    prompt = (1, 2, 3, 4, 5)
    engine = _engine(_pool(8, 4), cache=True, max_batch=2)
    engine.add_request(Request("a", prompt, max_new_tokens=4))
    engine.add_request(Request("b", prompt, max_new_tokens=4))
    first = engine.step()
    assert [out.request_id for out in first] == ["a"]
    assert first[0].cached_tokens == 0
    second = engine.step()
    assert [out.request_id for out in second] == ["a", "b"]
    assert second[1].cached_tokens == 4
    running = {slot.request_id: slot for slot in engine._running}
    assert running["a"].seq.blocks[0] == running["b"].seq.blocks[0]
    assert running["a"].seq.blocks[1] != running["b"].seq.blocks[1]
    assert engine.pool.refcount[running["a"].seq.blocks[0]] == 2
    engine.abort("b")
    assert engine.pool.refcount[running["a"].seq.blocks[0]] == 1
    assert running["a"].seq.blocks[0] not in engine.pool.free
    _pages_engine(engine.pool, engine)


def _hit_tokens_land_on_the_first_output():
    prompt = (1, 2, 3, 4, 5, 6, 7, 8)
    engine = _engine(_pool(8, 4), cache=True)
    engine.add_request(Request("a", prompt, max_new_tokens=3))
    engine.add_request(Request("b", prompt, max_new_tokens=1))
    outs = []
    for _ in range(8):
        outs.extend(engine.step())
        if engine.stats().requests_running == 0 and engine.stats().requests_waiting == 0:
            break
    by_id = {}
    for out in outs:
        by_id.setdefault(out.request_id, []).append(out)
    assert by_id["a"][0].cached_tokens == 0
    assert all(out.cached_tokens == 0 for out in by_id["a"][1:])
    assert by_id["b"][0].cached_tokens == 4
    assert by_id["b"][0].finish_reason == "length"
    stats = engine.stats()
    assert stats.counters.prefix_cache_hit_tokens == 4
    assert stats.counters.prefix_cache_query_tokens == 16
    assert stats.blocks_free == stats.blocks_total
    assert len(engine.pool.free) < stats.blocks_total
    _pages_engine(engine.pool, engine)


def _an_empty_pool_refuses():
    pool = _pool(1, 4)
    assert take_page(pool) == 0
    try:
        take_page(pool)
    except RuntimeError as exc:
        assert "no free page" in str(exc)
    else:
        raise AssertionError("an empty pool popped its free list")


def _a_preempted_full_block_is_reused():
    # old and new each need two pages. The spare third page is what old takes, so new's block stays cached.
    record = {}
    pool = _pool(3, 4)
    forwards = _suffix_only(record)
    engine = _engine(pool, forwards, cache=True, max_batch=2, growth_margin=0)
    engine.add_request(Request("old", (1,), max_new_tokens=8))
    engine.add_request(Request("new", (2,), max_new_tokens=8))
    cached_page = None
    saw_cached = False
    saw_repin = False
    outs = []
    for _ in range(40):
        outs.extend(engine.step())
        _pages_engine(pool, engine)
        running = {slot.request_id: slot for slot in engine._running}
        waiting = {item.request.request_id for item in engine._waiting}
        if "new" in running and running["new"].seq.seq_len >= 4 and cached_page is None:
            cached_page = running["new"].seq.blocks[0]
        if cached_page is not None and "new" in waiting:
            assert pool.cache.holds(cached_page)
            assert pool.refcount[cached_page] == 0
            saw_cached = True
        if saw_cached and "new" in running and running["new"].seq.blocks[0] == cached_page:
            assert running["new"].seq.seq_len >= 4
            saw_repin = True
        stats = engine.stats()
        if stats.requests_waiting == 0 and stats.requests_running == 0:
            break
    else:
        raise AssertionError("preempted prefix did not finish")
    assert saw_cached and saw_repin
    assert all(out.cached_tokens == 0 for out in outs)
    roomy = _engine(_pool(8, 4), growth_margin=0, max_batch=2)
    roomy.add_request(Request("old", (1,), max_new_tokens=8))
    roomy.add_request(Request("new", (2,), max_new_tokens=8))
    expected = {}
    for _ in range(40):
        for out in roomy.step():
            if out.token_id is not None:
                expected.setdefault(out.request_id, []).append(out.token_id)
        if roomy.stats().requests_running == 0 and roomy.stats().requests_waiting == 0:
            break
    got = {}
    for out in outs:
        if out.token_id is not None:
            got.setdefault(out.request_id, []).append(out.token_id)
    assert got == expected
    assert engine.stats().blocks_free == engine.stats().blocks_total
    assert engine.stats().counters.prefix_cache_hit_tokens > 0


def _suffix_only(record):
    def forward_one(ids, seq):
        n_new = ids.shape[1] - seq.seq_len
        if n_new <= 0:
            raise ValueError("paged forward was given no new tokens")
        _expect(seq, record)
        row, seq = _write(seq, ids[0, -n_new:].tolist(), record)
        return row.view(1, -1), seq

    def forward_batch(ids, seqs):
        rows = []
        for row, seq in enumerate(seqs):
            _expect(seq, record)
            rows.append(_write(seq, [int(ids[row, -1])], record)[0])
        return torch.stack(rows), seqs

    return forward_one, forward_batch


def _checking(record):
    def forward_one(ids, seq):
        _expect(seq, record)
        return _write(seq, ids[0].tolist(), record)
    def forward_batch(ids, seqs):
        rows = []
        for row, seq in enumerate(seqs):
            _expect(seq, record)
            rows.append(_write(seq, [int(ids[row, 0])], record)[0])
        return torch.stack(rows), seqs
    return forward_one, forward_batch


def _expect(seq, record):
    bs = seq.pool.block_size
    for pos in range(seq.seq_len):
        page = seq.blocks[pos // bs]
        slot = pos % bs
        got = int(seq.pool.k[0, page, 0, slot, 0].item())
        assert got == record[(page, slot)], (page, slot, got, record[(page, slot)])


def _write(seq, tokens, record):
    from engine.paged import reserve_pages as reserve
    start = seq.seq_len
    reserve(seq, start + len(tokens))
    bs = seq.pool.block_size
    for offset, token in enumerate(tokens):
        pos = start + offset
        page = seq.blocks[pos // bs]
        slot = pos % bs
        seq.pool.k[0, page, 0, slot, 0] = int(token)
        record[(page, slot)] = int(token)
    seq.seq_len = start + len(tokens)
    nxt = (int(tokens[-1]) + 1) % VOCAB
    row = torch.zeros(VOCAB)
    row[nxt] = 10
    return row, seq


def _a_preempted_chunk_keeps_its_full_block():
    # old needs its second page while new holds a full block and a partial tail.
    # The tail is what old takes, so the full block stays cached and new pins it again.
    record = {}
    pool = _pool(3, 4)
    engine = _engine(pool, _suffix_only(record), cache=True, max_batch=2, growth_margin=0, prefill_budget=2)
    engine.add_request(Request("old", (1,), max_new_tokens=8))
    engine.step()
    prompt = (2, 2, 2, 2, 3, 3, 3, 3, 4, 4)
    engine.add_request(Request("new", prompt, max_new_tokens=1))
    cached_page = None
    saw_cached = False
    saw_repin = False
    outs = []
    for _ in range(40):
        outs.extend(engine.step())
        _pages_engine(pool, engine)
        running = {slot.request_id: slot for slot in engine._running}
        waiting = {item.request.request_id for item in engine._waiting}
        if "new" in running and len(running["new"].written) >= 4 and cached_page is None:
            cached_page = running["new"].seq.blocks[0]
        if cached_page is not None and "new" in waiting:
            assert pool.cache.holds(cached_page)
            assert pool.refcount[cached_page] == 0
            saw_cached = True
        if saw_cached and "new" in running and running["new"].seq.blocks and running["new"].seq.blocks[0] == cached_page:
            assert running["new"].seq.seq_len >= 4
            saw_repin = True
        stats = engine.stats()
        if stats.requests_waiting == 0 and stats.requests_running == 0:
            break
    else:
        raise AssertionError("chunked prefill did not finish")
    assert saw_cached and saw_repin
    got = [out.token_id for out in outs if out.request_id == "new" and out.token_id is not None]
    assert got == [(prompt[-1] + 1) % VOCAB]
    assert engine.stats().blocks_free == engine.stats().blocks_total
    assert engine.stats().counters.prefix_cache_hit_tokens > 0
    _pages_engine(pool, engine)


def _engine(pool, forwards=None, cache=False, max_batch=2, growth_margin=1, prefill_budget=None):
    if cache:
        attach_prefix_cache(pool, "m")
    if forwards is None:
        one, batch = _forward_one, _forward_batch
    else:
        one, batch = forwards
    return EngineCore(one, batch, pool, max_batch, growth_margin=growth_margin, prefill_budget=prefill_budget)


def _forward_one(ids, seq):
    n_new = ids.shape[1] - seq.seq_len
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


def _pool(n_pages, block_size):
    shape = (1, n_pages, 1, block_size, 1)
    return PagePool(
        k=torch.zeros(shape),
        v=torch.zeros(shape),
        free=list(range(n_pages)),
        block_size=block_size,
    )


def _pages(pool, seqs):
    _check(pool, [seq.blocks for seq in seqs])


def _pages_engine(pool, engine):
    _check(pool, [slot.seq.blocks for slot in engine._running])


def _check(pool, held):
    n = pool.k.shape[1]
    counts = [0] * n
    for blocks in held:
        assert len(blocks) == len(set(blocks)), blocks
        for page in blocks:
            counts[page] += 1
    assert counts == pool.refcount
    free = set(pool.free)
    assert len(free) == len(pool.free)
    cache = pool.cache
    cached = set() if cache is None else set(cache.by_page)
    evictable = set() if cache is None else set(cache.lru)
    assert free.isdisjoint(cached)
    assert evictable <= cached
    for page, count in enumerate(counts):
        if count:
            assert page not in free
            assert page not in evictable
        elif page in cached:
            assert page in evictable
        else:
            assert page in free
    assert len(free) + (n - len(free)) == n


if __name__ == "__main__":
    main()
