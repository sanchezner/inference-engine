import json
import tempfile
from pathlib import Path

import torch

from bench.e2e.trace import (
    TraceRequest,
    percentile,
    poisson_trace,
    read_trace,
    replay,
    shared_prefix_trace,
    summarize,
    write_trace,
)
from engine.kv.prefix_cache import attach_prefix_cache
from engine.paged import PagePool, reserve_pages
from engine.sched.engine_core import EngineCore
from engine.sched.request import OutputTimestamps, StepOutput

VOCAB = 32


def main():
    _percentile_is_nearest_rank()
    _summary_counts_tokens_and_gaps()
    _poisson_trace_is_seeded_and_fixed_length()
    _shared_prefix_trace_is_one_head_and_many_tails()
    _a_shared_burst_hits_only_when_the_cache_is_on()
    _trace_file_roundtrip_and_rejects()
    _instant_engine_finishes_one_before_the_next()
    _a_slow_step_admits_the_next_arrival()
    _abort_mid_trace_returns_every_page()


def _percentile_is_nearest_rank():
    assert percentile([], 50) is None
    assert percentile([4], 50) == 4
    assert percentile([4], 99) == 4
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([1, 2, 3, 4], 99) == 4


def _summary_counts_tokens_and_gaps():
    by_request = {
        "a": [
            _out("a", 1, None, 0.0, 1.0),
            _out("a", 2, None, 0.0, 1.5),
            _out("a", 3, "length", 0.0, 2.5),
        ],
        "b": [_out("b", None, "abort", 0.2, 0.4)],
    }
    summary = summarize(by_request, elapsed=2.5)
    assert summary["requests"] == 2
    assert summary["finished"] == 1 and summary["aborted"] == 1
    assert summary["tokens"] == 3
    assert summary["throughput_tok_s"] == 3 / 2.5
    assert summary["ttft_s"] == {"p50": 1.0, "p99": 1.0}
    assert summary["itl_s"] == {"p50": 0.5, "p99": 1.0}
    assert summarize({}, 0.0)["throughput_tok_s"] is None


def _poisson_trace_is_seeded_and_fixed_length():
    first = poisson_trace(6, rate=2.0, prompt_len=5, max_tokens=7, seed=3)
    assert poisson_trace(6, rate=2.0, prompt_len=5, max_tokens=7, seed=3) == first
    assert poisson_trace(6, rate=2.0, prompt_len=5, max_tokens=7, seed=4) != first
    assert first[0].t == 0.0
    assert all(len(item.prompt) == 5 and item.max_tokens == 7 for item in first)
    assert all(b.t > a.t for a, b in zip(first, first[1:]))
    try:
        poisson_trace(1, rate=0, prompt_len=1, max_tokens=1, seed=0)
    except ValueError as exc:
        assert "rate" in str(exc)
    else:
        raise AssertionError("a zero arrival rate was accepted")


def _shared_prefix_trace_is_one_head_and_many_tails():
    rows = shared_prefix_trace(4, prefix_len=8, suffix_len=4, max_tokens=2, seed=1, vocab=50)
    assert shared_prefix_trace(4, prefix_len=8, suffix_len=4, max_tokens=2, seed=1, vocab=50) == rows
    assert shared_prefix_trace(4, prefix_len=8, suffix_len=4, max_tokens=2, seed=2, vocab=50) != rows
    assert all(item.t == 0.0 and item.max_tokens == 2 for item in rows)
    head = rows[0].prompt[:8]
    tails = [item.prompt[8:] for item in rows]
    assert all(item.prompt[:8] == head and len(item.prompt) == 12 for item in rows)
    assert len(set(tails)) == 4
    assert [tail[0] for tail in tails] == [0, 1, 2, 3]
    spaced = shared_prefix_trace(3, prefix_len=4, suffix_len=2, max_tokens=1, seed=1, rate=5)
    assert spaced[0].t == 0.0 and spaced[2].t > spaced[1].t > 0.0
    assert all(item.prompt[:4] == spaced[0].prompt[:4] for item in spaced)
    try:
        shared_prefix_trace(1, prefix_len=0, suffix_len=1, max_tokens=1, seed=0)
    except ValueError as exc:
        assert "prefix_len" in str(exc)
    else:
        raise AssertionError("a zero-length shared prefix was accepted")


def _a_shared_burst_hits_only_when_the_cache_is_on():
    trace = shared_prefix_trace(2, prefix_len=8, suffix_len=4, max_tokens=2, seed=0, vocab=VOCAB)
    cold_summary, cold, cold_stats = _replay_shared(trace, cache=False)
    warm_summary, warm, warm_stats = _replay_shared(trace, cache=True)
    assert cold_stats.counters.prefix_cache_hit_tokens == 0
    assert cold_stats.counters.prefix_cache_query_tokens == 0
    assert all(out.cached_tokens == 0 for outs in cold.values() for out in outs)
    assert warm["0"][0].cached_tokens == 0
    assert warm["1"][0].cached_tokens == 8
    assert all(out.cached_tokens == 0 for out in warm["1"][1:])
    assert warm_stats.counters.prefix_cache_hit_tokens == 8
    assert warm_stats.counters.prefix_cache_query_tokens == 24
    assert _tokens(cold) == _tokens(warm)
    assert cold_summary["finished"] == 2 and warm_summary["finished"] == 2
    assert cold_stats.blocks_free == cold_stats.blocks_total
    assert warm_stats.blocks_free == warm_stats.blocks_total


def _replay_shared(trace, cache):
    pool = _pool(8, 4)
    if cache:
        attach_prefix_cache(pool, "m")
    engine = EngineCore(_forward_one, _forward_batch, pool, max_batch=2)
    summary, by_request = replay(engine, trace)
    return summary, by_request, engine.stats()


def _tokens(by_request):
    return {
        rid: [out.token_id for out in outs if out.token_id is not None]
        for rid, outs in by_request.items()
    }


def _trace_file_roundtrip_and_rejects():
    rows = [
        TraceRequest(0.0, (1, 2), 4, "a"),
        TraceRequest(0.25, (3,), 2, "b"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.jsonl"
        write_trace(path, rows)
        assert read_trace(path) == rows
        text = path.read_text().splitlines()
        assert json.loads(text[0])["id"] == "a"
        path.write_text(text[1] + "\n" + text[0] + "\n")
        try:
            read_trace(path)
        except ValueError as exc:
            assert "backwards" in str(exc)
        else:
            raise AssertionError("a backwards trace was accepted")


def _instant_engine_finishes_one_before_the_next():
    clock = _Clock()
    engine = _stub(clock, max_batch=4)
    trace = [
        TraceRequest(0.0, (1,), 3, "a"),
        TraceRequest(0.5, (4,), 3, "b"),
    ]
    summary, by_request = replay(engine, trace, clock=clock, sleep=clock.sleep, step_cost=0.0)
    assert summary["peak_running"] == 1
    assert [out.token_id for out in by_request["a"]] == [2, 3, 4]
    assert by_request["a"][-1].finish_reason == "length"
    assert [out.token_id for out in by_request["b"]] == [5, 6, 7]
    assert summary["ttft_s"]["p50"] == 0.0
    assert engine.stats().blocks_free == engine.stats().blocks_total


def _a_slow_step_admits_the_next_arrival():
    clock = _Clock()
    engine = _stub(clock, max_batch=4)
    trace = [
        TraceRequest(0.0, (1,), 6, "a"),
        TraceRequest(0.5, (4,), 6, "b"),
    ]
    summary, by_request = replay(engine, trace, clock=clock, sleep=clock.sleep, step_cost=1.0)
    assert summary["peak_running"] == 2
    assert [out.token_id for out in by_request["a"] if out.token_id is not None] == [2, 3, 4, 5, 6, 7]
    assert [out.token_id for out in by_request["b"] if out.token_id is not None] == [5, 6, 7, 8, 9, 10]
    assert engine.stats().blocks_free == engine.stats().blocks_total


def _abort_mid_trace_returns_every_page():
    clock = _Clock()
    engine = _stub(clock, max_batch=2)
    trace = [TraceRequest(0.0, (1, 1), 5, str(i)) for i in range(4)]
    summary, by_request = replay(
        engine, trace, clock=clock, sleep=clock.sleep, abort_ids={"0", "1"},
    )
    assert summary["aborted"] == 2 and summary["finished"] == 2
    for rid in ("0", "1"):
        tokens = [out.token_id for out in by_request[rid] if out.token_id is not None]
        assert tokens and tokens == _expect((1, 1), len(tokens))
        assert by_request[rid][-1].finish_reason == "abort" and by_request[rid][-1].token_id is None
    for rid in ("2", "3"):
        tokens = [out.token_id for out in by_request[rid]]
        assert tokens == _expect((1, 1), 5)
        assert by_request[rid][-1].finish_reason == "length"
    assert engine.stats().blocks_free == engine.stats().blocks_total
    assert engine.stats().counters.requests_aborted == 2


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        if seconds < 0:
            raise ValueError(f"sleep of {seconds}")
        self.now += seconds


def _stub(clock, max_batch):
    return EngineCore(_forward_one, _forward_batch, _pool(16, 4), max_batch, clock=clock)


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


def _expect(prompt, budget):
    tokens = []
    last = prompt[-1]
    for _ in range(budget):
        last = (last + 1) % VOCAB
        tokens.append(last)
    return tokens


def _out(request_id, token_id, reason, arrival, emitted):
    return StepOutput(
        request_id=request_id,
        token_id=token_id,
        finish_reason=reason,
        timestamps=OutputTimestamps(arrival=arrival, emitted=emitted),
    )


if __name__ == "__main__":
    main()
