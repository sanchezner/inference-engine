import json
import tempfile
from pathlib import Path

import torch

from bench.e2e.trace import TraceRequest, percentile, poisson_trace, read_trace, replay, summarize, write_trace
from engine.paged import PagePool, reserve_pages
from engine.sched.engine_core import EngineCore
from engine.sched.request import OutputTimestamps, StepOutput

VOCAB = 32


def main():
    _percentile_is_nearest_rank()
    _summary_counts_tokens_and_gaps()
    _poisson_trace_is_seeded_and_fixed_length()
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
