from dataclasses import dataclass

import torch

from engine.paged import _advance, _decode_math, _logits, _prepare_decode, _stage


# A batch outside these sizes stays eager. A dummy row would change the matmul shape.
_BUCKETS = (1, 2, 4, 8, 16, 32)


def _table_bucket(tight: int, n_pages: int) -> int:
    # One new column every page would capture again. Powers of two, and never wider than the pool.
    if tight > n_pages:
        return tight
    width = 1
    while width < tight:
        nxt = width * 2
        if nxt > n_pages:
            return n_pages
        width = nxt
    return width


@dataclass
class _Replay:
    graph: torch.cuda.CUDAGraph
    ids: torch.Tensor
    positions: torch.Tensor
    tables: torch.Tensor
    seq_lens: torch.Tensor
    slots: torch.Tensor
    logits: torch.Tensor


class DecodeGraphs:
    def __init__(self):
        self._graphs = {}
        self._pool = None

    def try_decode(self, weights, cfg, rope, ids, seqs):
        device = weights.embed.device
        if device.type != "cuda" or ids.ndim != 2 or ids.shape[1] != 1 or ids.shape[0] != len(seqs):
            return None
        if ids.shape[0] not in _BUCKETS:
            return None
        ids = ids.to(device=device, dtype=torch.long)
        lens = [seq.seq_len for seq in seqs]
        pool, tables, seq_lens, slots = _prepare_decode(seqs, lens, device, check=False)
        if slots is None:
            return None
        width = _table_bucket(tables.shape[1], pool.k.shape[1])
        key = (pool.k.data_ptr(), weights.embed.data_ptr(), ids.shape[0], width)
        replay = self._graphs.get(key)
        if replay is None:
            replay = self._capture(key, weights, cfg, rope, seqs, lens, ids, tables, seq_lens, slots, width)
        else:
            self._fill(replay, ids, lens, tables, seq_lens, slots)
            replay.graph.replay()
        _advance(seqs, lens)
        # The replay writes the next step into the same logits. Hand back a copy.
        return replay.logits.clone(), seqs

    def _capture(self, key, weights, cfg, rope, seqs, lens, ids, tables, seq_lens, slots, width) -> _Replay:
        device = ids.device
        batch = tables.shape[0]
        replay = _Replay(
            graph=torch.cuda.CUDAGraph(),
            ids=torch.empty(batch, 1, dtype=torch.long, device=device),
            positions=torch.empty(batch, 1, dtype=torch.long, device=device),
            tables=torch.empty(batch, width, dtype=torch.int32, device=device),
            seq_lens=torch.empty(batch, dtype=torch.int32, device=device),
            slots=torch.empty(batch, 1, dtype=torch.int32, device=device),
            logits=torch.empty(0, device=device),
        )
        self._fill(replay, ids, lens, tables, seq_lens, slots)
        # Warmup compiles the kernels. Replay fills the static logits; the capture pass leaves them zero.
        for _ in range(3):
            _body(weights, cfg, rope, seqs, lens, replay)
        torch.cuda.synchronize()
        with torch.cuda.graph(replay.graph, pool=self._handle()):
            replay.logits = _body(weights, cfg, rope, seqs, lens, replay)
        replay.graph.replay()
        self._graphs[key] = replay
        return replay

    def _handle(self):
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        return self._pool

    def _fill(self, replay: _Replay, ids, lens, tables, seq_lens, slots) -> None:
        replay.ids.copy_(ids)
        positions = _stage(torch.tensor(lens, dtype=torch.long), ids.device).unsqueeze(1)
        replay.positions.copy_(positions)
        tight = tables.shape[1]
        # A shorter row in this bucket must not keep a page id from a longer one.
        replay.tables[:, :tight].copy_(tables)
        if tight < replay.tables.shape[1]:
            replay.tables[:, tight:].zero_()
        replay.seq_lens.copy_(seq_lens)
        replay.slots.copy_(slots)


def greedy_tokens(logits) -> list[int]:
    # One transfer for the whole batch. A per-row .item() would pause once per sequence.
    return [int(token) for token in torch.argmax(logits, dim=-1).tolist()]


def _body(weights, cfg, rope, seqs, lens, replay: _Replay):
    x = weights.embed[replay.ids]
    cos, sin = rope.gather(replay.positions)
    hidden = _decode_math(
        weights, cfg, x, seqs, lens, replay.tables, replay.seq_lens, replay.slots, cos, sin,
    )
    return _logits(weights, cfg, hidden)
