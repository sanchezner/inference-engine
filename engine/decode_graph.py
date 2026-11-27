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


@dataclass
class _Scratch:
    # Allocated before a replay. A fresh cudaMalloc during replay would sync the device.
    host_tables: torch.Tensor
    dev_tables: torch.Tensor
    host_lens: torch.Tensor
    dev_lens: torch.Tensor
    host_slots: torch.Tensor
    dev_slots: torch.Tensor
    stream: torch.cuda.Stream
    # The copy that last read the host buffer has finished.
    copied: torch.cuda.Event
    # The decode step that last read the device buffer has queued that read.
    free: torch.cuda.Event
    n_pages: int
    device: torch.device


@dataclass
class _Staged:
    seq_ids: tuple
    lens: tuple
    blocks: tuple
    tables: torch.Tensor
    seq_lens: torch.Tensor
    slots: torch.Tensor
    event: torch.cuda.Event


class DecodeGraphs:
    def __init__(self):
        self._graphs = {}
        self._pool = None
        self._scratch = None
        self._staged = None
        # How many replays started from a table booked during the previous one.
        self.reused = 0

    def __del__(self):
        # The side-stream copy outlives the token read. Dropping the buffers has to wait for it.
        scratch = self.__dict__.get("_scratch")
        if scratch is None:
            return
        scratch.stream.synchronize()

    def try_decode(self, weights, cfg, rope, ids, seqs, on_replay=None):
        device = weights.embed.device
        if device.type != "cuda" or ids.ndim != 2 or ids.shape[1] != 1 or ids.shape[0] != len(seqs):
            self._staged = None
            return None
        if ids.shape[0] not in _BUCKETS:
            self._staged = None
            return None
        ids = ids.to(device=device, dtype=torch.long)
        lens = [seq.seq_len for seq in seqs]
        self._ensure_scratch(seqs[0].pool)
        staged = self._take_staged(seqs, lens)
        if staged is None:
            pool, tables, seq_lens, slots = _prepare_decode(seqs, lens, device, check=False)
            if slots is None:
                return None
        else:
            pool = seqs[0].pool
            tables, seq_lens, slots = staged
        width = _table_bucket(tables.shape[1], pool.k.shape[1])
        key = (pool.k.data_ptr(), weights.embed.data_ptr(), ids.shape[0], width)
        replay = self._graphs.get(key)
        if replay is None:
            replay = self._capture(key, weights, cfg, rope, seqs, lens, ids, tables, seq_lens, slots, width)
            _advance(seqs, lens)
        else:
            self._fill(replay, ids, lens, tables, seq_lens, slots)
            # The upload of the next table waits for this read, not for the graph that follows.
            self._scratch.free.record()
            replay.graph.replay()
            _advance(seqs, lens)
            if on_replay is not None:
                on_replay()
        # The replay writes the next step into the same logits. Hand back a copy.
        return replay.logits.clone(), seqs

    def stage(self, seqs) -> None:
        # Host fill overlaps the graph. The device copy uses the side stream, into buffers already allocated.
        self._staged = None
        scratch = self._scratch
        if scratch is None or not seqs:
            return
        if not scratch.copied.query():
            scratch.copied.synchronize()
        width = self._fill_host(scratch, seqs)
        if width is None:
            return
        batch = len(seqs)
        side = scratch.stream
        side.wait_event(scratch.free)
        with torch.cuda.stream(side):
            scratch.dev_tables[:batch, :width].copy_(scratch.host_tables[:batch, :width], non_blocking=True)
            scratch.dev_lens[:batch].copy_(scratch.host_lens[:batch], non_blocking=True)
            scratch.dev_slots[:batch].copy_(scratch.host_slots[:batch], non_blocking=True)
        event = torch.cuda.Event()
        event.record(side)
        scratch.copied.record(side)
        self._staged = _Staged(
            seq_ids=tuple(id(seq) for seq in seqs),
            lens=tuple(seq.seq_len for seq in seqs),
            blocks=tuple(tuple(seq.blocks) for seq in seqs),
            tables=scratch.dev_tables[:batch, :width],
            seq_lens=scratch.dev_lens[:batch],
            slots=scratch.dev_slots[:batch],
            event=event,
        )

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

    def _ensure_scratch(self, pool) -> None:
        device = pool.k.device
        n_pages = pool.k.shape[1]
        scratch = self._scratch
        if scratch is not None and scratch.device == device and scratch.n_pages >= n_pages:
            return
        self._staged = None
        cap = _BUCKETS[-1]
        stream = torch.cuda.Stream()
        copied = torch.cuda.Event()
        copied.record(stream)
        free = torch.cuda.Event()
        free.record()
        self._scratch = _Scratch(
            host_tables=torch.zeros(cap, n_pages, dtype=torch.int32).pin_memory(),
            dev_tables=torch.zeros(cap, n_pages, dtype=torch.int32, device=device),
            host_lens=torch.zeros(cap, dtype=torch.int32).pin_memory(),
            dev_lens=torch.zeros(cap, dtype=torch.int32, device=device),
            host_slots=torch.zeros(cap, 1, dtype=torch.int32).pin_memory(),
            dev_slots=torch.zeros(cap, 1, dtype=torch.int32, device=device),
            stream=stream,
            copied=copied,
            free=free,
            n_pages=n_pages,
            device=device,
        )

    def _fill_host(self, scratch: _Scratch, seqs):
        batch = len(seqs)
        width = max(len(seq.blocks) for seq in seqs)
        if width < 1 or width > scratch.n_pages or batch > scratch.host_tables.shape[0]:
            return None
        bs = seqs[0].pool.block_size
        tables = scratch.host_tables
        tables[:batch, :width].zero_()
        for row, seq in enumerate(seqs):
            n = len(seq.blocks)
            if n:
                tables[row, :n] = torch.tensor(seq.blocks, dtype=torch.int32)
            pos = seq.seq_len
            page_index = pos // bs
            if page_index >= n:
                return None
            scratch.host_lens[row] = pos + 1
            scratch.host_slots[row, 0] = seq.blocks[page_index] * bs + (pos % bs)
        return width

    def _take_staged(self, seqs, lens):
        staged = self._staged
        self._staged = None
        if staged is None:
            return None
        if staged.seq_ids != tuple(id(seq) for seq in seqs):
            return None
        if staged.lens != tuple(lens) or staged.blocks != tuple(tuple(seq.blocks) for seq in seqs):
            return None
        torch.cuda.current_stream().wait_event(staged.event)
        self.reused += 1
        return staged.tables, staged.seq_lens, staged.slots

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
