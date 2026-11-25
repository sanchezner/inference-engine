import time
from collections import deque
from dataclasses import dataclass

import torch

from engine.kv.prefix_cache import match_prefix, publish_written
from engine.paged import (
    Sequence,
    available_pages,
    new_sequence,
    pages_for,
    pin_page,
    release_pages,
    reserve_pages,
)
from engine.sched.request import EngineCounters, EngineStats, OutputTimestamps, QueueFull, Request, StepOutput


@dataclass
class _Waiting:
    request: Request
    arrival: float
    generated: int = 0
    # Already returned to the caller. The last of these is not in a page yet.
    emitted: tuple[int, ...] = ()


@dataclass
class _Running:
    request: Request
    request_id: str
    seq: Sequence
    last_token: int
    generated: int
    max_new_tokens: int
    stop_ids: frozenset[int]
    arrival: float
    written: list[int]


# Ticket window in front of the paged loop. No threads: the caller decides when a step runs.
class EngineCore:
    def __init__(
        self, forward_one, forward_batch, pool, max_batch, clock=None, max_waiting=256, growth_margin=1,
    ):
        if isinstance(max_batch, bool) or not isinstance(max_batch, int) or max_batch < 1:
            raise ValueError(f"max_batch must be an int >= 1, got {max_batch!r}")
        if isinstance(max_waiting, bool) or not isinstance(max_waiting, int) or max_waiting < 1:
            raise ValueError(f"max_waiting must be an int >= 1, got {max_waiting!r}")
        # Pages past the next token that admission leaves alone. 0 protects only that next write.
        if isinstance(growth_margin, bool) or not isinstance(growth_margin, int) or growth_margin < 0:
            raise ValueError(f"growth_margin must be an int >= 0, got {growth_margin!r}")
        self.forward_one = forward_one
        self.forward_batch = forward_batch
        self.pool = pool
        self.max_batch = max_batch
        self.max_waiting = max_waiting
        self.growth_margin = growth_margin
        self.clock = time.perf_counter if clock is None else clock
        self._waiting = deque()
        self._running = []
        self._pending = []
        self._held = set()
        self._finished = 0
        self._aborted = 0
        self._rejected = 0
        self._tokens = 0
        self._hit_tokens = 0
        self._query_tokens = 0

    def add_request(self, req: Request) -> None:
        if not isinstance(req, Request):
            raise TypeError(f"add_request expected Request, got {type(req).__name__}")
        if req.request_id in self._held:
            raise ValueError(f"request {req.request_id!r} is already in the engine")
        arrival = self.clock()
        # Bigger than the whole pool, not merely bigger than what is free right now.
        # Does not take a waiting seat: the server maps this to "never fits", not "line full".
        if self._demand(len(req.prompt_token_ids), req.max_new_tokens) > self.pool.k.shape[1]:
            self._held.add(req.request_id)
            self._pending.append(self._terminal(req.request_id, "rejected", arrival))
            self._rejected += 1
            return
        if len(self._waiting) >= self.max_waiting:
            # Not held, so the caller may retry this same id once a seat frees.
            raise QueueFull(len(self._waiting), self.max_waiting)
        self._held.add(req.request_id)
        self._waiting.append(_Waiting(req, arrival))

    # Pages are freed here. The abort output waits for the next step so tokens and finishes share one stream.
    def abort(self, request_id: str) -> None:
        for i, item in enumerate(self._waiting):
            if item.request.request_id == request_id:
                del self._waiting[i]
                self._pending.append(self._terminal(request_id, "abort", item.arrival))
                self._aborted += 1
                return
        for i, slot in enumerate(self._running):
            if slot.request_id == request_id:
                release_pages(slot.seq)
                del self._running[i]
                self._pending.append(self._terminal(request_id, "abort", slot.arrival))
                self._aborted += 1
                return

    def step(self) -> list[StepOutput]:
        # A caller that forgets no_grad would keep a graph alive on every forward.
        with torch.no_grad():
            outputs = self._pending
            self._pending = []
            for out in outputs:
                self._held.discard(out.request_id)
            # Decode whoever is already seated, then seat new requests for one token.
            # A request resumed below is not decoded again in this call.
            self._make_room_for_decode()
            if self._running:
                outputs.extend(self._decode())
            self._admit(outputs)
            return outputs

    def stats(self) -> EngineStats:
        # Waiting sequences, including tokens a preempted request has to recompute.
        pending = sum(len(item.request.prompt_token_ids) + len(item.emitted) for item in self._waiting)
        return EngineStats(
            requests_waiting=len(self._waiting),
            requests_running=len(self._running),
            blocks_total=self.pool.k.shape[1],
            blocks_free=available_pages(self.pool),
            block_size=self.pool.block_size,
            pending_prefill_tokens=pending,
            counters=EngineCounters(
                requests_finished=self._finished,
                requests_aborted=self._aborted,
                requests_rejected=self._rejected,
                tokens_generated=self._tokens,
                prefix_cache_hit_tokens=self._hit_tokens,
                prefix_cache_query_tokens=self._query_tokens,
            ),
        )

    def _admit(self, outputs: list[StepOutput]) -> None:
        while len(self._running) < self.max_batch and self._waiting:
            item = self._waiting[0]
            req = item.request
            tokens = req.prompt_token_ids + item.emitted
            blocks, cached_tokens = match_prefix(self.pool.cache, tokens)
            cost = self._seat_cost(len(tokens), req.max_new_tokens, item.generated, len(blocks))
            if cost > self._slack(set(blocks)):
                # Oldest first. A ticket that does not fit blocks the ones behind it.
                break
            self._waiting.popleft()
            outputs.extend(self._prefill(item, tokens, blocks, cached_tokens))

    def _make_room_for_decode(self) -> None:
        # The newest running request is the victim. An older decode keeps its pages.
        while self._running and self._decode_demand() > self._allocatable():
            self._preempt(len(self._running) - 1)
        for slot in self._running:
            # Taken now, so a prefill seated after this cannot use the page the next write needs.
            reserve_pages(slot.seq, slot.seq.seq_len + 1)

    def _preempt(self, index: int) -> None:
        slot = self._running.pop(index)
        prompt = slot.request.prompt_token_ids
        if tuple(slot.written[:len(prompt)]) != prompt:
            raise RuntimeError(f"request {slot.request_id} stored tokens do not start with its prompt")
        emitted = tuple(slot.written[len(prompt):]) + (slot.last_token,)
        if len(emitted) != slot.generated:
            raise RuntimeError(
                f"request {slot.request_id} has {len(emitted)} emitted tokens but generated count {slot.generated}"
            )
        # Full blocks stay cached. The partial tail is freed and recomputed on resume.
        release_pages(slot.seq)
        self._waiting.appendleft(_Waiting(slot.request, slot.arrival, slot.generated, emitted))

    def _decode_demand(self) -> int:
        return sum(self._decode_need(slot) for slot in self._running)

    def _decode_need(self, slot: _Running) -> int:
        need = pages_for(self.pool.block_size, slot.seq.seq_len + 1) - len(slot.seq.blocks)
        if need < 0:
            raise RuntimeError(f"request {slot.request_id} holds more pages than its next token can use")
        return need

    def _allocatable(self) -> int:
        cache = self.pool.cache
        evictable = 0 if cache is None else cache.evictable_count()
        return len(self.pool.free) + evictable

    def _demand(self, prompt_len: int, max_new_tokens: int) -> int:
        # The token just sampled is written on the next decode, so the cache stops one short of the budget.
        return pages_for(self.pool.block_size, prompt_len + max_new_tokens - 1)

    def _seat_cost(self, n_tokens: int, max_new_tokens: int, generated: int, n_cached: int) -> int:
        bs = self.pool.block_size
        have = pages_for(bs, n_tokens)
        final_len = n_tokens + max_new_tokens - generated - 1
        if final_len < n_tokens:
            raise RuntimeError(
                f"token budget {max_new_tokens} is already exhausted after {generated} generated tokens"
            )
        extra = pages_for(bs, final_len) - have
        fresh = have - n_cached
        if extra < 0 or fresh < 0:
            raise RuntimeError(
                f"prefix of {n_cached} pages does not fit a {n_tokens}-token sequence ending at {final_len}"
            )
        return fresh + min(extra, self.growth_margin)

    def _slack(self, keep: set[int] | None = None) -> int:
        # Spare pages are the free list plus cached blocks we may evict, excluding the hit about to be pinned.
        keep = set() if keep is None else keep
        cache = self.pool.cache
        evictable = 0 if cache is None else cache.evictable_excluding(keep)
        reserved = sum(self._reserved(slot) for slot in self._running)
        available = len(self.pool.free) + evictable
        if reserved >= available:
            return 0
        return available - reserved

    def _reserved(self, slot: _Running) -> int:
        return min(self._owed(slot), self.growth_margin)

    def _owed(self, slot: _Running) -> int:
        final = slot.seq.seq_len + (slot.max_new_tokens - slot.generated)
        owed = pages_for(self.pool.block_size, final) - len(slot.seq.blocks)
        if owed < 0:
            raise RuntimeError(f"request {slot.request_id} holds more pages than its budget can use")
        return owed

    def _prefill(self, item: _Waiting, tokens: tuple[int, ...], blocks: list[int], cached_tokens: int) -> list[StepOutput]:
        req = item.request
        seq = new_sequence(self.pool)
        if blocks:
            for page in blocks:
                pin_page(self.pool, page)
            seq.blocks = list(blocks)
            seq.seq_len = cached_tokens
        # forward_one reads the whole sequence and scores only the tokens after seq_len.
        ids = torch.tensor(tokens, dtype=torch.long).view(1, -1)
        logits, seq = self.forward_one(ids, seq)
        written = list(tokens)
        if seq.seq_len != len(written):
            raise RuntimeError(
                f"request {req.request_id!r} prefill wrote {seq.seq_len} tokens of a {len(written)}-token sequence"
            )
        publish_written(self.pool.cache, seq.blocks, written)
        if self.pool.cache is not None:
            self._hit_tokens += cached_tokens
            self._query_tokens += len(written)
        token = int(torch.argmax(logits, dim=-1).reshape(-1)[0].item())
        generated = item.generated + 1
        reason = _finish_reason(token, generated, req.max_new_tokens, req.stop_token_ids)
        # cached_tokens is only reported on the request's first output.
        shown = 0 if item.generated else cached_tokens
        output = self._token(req.request_id, token, reason, item.arrival, shown)
        if reason is None:
            self._running.append(_Running(
                request=req,
                request_id=req.request_id,
                seq=seq,
                last_token=token,
                generated=generated,
                max_new_tokens=req.max_new_tokens,
                stop_ids=frozenset(req.stop_token_ids),
                arrival=item.arrival,
                written=written,
            ))
        else:
            release_pages(seq)
            self._finished += 1
            self._held.discard(req.request_id)
        return [output]

    def _decode(self) -> list[StepOutput]:
        last = torch.tensor([slot.last_token for slot in self._running], dtype=torch.long).view(-1, 1)
        logits, _seqs = self.forward_batch(last, [slot.seq for slot in self._running])
        tokens = [int(token) for token in torch.argmax(logits, dim=-1).tolist()]
        outputs = []
        alive = []
        for slot, token in zip(self._running, tokens):
            # The forward just wrote last_token into the page. The token sampled here is not stored yet.
            slot.written.append(slot.last_token)
            if len(slot.written) != slot.seq.seq_len:
                raise RuntimeError(
                    f"request {slot.request_id} has {len(slot.written)} stored tokens but cache length {slot.seq.seq_len}"
                )
            publish_written(self.pool.cache, slot.seq.blocks, slot.written)
            generated = slot.generated + 1
            reason = _finish_reason(token, generated, slot.max_new_tokens, slot.stop_ids)
            outputs.append(self._token(slot.request_id, token, reason, slot.arrival))
            if reason is None:
                slot.last_token = token
                slot.generated = generated
                alive.append(slot)
            else:
                release_pages(slot.seq)
                self._finished += 1
                self._held.discard(slot.request_id)
        self._running = alive
        return outputs

    def _token(self, request_id, token, reason, arrival, cached_tokens=0) -> StepOutput:
        self._tokens += 1
        return StepOutput(
            request_id=request_id,
            token_id=token,
            finish_reason=reason,
            timestamps=OutputTimestamps(arrival=arrival, emitted=self.clock()),
            cached_tokens=cached_tokens,
        )

    def _terminal(self, request_id, reason, arrival) -> StepOutput:
        return StepOutput(
            request_id=request_id,
            token_id=None,
            finish_reason=reason,
            timestamps=OutputTimestamps(arrival=arrival, emitted=self.clock()),
        )

def _finish_reason(token, generated, budget, stop_ids):
    # A token that is both a stop id and the last allowed token reports stop.
    if token in stop_ids:
        return "stop"
    if generated >= budget:
        return "length"
    return None
