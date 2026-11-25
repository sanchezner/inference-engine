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
    # None while the prompt is still being scored. Decode only runs once a token has been sampled.
    last_token: int | None
    generated: int
    max_new_tokens: int
    stop_ids: frozenset[int]
    arrival: float
    written: list[int]
    # Prompt tokens reused from the prefix cache. Reported on the first generated token.
    cached_tokens: int = 0
    # Tokens already returned before this run. A preempted prefill requeues these whole.
    recompute: tuple[int, ...] = ()


# Ticket window in front of the paged loop. No threads: the caller decides when a step runs.
class EngineCore:
    def __init__(
        self, forward_one, forward_batch, pool, max_batch, clock=None, max_waiting=256, growth_margin=1,
        prefill_budget=None,
    ):
        if isinstance(max_batch, bool) or not isinstance(max_batch, int) or max_batch < 1:
            raise ValueError(f"max_batch must be an int >= 1, got {max_batch!r}")
        if isinstance(max_waiting, bool) or not isinstance(max_waiting, int) or max_waiting < 1:
            raise ValueError(f"max_waiting must be an int >= 1, got {max_waiting!r}")
        # Pages past the next token that admission leaves alone. 0 protects only that next write.
        if isinstance(growth_margin, bool) or not isinstance(growth_margin, int) or growth_margin < 0:
            raise ValueError(f"growth_margin must be an int >= 0, got {growth_margin!r}")
        # None scores a prompt in the step that seats it. A cap spreads the rest over later steps.
        if prefill_budget is not None and (
            isinstance(prefill_budget, bool) or not isinstance(prefill_budget, int) or prefill_budget < 1
        ):
            raise ValueError(f"prefill_budget must be an int >= 1 or None, got {prefill_budget!r}")
        self.forward_one = forward_one
        self.forward_batch = forward_batch
        self.pool = pool
        self.max_batch = max_batch
        self.max_waiting = max_waiting
        self.growth_margin = growth_margin
        self.prefill_budget = prefill_budget
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
            # Decode first, then score at most one budget of prompt tokens, so a long prefill cannot hold up tokens already in flight.
            self._make_room_for_decode()
            if self._decoding():
                outputs.extend(self._decode())
            budget = self._continue_prefills(outputs, self.prefill_budget)
            self._admit(outputs, budget)
            return outputs

    def stats(self) -> EngineStats:
        # Waiting prompts, tokens a preemption must recompute, and any seated prompt not scored yet.
        pending = sum(len(item.request.prompt_token_ids) + len(item.emitted) for item in self._waiting)
        pending += sum(self._unscored(slot) for slot in self._running if slot.last_token is None)
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

    def _admit(self, outputs: list[StepOutput], budget: int | None) -> None:
        while len(self._running) < self.max_batch and self._waiting and budget != 0:
            item = self._waiting[0]
            req = item.request
            tokens = req.prompt_token_ids + item.emitted
            blocks, cached_tokens = match_prefix(self.pool.cache, tokens)
            remaining = len(tokens) - cached_tokens
            if remaining <= 0:
                raise RuntimeError(f"request {req.request_id} has no token left to score")
            want = remaining if budget is None else min(budget, remaining)
            keep = set(blocks)
            if budget is None:
                # The whole prompt is written this step, so it has to fit or it waits.
                if self._seat_cost(len(tokens), req.max_new_tokens, item.generated, len(blocks), len(tokens)) > self._slack(keep):
                    break
            else:
                # A cap may take a shorter prefix of the same request. It still does not skip it.
                want = self._affordable(len(tokens), req.max_new_tokens, item.generated, len(blocks), cached_tokens, want, keep)
                if want < 1:
                    break
            self._waiting.popleft()
            self._seat(item, tokens, blocks, cached_tokens, cached_tokens + want, outputs)
            if budget is not None:
                budget -= want

    def _continue_prefills(self, outputs: list[StepOutput], budget: int | None) -> int | None:
        for slot in list(self._running):
            if slot.last_token is not None:
                continue
            if budget == 0:
                break
            took = self._extend(slot, outputs, budget)
            if took == 0:
                # The oldest unfinished prompt cannot grow, so this step seats nothing else.
                return 0
            if budget is not None:
                budget -= took
        return budget

    def _make_room_for_decode(self) -> None:
        # The newest running request is the victim. An older decode keeps its pages.
        while self._decoding() and self._decode_demand() > self._allocatable():
            self._preempt(len(self._running) - 1)
        for slot in self._decoding():
            # Taken now, so a prefill seated after this cannot use the page the next write needs.
            reserve_pages(slot.seq, slot.seq.seq_len + 1)

    def _preempt(self, index: int) -> None:
        slot = self._running.pop(index)
        prompt = slot.request.prompt_token_ids
        if slot.last_token is None:
            # A partial prompt has not sampled. Tokens already returned are recomputed whole on resume.
            if slot.generated != len(slot.recompute):
                raise RuntimeError(
                    f"request {slot.request_id} generated {slot.generated} during prefill "
                    f"but {len(slot.recompute)} tokens were already returned"
                )
            target = prompt + slot.recompute
            if tuple(slot.written) != target[:len(slot.written)]:
                raise RuntimeError(f"request {slot.request_id} stored a different prefix than it is scoring")
            emitted = slot.recompute
        else:
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

    def _decoding(self) -> list[_Running]:
        return [slot for slot in self._running if slot.last_token is not None]

    def _decode_demand(self) -> int:
        return sum(self._decode_need(slot) for slot in self._decoding())

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

    def _seat_cost(self, n_tokens: int, max_new_tokens: int, generated: int, n_cached: int, chunk_end: int) -> int:
        bs = self.pool.block_size
        # Pages for the span this step will actually write, not for the rest of a capped prompt.
        have = pages_for(bs, chunk_end)
        final_len = n_tokens + max_new_tokens - generated - 1
        if final_len < n_tokens or chunk_end > n_tokens:
            raise RuntimeError(
                f"token budget {max_new_tokens} cannot score {chunk_end} tokens of a {n_tokens}-token sequence "
                f"after {generated} generated tokens"
            )
        extra = pages_for(bs, final_len) - have
        fresh = have - n_cached
        if extra < 0 or fresh < 0:
            raise RuntimeError(
                f"prefix of {n_cached} pages does not fit a {n_tokens}-token sequence ending at {final_len}"
            )
        return fresh + min(extra, self.growth_margin)

    def _affordable(self, n_tokens, max_new_tokens, generated, n_cached, cached_tokens, want, keep) -> int:
        slack = self._slack(keep)
        best = 0
        lo = 1
        hi = want
        while lo <= hi:
            mid = (lo + hi) // 2
            cost = self._seat_cost(n_tokens, max_new_tokens, generated, n_cached, cached_tokens + mid)
            if cost <= slack:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

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
        # Tokens that will sit in the cache when the budget is spent. The finishing token is not stored.
        final = len(slot.request.prompt_token_ids) + slot.max_new_tokens - 1
        owed = pages_for(self.pool.block_size, final) - len(slot.seq.blocks)
        if owed < 0:
            raise RuntimeError(f"request {slot.request_id} holds more pages than its budget can use")
        return owed

    def _unscored(self, slot: _Running) -> int:
        return len(slot.request.prompt_token_ids) + len(slot.recompute) - len(slot.written)

    def _seat(self, item: _Waiting, tokens, blocks, cached_tokens, chunk_end, outputs) -> None:
        req = item.request
        seq = new_sequence(self.pool)
        if blocks:
            for page in blocks:
                pin_page(self.pool, page)
            seq.blocks = list(blocks)
            seq.seq_len = cached_tokens
        if self.pool.cache is not None:
            self._hit_tokens += cached_tokens
            self._query_tokens += len(tokens)
        slot = _Running(
            request=req,
            request_id=req.request_id,
            seq=seq,
            last_token=None,
            generated=item.generated,
            max_new_tokens=req.max_new_tokens,
            stop_ids=frozenset(req.stop_token_ids),
            arrival=item.arrival,
            written=list(tokens[:cached_tokens]),
            cached_tokens=cached_tokens,
            recompute=item.emitted,
        )
        self._running.append(slot)
        self._extend_to(slot, tokens, chunk_end, outputs)

    def _extend(self, slot: _Running, outputs: list[StepOutput], budget: int | None) -> int:
        tokens = slot.request.prompt_token_ids + slot.recompute
        remaining = len(tokens) - len(slot.written)
        if remaining <= 0:
            raise RuntimeError(f"request {slot.request_id} is prefilling with nothing left to score")
        want = remaining if budget is None else min(budget, remaining)
        room = self._tokens_that_fit(slot)
        if want > room:
            want = room
        if want < 1:
            return 0
        self._extend_to(slot, tokens, len(slot.written) + want, outputs)
        return want

    def _tokens_that_fit(self, slot: _Running) -> int:
        # The open page is already held. Other seated requests keep the margin admission promised them.
        bs = self.pool.block_size
        room = len(slot.seq.blocks) * bs - slot.seq.seq_len
        if room < 0:
            raise RuntimeError(
                f"request {slot.request_id} holds {len(slot.seq.blocks)} pages for length {slot.seq.seq_len}"
            )
        reserved = sum(self._reserved(other) for other in self._running if other is not slot)
        available = self._allocatable()
        spare = 0 if reserved >= available else available - reserved
        return room + spare * bs

    def _extend_to(self, slot: _Running, tokens, chunk_end: int, outputs: list[StepOutput]) -> None:
        # A later chunk already has pages, so this forward gathers them. A varlen kernel can replace that copy.
        ids = torch.tensor(tokens[:chunk_end], dtype=torch.long).view(1, -1)
        logits, seq = self.forward_one(ids, slot.seq)
        written = list(tokens[:chunk_end])
        if seq.seq_len != len(written):
            raise RuntimeError(
                f"request {slot.request_id!r} prefill wrote {seq.seq_len} tokens of a {len(written)}-token sequence"
            )
        slot.seq = seq
        slot.written = written
        publish_written(self.pool.cache, seq.blocks, written)
        if chunk_end < len(tokens):
            return
        token = int(torch.argmax(logits, dim=-1).reshape(-1)[0].item())
        generated = slot.generated + 1
        reason = _finish_reason(token, generated, slot.max_new_tokens, slot.stop_ids)
        # cached_tokens is only reported on the request's first output.
        shown = 0 if slot.generated else slot.cached_tokens
        outputs.append(self._token(slot.request_id, token, reason, slot.arrival, shown))
        if reason is None:
            slot.last_token = token
            slot.generated = generated
            return
        self._running = [other for other in self._running if other is not slot]
        release_pages(seq)
        self._finished += 1
        self._held.discard(slot.request_id)

    def _decode(self) -> list[StepOutput]:
        decoding = self._decoding()
        last = torch.tensor([slot.last_token for slot in decoding], dtype=torch.long).view(-1, 1)
        logits, _seqs = self.forward_batch(last, [slot.seq for slot in decoding])
        tokens = [int(token) for token in torch.argmax(logits, dim=-1).tolist()]
        outputs = []
        finished = set()
        for slot, token in zip(decoding, tokens):
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
            else:
                release_pages(slot.seq)
                self._finished += 1
                self._held.discard(slot.request_id)
                finished.add(id(slot))
        if finished:
            self._running = [slot for slot in self._running if id(slot) not in finished]
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
