import time
from collections import deque
from dataclasses import dataclass

import torch

from engine.paged import Sequence, new_sequence, pages_for, release_pages
from engine.sched.request import EngineCounters, EngineStats, OutputTimestamps, QueueFull, Request, StepOutput


@dataclass
class _Waiting:
    request: Request
    arrival: float


@dataclass
class _Running:
    request_id: str
    seq: Sequence
    last_token: int
    generated: int
    max_new_tokens: int
    stop_ids: frozenset[int]
    arrival: float


# Ticket window in front of the paged loop. No threads: the caller decides when a step runs.
class EngineCore:
    def __init__(self, forward_one, forward_batch, pool, max_batch, clock=None, max_waiting=256):
        if isinstance(max_batch, bool) or not isinstance(max_batch, int) or max_batch < 1:
            raise ValueError(f"max_batch must be an int >= 1, got {max_batch!r}")
        if isinstance(max_waiting, bool) or not isinstance(max_waiting, int) or max_waiting < 1:
            raise ValueError(f"max_waiting must be an int >= 1, got {max_waiting!r}")
        self.forward_one = forward_one
        self.forward_batch = forward_batch
        self.pool = pool
        self.max_batch = max_batch
        self.max_waiting = max_waiting
        self.clock = time.perf_counter if clock is None else clock
        self._waiting = deque()
        self._running = []
        self._pending = []
        self._held = set()
        self._finished = 0
        self._aborted = 0
        self._rejected = 0
        self._tokens = 0

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
            # One iteration of the old loop: prefill everyone who fits, then decode the batch once.
            # A request admitted here therefore also receives its second token in this call.
            self._admit(outputs)
            if self._running:
                outputs.extend(self._decode())
            return outputs

    def stats(self) -> EngineStats:
        # Waiting prompts only. An admitted request has already written its whole prompt.
        pending = sum(len(item.request.prompt_token_ids) for item in self._waiting)
        return EngineStats(
            requests_waiting=len(self._waiting),
            requests_running=len(self._running),
            blocks_total=self.pool.k.shape[1],
            blocks_free=len(self.pool.free),
            block_size=self.pool.block_size,
            pending_prefill_tokens=pending,
            counters=EngineCounters(
                requests_finished=self._finished,
                requests_aborted=self._aborted,
                requests_rejected=self._rejected,
                tokens_generated=self._tokens,
            ),
        )

    def _admit(self, outputs: list[StepOutput]) -> None:
        while len(self._running) < self.max_batch and self._waiting:
            item = self._waiting[0]
            req = item.request
            if self._demand(len(req.prompt_token_ids), req.max_new_tokens) > self._slack():
                # Oldest first. A ticket that does not fit blocks the ones behind it.
                break
            self._waiting.popleft()
            outputs.extend(self._prefill(item))

    def _demand(self, prompt_len: int, max_new_tokens: int) -> int:
        # The token just sampled is written on the next decode, so the cache stops one short of the budget.
        return pages_for(self.pool.block_size, prompt_len + max_new_tokens - 1)

    def _slack(self) -> int:
        # Free pages minus the growth running sequences are already promised.
        owed = sum(self._owed(slot) for slot in self._running)
        free = len(self.pool.free)
        if owed > free:
            raise RuntimeError(f"running sequences are owed {owed} pages but only {free} are free")
        return free - owed

    def _owed(self, slot: _Running) -> int:
        final = slot.seq.seq_len + (slot.max_new_tokens - slot.generated)
        owed = pages_for(self.pool.block_size, final) - len(slot.seq.blocks)
        if owed < 0:
            raise RuntimeError(f"request {slot.request_id} holds more pages than its budget can use")
        return owed

    def _prefill(self, item: _Waiting) -> list[StepOutput]:
        req = item.request
        seq = new_sequence(self.pool)
        # forward_one reads the whole prompt. The decode forward reads one new token.
        ids = torch.tensor(req.prompt_token_ids, dtype=torch.long).view(1, -1)
        logits, seq = self.forward_one(ids, seq)
        token = int(torch.argmax(logits, dim=-1).reshape(-1)[0].item())
        reason = _finish_reason(token, 1, req.max_new_tokens, req.stop_token_ids)
        output = self._token(req.request_id, token, reason, item.arrival)
        if reason is None:
            self._running.append(_Running(
                request_id=req.request_id,
                seq=seq,
                last_token=token,
                generated=1,
                max_new_tokens=req.max_new_tokens,
                stop_ids=frozenset(req.stop_token_ids),
                arrival=item.arrival,
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

    def _token(self, request_id, token, reason, arrival) -> StepOutput:
        self._tokens += 1
        return StepOutput(
            request_id=request_id,
            token_id=token,
            finish_reason=reason,
            timestamps=OutputTimestamps(arrival=arrival, emitted=self.clock()),
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
