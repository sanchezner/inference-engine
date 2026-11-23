from dataclasses import dataclass, field


# The waiting rail is full. The caller retries; the id was not taken.
class QueueFull(Exception):
    def __init__(self, waiting: int, max_waiting: int):
        self.waiting = waiting
        self.max_waiting = max_waiting
        super().__init__(f"waiting queue is full ({waiting}/{max_waiting})")


# Only greedy is legal until M5. A non-zero temperature must fail here, not silently argmax.
@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0

    def __post_init__(self):
        if self.temperature != 0.0:
            raise ValueError(
                f"only greedy sampling is accepted until M5, got temperature {self.temperature}"
            )


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    stop_token_ids: tuple[int, ...] = ()
    sampling: SamplingParams = field(default_factory=SamplingParams)

    def __post_init__(self):
        if not isinstance(self.request_id, str) or self.request_id == "":
            raise ValueError("request_id must be a non-empty string")
        prompt = tuple(int(token) for token in self.prompt_token_ids)
        if len(prompt) == 0:
            raise ValueError(f"request {self.request_id!r} has an empty prompt")
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int):
            raise TypeError(
                f"request {self.request_id!r} max_new_tokens must be an int, got {self.max_new_tokens!r}"
            )
        if self.max_new_tokens < 1:
            raise ValueError(
                f"request {self.request_id!r} max_new_tokens must be >= 1, got {self.max_new_tokens}"
            )
        if not isinstance(self.sampling, SamplingParams):
            raise TypeError(
                f"request {self.request_id!r} sampling must be SamplingParams, got {type(self.sampling).__name__}"
            )
        object.__setattr__(self, "prompt_token_ids", prompt)
        object.__setattr__(self, "stop_token_ids", tuple(int(token) for token in self.stop_token_ids))


# arrival is when add_request accepted it. emitted is when this output was produced.
# TTFT is the first emitted minus arrival. Inter-token latency is the gap between emitted times.
@dataclass(frozen=True)
class OutputTimestamps:
    arrival: float
    emitted: float


# finish_reason is None while the request continues, else stop, length, abort, or rejected.
# cached_tokens is non-zero only on the first output of a hit: prompt tokens reused, not recomputed.
@dataclass(frozen=True)
class StepOutput:
    request_id: str
    token_id: int | None
    finish_reason: str | None
    timestamps: OutputTimestamps
    cached_tokens: int = 0


@dataclass(frozen=True)
class EngineCounters:
    requests_finished: int = 0
    requests_aborted: int = 0
    requests_rejected: int = 0
    tokens_generated: int = 0
    prefix_cache_hit_tokens: int = 0
    prefix_cache_query_tokens: int = 0


@dataclass(frozen=True)
class EngineStats:
    requests_waiting: int
    requests_running: int
    blocks_total: int
    blocks_free: int
    block_size: int
    pending_prefill_tokens: int
    counters: EngineCounters
