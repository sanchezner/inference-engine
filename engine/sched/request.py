import math
from dataclasses import dataclass, field


# The waiting rail is full. The caller retries; the id was not taken.
class QueueFull(Exception):
    def __init__(self, waiting: int, max_waiting: int):
        self.waiting = waiting
        self.max_waiting = max_waiting
        super().__init__(f"waiting queue is full ({waiting}/{max_waiting})")


# temperature 0 is argmax and ignores the filters. top_k 0 and top_p 1 leave the row unfiltered.
@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None

    def __post_init__(self):
        temperature = self.temperature
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature < 0
        ):
            raise ValueError(f"temperature must be a finite number >= 0, got {temperature!r}")
        top_k = self.top_k
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError(f"top_k must be an int >= 0, got {top_k!r}")
        top_p = self.top_p
        if (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not math.isfinite(top_p)
            or not 0 < top_p <= 1
        ):
            raise ValueError(f"top_p must be a finite number in (0, 1], got {top_p!r}")
        seed = self.seed
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64
        ):
            raise ValueError(f"seed must be an int in [0, 2**64) or None, got {seed!r}")
        object.__setattr__(self, "temperature", float(temperature))
        object.__setattr__(self, "top_p", float(top_p))


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
