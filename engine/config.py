import json
from dataclasses import dataclass
from pathlib import Path


# llama config from raw config.json. transformers is not used here as recent versions hide rope_theta inside rope_parameters, and this parser reads the file directly 
@dataclass(frozen=True)
class ModelConfig:
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    max_position: int
    rope_theta: float
    rope_type: str
    tie_word_embeddings: bool
    eos_token_ids: tuple[int, ...]
    bos_token_id: int | None

    def __post_init__(self):
        # unscaled RoPE; a Llama-3 scaling config must fail here rather than run with the wrong frequencies.
        if self.rope_type != "default":
            raise ValueError(
                f"RoPE type {self.rope_type!r} is not implemented; only 'default' is supported"
            )
        if self.n_kv_heads <= 0 or self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        if self.hidden_size % self.n_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) is not a multiple of n_heads ({self.n_heads})"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even")

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelConfig":
        model_type = raw.get("model_type")
        if model_type != "llama":
            raise ValueError(f"model_type {model_type!r} is not a Llama config")

        act = raw.get("hidden_act", "silu")
        if act != "silu":
            raise ValueError(f"hidden_act {act!r} is not implemented; only 'silu' is supported")
        for key in ("attention_bias", "mlp_bias", "rope_interleaved"):
            if raw.get(key, False):
                raise ValueError(f"{key} is not implemented")

        n_heads = int(_require(raw, "num_attention_heads"))
        hidden_size = int(_require(raw, "hidden_size"))
        # explicit head_dim wins and configs omit it when it equals hidden_size // n_heads.
        head_dim = raw.get("head_dim")
        if head_dim is None:
            head_dim = hidden_size // n_heads
        rope_type, rope_theta = _rope(raw)

        return cls(
            n_layers=int(_require(raw, "num_hidden_layers")),
            n_heads=n_heads,
            n_kv_heads=int(raw.get("num_key_value_heads", n_heads)),
            head_dim=int(head_dim),
            hidden_size=hidden_size,
            intermediate_size=int(_require(raw, "intermediate_size")),
            rms_norm_eps=float(_require(raw, "rms_norm_eps")),
            vocab_size=int(_require(raw, "vocab_size")),
            max_position=int(_require(raw, "max_position_embeddings")),
            rope_theta=rope_theta,
            rope_type=rope_type,
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            eos_token_ids=_eos_ids(raw),
            bos_token_id=_bos_id(raw),
        )


def _require(raw, key):
    if key not in raw:
        raise ValueError(f"config.json is missing {key}")
    return raw[key]


def _eos_ids(raw) -> tuple[int, ...]:
    eos = _require(raw, "eos_token_id")
    if isinstance(eos, int):
        return (eos,)
    if isinstance(eos, list) and eos and all(isinstance(x, int) for x in eos):
        return tuple(eos)
    raise ValueError(f"eos_token_id must be an int or a non-empty list of ints, got {eos!r}")


def _bos_id(raw) -> int | None:
    bos = raw.get("bos_token_id")
    if bos is not None and not isinstance(bos, int):
        raise ValueError(f"bos_token_id must be an int or null, got {bos!r}")
    return bos


def _rope(raw) -> tuple[str, float]:
    scaling = raw.get("rope_scaling")
    params = raw.get("rope_parameters")
    if scaling is not None and not isinstance(scaling, dict):
        raise ValueError(f"rope_scaling must be an object or null, got {scaling!r}")
    if params is not None and not isinstance(params, dict):
        raise ValueError(f"rope_parameters must be an object or null, got {params!r}")

    # rope_scaling is the legacy key. rope_parameters is the same dict in newer files. a null rope_scaling must not hide rope_parameters.
    chosen = scaling if isinstance(scaling, dict) else params
    theta = raw.get("rope_theta", 10_000.0)
    if chosen is None:
        _full_rotary(raw)
        return "default", float(theta)
    if any(isinstance(value, dict) for value in chosen.values()):
        raise ValueError("per-layer RoPE parameters are not implemented")

    rope_type = chosen.get("rope_type", chosen.get("type", "default"))
    if "rope_theta" in chosen:
        theta = chosen["rope_theta"]
    _full_rotary(raw)
    _full_rotary(chosen)
    return rope_type, float(theta)


def _full_rotary(raw) -> None:
    factor = raw.get("partial_rotary_factor", 1.0)
    if factor is not None and float(factor) != 1.0:
        raise ValueError(f"partial_rotary_factor {factor} is not implemented")
