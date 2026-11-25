import secrets

import torch

from engine.sched.request import SamplingParams


def sample_token(logits, params: SamplingParams, generator) -> int:
    row = _row(logits)
    if params.temperature == 0.0:
        return int(torch.argmax(row).item())
    if generator is None:
        raise RuntimeError("a non-zero temperature needs a generator")
    scores = _restrict(row.float() / params.temperature, params.top_k, params.top_p)
    probs = torch.softmax(scores, dim=-1)
    drawn = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(drawn.item())


def ensure_generator(params: SamplingParams, generator, device):
    if params.temperature == 0.0:
        return generator
    device = torch.device(device)
    if generator is not None:
        if generator.device != device:
            raise RuntimeError(f"sampler generator is on {generator.device}, logits are on {device}")
        return generator
    # Created on the logit device: multinomial refuses a generator from another device.
    created = torch.Generator(device=device)
    seed = secrets.randbits(63) if params.seed is None else params.seed
    created.manual_seed(seed)
    return created


def _row(logits):
    if logits.ndim == 2 and logits.shape[0] == 1:
        logits = logits.view(-1)
    if logits.ndim != 1 or logits.shape[0] < 1:
        raise ValueError(f"sample_token expected one logit row, got shape {tuple(logits.shape)}")
    return logits


def _restrict(scores, top_k, top_p):
    # Nucleus is a fraction of the surviving shortlist, so top-k is applied first.
    if 0 < top_k < scores.numel():
        picked = torch.topk(scores, top_k).indices
        kept = torch.full_like(scores, float("-inf"))
        kept.scatter_(0, picked, scores.gather(0, picked))
        scores = kept
    if top_p < 1.0:
        scores = _nucleus(scores, top_p)
    return scores


def _nucleus(scores, top_p):
    ordered, indices = torch.sort(scores, descending=True)
    cumulative = torch.cumsum(torch.softmax(ordered, dim=-1), dim=-1)
    # Keep the token that crosses the cap. The mask is shifted one to the right.
    remove = cumulative > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    ordered = ordered.masked_fill(remove, float("-inf"))
    restored = torch.full_like(scores, float("-inf"))
    restored.scatter_(0, indices, ordered)
    return restored
