import torch

from engine.sched.request import SamplingParams
from engine.sched.sampling import _restrict, ensure_generator, sample_batch, sample_token


def main():
    _temperature_zero_is_argmax()
    _top_k_then_nucleus()
    _a_disabled_cap_matches_the_full_row()
    _the_same_seed_repeats()
    _a_generator_stays_on_its_device()
    _a_batch_matches_one_row_at_a_time()


def _temperature_zero_is_argmax():
    logits = torch.tensor([0.0, 3.0, 2.9])
    params = SamplingParams(temperature=0, top_k=1, top_p=0.1, seed=7)
    assert sample_token(logits, params, None) == 1
    assert sample_token(logits.view(1, -1), params, None) == 1
    try:
        sample_token(logits, SamplingParams(temperature=1, seed=1), None)
    except RuntimeError as exc:
        assert "generator" in str(exc)
    else:
        raise AssertionError("sampling without a generator was accepted")


def _top_k_then_nucleus():
    # softmax recovers these weights. top_p 0.5 alone keeps two; top_k 2 then top_p 0.5 keeps the mode.
    weights = torch.tensor([0.34, 0.33, 0.33])
    logits = torch.log(weights)
    nucleus = _restrict(logits, top_k=0, top_p=0.5)
    both = _restrict(logits, top_k=2, top_p=0.5)
    assert int(torch.isfinite(nucleus).sum()) == 2
    assert int(torch.isfinite(both).sum()) == 1
    assert int(torch.argmax(both)) == 0

    peaked = torch.tensor([0.0, 0.0, 0.0, 8.0])
    assert int(torch.isfinite(_restrict(peaked, top_k=0, top_p=0.5)).nonzero()[0]) == 3
    short = _restrict(torch.tensor([0.0, 10.0, 9.0, -5.0]), top_k=2, top_p=1.0)
    assert torch.isfinite(short).tolist() == [False, True, True, False]


def _a_disabled_cap_matches_the_full_row():
    logits = torch.arange(8, dtype=torch.float32)
    open_row = SamplingParams(temperature=1, top_k=0, top_p=1, seed=1)
    wide = SamplingParams(temperature=1, top_k=100, top_p=1, seed=1)
    assert _draw(logits, open_row, 6) == _draw(logits, wide, 6)


def _the_same_seed_repeats():
    logits = torch.tensor([0.2, 0.7, 0.1, 0.4])
    params = SamplingParams(temperature=1, seed=5)
    other = SamplingParams(temperature=1, seed=6)
    assert _draw(logits, params, 8) == _draw(logits, params, 8)
    assert _draw(logits, params, 8) != _draw(logits, other, 8)


def _a_generator_stays_on_its_device():
    params = SamplingParams(temperature=1, seed=0)
    cpu = ensure_generator(params, None, "cpu")
    assert ensure_generator(params, cpu, "cpu") is cpu
    if not torch.cuda.is_available():
        return
    logits = torch.tensor([0.0, 1.0, 3.0], device="cuda")
    cuda = ensure_generator(params, None, logits.device)
    assert sample_token(logits, SamplingParams(temperature=1, top_k=1, seed=0), cuda) == 2
    try:
        ensure_generator(params, cpu, logits.device)
    except RuntimeError as exc:
        assert "generator" in str(exc)
    else:
        raise AssertionError("a cpu generator was used on cuda logits")


def _a_batch_matches_one_row_at_a_time():
    logits = torch.tensor([
        [0.2, 0.7, 0.1, 0.4],
        [0.4, 0.1, 0.7, 0.2],
        [1.0, 0.0, 0.2, 0.1],
    ])
    params = [
        SamplingParams(temperature=1, seed=5),
        SamplingParams(temperature=1, top_k=2, top_p=0.8, seed=6),
        SamplingParams(temperature=0, top_k=1, top_p=0.1, seed=7),
    ]
    _batch_matches_rows(logits, params)
    greedy = [SamplingParams() for _ in range(len(params))]
    assert sample_batch(logits, greedy, [None] * len(params)) == [int(torch.argmax(row)) for row in logits]
    if torch.cuda.is_available():
        _batch_matches_rows(logits.to("cuda"), params)


def _batch_matches_rows(logits, params):
    first_gens = _generators(params, logits.device)
    second_gens = _generators(params, logits.device)
    alone = [sample_token(logits[i], params[i], first_gens[i]) for i in range(len(params))]
    assert sample_batch(logits, params, second_gens) == alone
    # The batch draw has to consume each stream once, same as the row draw.
    again = [sample_token(logits[i], params[i], first_gens[i]) for i in range(len(params))]
    assert sample_batch(logits, params, second_gens) == again


def _generators(params, device):
    gens = []
    for item in params:
        if item.temperature == 0.0:
            gens.append(None)
            continue
        generator = torch.Generator(device=device)
        generator.manual_seed(item.seed)
        gens.append(generator)
    return gens


def _draw(logits, params, n):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(params.seed)
    return [sample_token(logits, params, generator) for _ in range(n)]


if __name__ == "__main__":
    main()
