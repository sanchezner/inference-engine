import torch

from engine.sched.engine_core import EngineCore
from engine.sched.request import Request


# One-shot driver over EngineCore so the paged check can still hand every prompt in up front.
def generate_continuous_paged(forward_one, forward_batch, pool, prompts, budgets, eos_id, max_batch):
    prompts = list(prompts)
    budgets = list(budgets)
    core = EngineCore(forward_one, forward_batch, pool, max_batch)
    for index, (prompt, budget) in enumerate(zip(prompts, budgets)):
        core.add_request(Request(
            request_id=str(index),
            prompt_token_ids=prompt.tolist(),
            max_new_tokens=budget,
            stop_token_ids=(eos_id,),
        ))

    generated = {str(index): [] for index in range(len(prompts))}
    steps = 0
    limit = sum(budgets) + len(prompts) + 1
    while True:
        # Step before the idle check, so a finish queued onto an idle engine is still delivered.
        if steps > limit:
            raise RuntimeError("paged generate exceeded its step budget")
        steps += 1
        for out in core.step():
            if out.token_id is not None:
                generated[out.request_id].append(out.token_id)
        stats = core.stats()
        if stats.requests_waiting == 0 and stats.requests_running == 0:
            break

    return [
        torch.tensor(
            list(prompts[index].tolist()) + generated[str(index)],
            dtype=prompts[index].dtype,
            device=prompts[index].device,
        )
        for index in range(len(prompts))
    ]
