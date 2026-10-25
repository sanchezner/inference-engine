import torch
from engine.paged import new_sequence, release_pages


# prefills a request into fresh pages via single sequence forward, samples its first token, and returns live slot or retires it
def admit_paged(forward_one, pool, req, outputs, eos_id):
    seq = new_sequence(pool)
    logits, seq = forward_one(req["prompt"].view(1, -1), seq)
    tok = torch.argmax(logits, dim=-1)[0].to(device=req["prompt"].device)
    ids = torch.cat([req["prompt"], tok.view(1)])

    if tok.item() == eos_id or req["budget"] == 1:
        outputs[req["index"]] = ids
        release_pages(seq)
        return None

    return {
        "index": req["index"],
        "seq": seq,
        "ids": ids,
        "generated": 1,
        "budget": req["budget"],
    }


# admit, step, retire loop
def generate_continuous_paged(forward_one, forward_batch, pool, prompts, budgets, eos_id, max_batch):
    waiting = [{"prompt": p, "budget": b, "index": i} for i, (p, b) in enumerate(zip(prompts, budgets))]
    slots = []
    outputs = [None] * len(prompts)

    while waiting or slots:
        while len(slots) < max_batch and waiting:
            slot = admit_paged(forward_one, pool, waiting.pop(0), outputs, eos_id)
            if slot is not None:
                slots.append(slot)

        if not slots:
            continue

        last = torch.stack([s["ids"][-1] for s in slots]).view(-1, 1)
        logits, _ = forward_batch(last, [s["seq"] for s in slots])
        next_id = torch.argmax(logits, dim=-1)
        alive = []

        for i, slot in enumerate(slots):
            tok = next_id[i].to(device=slot["ids"].device)
            slot["ids"] = torch.cat([slot["ids"], tok.view(1)])
            slot["generated"] += 1

            if tok.item() == eos_id or slot["generated"] == slot["budget"]:
                outputs[slot["index"]] = slot["ids"]
                release_pages(slot["seq"])
            else:
                alive.append(slot)

        slots = alive

    return outputs
