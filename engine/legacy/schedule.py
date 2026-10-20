import torch
from engine.legacy.cache import cache_cat, cache_pad_left, cache_row, left_pad


def admit_into_batch(forward_fn, slots, incoming, outputs, eos_id):
    prompts = [req["prompt"] for req in incoming]
    padded, group_pad = left_pad(prompts, eos_id)
    logits, cache = forward_fn(padded, [], group_pad)
    next_id = torch.argmax(logits, dim=-1)

    width = padded.shape[1]
    resident_t = slots[0]["cache"][0][0].shape[2]

    if width > resident_t:
        extra = width - resident_t
        align = 0

        for slot in slots:
            slot["cache"] = cache_pad_left(slot["cache"], extra)
            slot["pad_len"] += extra
            filler = torch.full((extra,), eos_id, dtype=slot["ids"].dtype)
            slot["ids"] = torch.cat([filler, slot["ids"]])
    else:
        align = resident_t - width
        cache = cache_pad_left(cache, align)

    accepted = []

    for i, req in enumerate(incoming):
        tok = next_id[i]
        prompt = req["prompt"]

        if tok.item() == eos_id or req["budget"] == 1:
            outputs[req["index"]] = torch.cat([prompt, tok.view(1)])
            continue

        n_left = (width - prompt.shape[0]) + align
        filler = torch.full((n_left,), eos_id, dtype=prompt.dtype)
        accepted.append({
            "index": req["index"],
            "ids": torch.cat([filler, prompt, tok.view(1)]),
            "pad_len": n_left,
            "cache": cache_row(cache, i),
            "generated": 1,
            "budget": req["budget"],
        })

    return slots + accepted


def generate_continuous(forward_fn, prompts, budgets, eos_id, max_batch):
    waiting = [{"prompt": p, "budget": b, "index": i} for i, (p, b) in enumerate(zip(prompts, budgets))]
    slots = []
    outputs = [None] * len(prompts)

    while waiting or slots:
        free = max_batch - len(slots)
        if free > 0 and waiting:
            incoming = waiting[:free]
            del waiting[:free]

            if not slots:
                padded, pad_lens = left_pad([req["prompt"] for req in incoming], eos_id)
                for row, req, n_pad in zip(padded, incoming, pad_lens.tolist()):
                    slots.append({
                        "index": req["index"],
                        "ids": row.clone(),
                        "pad_len": n_pad,
                        "cache": None,
                        "generated": 0,
                        "budget": req["budget"],
                    })
            else:
                slots = admit_into_batch(forward_fn, slots, incoming, outputs, eos_id)

        if not slots:
            continue

        ids = torch.stack([s["ids"] for s in slots])
        pad_lens = torch.tensor([s["pad_len"] for s in slots])

        if slots[0]["cache"] is None:
            logits, cache = forward_fn(ids, [], pad_lens)
        else:
            logits, cache = forward_fn(ids, cache_cat([s["cache"] for s in slots]), pad_lens)

        next_id = torch.argmax(logits, dim=-1)
        alive = []

        for i, slot in enumerate(slots):
            slot["cache"] = cache_row(cache, i)
            tok = next_id[i]
            slot["ids"] = torch.cat([slot["ids"], tok.view(1)])
            slot["generated"] += 1

            if tok.item() == eos_id or slot["generated"] == slot["budget"]:
                outputs[slot["index"]] = slot["ids"][slot["pad_len"]:]
            else:
                alive.append(slot)

        slots = alive

    return outputs
