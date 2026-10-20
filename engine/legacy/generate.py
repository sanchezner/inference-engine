import torch


# static batch where every row steps until the longest one is done
# a finished row keeps being forwarded as padding
def generate_batch(forward_fn, ids, max_new_tokens, eos_id, cache, pad_lens=None):
    finished = torch.zeros(ids.shape[0], dtype=torch.bool)

    for _ in range(max_new_tokens):
        logits, cache = forward_fn(ids, cache, pad_lens)
        next_id = torch.argmax(logits, dim=-1)
        next_id = torch.where(finished, torch.full_like(next_id, eos_id), next_id)
        ids = torch.cat([ids, next_id[:, None]], dim=1)
        finished = finished | (next_id == eos_id)

        if bool(finished.all()):
            break

    return ids
