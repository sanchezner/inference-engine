import torch


# generates tokens sequentially for a single prompt w/ greedy argmax decoding until max_new_tokens or eos
def generate(forward_fn, ids, max_new_tokens, eos_id, cache=None):
    for _ in range(max_new_tokens):
        if cache is None:
            logits = forward_fn(ids)
        else:
            logits, cache = forward_fn(ids, cache)
    
        next_id = torch.argmax(logits)
        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
        if next_id.item() == eos_id:
            break

    return ids


# runs multiple prompts in lockstep
# if one prompt needs 2 tokens and another needs 100, the first must wait while empty padding tokens are processed until the second one finishes
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