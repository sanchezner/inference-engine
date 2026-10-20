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