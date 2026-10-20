import torch


# helper function that pads input sequences with pad_id on the left instead of the right
# autoregressive generation appens new tokesn to the right; if right padding was used, new tkoens would sit in the middle, breaking position indices
# left padding keeps active generation heads aligned at the very last column index (-1)
def left_pad(rows, pad_id):
    width = max(r.shape[0] for r in rows)
    pad_lens = []
    padded = []
    
    for r in rows:
        n = width - r.shape[0]
        pad_lens.append(n)
        
        if n:
            r = torch.cat([torch.full((n,), pad_id, dtype=r.dtype), r])
        
        padded.append(r)
    
    return torch.stack(padded), torch.tensor(pad_lens)


# extracts KV cache of a single sequence out of a batched KV cache tensor
def cache_row(cache, index):
    return [(k[index:index + 1], v[index:index + 1]) for k, v in cache]


# stacks independent request caches along batch dimension so they can be processed in a single batched GEMM kernel
def cache_cat(caches):
    merged = []
    n_layers = len(caches[0])
    
    for layer in range(n_layers):
        k = torch.cat([c[layer][0] for c in caches], dim=0)
        v = torch.cat([c[layer][1] for c in caches], dim=0)
        merged.append((k, v))

    return merged


# prepends zeros to time dimension T of a sequence's KV cache
# when two requests have different prompt lengths or cache sizes, their KV caches must have matching sequence lengths to be batched together
# left padding the shorter cache with dummy vectors aligns their dims
def cache_pad_left(cache, n_pad):
    if n_pad == 0:
        return cache

    padded = []

    for k, v in cache:
        zk = k.new_zeros(k.shape[0], k.shape[1], n_pad, k.shape[3])
        zv = v.new_zeros(v.shape[0], v.shape[1], n_pad, v.shape[3])
        padded.append((torch.cat([zk, k], dim=2), torch.cat([zv, v], dim=2)))

    return padded