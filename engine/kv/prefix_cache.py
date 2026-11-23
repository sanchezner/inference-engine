import hashlib
import struct
from collections import OrderedDict


_FIRST_PARENT = bytes(32)


# Parent hash covers every earlier token, so the same tail misses when the head differs.
def block_hashes(model_id: str, block_size: int, token_ids) -> list[bytes]:
    tokens = tuple(int(token) for token in token_ids)
    parent = _FIRST_PARENT
    hashes = []
    for start in range(0, len(tokens) - block_size + 1, block_size):
        parent = _hash_block(model_id, block_size, parent, tokens[start:start + block_size])
        hashes.append(parent)
    return hashes


class PrefixCache:
    def __init__(self, model_id: str, block_size: int):
        if not isinstance(model_id, str) or model_id == "":
            raise ValueError("prefix cache needs a non-empty model id in the hash salt")
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
            raise ValueError(f"block_size must be an int >= 1, got {block_size!r}")
        self.model_id = model_id
        self.block_size = block_size
        self.by_hash: dict[bytes, int] = {}
        self.by_page: dict[int, bytes] = {}
        # Front is the oldest unreferenced block. Referenced blocks stay out of this map.
        self.lru: OrderedDict[int, None] = OrderedDict()

    def lookup(self, token_ids) -> tuple[list[int], int]:
        found = []
        for digest in block_hashes(self.model_id, self.block_size, token_ids):
            page = self.by_hash.get(digest)
            if page is None:
                break
            found.append(page)
        return found, len(found) * self.block_size

    def holds(self, page_id: int) -> bool:
        return page_id in self.by_page

    def insert(self, page_id: int, digest: bytes) -> None:
        current = self.by_page.get(page_id)
        if current == digest:
            return
        if current is not None:
            raise RuntimeError(f"page {page_id} is already cached under a different prefix")
        # The first sequence to finish a prefix keeps the page. A later copy stays private.
        if digest in self.by_hash:
            return
        self.by_hash[digest] = page_id
        self.by_page[page_id] = digest

    def mark_evictable(self, page_id: int) -> None:
        if page_id not in self.by_page:
            raise RuntimeError(f"page {page_id} is not a cached prefix block")
        self.lru[page_id] = None
        self.lru.move_to_end(page_id)

    def unmark(self, page_id: int) -> None:
        self.lru.pop(page_id, None)

    def evict(self) -> int | None:
        if not self.lru:
            return None
        page_id, _ = self.lru.popitem(last=False)
        digest = self.by_page.pop(page_id)
        owner = self.by_hash.pop(digest, None)
        if owner != page_id:
            raise RuntimeError(f"prefix hash for page {page_id} pointed at {owner}")
        return page_id

    def evictable_count(self) -> int:
        return len(self.lru)

    def evictable_excluding(self, keep: set[int]) -> int:
        if not keep:
            return len(self.lru)
        return sum(1 for page in self.lru if page not in keep)


def attach_prefix_cache(pool, model_id: str) -> PrefixCache:
    if pool.cache is not None:
        raise ValueError("this page pool already has a prefix cache")
    cache = PrefixCache(model_id, pool.block_size)
    pool.cache = cache
    return cache


def match_prefix(cache: PrefixCache | None, token_ids) -> tuple[list[int], int]:
    if cache is None:
        return [], 0
    blocks, n_tokens = cache.lookup(token_ids)
    # A full hit leaves no token to score, and the paged forward rejects an empty span.
    if n_tokens >= len(token_ids) and blocks:
        blocks = blocks[:-1]
        n_tokens = len(blocks) * cache.block_size
    return blocks, n_tokens


def publish_written(cache: PrefixCache | None, block_ids: list[int], token_ids) -> None:
    if cache is None:
        return
    hashes = block_hashes(cache.model_id, cache.block_size, token_ids)
    if len(block_ids) < len(hashes):
        raise RuntimeError(
            f"{len(hashes)} full blocks of tokens but the sequence only holds {len(block_ids)} pages"
        )
    for page, digest in zip(block_ids, hashes):
        cache.insert(page, digest)


def _hash_block(model_id: str, block_size: int, parent: bytes, tokens: tuple[int, ...]) -> bytes:
    digest = hashlib.sha256()
    digest.update(b"v1\0")
    digest.update(model_id.encode())
    digest.update(b"\0")
    digest.update(block_size.to_bytes(4, "little"))
    digest.update(parent)
    digest.update(struct.pack(f"<{len(tokens)}q", *tokens))
    return digest.digest()
