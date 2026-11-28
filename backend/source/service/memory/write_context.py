"""Related, already-stored facts shown to extraction so it can tell which one a new statement changes."""
from backend.source.dal.repositories.search_repository import SearchRepository
from backend.source.utils.text import clip


def related_previous_facts(search: SearchRepository, embedder, text: str, n: int = 6, max_tokens: int = 1000):
    """Returns (rows, rendered text). Only active facts; bounded in count and tokens."""
    qvec = embedder.embed_one(text[:500]) if embedder else None  # semantic match finds a correction worded differently
    rows, _, _ = search.search(text[:600], n_facts=n, qvec=qvec)
    lines = [f"[F{f['id']}] {f['subject']} | {f['relation']} | {f['value']} "
             f"({f['kind'].lower()}, noted {(f['created_ts'] or '')[:10]})" for f in rows]
    return rows, clip("\n".join(lines), max_tokens) or "none"
