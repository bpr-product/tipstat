"""Candidate search over all ages: lexical (BM25 / FTS5) and, when a local embedding model is available, semantic
(cosine over stored vectors), merged with Reciprocal Rank Fusion. The facts table stays the source of truth."""
import re
import sqlite3

from backend.source.dal.db_session import Database
from backend.source.dal.repositories.memory_repository import MemoryRepository
from backend.source.utils import config
from backend.source.utils.text import STOP

RRF_K = 60


def fts_query(*texts: str) -> str | None:
    toks = [t for text in texts for t in re.findall(r"\w+", (text or "").lower()) if t not in STOP]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(toks)) or None


def fuse(*ranked_lists, k: int = RRF_K) -> list:
    """Reciprocal Rank Fusion: an item scores the sum of 1/(k + rank) over every list that ranks it."""
    score: dict = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            score[item] = score.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score, key=lambda i: -score[i])


class SearchRepository:
    def __init__(self, database: Database, repo: MemoryRepository) -> None:
        self.db, self.lock, self.repo = database.db, database.lock, repo

    def _semantic(self, qvec, prefix: str, limit: int = 30) -> list:
        """Ids ordered by cosine similarity to the query vector; weak matches are dropped (floor and gap to the best)."""
        if not qvec:
            return []
        scored = sorted(((sum(a * b for a, b in zip(qvec, vec)), ref[2:]) for ref, vec in self.repo.embeddings(prefix)), reverse=True)
        if not scored:
            return []
        keep = [(s, i) for s, i in scored[:limit] if s >= config.SEMANTIC_MIN and s >= scored[0][0] - 0.12]
        return [int(i) if prefix == "f:" else i for _, i in keep]

    def search(self, question: str, extra_terms: tuple | list = (), n_facts: int = 10, n_history: int = 5,
               n_sources: int = 5, qvec=None, diag: dict | None = None) -> tuple[list[sqlite3.Row], list[sqlite3.Row], list[sqlite3.Row]]:
        """Candidates: (active facts, earlier versions, sources). extra_terms are model-suggested keywords; qvec is the
        query embedding when semantic search is available.

        History = matching superseded/stale facts plus the predecessor chain of every matched active fact, so a
        'what changed / what was it before' question can reach the old value even if only the new one matched.
        Siblings = other active facts in the same relationship as a match (the rest of a set such as hobbies).
        """
        q = fts_query(question, " ".join(map(str, extra_terms)))
        lex_f, lex_s = [], []
        if q:
            with self.lock:
                lex_f = [r["id"] for r in self.db.execute(
                    "SELECT CAST(substr(ref,3) AS INTEGER) id FROM fts WHERE fts MATCH ? AND ref LIKE 'f:%' ORDER BY rank LIMIT ?",
                    (q, n_facts * 3))]
                lex_s = [r["mid"] for r in self.db.execute(
                    "SELECT substr(ref,3) mid FROM fts WHERE fts MATCH ? AND ref LIKE 's:%' ORDER BY rank LIMIT ?", (q, n_sources * 2))]
        sem_f, sem_s = self._semantic(qvec, "f:"), self._semantic(qvec, "s:")
        if diag is not None:
            diag.update(lex_f=lex_f, sem_f=sem_f, lex_s=lex_s, sem_s=sem_s)
        fact_ids = fuse(lex_f, sem_f)[: n_facts * 3]
        source_ids = fuse(lex_s, sem_s)[:n_sources]
        if not fact_ids and not source_ids:
            return [], [], []
        with self.lock:
            rows = {r["id"]: r for r in self.repo.facts_by_id(fact_ids)}
            ranked = [rows[i] for i in fact_ids if i in rows]
            active = [r for r in ranked if r["status"] == "active"][:n_facts]
            hist: dict[int, sqlite3.Row] = {}
            for a in active:  # walk predecessor chains (bounded)
                pid, hops = a["predecessor_id"], 0
                while pid and hops < 3:
                    p = self.repo.fact(pid)
                    if not p:
                        break
                    hist[p["id"]] = p
                    pid, hops = p["predecessor_id"], hops + 1
            for r in ranked:
                if r["status"] != "active":
                    hist.setdefault(r["id"], r)
            active = active + self.repo.siblings_of(active)
            srcs = [s for s in (self.repo.source(m) for m in source_ids) if s]
            return active, list(hist.values())[:n_history], srcs
