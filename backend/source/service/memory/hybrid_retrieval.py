"""Candidate discovery for Ask: lexical BM25 over all ages plus model-suggested keywords (paraphrase recall
without embeddings). One small model call; if it fails, lexical search alone still runs."""
import re
from datetime import date, timedelta

from backend.source.dal.repositories.search_repository import SearchRepository, fuse
from backend.source.utils.text import parse_json

EXPAND_SYS = ('You help search a personal memory of notes, chats and app events. Given a question, list up to 8 single-word '
              'search keywords: synonyms, related terms and the wording a note holding the answer would likely use. '
              'Output ONLY JSON: {"terms":["word", ...]}')


AGENDA = re.compile(r"\b(this week|next week|coming up|upcoming|important|keep in mind|today|tomorrow|tonight|pending|to-?do|"
                    r"deadlines?|schedule|agenda|reminders?|anything (due|new)|what('s| is) (on|next))\b", re.IGNORECASE)
PART_SPLIT = re.compile(r"[?;]|\band also\b|\band\b|,\s")


def _parts(question: str) -> list:
    """The separate asks inside one question ('X and Y?'), when there are at least two real ones."""
    parts = [p.strip() for p in PART_SPLIT.split(question) if len(p.split()) >= 3]
    return parts[:3] if len(parts) >= 2 else []


def _union(*lists, cap: int) -> list:
    out, seen = [], set()
    for lst in lists:
        for row in lst:
            key = row["id"] if "id" in row.keys() else row["message_id"]
            if key not in seen and len(out) < cap:
                seen.add(key)
                out.append(row)
    return out


def retrieve(search: SearchRepository, llm, question: str, has_data: bool, embedder=None):
    """Returns (active facts, earlier versions, sources, trace entries). Candidates only: nothing here is evidence yet."""
    terms, trace = [], []
    if has_data:  # nothing stored yet: no point spending a model call
        try:
            out = parse_json(llm([{"role": "system", "content": EXPAND_SYS}, {"role": "user", "content": question}],
                                 max_tokens=120, temperature=0, timeout=10))
            terms = [t for t in (x.strip() for x in out["terms"][:8] if isinstance(x, str)) if re.fullmatch(r"\w{2,30}", t)]
            trace.append({"step": "expand", "terms": terms})
        except Exception as e:
            trace.append({"step": "expand", "error": type(e).__name__})
    qvec = embedder.embed_one(question + " " + " ".join(terms)) if embedder and has_data else None
    trace.append({"step": "semantic", "used": qvec is not None})
    diags: list = [{}]  # one diagnostics dict per search run: the whole question, then each part of it
    facts, hist, srcs = search.search(question, extra_terms=terms, qvec=qvec, n_history=8, diag=diags[0])
    agenda_ids: set = set()
    if has_data and AGENDA.search(question):  # such questions share no words with what matters: add what is due soon and still open
        repo = search.repo
        try:
            today = repo.get_meta("now", "")[:10]
            horizon = (date.fromisoformat(today) + timedelta(days=14)).isoformat()
            agenda = repo.upcoming(today, horizon, 8) + repo.active_by_kind("TODO")[:5]
            agenda_ids = {f["id"] for f in agenda}
            facts = _union(agenda, facts, cap=20)  # first: what is due soon is the point of such a question, and a cap must not drop it
            trace.append({"step": "agenda", "n": len(agenda)})
        except ValueError:  # no message seen yet, so no "now"
            pass
    parts = _parts(question) if has_data else []
    if parts:  # a question with two asks must find evidence for both, not only for the louder one
        found = []
        for p in parts:
            diags.append({})
            found.append(search.search(p, qvec=embedder.embed_one(p) if embedder else None, n_history=8, diag=diags[-1]))
        facts = _union(facts, *[f[0] for f in found], cap=16)
        hist = _union(hist, *[f[1] for f in found], cap=8)
        srcs = _union(srcs, *[f[2] for f in found], cap=7)
        trace.append({"step": "parts", "n": len(parts)})

    def ids(key):
        return {i for d in diags for i in d.get(key, [])}
    lex_f, sem_f, lex_s, sem_s = ids("lex_f"), ids("sem_f"), ids("lex_s"), ids("sem_s")
    fused = {("f", i) for d in diags for i in fuse(d.get("lex_f", []), d.get("sem_f", []))} | \
            {("s", i) for d in diags for i in fuse(d.get("lex_s", []), d.get("sem_s", []))}

    def via(i, lex, sem):  # which method produced the candidate; "due soon" = agenda; "related" = siblings or version chain
        return "both" if i in lex and i in sem else "lexical" if i in lex else "semantic" if i in sem else "due soon" if i in agenda_ids else "related"
    candidates = [{"ref": f"F{f['id']}", "via": via(f["id"], lex_f, sem_f), "status": f["status"]} for f in list(facts) + list(hist)]
    candidates += [{"ref": f"S:{s['message_id']}", "via": via(s["message_id"], lex_s, sem_s), "status": "source"} for s in srcs]
    trace.append({"step": "retrieve", "candidate_facts": len(facts), "earlier_versions": len(hist), "candidate_sources": len(srcs),
                  "lexical": len(lex_f) + len(lex_s), "semantic": len(sem_f) + len(sem_s),   # facts + sources, over every search run
                  "fused": len(fused),                                                         # size of the RRF ranking (facts + sources)
                  "rrf_applied": bool(sem_f or sem_s),                                         # false when only one method returned anything
                  "candidates": candidates[:40]})
    return facts, hist, srcs, trace
