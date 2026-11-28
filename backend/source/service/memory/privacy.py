"""Forget step 2: the read-only erasure plan. Pure deterministic closure over what the model selected.

Closure rules (repeat until stable):
- every earlier and later version of a selected fact goes (the whole relationship history);
- every fact derived from a removed fact goes (dependency edges, transitively);
- every raw source that evidences a removed fact goes (its text is erased). A fact supported only by removed sources goes
  too only if it is about what the user asked to forget (shares a content word with the request or a removed fact); an
  independent fact that merely arrived in the same message survives with its own quote;
- anything still stored that textually contains a removed fact's quote (fact, evidence quote, source text) goes:
  that is how derived or mixed-source copies are caught without trusting the model to list them all;
- anything still stored that shares two or more distinctive words with one removed fact (words found in at most
  RARE_DOCS messages) goes: this catches paraphrased copies. One shared word is never enough, so unrelated facts that
  happen to mention a name survive.
"""
from typing import NamedTuple

from backend.source.dal.repositories.memory_repository import MemoryRepository
from backend.source.utils.text import norm, words

RARE_DOCS = 2   # a word is distinctive if it occurs in at most this many messages (the erased one plus one copy)
GENERIC = {"forget", "forgot", "delete", "erase", "remove", "please", "wipe", "purge", "anything", "everything", "about",
           "thing", "things", "stuff", "that", "this", "told", "tell", "keep", "anywhere", "remember", "mentioned", "never"}
MIN_PHRASE = 12  # shorter quotes would match too much unrelated text to be a safe cascade key


class Plan(NamedTuple):
    facts: set
    sources: set
    phrases: set     # normalized quotes that must not survive anywhere
    cascaded: int    # items added beyond the model's selection, version chains and direct evidence sources


def _text(f) -> str:
    return f"{f['subject']} {f['relation']} {f['value']} {f['quote']}"


def plan_forget(repo: MemoryRepository, fact_ids, source_ids, request: str = "") -> Plan:
    facts, sources, phrases = repo.chain_ids(fact_ids), set(source_ids), set()
    anchor = (words(request, 4) - GENERIC) | set().union(*[words(_text(f), 4) for f in repo.facts_by_id(list(facts))], set())
    base = len(facts) + len(sources)
    evidence = repo.all_evidence()
    rare = {w for w, n in repo.doc_freq(5).items() if n <= RARE_DOCS}  # words found in few messages are distinctive
    seen = None
    while seen != (len(facts), len(sources), len(phrases)):
        seen = (len(facts), len(sources), len(phrases))
        facts = repo.chain_ids(facts) | repo.descendants(facts)  # versions, and facts derived from anything removed
        about = {f["id"] for f in repo.all_facts() if words(_text(f), 4) & anchor}
        facts |= {e["fact_id"] for e in evidence if e["message_id"] in sources and e["fact_id"] in about
                  and all(x["message_id"] in sources for x in evidence if x["fact_id"] == e["fact_id"])}
        sources |= repo.evidence_sources(facts)
        for f in repo.all_facts():
            if f["id"] in facts and len(norm(f["quote"])) >= MIN_PHRASE:
                phrases.add(norm(f["quote"]))
        for f in repo.all_facts():
            if f["id"] not in facts and any(p in norm(f["quote"]) for p in phrases):
                facts.add(f["id"])
        for e in evidence:
            if e["fact_id"] not in facts and any(p in norm(e["quote"]) for p in phrases):
                facts.add(e["fact_id"])
        groups = [g for g in (words(_text(f)) & rare for f in repo.all_facts() if f["id"] in facts) if len(g) >= 2]
        for f in repo.all_facts():
            if f["id"] not in facts and any(len(words(_text(f)) & g) >= 2 for g in groups):
                facts.add(f["id"])
        for s in repo.all_sources():
            if s["message_id"] not in sources and any(len(words(s["text"]) & g) >= 2 for g in groups):
                sources.add(s["message_id"])
        for s in repo.all_sources():
            if s["message_id"] not in sources and any(p in norm(s["text"]) for p in phrases):
                sources.add(s["message_id"])
    return Plan(facts, sources, phrases, max(0, len(facts) + len(sources) - base))
