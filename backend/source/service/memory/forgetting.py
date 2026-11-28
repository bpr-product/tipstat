"""Forget step 3: execute the plan, verify the application's own stores, and keep a payload-free receipt.

Limits (stated in MEMORY.md): this covers this application's database only. It cannot reach the model provider's
logs, the organizer's harness records, or earlier answers already returned to the caller.
"""
import re
import sys

from backend.source.dal.repositories.memory_repository import MemoryRepository
from backend.source.dal.repositories.search_repository import STOP, SearchRepository
from backend.source.service.memory.forget_scope_search import candidates, select_targets
from backend.source.service.memory.intent import FORGET_HINT
from backend.source.service.memory.privacy import GENERIC, MIN_PHRASE, Plan, plan_forget
from backend.source.utils.text import leak_terms, norm, words

IMPERATIVE = re.compile(r"^\s*(?:please\s+|can you\s+|could you\s+|actually,?\s+)?(?:forget|delete|erase|remove|wipe|disregard)\b", re.IGNORECASE)
SENTENCES = re.compile(r"(?<=[.!?])\s+|\n")


def _source_terms(repo: MemoryRepository, source_ids) -> tuple:
    """What the raw sources about to be erased contained: names/numbers, word pairs and the user's own lines. Needed so a
    forget that removes only a source (no fact was extracted from it) still clears conversations and titles."""
    uni, bi, lines = set(), set(), set()
    for s in repo.all_sources():
        if s["message_id"] in source_ids:
            u, b = leak_terms("", "", "", s["text"] or "")
            uni |= u
            bi |= b
            lines |= {norm(ln[6:]) for ln in (s["text"] or "").splitlines() if ln.startswith("USER: ") and len(norm(ln[6:])) >= MIN_PHRASE}
    return uni, bi, lines


def forget(repo: MemoryRepository, plan: Plan, ts: str, request_id: str) -> bool:
    """Erase the plan, then check nothing of it remains. Returns True only if verification found no problem."""
    erased_uni, erased_bi = set(), set()
    for f in repo.facts_by_id(list(plan.facts)):
        u, b = leak_terms(f["subject"], f["relation"], f["value"], f["quote"])
        erased_uni |= u
        erased_bi |= b
    src_uni, src_bi, src_lines = _source_terms(repo, plan.sources)
    erased_uni |= src_uni
    erased_bi |= src_bi
    chat_phrases = set(plan.phrases) | src_lines
    repo.erase(plan.facts, plan.sources)
    # Keep only salted hashes, in process memory, of names/numbers and word pairs that no longer occur anywhere, so the
    # assistant can still refuse to write them into a message later. Nothing derived from the text reaches the disk.
    left_uni, left_bi = repo.remaining_terms()
    gone_uni, gone_bi = erased_uni - left_uni, erased_bi - left_bi
    repo.add_tombstones({"u:" + w for w in gone_uni} | {"b:" + p for p in gone_bi})
    repo.redact_chats(chat_phrases, gone_uni, gone_bi)  # saved conversations and their titles must not keep it either
    problems = repo.dangling() + repo.chat_leftovers(chat_phrases)
    problems += sum(1 for f in repo.all_facts()
                    if any(p in norm(f"{f['subject']} {f['relation']} {f['value']} {f['quote']} {f['data'] or ''}") for p in plan.phrases))
    problems += sum(1 for e in repo.all_evidence() if any(p in norm(e["quote"]) for p in plan.phrases))
    problems += sum(1 for s in repo.all_sources() if any(p in norm(s["text"]) for p in plan.phrases))
    repo.log_forget(ts, request_id, len(plan.facts), len(plan.sources), problems == 0, problems)
    return problems == 0


def _run(repo: MemoryRepository, plan: Plan, ts: str, message_id: str) -> None:
    plan.sources.discard(message_id)  # the (already scrubbed) request row stays so a replay of this message id is a no-op
    if not forget(repo, plan, ts, message_id):
        print(f"forget verification found leftovers for {message_id}", file=sys.stderr)


def handle_forget_request(repo: MemoryRepository, search: SearchRepository, llm, embedder, message_id: str, ts: str, user_text: str) -> bool:
    """Called only for text the user wrote. Returns True if it was a forget request that was carried out (or matched
    nothing). The request must never become another source of what it asks to forget: its own text is scrubbed and any
    fact the extraction call already made from it is removed too."""
    facts, srcs = candidates(search, embedder, user_text)
    srcs = [s for s in srcs if s["message_id"] != message_id]  # the request is not its own target
    try:
        is_forget, fids, sids = select_targets(repo, llm, user_text, facts, srcs)
    except Exception as e:  # cannot tell what to erase: change nothing stored, do not keep the request text, flag it
        print(f"forget decision failed for {message_id}: {type(e).__name__}: {e}", file=sys.stderr)
        repo.scrub_source(message_id, "forget_unresolved")
        echoes = _clause_facts(repo, message_id, user_text, imperative_only=True)  # only a plain "forget ..." sentence is certain
        if echoes:
            _run(repo, plan_forget(repo, echoes, [], user_text), ts, message_id)
        repo.log_forget(ts, message_id, 0, 0, False, 1)  # shows as an unverified receipt
        return False
    if not is_forget:
        return False
    repo.scrub_source(message_id, "forget_request")  # first, so it cannot be cascaded or trip the verification
    if fids or sids:
        plan = plan_forget(repo, fids, sids, user_text)
        plan.facts.update(_request_echoes(repo, plan, message_id))
        plan.facts.update(_clause_facts(repo, message_id, user_text))
        _run(repo, plan, ts, message_id)
    else:  # nothing stored matched, but the request text itself may have produced a fact that restates the thing to forget
        echoes = _clause_facts(repo, message_id, user_text)
        if echoes:
            _run(repo, plan_forget(repo, echoes, [], user_text), ts, message_id)
    return True


def _tokens(rows) -> set:
    return {t for r in rows for k in ("value", "quote") for t in re.findall(r"\w{4,}", r[k].lower()) if t not in STOP}


def _request_echoes(repo: MemoryRepository, plan: Plan, message_id: str) -> set:
    """Facts extracted from the forget request itself that mention what is being erased (e.g. 'asked to forget the
    job offer') would reveal that it existed. Facts from the same message about other things are kept."""
    erased = _tokens(repo.facts_by_id(list(plan.facts)))
    return {f["id"] for f in repo.facts_from_message(message_id) if _tokens([f]) & erased}


def _clause_facts(repo: MemoryRepository, message_id: str, user_text: str, imperative_only: bool = False) -> set:
    """Facts from this message that were made out of the sentence(s) asking to forget (they restate the very thing to
    forget). Facts from the message's other sentences are kept."""
    pick = IMPERATIVE if imperative_only else FORGET_HINT
    clauses = [s for s in SENTENCES.split(user_text or "") if pick.search(s)]
    anchor = words(" ".join(clauses), 4) - GENERIC
    flat = [norm(c) for c in clauses]
    return {f["id"] for f in repo.facts_from_message(message_id)
            if any(norm(f["quote"]) in c for c in flat) or (words(f"{f['value']} {f['quote']}", 4) & anchor)}
