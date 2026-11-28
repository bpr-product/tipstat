"""Update rules the backend enforces on top of what the model proposes (the model suggests, this decides).

- A `replaces` target must be a related fact we showed it AND share the relationship with the new fact (same subject
  or relation words); a model that points at an unrelated fact changes nothing.
- A new fact that restates an active fact (same value, same source type, same subject and a similar relation) is a
  paraphrase duplicate: it adds evidence instead of a second fact.
- An EXCLUSIVE fact that lands on the same subject and a similar relation as an active fact, without a valid `replaces`,
  is flagged as conflicting with it. Both stay current, Ask is told, and nothing is overwritten on a guess.
Independent members of a set never reach these rules as replacements: only an explicit, validated `replaces` supersedes.
"""
from backend.source.utils.text import norm, words


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def _subject_words(subject: str) -> set:
    return words(subject, 3)


def same_slot(a_subject: str, a_relation: str, b_subject: str, b_relation: str, threshold: float = 0.5) -> bool:
    """Same subject (words overlap) and a similar relation (word overlap by Jaccard)."""
    same_subject = norm(a_subject) == norm(b_subject) or bool(_subject_words(a_subject) & _subject_words(b_subject))
    return same_subject and _jaccard(words(a_relation, 3), words(b_relation, 3)) >= threshold


def related_enough(old, new: dict) -> bool:
    a = words(f"{old['subject']} {old['relation']} {old['value']}", 3)
    b = words(f"{new['subject']} {new['relation']} {new['value']}", 3)
    return bool(a & b)


def reconcile(repo, clean: dict, related) -> tuple[dict, int | None]:
    """Returns (candidate with a validated `replaces` and maybe `conflict_with`, id of the fact it duplicates or None)."""
    by_id = {r["id"]: r for r in related}
    rid = clean.get("replaces")
    if rid is not None and (rid not in by_id or not related_enough(by_id[rid], clean)):
        clean["replaces"] = rid = None
    if rid is None:
        for c in repo.equivalent_candidates(clean["value"], clean["attribution"]):
            if same_slot(c["subject"], c["relation"], clean["subject"], clean["relation"]):
                return clean, c["id"]
    if rid is None and clean.get("cardinality") == "EXCLUSIVE":
        for c in repo.same_subject_active(clean["subject"]):
            if c["kind"] == clean["kind"] and c["attribution"] == clean["attribution"] \
                    and _jaccard(words(c["relation"], 3), words(clean["relation"], 3)) >= 0.6:
                clean["conflict_with"] = c["id"]
                break
    return clean, None
