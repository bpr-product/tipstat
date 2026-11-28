"""Text helpers shared by services: normalization, token estimate (the proxy's rule), clipping, JSON parsing."""
import json
import re


STOP = set("a an the of to in on at is are was were be been do does did i my me we our you your it its "
           "and or for with what when where which who whom how many much this that these those "
           "have has had about from by as so if not no am will would can could should user".split())  # "user" is on every fact


def words(text: str, min_len: int = 5) -> set:
    """Lower-case words of at least min_len characters that are not stop words (used for guards and sweeps)."""
    return {t for t in re.findall(r"\w+", (text or "").lower()) if len(t) >= min_len and t not in STOP}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


def est_tokens(s: str) -> float:
    """The proxy's rule: ~4 ASCII chars per token, one token per non-ASCII character."""
    return sum(1.0 if ord(c) > 127 else 0.25 for c in s)


def user_view(text: str) -> str:
    """A chat source without the assistant's lines. The assistant's own words are not facts, so they are not shown as evidence."""
    return "\n".join(ln for ln in (text or "").splitlines() if not ln.startswith("ASSISTANT:"))


def clip(s: str, max_tokens: int) -> str:
    while est_tokens(s) > max_tokens:
        s = s[: int(len(s) * 0.9)]
    return s


def parse_json(s: str) -> dict:
    """The JSON object in a model reply (fences tolerated). Raises ValueError if there is none."""
    s = re.sub(r"^```(?:json)?|```$", "", (s or "").strip(), flags=re.M).strip()
    try:
        out = json.loads(s)
    except ValueError:
        m = re.search(r"\{.*\}", s, re.S)
        try:
            out = json.loads(m.group(0)) if m else None
        except ValueError:
            out = None
    if not isinstance(out, dict):
        raise ValueError(f"model returned no JSON object (got {len(s)} chars)")
    return out


def _seq(text: str, short_field: bool = False) -> list:
    """[(lower word, looks like a name or number)] for words of 3+ characters, stop words dropped.
    A capitalized word counts as a name unless it merely starts a sentence (short fields have no sentences)."""
    out, text = [], text or ""
    for m in re.finditer(r"\w+", text):
        w = m.group(0)
        low, digit = w.lower(), any(c.isdigit() for c in w)
        if low in STOP or len(w) < 3:
            continue
        before = text[: m.start()].rstrip()
        initial = not before or before[-1] in ".!?:\n"
        out.append((low, digit or (w[0].isupper() and not (initial and not short_field))))
    return out


def leak_terms(subject: str, relation: str, value: str, quote: str) -> tuple:
    """(distinctive single words, adjacent word pairs) of a fact: what a message would contain if it revealed the fact.
    Single words are names and numbers (plus the words of a short value); pairs catch phrases like 'job offer'.
    Common words alone are never terms, so unrelated text is not caught."""
    uni, bi = set(), set()
    for text in (subject, relation, value, quote):
        seq = _seq(text, short_field=len((text or "").split()) <= 4)
        uni |= {w for w, proper in seq if proper}
        bi |= {f"{a} {b}" for (a, _), (b, _) in zip(seq, seq[1:])}
    if len((value or "").split()) <= 3:
        uni |= {w for w, _ in _seq(value, True)}
    return uni, bi


def message_terms(text: str) -> tuple:
    """(all content words, adjacent word pairs) of any text, to compare against leak_terms."""
    seq = _seq(text, short_field=True)
    return {w for w, _ in seq}, {f"{a} {b}" for (a, _), (b, _) in zip(seq, seq[1:])}
