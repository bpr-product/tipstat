"""Read-only routing: does a user's text possibly ask to forget something? Cheap, deterministic, deliberately broad.

A hit never deletes anything by itself; it only buys one model call that decides, and code then verifies.
Negations ("don't forget that ...") and "I forgot my password" match here and are rejected by that decision.
"""
import re

FORGET_HINT = re.compile(
    r"\b(forget|forgot|delete|erase|remove|wipe|purge|scrub|disregard|unsave|expunge|unremember)\b"
    r"|\b(?:get\s+rid\s+of|drop|clear|stop\s+(?:tracking|remembering))\s+(?:that|this|it|all|everything|what)\b"
    r"|\b(?:shouldn'?t|should\s+not)\s+have\s+(?:told|said|mentioned)|\bwasn'?t\s+supposed\s+to"
    r"|\b(?:don'?t|do not|stop|no longer|never)\s+(?:keep|store|save|remember|retain|log|record)"
    r"|\bnever\s+(?:said|mentioned|told)|\bignore\s+(?:that|what)|\b(?:take|took)\s+(?:that|it)\s+back"
    r"|\bscratch\s+that|\bpretend\b|\bnot\s+to\s+(?:keep|store|remember)",
    re.IGNORECASE)


INTERROGATIVE = re.compile(r"^\s*(what|who|whom|whose|when|where|which|how|why|did|do|does|is|are|was|were|am|can|could|should|"
                         r"will|would|have|has|tell me|show me|list|any)\b", re.IGNORECASE)
REMEMBER_CMD = re.compile(r"^\s*(please\s+)?(remember|note|save|store|keep\s+in\s+mind|add\s+to\s+(my\s+)?memory)\b", re.IGNORECASE)


def looks_like_question(text: str) -> bool:
    """A question, not something to remember: it has a question mark and starts with a question word or ends with one."""
    t = (text or "").strip()
    return "?" in t and (bool(INTERROGATIVE.search(t)) or t.endswith("?"))


def is_change_request(text: str) -> bool:
    """A command to store or forget something, not a question. /answer is read-only, so it must refuse these honestly."""
    if "?" in (text or "") or INTERROGATIVE.search(text or ""):
        return False
    return bool(REMEMBER_CMD.search(text or "")) or may_ask_to_forget(text)


def may_ask_to_forget(user_text: str) -> bool:
    return bool(FORGET_HINT.search(user_text or ""))
