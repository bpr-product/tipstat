"""The product workspace: projects, persistent conversations, and honest receipts for Tell / Ask / Do.

Every project is its own MemorySystem over its own SQLite file (see ProjectRepository), so extraction, reconciliation,
retrieval, evidence, dependencies, embeddings, forgotten-term hashes and deletion are isolated by construction.
This workspace is separate from the evaluation service's database: the official /reset cannot touch it and it cannot
read the evaluation memory. It runs in the UI process; server.py is unchanged.

Each turn stores what really happened with it: a receipt built from before/after state (Tell), the retrieval flow and cited
evidence with their details (Ask), the tool steps (Do), and the model usage measured for that turn.
"""
import re
import time
import uuid
from pathlib import Path

from backend.source.api.memory_system import MemorySystem
from backend.source.dal.repositories.chat_repository import ChatRepository, now_iso
from backend.source.dal.repositories.memory_view import snapshot
from backend.source.dal.repositories.project_repository import ProjectRepository
from backend.source.service.memory.intent import looks_like_question
from backend.source.utils.text import STOP
from backend.source.utils import config, metrics

MODES = {"tell", "ask", "do"}
FORGOTTEN_REQUEST = "(You asked me to forget something. The request text was not kept.)"
CONNECTIVE = re.compile(r"^\s*(and|also|what about|how about|then|so|but)\b", re.IGNORECASE)
PRONOUN = re.compile(r"\b(it|its|that|those|them|they|he|she|him|her|there|this one)\b", re.IGNORECASE)
THIN = {"when", "where", "which", "whose", "whom", "why", "does", "tell", "show", "about", "next", "also", "then", "with"}
QUESTION_GUIDANCE = "That looks like a question, so I did not store it. Tell is for things to remember; use Ask to look something up."


class WorkspaceError(Exception):
    """A request the caller got wrong (unknown project or chat, bad input): shown to the user as is."""


class Workspace:
    def __init__(self, root, llm_fn=None, chat_fn=None, embedder=None) -> None:
        self.projects_repo = ProjectRepository(Path(root))
        self._llm, self._chat, self._embedder = llm_fn, chat_fn, embedder
        self._systems: dict = {}
        if not self.projects_repo.list():
            self.projects_repo.create("General")

    # -- projects -----------------------------------------------------------------
    def list_projects(self) -> list[dict]:
        return self.projects_repo.list()

    def create_project(self, name: str) -> dict:
        try:
            return self.projects_repo.create(name)
        except ValueError as e:
            raise WorkspaceError(str(e))

    def rename_project(self, pid: str, name: str) -> dict:
        try:
            return self.projects_repo.rename(pid, name)
        except KeyError:
            raise WorkspaceError("Unknown project.")
        except ValueError as e:
            raise WorkspaceError(str(e))

    def delete_project(self, pid: str) -> None:
        """Removes the project's whole database file: memories, conversations, vectors, everything."""
        path = self._path(pid)
        system = self._systems.pop(pid, None)
        if system:
            system.close()
        self.projects_repo.delete(pid)
        for suffix in ("", "-journal", "-wal", "-shm"):
            Path(str(path) + suffix).unlink(missing_ok=True)
        if not self.projects_repo.list():
            self.projects_repo.create("General")

    def reset_all(self) -> None:
        for p in self.projects_repo.list():
            self.delete_project(p["id"])

    def _path(self, pid: str) -> Path:
        try:
            self.projects_repo.get(pid)
            return self.projects_repo.path(pid)
        except KeyError:
            raise WorkspaceError("Unknown project.")

    def system(self, pid: str) -> MemorySystem:
        if pid not in self._systems:
            self._systems[pid] = MemorySystem(llm_fn=self._llm, chat_fn=self._chat, embedder=self._embedder, db_path=str(self._path(pid)))
        return self._systems[pid]

    def _chats(self, pid: str) -> ChatRepository:
        return ChatRepository(self.system(pid).database)

    # -- conversations ------------------------------------------------------------
    def list_chats(self, pid: str) -> list[dict]:
        return self._chats(pid).all_chats()

    def new_chat(self, pid: str) -> dict:
        return self._chats(pid).create()

    def open_chat(self, pid: str, cid: int) -> dict:
        chats = self._chats(pid)
        conv = chats.get(cid)
        if not conv:
            raise WorkspaceError("Unknown conversation.")
        return {**conv, "messages": chats.messages(cid)}

    def rename_chat(self, pid: str, cid: int, title: str) -> None:
        if not self._chats(pid).rename(cid, title):
            raise WorkspaceError("Unknown conversation.")

    def delete_chat(self, pid: str, cid: int) -> None:
        if not self._chats(pid).delete(cid):  # memories stay: a chat is a log, not the memory
            raise WorkspaceError("Unknown conversation.")

    def clear_chats(self, pid: str) -> None:
        self._chats(pid).clear()

    # -- Tell / Ask / Do -----------------------------------------------------------
    def send(self, pid: str, cid: int, mode: str, text: str, world_url: str = "", request_id=None) -> dict:
        """One turn. A request id makes it idempotent: sending the same id again returns the stored turn and changes nothing."""
        text = (text or "").strip()
        if mode not in MODES or not text or len(text) > 4000:
            raise WorkspaceError("Choose Tell, Ask or Do and write between 1 and 4000 characters.")
        system, chats = self.system(pid), self._chats(pid)
        if not chats.get(cid):
            raise WorkspaceError("Unknown conversation.")
        rid = re.sub(r"[^A-Za-z0-9]", "", str(request_id or ""))[:32] or None
        if rid and (done := chats.by_request(cid, rid)):
            return {**done, "replayed": True}
        stored_user, started = text, time.monotonic()
        with metrics.metering() as meter:
            if mode == "tell" and looks_like_question(text):
                payload = {"headline": QUESTION_GUIDANCE, "guidance": "ask", "extraction": "not_sent", "saved": [], "updated": [],
                           "history": [], "stale": [], "already_known": [], "conflicts": [], "forgotten": 0, "source_kept": False}
                reply = QUESTION_GUIDANCE
            elif mode == "tell":
                mid, before = "u" + (rid or uuid.uuid4().hex)[:24], self._state(system)
                system.add_message({"id": mid, "ts": now_iso(), "type": "chat", "turns": [{"role": "user", "content": text}]})
                payload = self._receipt(system, before, mid)
                reply = payload["headline"]
                if payload["extraction"] == "forget_request":
                    stored_user = FORGOTTEN_REQUEST  # the transcript must not keep what the user asked to forget
            elif mode == "ask":
                hint = self._follow_up_hint(chats, cid, text)
                out = system.answer(text, hint=hint)
                reply, payload = out["answer"], {"evidence": out["evidence"], "trace": out["trace"], "follow_up_of": hint, **self._ask_view(system, out["trace"])}
            else:
                if not world_url.startswith(("http://", "https://")):
                    raise WorkspaceError("Enter the tools URL of the mock world (http://...) to run a job.")
                out = system.act(text, {"tools_url": world_url, "mcp_url": world_url.rsplit("/tools", 1)[0] + "/mcp"})
                reply, payload = out["reply"], {"trace": out["trace"]}
        payload["usage"] = {**meter.summary(), "latency_ms": round((time.monotonic() - started) * 1000)}
        return {"user": chats.append(cid, "user", mode, stored_user, request_id=rid),
                "assistant": chats.append(cid, "assistant", mode, reply, payload, request_id=rid)}

    @staticmethod
    def _follow_up_hint(chats: ChatRepository, cid: int, text: str):
        """For a thin question ("and with whom?", "when is it?", "where?") in a chat that already had an Ask: the previous
        QUESTION, as a hint about what 'it' or 'that' means. A question that carries its own subject is never a follow-up.
        The hint is never evidence and the previous answer is not used at all."""
        own = {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOP and w not in THIN}
        if len(text.split()) > 12 or not (CONNECTIVE.match(text) or not own or (PRONOUN.search(text) and len(own) <= 1)):
            return None
        prior = [m["content"] for m in chats.messages(cid) if m["role"] == "user" and m["mode"] == "ask" and not m["content"].startswith(("[removed", "("))]
        return prior[-1][:200] if prior else None

    # -- Ask details: what was retrieved, supplied and cited, with the memories behind them -----------------
    def _label(self, system: MemorySystem, ref: str) -> dict:
        if ref.startswith("F") and ref[1:].isdigit():
            f = system.repo.fact(int(ref[1:]))
            return {"label": f"{f['subject']} · {f['relation']}: {f['value']}", "status": f["status"], "kind": f["kind"]} if f else {"label": "(removed)"}
        src = system.repo.source(ref[2:])
        return {"label": ((src["text"] if src else "") or "(source text not kept)")[:110], "status": "source", "kind": "source"}

    def _ask_view(self, system: MemorySystem, trace: list) -> dict:
        step = lambda name: next((t for t in trace if t["step"] == name), {})  # noqa: E731
        expand, semantic, retrieve, supply, answer = step("expand"), step("semantic"), step("retrieve"), step("supply"), step("answer")
        cited = []
        for ref in answer.get("cited", []):
            if ref.startswith("F") and (f := system.repo.fact(int(ref[1:]))):
                src = system.repo.source(f["first_message_id"])
                cited.append({"ref": ref, "fact_id": f["id"], "label": f"{f['subject']} · {f['relation']}: {f['value']}", "quote": f["quote"],
                              "kind": f["kind"], "attribution": f["attribution"], "status": f["status"], "message_id": f["first_message_id"],
                              "source_kind": src["kind"] if src else None, "source_origin": src["origin"] if src else None,
                              "noted": f["created_ts"], "replaced": f["ended_ts"]})
            elif ref.startswith("S:"):
                src = system.repo.source(ref[2:])
                cited.append({"ref": ref, "fact_id": None, "label": "source message", "quote": ((src["text"] if src else "") or "")[:300],
                              "kind": "source", "attribution": src["origin"] if src else None, "status": "source", "message_id": ref[2:],
                              "source_kind": src["kind"] if src else None, "source_origin": src["origin"] if src else None})
        # Only facts the answer cited tie a whole answer to a fact: an uncited candidate that is later erased is removed from the
        # answer's retrieval list instead (see MemoryRepository.redact_chats), so unrelated answers are not redacted.
        return {"fact_refs": sorted({c["fact_id"] for c in cited if c["fact_id"]}), "flow": {"expansion": {"terms": expand.get("terms"), "error": expand.get("error")}, "semantic_used": semantic.get("used"),
                         "lexical": retrieve.get("lexical"), "semantic_hits": retrieve.get("semantic"), "fused": retrieve.get("fused"),
                         "rrf_applied": retrieve.get("rrf_applied"),
                         "candidates": [{**c, **self._label(system, c["ref"])} for c in retrieve.get("candidates", [])],
                         "supplied": [{"ref": r, **self._label(system, r)} for r in supply.get("supplied", [])],
                         "supplied_tokens": supply.get("est_tokens"), "extra": [t for t in trace if t["step"] in ("parts", "agenda", "read_only", "abstain")]},
                "cited": cited}

    # -- receipts: what a Tell really did, from before/after states, never from the model's say-so -----------
    @staticmethod
    def _state(system: MemorySystem) -> dict:
        with system.repo.lock:
            facts = {r["id"]: dict(r) for r in system.repo.db.execute(
                "SELECT f.id, f.subject, f.relation, f.value, f.status, f.conflict_with, f.predecessor_id, "
                "(SELECT COUNT(*) FROM evidence e WHERE e.fact_id=f.id) AS ev FROM facts f")}
            forget_max = system.repo.db.execute("SELECT COALESCE(MAX(id), 0) FROM forget_log").fetchone()[0]
        return {"facts": facts, "forget_max": forget_max}

    def _receipt(self, system: MemorySystem, before: dict, mid: str) -> dict:
        after = self._state(system)
        text = lambda f: f"{f['subject']} · {f['relation']}: {f['value']}"  # noqa: E731
        b, a = before["facts"], after["facts"]
        new = {i: f for i, f in a.items() if i not in b}
        updated = [{"was": text(b[f["predecessor_id"]]), "now": text(f), "fact": i} for i, f in new.items()   # real version links only
                   if f["predecessor_id"] in b and b[f["predecessor_id"]]["status"] == "active" and a[f["predecessor_id"]]["status"] == "superseded"]
        new_versions = {u["fact"] for u in updated}
        saved = [text(f) for i, f in new.items() if f["status"] == "active" and i not in new_versions]
        history = [text(f) for f in new.values() if f["status"] != "active"]
        stale = [text(b[i]) for i, f in a.items() if i in b and b[i]["status"] == "active" and f["status"] == "stale"]
        again = [text(f) for i, f in a.items() if i in b and f["ev"] > b[i]["ev"] and f["status"] == "active"]
        conflicts = [text(f) for f in new.values() if f["conflict_with"]]
        with system.repo.lock:
            rows = system.repo.db.execute("SELECT facts_removed, sources_removed, verified, problems FROM forget_log WHERE id>?", (before["forget_max"],)).fetchall()
        src = system.repo.source(mid)
        extraction = src["extraction"] if src else "unknown"
        kept = bool(src and src["text"])  # the raw message stays searchable even when no fact was extracted from it
        removed, erased = sum(r["facts_removed"] for r in rows), sum(r["sources_removed"] for r in rows)
        verified, problems = (all(r["verified"] for r in rows), sum(r["problems"] for r in rows)) if rows else (None, 0)
        if extraction == "forget_unresolved":
            headline = "I couldn't work out what to forget, so nothing was removed. Please say it again more specifically."
        elif rows and removed + erased:
            headline = (f"Forgot {removed} fact(s) and erased {erased} source(s). The removal check found nothing left." if verified else
                        f"Removed {removed} fact(s) and {erased} source(s), but the removal check found {problems} problem(s): something may remain, "
                        "so treat it as not fully forgotten.")
        elif extraction == "forget_request":
            headline = "I found nothing stored that matches, so nothing was removed."
        elif extraction == "failed":
            headline = "I couldn't process that just now (the model was unavailable), so no facts were saved. The message itself is kept as a searchable source."
        elif saved or updated or again or history:
            bits = [f"saved {len(saved)}" if saved else "", f"updated {len(updated)}" if updated else "",
                    f"already knew {len(again)}" if again else "", f"kept {len(history)} as history" if history else ""]
            headline = "Done: " + ", ".join(x for x in bits if x) + "." + (" The message itself is also kept as a searchable source." if kept else "")
        elif extraction == "skipped":
            headline = "Nothing to remember in that. The message is kept as a searchable source, but no memory was saved."
        else:
            headline = "No structured facts were extracted from that. The message itself is kept as a searchable source, so Ask can still find it, but nothing was saved as a memory."
        return {"headline": headline, "extraction": extraction, "saved": saved, "updated": updated, "history": history, "stale": stale,
                "already_known": again, "conflicts": conflicts, "forgotten": removed, "sources_erased": erased, "source_kept": kept,
                "fact_refs": sorted(set(new) | {f["predecessor_id"] for f in new.values() if f["predecessor_id"]}
                                    | {i for i, f in a.items() if i in b and (f["status"] == "stale" or f["ev"] > b[i]["ev"])}),
                "verification": None if verified is None else {"verified": bool(verified), "problems": problems}}

    # -- memory timeline, inspection, indexing -----------------------------------------------------------
    def memory(self, pid: str) -> dict:
        return snapshot(self.system(pid).database.db)

    def memory_item(self, pid: str, fid: int) -> dict:
        repo = self.system(pid).repo
        f = repo.fact(fid)
        if not f:
            raise WorkspaceError("Unknown memory.")
        short = lambda r: {"id": r["id"], "label": f"{r['subject']} · {r['relation']}: {r['value']}", "status": r["status"],  # noqa: E731
                           "noted": r["created_ts"], "ended": r["ended_ts"], "quote": r["quote"]}
        evidence = []
        for e in repo.evidence_for(fid):
            src = repo.source(e["message_id"])
            evidence.append({"message_id": e["message_id"], "quote": e["quote"], "source_kind": src["kind"] if src else None,
                             "origin": src["origin"] if src else None, "ts": src["ts"] if src else None,
                             "source_text": ((src["text"] if src else "") or "(source text not kept)")[:600]})
        chain, cur, seen = [], f["predecessor_id"], set()
        while cur and cur not in seen:
            seen.add(cur)
            row = repo.fact(cur)
            if not row:
                break
            chain.append(short(row))
            cur = row["predecessor_id"]
        with repo.lock:
            parents = [r[0] for r in repo.db.execute("SELECT parent_id FROM dependencies WHERE child_id=?", (fid,))]
            children = [r[0] for r in repo.db.execute("SELECT child_id FROM dependencies WHERE parent_id=?", (fid,))]
            successors = [r[0] for r in repo.db.execute("SELECT id FROM facts WHERE predecessor_id=?", (fid,))]
        return {"fact": {k: f[k] for k in ("id", "kind", "subject", "relation", "value", "attribution", "when_date", "when_ts", "status",
                                           "created_ts", "ended_ts", "quote", "sensitive", "conflict_with", "cardinality", "first_message_id")},
                "evidence": evidence, "earlier_versions": chain, "replaced_by": successors,
                "derived_from": [short(repo.fact(i)) for i in parents if repo.fact(i)],
                "derived_facts": [short(repo.fact(i)) for i in children if repo.fact(i)]}

    def _index_counts(self, system: MemorySystem) -> dict:
        db = system.repo.db
        with system.repo.lock:
            n = lambda q: db.execute(q).fetchone()[0]  # noqa: E731
            return {"facts": n("SELECT COUNT(*) FROM facts"), "facts_indexed": n("SELECT COUNT(*) FROM embeddings WHERE ref LIKE 'f:%'"),
                    "sources": n("SELECT COUNT(*) FROM sources WHERE text != ''"), "sources_indexed": n("SELECT COUNT(*) FROM embeddings WHERE ref LIKE 's:%'")}

    def reindex(self, pid: str) -> dict:
        """Embeds whatever has no vector yet (for example after the model became available). No language-model calls."""
        system = self.system(pid)
        emb = system.embedder
        if not getattr(emb, "available", False):
            raise WorkspaceError("Semantic search is not available: " + (getattr(emb, "error", "") or "no embedding model"))
        have = {r[0] for r in system.repo.db.execute("SELECT ref FROM embeddings")}
        todo = {f"f:{r['id']}": f"{r['subject']} {r['relation']} {r['value']} {r['quote']}" for r in system.repo.db.execute("SELECT * FROM facts")}
        todo.update({f"s:{r['message_id']}": r["text"][:500] for r in system.repo.db.execute("SELECT * FROM sources WHERE text != ''")})
        todo = {k: v for k, v in todo.items() if k not in have}
        vecs = emb.embed(list(todo.values())) if todo else []
        for ref, vec in zip(todo, vecs or []):
            system.repo.put_embedding(ref, vec)
        return {"indexed_now": len(vecs or []), "still_missing": len(todo) - len(vecs or []), **self._index_counts(system)}

    def system_info(self) -> dict:
        emb = self._embedder
        ready = bool(getattr(emb, "available", False)) if emb else False
        per_project = []
        for p in self.projects_repo.list():
            per_project.append({"id": p["id"], "name": p["name"], **self._index_counts(self.system(p["id"]))})
        return {"projects": len(per_project), "root": str(self.projects_repo.root), "model": config.MODEL,
                "model_configured": bool(config.API_KEY),
                "semantic": {"ready": ready, "model": config.EMBED_MODEL,
                             "reason": "" if ready else (getattr(emb, "error", "") if emb else "no embedding model configured") or "unavailable"},
                "index": per_project}
