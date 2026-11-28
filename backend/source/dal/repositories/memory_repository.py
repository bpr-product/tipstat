"""Data access for sources, facts, evidence, dependencies, embeddings and run metadata. Facts and their FTS rows are
written in one transaction, so a durable fact is never left unindexed; embeddings are a separate, optional index."""
import hashlib
import json
import re
import secrets
import sqlite3
from array import array
from collections import Counter

from backend.source.dal.db_session import Database
from backend.source.utils.text import message_terms, norm, words


def _strings(o):
    """Every string value inside a payload (never keys, numbers or timestamps' structure)."""
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _strings(v)


HANDLE_AFTER = re.compile(r"(?:from|to): (@[\w.]*\w)")


class MemoryRepository:
    def __init__(self, database: Database) -> None:
        self.db, self.lock = database.db, database.lock
        # Hashes of forgotten terms live in process memory only: nothing derived from erased text is written to disk.
        self._salt, self._tomb = secrets.token_hex(8), set()

    # -- sources -------------------------------------------------------------
    def put_source(self, message_id: str, ts: str, kind: str, origin: str, text: str) -> bool:
        """Idempotent on message id. Returns True if this call stored it (False = replay)."""
        with self.lock, self.db:
            cur = self.db.execute("INSERT OR IGNORE INTO sources(message_id,ts,kind,origin,text) VALUES(?,?,?,?,?)",
                                  (message_id, ts, kind, origin, text))
            if cur.rowcount:
                self.db.execute("INSERT INTO fts(ref,text) VALUES(?,?)", (f"s:{message_id}", text))
                if ts:
                    self.set_meta("now", ts)
            return bool(cur.rowcount)

    def set_extraction(self, message_id: str, status: str) -> None:
        with self.lock, self.db:
            self.db.execute("UPDATE sources SET extraction=? WHERE message_id=?", (status, message_id))

    def active_by_kind(self, kind: str, attribution: str | None = None) -> list[sqlite3.Row]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM facts WHERE kind=? AND status='active' ORDER BY id", (kind,)).fetchall()
        return [r for r in rows if attribution is None or r["attribution"] == attribution]

    def sensitive_facts(self) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute("SELECT * FROM facts WHERE sensitive=1 AND status='active'").fetchall()

    def upcoming(self, from_date: str, to_date: str, limit: int = 8) -> list[sqlite3.Row]:
        """Active dated events/plans/to-dos in [from_date, to_date], nearest first. Structured calendar entities are
        skipped on purpose: the calendar tools already show those; this is what only memory knows (e.g. a flight)."""
        with self.lock:
            return self.db.execute(
                "SELECT * FROM facts WHERE status='active' AND ext_id IS NULL AND kind IN ('EVENT','PLAN','TODO','DERIVED') "
                "AND when_date IS NOT NULL AND when_date>=? AND when_date<=? ORDER BY when_date, id LIMIT ?",
                (from_date, to_date, limit)).fetchall()

    def outdated_dates(self) -> set:
        """Dates that only appear on replaced (superseded or stale) facts, never on a current one."""
        with self.lock:
            old = {r[0] for r in self.db.execute("SELECT when_date FROM facts WHERE status IN ('superseded','stale') AND when_date IS NOT NULL")}
            now = {r[0] for r in self.db.execute("SELECT when_date FROM facts WHERE status='active' AND when_date IS NOT NULL")}
        return old - now

    def people_note(self) -> str:
        """Identity facts that follow from stored app events, nothing more: which handle is the user, and which handles
        belong to different people who share a first name. Empty when nothing is known."""
        own, dm, handles = Counter(), Counter(), set()
        for s in self.all_sources():
            t = s["text"] or ""
            handles |= set(HANDLE_AFTER.findall(t))
            if " by user - " in t:
                own.update(re.findall(r"from: (@[\w.]*\w)", t))
            elif " by inbound - " in t and "channel:" not in t:
                dm.update(re.findall(r"to: (@[\w.]*\w)", t))
        counts = own or dm
        parts = []
        if counts:
            parts.append(f"The user's own handle is {counts.most_common(1)[0][0]}; a message addressed to it was addressed to the user.")
        groups: dict = {}
        for h in handles:
            groups.setdefault(h.lstrip("@").split(".")[0].lower(), set()).add(h)
        clash = ["/".join(sorted(g)) for g in groups.values() if len(g) > 1]
        if clash:
            parts.append("People who share a first name are different people, never merge them: " + "; ".join(sorted(clash)) + ".")
        return " ".join(parts)

    def has_data(self) -> bool:
        with self.lock:
            return self.db.execute("SELECT 1 FROM sources LIMIT 1").fetchone() is not None

    def source(self, message_id: str):
        with self.lock:
            return self.db.execute("SELECT * FROM sources WHERE message_id=?", (message_id,)).fetchone()

    # -- facts ---------------------------------------------------------------
    def add_fact(self, f: dict, message_id: str, ts: str, replaces: int | None = None) -> tuple[int, bool, bool]:
        """Insert a fact or attach evidence to the identical active fact; optionally supersede one active fact.

        Returns (id, created, superseded). A supersede happens only when the new fact is genuinely new, the target is an
        active, different fact, and the new fact is not effective EARLIER than the target (a backdated statement is
        stored as history and the newer-effective fact stays current). Facts derived from a superseded fact go stale.
        """
        key = "|".join(norm(f[k]) for k in ("subject", "relation", "value"))
        with self.lock, self.db:
            row = self.db.execute("SELECT id FROM facts WHERE key=? AND status='active' AND attribution=?", (key, f["attribution"])).fetchone()
            created = row is None
            superseded = False
            if created:
                target = None
                if replaces is not None:
                    target = self.db.execute("SELECT * FROM facts WHERE id=? AND status='active'", (replaces,)).fetchone()
                historical = bool(target and f.get("when") and target["when_date"] and f["when"] < target["when_date"])
                former = bool(f.get("historical")) and target is None  # the text itself says it is over: never stored as current
                cur = self.db.execute(
                    "INSERT INTO facts(kind,subject,relation,value,attribution,when_date,quote,key,first_message_id,created_ts,"
                    "cardinality,ext_id,data,predecessor_id,when_ts,sensitive,conflict_with,vkey,skey,status,ended_ts) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (f["kind"], f["subject"], f["relation"], f["value"], f["attribution"], f.get("when"), f["quote"], key,
                     message_id, ts, f.get("cardinality"), f.get("ext_id"), f.get("data"),
                     target["id"] if target and not historical else None, f.get("when_ts"), int(bool(f.get("sensitive"))),
                     f.get("conflict_with"), norm(f["value"]), norm(f["subject"]) + "|" + norm(f["relation"]),
                     "superseded" if historical or former else "active",
                     target["when_date"] if historical else (f.get("when") or ts[:10]) if former else None))
                fid = cur.lastrowid
                self.db.execute("INSERT INTO fts(ref,text) VALUES(?,?)",
                                (f"f:{fid}", f"{f['subject']} {f['relation']} {f['value']} {f['quote']}"))
                for parent in f.get("derived_from") or []:
                    self.db.execute("INSERT OR IGNORE INTO dependencies(child_id,parent_id) "
                                    "SELECT ?, id FROM facts WHERE id=?", (fid, parent))
                if target and historical:
                    self.db.execute("UPDATE facts SET predecessor_id=? WHERE id=? AND predecessor_id IS NULL", (fid, target["id"]))
                elif target:
                    self.db.execute("UPDATE facts SET status='superseded', ended_ts=? WHERE id=?", (ts, target["id"]))
                    self._stale_descendants(target["id"], fid, ts)
                    superseded = True
            else:
                fid = row["id"]
                if f.get("sensitive"):  # restating a fact with "keep it private" makes the stored one private; never the reverse
                    self.db.execute("UPDATE facts SET sensitive=1 WHERE id=?", (fid,))
                if f.get("historical"):  # "I used to ..." about a fact that is stored as current closes it
                    self.db.execute("UPDATE facts SET status='superseded', ended_ts=? WHERE id=? AND status='active'", (f.get("when") or ts[:10], fid))
            self.db.execute("INSERT OR IGNORE INTO evidence(fact_id,message_id,quote) VALUES(?,?,?)",
                            (fid, message_id, f["quote"]))
            return fid, created, superseded

    def _stale_descendants(self, parent: int, keep: int, ts: str) -> None:
        """Facts derived (directly or not) from a superseded fact no longer have their basis: mark them stale."""
        frontier = [parent]
        while frontier:
            marks = ",".join("?" * len(frontier))
            kids = [r[0] for r in self.db.execute(
                f"SELECT child_id FROM dependencies WHERE parent_id IN ({marks})", frontier) if r[0] != keep]
            for k in kids:
                self.db.execute("UPDATE facts SET status='stale', ended_ts=? WHERE id=? AND status='active'", (ts, k))
            frontier = kids

    def has_successor(self, fact_id: int) -> bool:
        with self.lock:
            return self.db.execute("SELECT 1 FROM facts WHERE predecessor_id=? LIMIT 1", (fact_id,)).fetchone() is not None

    def close_fact(self, fact_id: int, ended: str) -> None:
        """The text said this is no longer true: it stays as history, no longer as a current fact."""
        with self.lock, self.db:
            self.db.execute("UPDATE facts SET status='superseded', ended_ts=? WHERE id=? AND status='active'", (ended, fact_id))

    def escalate_sensitive(self, fact_id: int) -> None:
        with self.lock, self.db:
            self.db.execute("UPDATE facts SET sensitive=1 WHERE id=?", (fact_id,))

    def add_evidence(self, fact_id: int, message_id: str, quote: str) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO evidence(fact_id,message_id,quote) VALUES(?,?,?)", (fact_id, message_id, quote))

    def equivalent_candidates(self, value: str, attribution: str) -> list[sqlite3.Row]:
        """Active facts holding the same (normalized) value from the same kind of source: possible paraphrase duplicates."""
        with self.lock:
            return self.db.execute("SELECT * FROM facts WHERE status='active' AND vkey=? AND attribution=?",
                                   (norm(value), attribution)).fetchall()

    def same_subject_active(self, subject: str) -> list[sqlite3.Row]:
        """Active facts about the same subject (any relation): candidates for a conflicting current value."""
        with self.lock:
            return self.db.execute("SELECT * FROM facts WHERE status='active' AND skey LIKE ?",
                                   (norm(subject).replace("%", "") + "|%",)).fetchall()

    def siblings_of(self, rows, limit: int = 6) -> list[sqlite3.Row]:
        """Other active facts in the same relationship (same subject and relation) as the given facts."""
        out, have = [], {r["id"] for r in rows}
        with self.lock:
            for r in rows:
                for s in self.db.execute("SELECT * FROM facts WHERE status='active' AND skey=? AND id!=? ORDER BY id",
                                         (r["skey"], r["id"])):
                    if s["id"] not in have and len(out) < limit:
                        have.add(s["id"])
                        out.append(s)
        return out

    def descendants(self, ids) -> set:
        """Every fact derived (transitively) from the given facts."""
        out, frontier = set(), set(ids)
        with self.lock:
            while frontier:
                marks = ",".join("?" * len(frontier))
                kids = {r[0] for r in self.db.execute(f"SELECT child_id FROM dependencies WHERE parent_id IN ({marks})", list(frontier))}
                frontier, out = kids - out - set(ids), out | kids
        return out

    def active_by_ext(self, ext_id: str):
        with self.lock:
            return self.db.execute("SELECT * FROM facts WHERE ext_id=? AND status='active' ORDER BY id DESC",
                                   (ext_id,)).fetchone()

    def facts_by_id(self, ids: list[int]) -> list[sqlite3.Row]:
        with self.lock:
            q = ",".join("?" * len(ids))
            return self.db.execute(f"SELECT * FROM facts WHERE id IN ({q})", ids).fetchall() if ids else []

    def fact(self, fact_id: int):
        with self.lock:
            return self.db.execute("SELECT * FROM facts WHERE id=?", (fact_id,)).fetchone()

    def evidence_for(self, fact_id: int) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute("SELECT message_id, quote FROM evidence WHERE fact_id=?", (fact_id,)).fetchall()

    # -- embeddings (optional semantic index) --------------------------------------------
    def put_embedding(self, ref: str, vec) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO embeddings(ref,vec) VALUES(?,?)", (ref, array("f", vec).tobytes()))

    def embeddings(self, prefix: str) -> list:
        with self.lock:
            rows = self.db.execute("SELECT ref, vec FROM embeddings WHERE ref LIKE ?", (prefix + "%",)).fetchall()
        out = []
        for r in rows:
            a = array("f")
            a.frombytes(r["vec"])
            out.append((r["ref"], a))
        return out

    # -- meta ----------------------------------------------------------------
    def set_meta(self, k: str, v: str) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))

    def get_meta(self, k: str, default: str = "") -> str:
        with self.lock:
            r = self.db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
            return r["v"] if r else default

    # -- forgotten-term tombstones (process memory only, salted hashes) -----------------------
    def hash_term(self, term: str) -> str:
        return hashlib.sha256((self._salt + term).encode()).hexdigest()[:20]

    def add_tombstones(self, terms) -> None:
        self._tomb |= {self.hash_term(t) for t in terms}

    def tombstones(self) -> set:
        return set(self._tomb)

    def clear_tombstones(self) -> None:
        self._tomb.clear()

    # -- statistics used by guards and sweeps --------------------------------------------
    def doc_freq(self, min_len: int = 5) -> Counter:
        """For each word, in how many distinct messages it occurs (sources, facts and evidence counted per message)."""
        docs: dict = {}
        for s in self.all_sources():
            docs.setdefault(s["message_id"], set()).update(words(s["text"], min_len))
        for f in self.all_facts():
            docs.setdefault(f["first_message_id"], set()).update(
                words(f"{f['subject']} {f['relation']} {f['value']} {f['quote']}", min_len))
        for e in self.all_evidence():
            docs.setdefault(e["message_id"], set()).update(words(e["quote"], min_len))
        return Counter(w for ws in docs.values() for w in ws)

    def remaining_terms(self) -> tuple:
        """(content words, word pairs) still stored anywhere: facts, evidence quotes and raw sources."""
        uni, bi = set(), set()
        texts = [f"{f['subject']} {f['relation']} {f['value']} {f['quote']}" for f in self.all_facts()]
        texts += [e["quote"] for e in self.all_evidence()] + [s["text"] for s in self.all_sources()]
        for t in texts:
            u, b = message_terms(t)
            uni |= u
            bi |= b
        return uni, bi

    # -- erasure support (forget) ----------------------------------------------
    def all_facts(self) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute("SELECT id, subject, relation, value, quote, data, first_message_id FROM facts").fetchall()

    def all_evidence(self) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute("SELECT fact_id, message_id, quote FROM evidence").fetchall()

    def all_sources(self) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute("SELECT message_id, text FROM sources").fetchall()

    def facts_from_message(self, message_id: str) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute("SELECT id, subject, relation, value, quote FROM facts WHERE first_message_id=?",
                                   (message_id,)).fetchall()

    def chain_ids(self, ids) -> set[int]:
        """The ids plus every earlier and later version linked to them (the whole relationship history)."""
        out, frontier = set(ids), set(ids)
        with self.lock:
            while frontier:
                marks = ",".join("?" * len(frontier))
                rows = self.db.execute(f"SELECT id, predecessor_id FROM facts WHERE id IN ({marks}) "
                                       f"OR predecessor_id IN ({marks})", [*frontier, *frontier]).fetchall()
                found = {r["id"] for r in rows} | {r["predecessor_id"] for r in rows if r["predecessor_id"]}
                frontier, out = found - out, out | found
        return out

    def evidence_sources(self, fact_ids) -> set[str]:
        with self.lock:
            return {r["message_id"] for f in fact_ids for r in
                    self.db.execute("SELECT message_id FROM evidence WHERE fact_id=?", (f,))}

    def erase(self, fact_ids, source_ids) -> None:
        """Remove facts, their evidence, dependency edges and index rows, and the raw source text, in one transaction."""
        with self.lock, self.db:
            for f in fact_ids:
                self.db.execute("DELETE FROM fts WHERE ref=?", (f"f:{f}",))
                self.db.execute("DELETE FROM embeddings WHERE ref=?", (f"f:{f}",))
                self.db.execute("DELETE FROM evidence WHERE fact_id=?", (f,))
                self.db.execute("DELETE FROM dependencies WHERE child_id=? OR parent_id=?", (f, f))
                self.db.execute("DELETE FROM facts WHERE id=?", (f,))
                self.db.execute("UPDATE facts SET predecessor_id=NULL WHERE predecessor_id=?", (f,))
                self.db.execute("UPDATE facts SET conflict_with=NULL WHERE conflict_with=?", (f,))
            for m in source_ids:
                self.db.execute("DELETE FROM fts WHERE ref=?", (f"s:{m}",))
                self.db.execute("DELETE FROM embeddings WHERE ref=?", (f"s:{m}",))
                self.db.execute("DELETE FROM sources WHERE message_id=?", (m,))
        self.compact()

    def scrub_source(self, message_id: str, status: str) -> None:
        """Keep the row (so a replay of this message id stays a no-op) but drop its text and index entries."""
        with self.lock, self.db:
            self.db.execute("DELETE FROM fts WHERE ref=?", (f"s:{message_id}",))
            self.db.execute("DELETE FROM embeddings WHERE ref=?", (f"s:{message_id}",))
            self.db.execute("UPDATE sources SET text='', extraction=? WHERE message_id=?", (status, message_id))
        self.compact()

    def redact_chats(self, phrases, uni, bi, fact_ids=()) -> int:
        """After a Forget: replace every stored chat message that carried the erased information, and the other half of
        its request (the question or the answer). A message is hit by: a receipt/answer that references an erased fact id,
        an erased phrase, a name or long number that is gone from memory, or a word pair that is gone. Only visible text is
        compared (string values), so dates, ids and numbers in the metadata never cause an unrelated message to be redacted."""
        note, n, fact_ids, requests = "[removed: this message contained information you asked me to forget]", 0, set(fact_ids), set()
        with self.lock, self.db:
            for r in self.db.execute("SELECT id, content, payload, request_id FROM chat_messages").fetchall():
                payload = json.loads(r["payload"]) if r["payload"] else None
                if fact_ids and isinstance(payload, dict) and isinstance(payload.get("flow"), dict):  # an unused candidate that was erased:
                    flow, changed = payload["flow"], False                                          # drop that entry, not the whole answer
                    for key in ("candidates", "supplied"):
                        kept = [c for c in flow.get(key, []) if not (str(c.get("ref", "")).startswith("F") and str(c["ref"])[1:].isdigit()
                                                                     and int(c["ref"][1:]) in fact_ids)]
                        changed |= len(kept) != len(flow.get(key, []))
                        flow[key] = kept
                    if changed:
                        self.db.execute("UPDATE chat_messages SET payload=? WHERE id=?", (json.dumps(payload, ensure_ascii=False), r["id"]))
                text = " ".join([r["content"], *(_strings(payload) if payload else [])])
                refs = set(payload.get("fact_refs", [])) if isinstance(payload, dict) else set()
                words_, pairs = message_terms(text)
                if refs & fact_ids or any(p in norm(text) for p in phrases) or words_ & uni or pairs & bi:
                    self.db.execute("UPDATE chat_messages SET content=?, payload=NULL WHERE id=?", (note, r["id"]))
                    n += 1
                    if r["request_id"]:
                        requests.add(r["request_id"])
            for rid in requests:  # the turn's other half (what was asked, or what was answered) goes with it
                n += self.db.execute("UPDATE chat_messages SET content=?, payload=NULL WHERE request_id=? AND content != ?", (note, rid, note)).rowcount
            for c in self.db.execute("SELECT id, title FROM conversations").fetchall():  # titles come from the first message
                w2, p2 = message_terms(c["title"])
                if any(p in norm(c["title"]) for p in phrases) or w2 & uni or p2 & bi:
                    self.db.execute("UPDATE conversations SET title='Conversation' WHERE id=?", (c["id"],))
        return n

    def chat_leftovers(self, phrases) -> int:
        with self.lock:
            rows = self.db.execute("SELECT content, payload FROM chat_messages").fetchall()
            titles = self.db.execute("SELECT title AS content, NULL AS payload FROM conversations").fetchall()
        return sum(1 for r in [*rows, *titles] if any(p in norm(f"{r['content']} {r['payload'] or ''}") for p in phrases))

    def compact(self) -> None:
        """Merge the FTS segments and rewrite the file so erased text does not linger in free pages."""
        with self.lock:
            self.db.execute("INSERT INTO fts(fts) VALUES('optimize')")
            self.db.commit()
            self.db.execute("VACUUM")

    def dangling(self) -> int:
        """Referential problems: evidence/index/version/dependency links that point at something that no longer exists."""
        with self.lock:
            q = ("SELECT (SELECT COUNT(*) FROM evidence WHERE fact_id NOT IN (SELECT id FROM facts))"
                 " + (SELECT COUNT(*) FROM fts WHERE ref LIKE 'f:%' AND CAST(substr(ref,3) AS INTEGER) NOT IN (SELECT id FROM facts))"
                 " + (SELECT COUNT(*) FROM fts WHERE ref LIKE 's:%' AND substr(ref,3) NOT IN (SELECT message_id FROM sources))"
                 " + (SELECT COUNT(*) FROM embeddings WHERE ref LIKE 'f:%' AND CAST(substr(ref,3) AS INTEGER) NOT IN (SELECT id FROM facts))"
                 " + (SELECT COUNT(*) FROM embeddings WHERE ref LIKE 's:%' AND substr(ref,3) NOT IN (SELECT message_id FROM sources))"
                 " + (SELECT COUNT(*) FROM dependencies WHERE child_id NOT IN (SELECT id FROM facts) OR parent_id NOT IN (SELECT id FROM facts))"
                 " + (SELECT COUNT(*) FROM facts WHERE predecessor_id IS NOT NULL AND predecessor_id NOT IN (SELECT id FROM facts))")
            return self.db.execute(q).fetchone()[0]

    def log_forget(self, ts: str, message_id: str, facts: int, sources: int, verified: bool, problems: int) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT INTO forget_log(ts,message_id,facts_removed,sources_removed,verified,problems) VALUES(?,?,?,?,?,?)",
                            (ts, message_id, facts, sources, int(verified), problems))
