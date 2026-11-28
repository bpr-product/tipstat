"""Read-only projection of one project's store for the Memories page: current facts, version chains (timeline), derived-from
edges, conflicts and forget receipts. Takes an open connection; never writes."""
import sqlite3


def snapshot(conn: sqlite3.Connection) -> dict:
    conn.row_factory = sqlite3.Row
    facts = [dict(r) for r in conn.execute(
        "SELECT f.id, f.kind, f.subject, f.relation, f.value, f.attribution, f.when_date, f.when_ts, f.status, f.predecessor_id, "
        "f.created_ts, f.ended_ts, f.first_message_id, f.quote, f.sensitive, f.conflict_with, "
        "(SELECT COUNT(*) FROM evidence e WHERE e.fact_id=f.id) AS evidence_count "
        "FROM facts f ORDER BY f.id DESC")]
    parents: dict = {}
    for r in conn.execute("SELECT child_id, parent_id FROM dependencies"):
        parents.setdefault(r["child_id"], []).append(r["parent_id"])
    by_id = {f["id"]: f for f in facts}
    older = {f["predecessor_id"] for f in facts if f["predecessor_id"]}
    for f in facts:
        f["derived_from"] = parents.get(f["id"], [])
        chain, cur, seen = [], f["predecessor_id"], set()
        while cur and cur in by_id and cur not in seen:  # earlier versions, newest first
            seen.add(cur)
            chain.append(cur)
            cur = by_id[cur]["predecessor_id"]
        f["versions"] = chain
        f["is_head"] = f["id"] not in older
    return {
        "available": True,
        "now": (conn.execute("SELECT v FROM meta WHERE k='now'").fetchone() or [""])[0],
        "facts": facts,
        "counts": {"sources": conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
                   "active": sum(f["status"] == "active" for f in facts),
                   "history": sum(f["status"] != "active" for f in facts)},
        "forgets": [dict(r) for r in conn.execute(  # receipts hold counts only, never the erased text
            "SELECT ts, facts_removed, sources_removed, verified, problems FROM forget_log ORDER BY id DESC")],
    }
