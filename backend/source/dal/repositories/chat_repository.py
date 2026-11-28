"""Persistent conversations inside one project's database. Chats are a log of what was said, not a memory source:
deleting a chat never deletes memories, and a Forget redacts any chat message that carried the erased information."""
import json
from datetime import datetime

from backend.source.dal.db_session import Database


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class ChatRepository:
    def __init__(self, database: Database) -> None:
        self.db, self.lock = database.db, database.lock

    def create(self, title: str = "New chat") -> dict:
        with self.lock, self.db:
            cur = self.db.execute("INSERT INTO conversations(title, created_ts, updated_ts) VALUES(?,?,?)", (title[:80] or "New chat", now_iso(), now_iso()))
        return self.get(cur.lastrowid)

    def get(self, cid: int) -> dict | None:
        with self.lock:
            r = self.db.execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
        return dict(r) if r else None

    def all_chats(self) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute(
                "SELECT c.id, c.title, c.updated_ts, (SELECT COUNT(*) FROM chat_messages m WHERE m.conversation_id=c.id) AS messages "
                "FROM conversations c ORDER BY c.updated_ts DESC, c.id DESC")]

    def rename(self, cid: int, title: str) -> bool:
        with self.lock, self.db:
            return self.db.execute("UPDATE conversations SET title=? WHERE id=?", (title.strip()[:80] or "Untitled", cid)).rowcount > 0

    def delete(self, cid: int) -> bool:
        with self.lock, self.db:
            self.db.execute("DELETE FROM chat_messages WHERE conversation_id=?", (cid,))
            return self.db.execute("DELETE FROM conversations WHERE id=?", (cid,)).rowcount > 0

    def clear(self) -> None:
        with self.lock, self.db:
            self.db.execute("DELETE FROM chat_messages")
            self.db.execute("DELETE FROM conversations")

    def messages(self, cid: int) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM chat_messages WHERE conversation_id=? ORDER BY id", (cid,)).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"]) if r["payload"] else None} for r in rows]

    def by_request(self, cid: int, request_id: str):
        """The (user, assistant) pair already stored for this request id, so a retry or double submit changes nothing."""
        with self.lock:
            rows = self.db.execute("SELECT * FROM chat_messages WHERE conversation_id=? AND request_id=? ORDER BY id", (cid, request_id)).fetchall()
        if len(rows) < 2:
            return None
        msgs = [{**dict(r), "payload": json.loads(r["payload"]) if r["payload"] else None} for r in rows]
        return {"user": msgs[0], "assistant": msgs[1]}

    def append(self, cid: int, role: str, mode: str, content: str, payload=None, request_id: str | None = None) -> dict:
        with self.lock, self.db:
            cur = self.db.execute("INSERT INTO chat_messages(conversation_id, role, mode, content, payload, ts, request_id) VALUES(?,?,?,?,?,?,?)",
                                  (cid, role, mode, content, json.dumps(payload, ensure_ascii=False) if payload is not None else None, now_iso(), request_id))
            self.db.execute("UPDATE conversations SET updated_ts=?, title=CASE WHEN title='New chat' AND ?='user' THEN ? ELSE title END WHERE id=?",
                            (now_iso(), role, content[:40].strip() or "New chat", cid))
        return {"id": cur.lastrowid, "role": role, "mode": mode, "content": content, "payload": payload}
