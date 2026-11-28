"""Database boundary: one SQLite file (FTS5 for lexical search). One store per run; reset() wipes it."""
import sqlite3
import threading

TABLES = ("sources", "facts", "evidence", "meta", "fts", "forget_log", "dependencies", "embeddings", "conversations", "chat_messages")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources(
  message_id TEXT PRIMARY KEY, ts TEXT, kind TEXT, origin TEXT, text TEXT,
  extraction TEXT NOT NULL DEFAULT 'pending');
CREATE TABLE IF NOT EXISTS facts(
  id INTEGER PRIMARY KEY, kind TEXT, subject TEXT, relation TEXT, value TEXT,
  attribution TEXT, when_date TEXT, quote TEXT, key TEXT,
  status TEXT NOT NULL DEFAULT 'active', predecessor_id INTEGER,
  first_message_id TEXT NOT NULL, created_ts TEXT,
  cardinality TEXT, ext_id TEXT, data TEXT, ended_ts TEXT,
  when_ts TEXT, sensitive INTEGER NOT NULL DEFAULT 0, conflict_with INTEGER, vkey TEXT, skey TEXT);
CREATE INDEX IF NOT EXISTS facts_vkey ON facts(vkey, status);
CREATE INDEX IF NOT EXISTS facts_skey ON facts(skey, status);
CREATE INDEX IF NOT EXISTS facts_key ON facts(key, status);
CREATE INDEX IF NOT EXISTS facts_ext ON facts(ext_id, status);
CREATE TABLE IF NOT EXISTS evidence(
  fact_id INTEGER NOT NULL, message_id TEXT NOT NULL, quote TEXT,
  PRIMARY KEY(fact_id, message_id));
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS conversations(id INTEGER PRIMARY KEY, title TEXT NOT NULL, created_ts TEXT, updated_ts TEXT);
CREATE TABLE IF NOT EXISTS chat_messages(id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL, role TEXT NOT NULL,
  mode TEXT, content TEXT NOT NULL, payload TEXT, ts TEXT, request_id TEXT);
CREATE INDEX IF NOT EXISTS chat_conv ON chat_messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS dependencies(child_id INTEGER NOT NULL, parent_id INTEGER NOT NULL, PRIMARY KEY(child_id, parent_id));  -- a derived fact and the facts it rests on
CREATE TABLE IF NOT EXISTS embeddings(ref TEXT PRIMARY KEY, vec BLOB);  -- f:<fact id> or s:<message id>; the index state is separate from the fact
CREATE TABLE IF NOT EXISTS forget_log(  -- payload-free receipt: counts only, never the erased text
  id INTEGER PRIMARY KEY, ts TEXT, message_id TEXT, facts_removed INTEGER, sources_removed INTEGER,
  verified INTEGER, problems INTEGER);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(ref UNINDEXED, text, tokenize='porter unicode61');
"""


# Columns added after the first release. A database file from an earlier version gets them on open, so it keeps working.
ADDED = {"facts": [("cardinality", "TEXT"), ("ext_id", "TEXT"), ("data", "TEXT"), ("ended_ts", "TEXT"), ("when_ts", "TEXT"),
                   ("sensitive", "INTEGER NOT NULL DEFAULT 0"), ("conflict_with", "INTEGER"), ("vkey", "TEXT"), ("skey", "TEXT")],
         "chat_messages": [("request_id", "TEXT")]}


class Database:
    def __init__(self, path: str) -> None:
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA secure_delete=ON")  # deleted content is overwritten in the file, not just unlinked
        self.lock = threading.RLock()
        with self.lock:
            self._migrate()
            self.db.executescript(SCHEMA)
            self.db.commit()

    def _migrate(self) -> None:
        for table, columns in ADDED.items():
            have = {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}
            for name, decl in columns if have else []:
                if name not in have:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        self.db.commit()

    def reset(self) -> None:
        # Drop and recreate so a schema change can never leave a stale table behind.
        with self.lock:
            for t in TABLES:
                self.db.execute(f"DROP TABLE IF EXISTS {t}")
            self.db.executescript(SCHEMA)
            self.db.commit()
