"""The registry of product projects (id, name). Each project's memory is a separate SQLite file next to this registry,
so isolation is structural: nothing in one project's database can refer to or be queried together with another's.
Ids are generated here, never taken from a caller, so they cannot be used to reach other files."""
import re
import secrets
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

ID = re.compile(r"^p_[0-9a-f]{8}$")


class ProjectRepository:
    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "projects").mkdir(exist_ok=True)
        self.root, self.lock = root, threading.RLock()
        self.db = sqlite3.connect(str(root / "registry.db"), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY, name TEXT NOT NULL, created_ts TEXT NOT NULL)")
        self.db.commit()

    def path(self, pid: str) -> Path:
        if not ID.match(pid or ""):
            raise KeyError(pid)
        return self.root / "projects" / f"{pid}.db"

    def list(self) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT id, name, created_ts FROM projects ORDER BY created_ts, id")]

    def get(self, pid: str) -> dict:
        with self.lock:
            r = self.db.execute("SELECT id, name, created_ts FROM projects WHERE id=?", (pid,)).fetchone() if ID.match(pid or "") else None
        if not r:
            raise KeyError(pid)
        return dict(r)

    def create(self, name: str) -> dict:
        name = (name or "").strip()
        if not 1 <= len(name) <= 60:
            raise ValueError("A project name must be 1 to 60 characters.")
        with self.lock, self.db:
            if self.db.execute("SELECT 1 FROM projects WHERE lower(name)=lower(?)", (name,)).fetchone():
                raise ValueError("A project with that name already exists.")
            pid = "p_" + secrets.token_hex(4)
            self.db.execute("INSERT INTO projects(id, name, created_ts) VALUES(?,?,?)", (pid, name, datetime.now().astimezone().isoformat(timespec="seconds")))
        return self.get(pid)

    def rename(self, pid: str, name: str) -> dict:
        self.get(pid)
        name = (name or "").strip()
        if not 1 <= len(name) <= 60:
            raise ValueError("A project name must be 1 to 60 characters.")
        with self.lock, self.db:
            if self.db.execute("SELECT 1 FROM projects WHERE lower(name)=lower(?) AND id!=?", (name, pid)).fetchone():
                raise ValueError("A project with that name already exists.")
            self.db.execute("UPDATE projects SET name=? WHERE id=?", (name, pid))
        return self.get(pid)

    def delete(self, pid: str) -> None:
        self.get(pid)
        with self.lock, self.db:
            self.db.execute("DELETE FROM projects WHERE id=?", (pid,))
