"""Local product UI server. NOT part of the official service: server.py is untouched and stays the graded API on :8000.

  python frontend/serve.py            # product UI on http://127.0.0.1:8080

It serves the static page and a small JSON API over a Workspace (projects, conversations, Tell/Ask/Do). The workspace keeps
its own storage (default: <tempdir>/hush_product, or $HUSH_PRODUCT_DIR), completely separate from the evaluation service's
database: an official /reset cannot touch it and it cannot read evaluation memory. Model calls use the same
OPENAI_BASE_URL / OPENAI_API_KEY as the service (read from .env).
"""
import json
import os
import re
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

if (HERE.parent / ".env").exists():  # same convenience as server.py: never overrides variables already set
    for _line in open(HERE.parent / ".env", encoding="utf-8"):
        _k, _sep, _v = _line.strip().partition("=")
        if _sep and _k and not _k.startswith("#"):
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

from backend.source.service.projects.workspace import Workspace, WorkspaceError  # noqa: E402

STATIC = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/app.css": ("app.css", "text/css; charset=utf-8")}
ROUTES = [  # (method, pattern, handler name)
    ("GET", r"/p/system", "system"), ("POST", r"/p/reset-all", "reset_all"),
    ("GET", r"/p/projects", "projects"), ("POST", r"/p/projects", "create_project"),
    ("POST", r"/p/projects/(?P<pid>[^/]+)/rename", "rename_project"), ("POST", r"/p/projects/(?P<pid>[^/]+)/delete", "delete_project"),
    ("GET", r"/p/(?P<pid>[^/]+)/chats", "chats"), ("POST", r"/p/(?P<pid>[^/]+)/chats", "new_chat"),
    ("POST", r"/p/(?P<pid>[^/]+)/chats/clear", "clear_chats"),
    ("GET", r"/p/(?P<pid>[^/]+)/chats/(?P<cid>\d+)", "open_chat"), ("POST", r"/p/(?P<pid>[^/]+)/chats/(?P<cid>\d+)/rename", "rename_chat"),
    ("POST", r"/p/(?P<pid>[^/]+)/chats/(?P<cid>\d+)/delete", "delete_chat"), ("POST", r"/p/(?P<pid>[^/]+)/chats/(?P<cid>\d+)/send", "send"),
    ("GET", r"/p/(?P<pid>[^/]+)/memory", "memory"), ("GET", r"/p/(?P<pid>[^/]+)/memory/(?P<fid>\d+)", "memory_item"),
    ("POST", r"/p/(?P<pid>[^/]+)/reindex", "reindex"),
]


def route(ws: Workspace, method: str, path: str, body: dict) -> tuple:
    """Pure routing: (status, JSON-able). Unknown project/chat or bad input is a 4xx with a readable message."""
    for m, pattern, name in ROUTES:
        hit = re.fullmatch(pattern, path) if m == method else None
        if not hit:
            continue
        a = hit.groupdict()
        cid = int(a["cid"]) if "cid" in a else None
        fid = int(a["fid"]) if "fid" in a else None
        try:
            if name == "system": return 200, ws.system_info()
            if name == "reset_all": ws.reset_all(); return 200, {"ok": True}
            if name == "projects": return 200, {"projects": ws.list_projects()}
            if name == "create_project": return 200, ws.create_project(body.get("name", ""))
            if name == "rename_project": return 200, ws.rename_project(a["pid"], body.get("name", ""))
            if name == "delete_project": ws.delete_project(a["pid"]); return 200, {"ok": True}
            if name == "chats": return 200, {"chats": ws.list_chats(a["pid"])}
            if name == "new_chat": return 200, ws.new_chat(a["pid"])
            if name == "clear_chats": ws.clear_chats(a["pid"]); return 200, {"ok": True}
            if name == "open_chat": return 200, ws.open_chat(a["pid"], cid)
            if name == "rename_chat": ws.rename_chat(a["pid"], cid, body.get("title", "")); return 200, {"ok": True}
            if name == "delete_chat": ws.delete_chat(a["pid"], cid); return 200, {"ok": True}
            if name == "send": return 200, ws.send(a["pid"], cid, body.get("mode", ""), body.get("text", ""), body.get("world", ""), body.get("request_id"))
            if name == "memory": return 200, ws.memory(a["pid"])
            if name == "memory_item": return 200, ws.memory_item(a["pid"], fid)
            if name == "reindex": return 200, ws.reindex(a["pid"])
        except WorkspaceError as e:
            return 404 if "Unknown" in str(e) else 400, {"error": str(e)}
        except Exception as e:  # model or tool failure: reported, never faked
            return 500, {"error": f"{type(e).__name__}: {e}"}
    return 404, {"error": "not found"}


def make_handler(ws: Workspace):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode())

        def do_GET(self):
            path = self.path.split("?")[0]  # a query string (?theme=light) must not turn the page into a 404
            if path in STATIC:
                name, ctype = STATIC[path]
                return self._send(200, (HERE / name).read_bytes(), ctype)
            self._json(*route(ws, "GET", path, {}))

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                assert isinstance(body, dict)
            except (ValueError, AssertionError):
                return self._json(400, {"error": "body must be a JSON object"})
            self._json(*route(ws, "POST", self.path, body))
    return Handler


if __name__ == "__main__":
    from backend.source.service.ai.embeddings import Embedder
    root = os.environ.get("HUSH_PRODUCT_DIR") or os.path.join(tempfile.gettempdir(), "hush_product")
    workspace = Workspace(root, embedder=Embedder())
    port = int(os.environ.get("UI_PORT", "8080"))
    print(f"HUSH on http://127.0.0.1:{port}  (product storage {root}; separate from the evaluation database)")
    ThreadingHTTPServer(("127.0.0.1", port), make_handler(workspace)).serve_forever()
