#!/usr/bin/env python3
"""Static pre-flight check for a submission folder. Participants run it before
submitting; organizers run it on every submission.

  python scripts/check_submission.py path/to/submission [--live http://localhost:8000]

Checks
  - docker-compose.yml with a "memory" service and the ${PORT:-8000}:8000 port line
  - Dockerfile, MEMORY.md, prompt_history/ (exported coding-agent sessions)
  - no banned agent-memory libraries in dependency files or imports
  - no hard-coded LLM endpoints/keys (must use OPENAI_BASE_URL / OPENAI_API_KEY)
  --live: hits /health, /reset, /message, /answer, /act on a running service
Exit code 1 if anything fails.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

BANNED = ["mem0", "mem0ai", "supermemory", "letta", "memgpt", "zep", "zep-python", "zep-cloud", "graphiti",
          "graphiti-core", "cognee", "langmem", "memobase", "memoryscope", "motorhead"]
BANNED_IMPORT = re.compile(r"^\s*(from|import)\s+(mem0|supermemory|letta|memgpt|zep_python|zep_cloud|graphiti_core|"
                           r"cognee|langmem|memobase)\b", re.M)
BANNED_CLASSES = re.compile(r"\b(ConversationBufferMemory|ConversationSummaryMemory|VectorStoreRetrieverMemory|"
                            r"ChatMemoryBuffer|ConversationEntityMemory)\b")
HARDCODED = re.compile(r"(sk-or-v1-[A-Za-z0-9]{10,}|sk-ant-[A-Za-z0-9-]{10,}|sk-proj-[A-Za-z0-9]{10,}|"
                       r"api\.anthropic\.com|api\.openai\.com|generativelanguage\.googleapis\.com)")
SKIP = {"node_modules", ".git", ".venv", "venv", "__pycache__", "dist", "build", "prompt_history"}
DEP_FILES = ["requirements.txt", "requirements-dev.txt", "pyproject.toml", "package.json", "Pipfile", "setup.py",
             "uv.lock", "poetry.lock", "package-lock.json"]


def files(root: Path, exts):
    for f in root.rglob("*"):
        if f.is_file() and f.suffix in exts and not any(p in SKIP for p in f.parts) and f.name != "check_submission.py":
            yield f


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    live = sys.argv[sys.argv.index("--live") + 1] if "--live" in sys.argv else None
    fails, warns = [], []

    compose = root / "docker-compose.yml"
    if not compose.exists():
        fails.append("missing docker-compose.yml")
    else:
        c = compose.read_text()
        if not re.search(r"^\s{2}memory:\s*$", c, re.M):
            fails.append("docker-compose.yml has no 'memory' service")
        if "${PORT:-8000}:8000" not in c:
            fails.append("docker-compose.yml must map ports \"${PORT:-8000}:8000\" on the memory service")
    owned = root / "scripts" / ".owned.sha256"
    if owned.exists():
        import hashlib
        for line in owned.read_text().splitlines():
            want, _, fname = line.partition("  ")
            f = root / fname.strip()
            if not f.exists():
                fails.append(f"{fname.strip()} is missing (it is ours; do not delete it)")
            elif hashlib.sha256(f.read_bytes()).hexdigest() != want.strip():
                fails.append(f"{fname.strip()} has been modified. It is provided by the organizers "
                             f"and must stay as shipped - dependencies go in requirements.txt, system "
                             f"packages in apt-packages.txt, extra services in compose.extra.yml")
    for name in ("Dockerfile", "MEMORY.md"):
        if not (root / name).exists():
            fails.append(f"missing {name}")
    ph = root / "prompt_history"
    if not ph.exists() or not any(f.is_file() and f.name not in ("README.md", ".gitkeep") for f in ph.rglob("*")):
        fails.append("missing prompt_history/ (export your coding-agent sessions there)")

    for name in DEP_FILES:
        for f in root.rglob(name):
            if any(p in SKIP for p in f.parts):
                continue
            txt = f.read_text(errors="ignore").lower()
            for b in BANNED:
                if re.search(rf"(^|[\s\"'=\[,]){re.escape(b)}([\s\"'=<>~!\],;:]|$)", txt, re.M):
                    fails.append(f"banned memory library '{b}' in {f.relative_to(root)}")
    for f in files(root, {".py", ".ts", ".js", ".mjs"}):
        txt = f.read_text(errors="ignore")
        if BANNED_IMPORT.search(txt) or re.search(r"(require\(|from\s+)['\"](mem0ai|supermemory|@letta|@getzep|cognee)",
                                                  txt):
            fails.append(f"banned memory library imported in {f.relative_to(root)}")
        if BANNED_CLASSES.search(txt):
            fails.append(f"framework memory class used in {f.relative_to(root)} (build your own memory)")
        m = HARDCODED.search(txt)
        if m:
            fails.append(f"hard-coded key/endpoint '{m.group(0)[:20]}...' in {f.relative_to(root)}; use OPENAI_BASE_URL/OPENAI_API_KEY")

    if live:
        base = live.rstrip("/")

        def post(path, body, timeout=60):
            req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        try:
            urllib.request.urlopen(base + "/health", timeout=10)
            post("/reset", {})
            post("/message", {"message": {"id": "m001", "ts": "2026-11-02T09:00:00+05:30", "type": "chat",
                                          "turns": [{"role": "user", "content": "My locker number is 214."}]}}, 30)
            post("/message", {"message": {"id": "m002", "ts": "2026-11-02T10:00:00+05:30", "type": "event",
                                          "source": "inbound", "op": "add_message",
                                          "data": {"id": "x1", "channel": "#general", "from": "@someone",
                                                   "text": "Lunch is at one.", "ts": "2026-11-02T10:00:00+05:30"}}}, 30)
            out = post("/answer", {"question": "What's my locker number?"}, 30)
            if "answer" not in out:
                fails.append("/answer response has no 'answer'")
            print("live /answer ->", str(out.get("answer"))[:120])
            try:
                out = post("/act", {"request": "What's on my calendar today?", "mcp_url": "http://127.0.0.1:9/mcp",
                                    "tools_url": "http://127.0.0.1:9/tools"}, 100)
                print("live /act ->", str(out.get("reply"))[:120])
            except Exception as e:
                warns.append(f"/act failed against an unreachable world (fine if you handle errors): {e}")
        except Exception as e:
            fails.append(f"live check failed: {e}")

    for w in warns:
        print("WARN ", w)
    for f in fails:
        print("FAIL ", f)
    print("OK - submission looks good" if not fails else f"{len(fails)} problem(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
