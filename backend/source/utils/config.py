"""Runtime configuration from the environment (server.py loads .env before importing us)."""
import os
import tempfile

BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("MODEL", "deepseek/deepseek-v4.1-flash")


def db_path() -> str:
    return os.environ.get("MEMORY_DB") or os.path.join(tempfile.gettempdir(), "memory.db")

# Local semantic embeddings (optional). "off" disables them; otherwise they are used only if the model can be loaded.
EMBEDDINGS = os.environ.get("MEMORY_EMBEDDINGS", "on").lower()
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_CACHE = os.environ.get("EMBED_CACHE") or ("/opt/models" if os.path.isdir("/opt/models") else None)
SEMANTIC_MIN = float(os.environ.get("SEMANTIC_MIN", "0.5"))  # cosine floor for a semantic candidate
