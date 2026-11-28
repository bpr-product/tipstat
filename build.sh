#!/bin/sh
# Optional and best effort: the local semantic-search model (RULES.md section 9). It must never fail the build; if the
# library or model cannot be fetched, the service starts without semantic search and uses lexical retrieval only.
if pip install --no-cache-dir fastembed >/dev/null 2>&1; then
  mkdir -p /opt/models
  python - <<'PYEOF' || echo "build.sh: model download failed; semantic search disabled"
from fastembed import TextEmbedding
TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir="/opt/models")
print("build.sh: embedding model cached in /opt/models")
PYEOF
else
  echo "build.sh: fastembed not installed; semantic search disabled"
fi
exit 0
