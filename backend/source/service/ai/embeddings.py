"""Local semantic embeddings: fastembed (ONNX runtime) with a small sentence model, run inside the container.

No network is used at runtime: the model is downloaded into /opt/models by build.sh at image build time and loaded with
local_files_only. If the library or model is missing the embedder reports unavailable (embed returns None) and the
memory system falls back to lexical search; nothing else changes. These are not LLM calls (RULES section 5).
"""
import math
import os
import sys

from backend.source.utils import config


class Embedder:
    def __init__(self) -> None:
        self._model, self._tried, self.error = None, False, ""

    def _load(self):
        if self._tried:
            return self._model
        self._tried = True
        if config.EMBEDDINGS == "off":
            self.error = "switched off (MEMORY_EMBEDDINGS=off)"
            return None
        try:
            from fastembed import TextEmbedding
            kwargs = {"cache_dir": config.EMBED_CACHE} if config.EMBED_CACHE else {}
            if config.EMBED_CACHE and os.path.isdir(config.EMBED_CACHE):
                kwargs["local_files_only"] = True
            self._model = TextEmbedding(config.EMBED_MODEL, **kwargs)
        except Exception as e:  # missing library, missing model, no network: lexical search only
            print(f"semantic search unavailable: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
            self._model, self.error = None, f"{type(e).__name__}: {str(e)[:120]}"
        return self._model

    @property
    def available(self) -> bool:
        return self._load() is not None

    def embed(self, texts: list):
        """Unit-length vectors for the texts, or None if semantic search is unavailable or the call failed."""
        model = self._load()
        if model is None or not texts:
            return None
        try:
            out = []
            for v in model.embed([t[:1500] for t in texts]):
                v = [float(x) for x in v]
                n = math.sqrt(sum(x * x for x in v)) or 1.0
                out.append([x / n for x in v])
            return out
        except Exception as e:
            print(f"embedding failed: {type(e).__name__}", file=sys.stderr)
            return None

    def embed_one(self, text: str):
        r = self.embed([text])
        return r[0] if r else None
