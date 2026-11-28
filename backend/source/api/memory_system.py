"""The contract boundary: the four methods server.py calls. Validate, delegate to services, return the contract shape."""
from backend.source.dal.db_session import Database
from backend.source.dal.repositories.memory_repository import MemoryRepository
from backend.source.dal.repositories.search_repository import SearchRepository
import json
import time

from backend.source.service.ai.embeddings import Embedder
from backend.source.service.ai.provider import chat_with_tools, complete
from backend.source.service.memory.acting import act as run_act
from backend.source.service.memory.ingestion import ingest
from backend.source.service.memory.retrieval import answer_question
from backend.source.utils import config, metrics


class MemorySystem:
    def __init__(self, llm_fn=None, db_path: str | None = None, chat_fn=None, embedder=None) -> None:
        self.llm = llm_fn or complete
        self.embedder = embedder if embedder is not None else Embedder()
        getattr(self.embedder, "available", None)  # load the model at startup, not during the first message
        self.chat = chat_fn or (lambda messages, tools, tool_choice, timeout: chat_with_tools(
            messages, tools, tool_choice=tool_choice, timeout=timeout))
        self.database = Database(db_path or config.db_path())
        self.repo = MemoryRepository(self.database)
        self.search = SearchRepository(self.database, self.repo)

    def _bounded(self, seconds: float):
        """The model callable for one request: every call gets what is left of a shared deadline, so two sequential calls
        cannot add up past the official 30 s. When too little is left the call fails fast and callers fall back."""
        end = time.monotonic() + seconds

        def call(messages, **kw):
            left = end - time.monotonic()
            if left < 2:
                raise TimeoutError("request deadline reached")
            kw["timeout"] = min(kw.get("timeout", 25), int(left))
            started = time.monotonic()
            try:
                out = self.llm(messages, **kw)
            except Exception:
                metrics.record_call(messages, "", time.monotonic() - started, ok=False)
                raise
            metrics.record_call(messages, out, time.monotonic() - started)
            return out
        return call

    def _metered_chat(self, messages, tools, tool_choice, timeout):
        started = time.monotonic()
        try:
            msg = self.chat(messages, tools, tool_choice, timeout)
        except Exception:
            metrics.record_call(messages, "", time.monotonic() - started, ok=False, extra=json.dumps(tools))
            raise
        metrics.record_call(messages, (msg.get("content") or "") + json.dumps(msg.get("tool_calls") or []), time.monotonic() - started,
                            extra=json.dumps(tools))
        return msg

    def close(self) -> None:
        self.database.db.close()

    def reset(self) -> None:
        self.database.reset()
        self.repo.clear_tombstones()

    def add_message(self, message: dict) -> None:
        ingest(self.repo, self.search, self._bounded(27), self.embedder, message)

    def answer(self, question: str, hint: str | None = None) -> dict:
        return answer_question(self.repo, self.search, self._bounded(27), self.embedder, question, hint)

    def act(self, request: str, world: dict) -> dict:
        return run_act(self.repo, self.search, self._bounded(20), self._metered_chat, self.embedder, request, world)
