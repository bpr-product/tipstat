"""Official entry point: server.py loads `memory:MemorySystem`. The implementation lives in backend/source/
(api -> service -> dal, plus schemas and utils); see BUILD_STATE.md for the layout."""
from backend.source.api.memory_system import MemorySystem

__all__ = ["MemorySystem"]
