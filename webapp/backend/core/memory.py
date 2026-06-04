"""세션별 대화 기록 (인메모리, 자동 프루닝)."""
from collections import defaultdict, deque
from typing import List, Dict

MAX_TURNS = 20  # 세션당 최대 turn 수


class Memory:
    def __init__(self):
        self._store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_TURNS))

    def append(self, session_id: str, role: str, content: str):
        self._store[session_id].append({"role": role, "content": content})

    def get(self, session_id: str) -> List[Dict]:
        return list(self._store[session_id])

    def get_history(self, session_id: str) -> List[Dict]:
        return list(self._store[session_id])

    def reset(self, session_id: str):
        self._store[session_id] = deque(maxlen=MAX_TURNS)

    def clear_all(self):
        self._store.clear()


memory = Memory()
