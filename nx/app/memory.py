from collections import defaultdict, deque

class MemoryManager:
    def __init__(self, max_turns=20):
        """
        session_id 단위로 대화 히스토리를 저장
        - max_turns: 최대 대화 유지 개수
        """
        self.sessions = defaultdict(lambda: deque(maxlen=max_turns))

    def append(self, session_id: str, role: str, content: str):
        """대화 기록 추가"""
        self.sessions[session_id].append({"role": role, "content": content})

    def get_history(self, session_id: str):
        """현재 세션의 대화 기록 반환"""
        return list(self.sessions[session_id])

    def reset(self, session_id: str):
        """특정 세션의 기록 초기화"""
        self.sessions[session_id].clear()

    def get(self, session_id):
        return self.sessions.get(session_id, [])

# 인스턴스 전역으로 사용
memory = MemoryManager()