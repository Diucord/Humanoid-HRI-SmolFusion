# memory.py
from collections import defaultdict, deque

class MemoryManager:
    def __init__(self, max_turns=20):
        """
        Stores conversation history per session_id.
        
        Args:
            max_turns (int): Maximum number of conversation turns to retain per session.
                            When the limit is reached, the oldest turns are dropped.
        """
        self.sessions = defaultdict(lambda: deque(maxlen=max_turns))

    def append(self, session_id: str, role: str, content: str):
        """Append a new message to the conversation history."""
        self.sessions[session_id].append({"role": role, "content": content})

    def get_history(self, session_id: str):
        """Return the full conversation history for the given session."""
        return list(self.sessions[session_id])

    def reset(self, session_id: str):
        """Clear the conversation history for the given session."""
        self.sessions[session_id].clear()

    def get(self, session_id):
        """Get the conversation history deque directly (or empty list if not found)."""
        return self.sessions.get(session_id, [])

# Global instance for use across the app
memory = MemoryManager()