"""페르소나 로딩/조회/생성.

  - 기본 페르소나: config/personas.json (igris, custom)
  - 사용자 생성 페르소나: config/user_personas.json (런타임 추가, 영구 저장)
"""
import json
import uuid
import threading
from core.settings import CONFIG_DIR

_PERSONAS_PATH = CONFIG_DIR / "personas.json"
_USER_PATH = CONFIG_DIR / "user_personas.json"

_lock = threading.Lock()

with open(_PERSONAS_PATH, encoding="utf-8") as f:
    _BASE = json.load(f)["personas"]


def _load_user() -> list:
    if _USER_PATH.exists():
        try:
            with open(_USER_PATH, encoding="utf-8") as f:
                return json.load(f).get("personas", [])
        except Exception:
            return []
    return []


def _save_user(personas: list):
    with open(_USER_PATH, "w", encoding="utf-8") as f:
        json.dump({"personas": personas}, f, ensure_ascii=False, indent=2)


def _all() -> list:
    return _BASE + _load_user()


def list_personas() -> list:
    """프론트에 노출할 페르소나 목록 (프롬프트 제외)."""
    return [
        {k: v for k, v in p.items() if k != "system_prompt"}
        for p in _all()
    ]


def get_persona(persona_id: str) -> dict:
    for p in _all():
        if p["id"] == persona_id:
            return p
    return _BASE[0]


def is_user_persona(persona_id: str) -> bool:
    return any(p["id"] == persona_id for p in _load_user())


def create_persona(name: str, system_prompt: str = "", traits: dict = None,
                   voice: str = "ko-KR-InJoonNeural", language: str = "ko") -> dict:
    """사용자 정의 페르소나 생성 + 저장. 생성된 페르소나(메타) 반환."""
    pid = "user_" + uuid.uuid4().hex[:8]
    persona = {
        "id": pid,
        "name": name.strip() or "내 로봇",
        "emoji": "🛠️",
        "description": "사용자가 만든 커스텀 페르소나",
        "system_prompt": system_prompt.strip(),
        "language": language,
        "voice": voice,
        "llm": "general",          # 파인튜닝 안 된 일반 LLM 사용
        "customizable": True,
        "traits": traits or {"friendliness": 3, "knowledge": 3, "empathy": 3, "formality": 3},
        "user_created": True,
        "tags": ["커스텀"],
    }
    with _lock:
        users = _load_user()
        users.append(persona)
        _save_user(users)
    return {k: v for k, v in persona.items() if k != "system_prompt"}


def delete_persona(persona_id: str) -> bool:
    with _lock:
        users = _load_user()
        new = [p for p in users if p["id"] != persona_id]
        if len(new) == len(users):
            return False
        _save_user(new)
    return True


def resolve_system_prompt(persona_id: str, custom_prompt: str = "") -> str:
    p = get_persona(persona_id)
    if persona_id == "custom" and custom_prompt.strip():
        return custom_prompt.strip()
    return p.get("system_prompt", "")
