"""페르소나 로딩/조회."""
import json
from core.settings import CONFIG_DIR

_PERSONAS_PATH = CONFIG_DIR / "personas.json"

with open(_PERSONAS_PATH, encoding="utf-8") as f:
    _PERSONAS = json.load(f)["personas"]

_BY_ID = {p["id"]: p for p in _PERSONAS}


def list_personas() -> list:
    """프론트에 노출할 페르소나 목록 (프롬프트 제외)."""
    return [
        {k: v for k, v in p.items() if k != "system_prompt"}
        for p in _PERSONAS
    ]


def get_persona(persona_id: str) -> dict:
    return _BY_ID.get(persona_id, _PERSONAS[0])


def resolve_system_prompt(persona_id: str, custom_prompt: str = "") -> str:
    p = get_persona(persona_id)
    if persona_id == "custom" and custom_prompt.strip():
        return custom_prompt.strip()
    return p["system_prompt"]
