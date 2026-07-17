"""새 사람 감지 시 연령대/성별 기반 인사말 생성.

기존 nx/app/test.py 의 인사 로직을 그대로 옮김:
  greetings[ko][age_group] → (성별별 dict이면 gender로 한번 더)
"""
import json
from core.settings import CONFIG_DIR

with open(CONFIG_DIR / "general_responses.json", encoding="utf-8") as f:
    _RESPONSES = json.load(f)


def make_greeting(age_group: str = "young adult", gender: str = "unknown",
                  lang: str = "ko") -> str:
    greet_map = _RESPONSES.get("greetings", {}).get(lang, {})
    default = _RESPONSES.get("default", "안녕하세요!")

    msg = greet_map.get(age_group, default)
    if isinstance(msg, dict):
        msg = msg.get(gender, msg.get("unknown", default))
    return msg
