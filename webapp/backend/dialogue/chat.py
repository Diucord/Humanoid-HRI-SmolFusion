"""대화 라우팅.

igris 페르소나: 기존 4단계(robot_qa → 룰 → LLM) + RAG/Vision 유지.
그 외 페르소나: 페르소나 프롬프트 + RAG + Vision → LLM.
"""
import json
from typing import Optional

from core import settings
from core.memory import memory
from dialogue.personas import resolve_system_prompt, get_persona
from dialogue.llm import chat_llm
from dialogue.traits import traits_to_prompt
from rag.store import get_store

# 기존 이그리스 룰베이스 응답/로봇정보 재활용
with open(settings.CONFIG_DIR / "general_responses.json", encoding="utf-8") as f:
    GENERAL_RESPONSES = json.load(f)
with open(settings.CONFIG_DIR / "robot_info.json", encoding="utf-8") as f:
    ROBOT_INFO = json.load(f)

_FALLBACK_PHRASES = ["모르겠어요", "죄송", "잘 모르겠어요"]


def _classify_robot_question(text: str) -> str:
    # 가장 긴 키워드가 매칭되는 카테고리를 우선 (예: '이름 뜻' > '이름')
    best_cat = "fallback"
    best_len = 0
    for category, entry in GENERAL_RESPONSES.items():
        if category in ("default", "greetings"):
            continue
        for kw in entry.get("keywords", []):
            if len(kw) >= 2 and kw in text and len(kw) > best_len:
                best_cat = category
                best_len = len(kw)
    return best_cat


def _answer_about_robot(text: str, persona_id: str) -> Optional[str]:
    """igris 모델 정보 기반 룰 응답. 매치 없으면 None."""
    if persona_id != "igris":
        return None
    if "igris-C" not in ROBOT_INFO.get("models", {}):
        return None
    qtype = _classify_robot_question(text)
    if qtype != "fallback" and qtype in GENERAL_RESPONSES:
        return GENERAL_RESPONSES[qtype]["response"]
    return None


def respond(
    user_msg: str,
    session_id: str,
    persona_id: str = "igris",
    custom_prompt: str = "",
    vision_context: str = "",
    use_rag: bool = True,
    traits: dict = None,
) -> dict:
    user_msg = user_msg.strip()
    if len(user_msg) < 1:
        return {"source": "rule", "text": GENERAL_RESPONSES["default"]}

    is_igris = persona_id == "igris"
    llm_kind = "finetuned" if is_igris else "general"

    persona = get_persona(persona_id)

    # 시스템 프롬프트 결정
    if is_igris:
        system_prompt = persona.get("system_prompt", "")
    elif persona.get("user_created"):
        # 저장된 사용자 페르소나: 저장된 traits + 프롬프트 사용
        saved_traits = persona.get("traits", {})
        system_prompt = traits_to_prompt(saved_traits, base_prompt=persona.get("system_prompt", ""))
    else:
        # 즉석 custom: 요청으로 받은 traits + 프롬프트
        system_prompt = traits_to_prompt(traits or {}, base_prompt=custom_prompt)

    history = memory.get_history(session_id)

    # === 1) RAG 컨텍스트 (업로드 문서가 있으면 최우선) ===
    rag_context = ""
    if use_rag:
        store = get_store(persona_id)
        if store.count() > 0:
            rag_context = store.query(user_msg, top_k=settings.RAG_TOP_K)

    # === 2) igris 전용 로봇 QA 룰 (단, RAG 문서가 잡히면 건너뜀) ===
    if is_igris and not rag_context:
        robot_ans = _answer_about_robot(user_msg, persona_id)
        if robot_ans:
            _log(session_id, user_msg, robot_ans)
            return {"source": "robot", "text": robot_ans}

    # === 3) LLM (페르소나에 따라 finetuned/general 라우팅) ===
    reply = chat_llm(
        system_prompt=system_prompt,
        history=history,
        user_msg=user_msg,
        rag_context=rag_context,
        vision_context=vision_context,
        llm_kind=llm_kind,
    )

    # LLM 실패/fallback 문구 처리
    if not reply or any(p in reply for p in _FALLBACK_PHRASES):
        if is_igris:
            reply = reply or GENERAL_RESPONSES["default"]
            _log(session_id, user_msg, reply)
            return {"source": "llm-fallback", "text": reply}
        reply = reply or "죄송해요, 다시 한 번 말씀해 주시겠어요?"

    _log(session_id, user_msg, reply)
    return {"source": "llm", "text": reply, "rag_used": bool(rag_context), "llm": llm_kind}


def _log(session_id: str, user_msg: str, reply: str):
    memory.append(session_id, "user", user_msg)
    memory.append(session_id, "assistant", reply)
