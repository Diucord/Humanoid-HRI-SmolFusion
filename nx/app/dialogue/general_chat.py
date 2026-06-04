"""
general_chat.py

이 파일은 외부 클라이언트 또는 다른 시스템에서
포트 기반 FastAPI 서버에 HTTP 요청을 보내 LLM 응답을 받는 모듈입니다.

서버에 POST /chat 으로 요청을 보내며, fallback이나 로봇 QA는 로컬 처리합니다.
"""

import json
import requests
from dialogue.robot_qa import answer_about_robot
from memory import memory

# 기본 응답 로딩
GENERAL_RESPONSES = json.load(open("config/general_responses.json", encoding="utf-8"))

# 언어 추정 힌트 (필요 시 활용)
LANG_HINTS = {
    "ko": ["이", "너", "뭐", "왜", "있", "어디", "몇", "누구"],
    "en": ["what", "who", "is", "are", "do", "can", "where", "why"],
    "ja": ["何", "誰", "です", "する", "どこ", "なぜ"],
    "zh": ["什么", "谁", "是", "能", "为什么", "哪里"],
}


def infer_language(text: str) -> str:
    scores = {
        lang: sum(1 for word in hints if word in text)
        for lang, hints in LANG_HINTS.items()
    }
    return max(scores, key=scores.get) if scores else "ko"


def general_chat(user_input: str, model_type="igris-C", session_id="default") -> dict:
    """
    로봇 QA → 키워드 룰 → 로봇 비교 응답 → LLM 순서로 응답.
    fallback 감지 시 로봇 응답으로 재전환.
    """
    user_input = user_input.strip()
    if len(user_input) < 2:
        return {"source": "rule", "text": GENERAL_RESPONSES["default"]}

    # 기존 세션 대화 기록
    history = memory.get_history(session_id)
    messages = history + [{"role": "user", "content": user_input}]

    # === 로봇 관련 질문 먼저 응답 ===
    robot_response = answer_about_robot(user_input, model_type)
    if robot_response != "__fallback__":
        return {"source": "robot", "text": robot_response}

    # === 룰 기반 응답 체크 ===
    for category, entry in GENERAL_RESPONSES.items():
        if category in ["default", "greetings"]:
            continue
        if any(kw in user_input for kw in entry.get("keywords", [])):
            return {"source": "rule", "text": entry["response"]}

    # === instruction 기반 프롬프트 구성 ===
    system_prompt = "당신은 사람과 자연스럽게 대화하는 로브로스의 인공지능 휴머노이드 로봇 이그리스 C입니다. 질문자와 동일한 언어를 사용해 2~3문장 이내로 공손하고 정중하게 대답하세요."
    formatted_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # === LLM 호출 ===
    try:
        res = requests.post(
            "http://localhost:8000/chat",
            json={
                "prompt": formatted_prompt,
                "persona": model_type,
                "session_id": session_id
            },
            timeout=15
        )
        if res.status_code == 200:
            llm_response = res.json().get("response", "")
        else:
            llm_response = GENERAL_RESPONSES["default"]
    except Exception as e:
        print(f"[❌ LLM HTTP 오류]: {e}")
        llm_response = GENERAL_RESPONSES["default"]

    # fallback 문구 감지 시 → 로봇 응답으로 대체
    fallback_phrases = ["모르겠어요", "죄송", "잘 모르겠어요"]
    if any(phrase in llm_response for phrase in fallback_phrases):
        return {"source": "llm-fallback", "text": answer_about_robot(user_input, model_type)}

    return {"source": "llm", "text": llm_response}