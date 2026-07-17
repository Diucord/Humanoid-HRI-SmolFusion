"""LLM 클라이언트 — 페르소나별 llama.cpp 서버 라우팅.

  - 파인튜닝 igris (LLM_FINETUNED_URL, 8080): 이그리스 C
  - 일반 Qwen3-1.7B (LLM_GENERAL_URL, 8082): 커스텀 등

대화 라우팅(robot_qa → 룰 → LLM)은 chat.py가 담당하고,
여기서는 순수 LLM 호출만 한다.
"""
import requests
from typing import List, Dict

from core import settings


def _build_messages(system_prompt: str, history: List[Dict], user_msg: str,
                    rag_context: str = "", vision_context: str = "") -> List[Dict]:
    sys = system_prompt.strip()
    if rag_context:
        sys += f"\n\n[관련 지식 베이스]\n{rag_context}"
    if vision_context:
        sys += f"\n\n[현재 카메라 시각 정보]\n{vision_context}"

    messages = []
    if sys:
        messages.append({"role": "system", "content": sys})
    messages.extend(history)
    messages.append({"role": "user", "content": user_msg})
    return messages


def chat_llm(system_prompt: str, history: List[Dict], user_msg: str,
             rag_context: str = "", vision_context: str = "",
             llm_kind: str = "general", temperature: float = None) -> str:
    """llm_kind: 'finetuned'(igris) | 'general'(나머지)."""
    messages = _build_messages(system_prompt, history, user_msg, rag_context, vision_context)

    if llm_kind == "finetuned":
        url, model = settings.LLM_FINETUNED_URL, settings.LLM_FINETUNED_MODEL
        max_tokens = settings.LLM_MAX_TOKENS
        default_temp = settings.LLM_FINETUNED_TEMPERATURE  # 낮은 temp (일관성)
    else:
        url, model = settings.LLM_GENERAL_URL, settings.LLM_GENERAL_MODEL
        max_tokens = settings.LLM_GENERAL_MAX_TOKENS  # CPU라 짧게
        default_temp = settings.LLM_TEMPERATURE

    temp = default_temp if temperature is None else temperature

    try:
        res = requests.post(
            f"{url}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temp,
                # Qwen3 thinking 모드 비활성화 (속도↑, 답변 바로 출력)
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=60,
        )
        res.raise_for_status()
        content = res.json()["choices"][0]["message"]["content"].strip()
        # 혹시 남은 <think>...</think> 블록 제거
        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
    except Exception as e:
        print(f"[LLM] {llm_kind} 호출 오류 ({url}): {e}")
        return ""
