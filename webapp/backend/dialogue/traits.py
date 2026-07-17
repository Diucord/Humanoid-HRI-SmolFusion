"""커스텀 페르소나 슬라이더(1~5) → 시스템 프롬프트 변환.

항목: friendliness(친절도), knowledge(지식 수준), empathy(공감 능력), formality(말투/격식)
"""
from typing import Dict

# 각 항목의 레벨별(1~5) 문장 조각
_FRIENDLINESS = {
    1: "사무적이고 간결하게, 군더더기 없이 대답합니다.",
    2: "차분하고 담백한 어조로 대답합니다.",
    3: "친근하고 부드러운 어조로 대답합니다.",
    4: "다정하고 따뜻하게, 상대를 배려하며 대답합니다.",
    5: "매우 다정하고 살갑게, 상대가 편안함을 느끼도록 대답합니다.",
}
_KNOWLEDGE = {
    1: "초등학생도 이해할 만큼 쉽고 단순하게 설명합니다.",
    2: "쉬운 용어로 핵심만 간단히 설명합니다.",
    3: "적당한 수준의 정보와 함께 명확하게 설명합니다.",
    4: "전문적인 내용을 근거와 함께 자세히 설명합니다.",
    5: "해당 분야 전문가 수준으로 깊이 있고 정밀하게 설명합니다.",
}
_EMPATHY = {
    1: "감정 표현은 자제하고 사실 위주로 답합니다.",
    2: "상대의 감정을 가볍게 인지하며 답합니다.",
    3: "상대의 감정을 이해하고 공감하며 답합니다.",
    4: "상대의 감정을 먼저 헤아리고 깊이 공감하며 답합니다.",
    5: "상대의 감정에 진심으로 공감하고 정서적으로 지지하며 답합니다.",
}
_FORMALITY = {
    1: "친구처럼 편한 반말로 대화합니다.",
    2: "가벼운 존댓말로 친근하게 대화합니다.",
    3: "정중한 존댓말로 대화합니다.",
    4: "격식 있는 존댓말로 예의 바르게 대화합니다.",
    5: "매우 격식 있고 정중한 경어체로 대화합니다.",
}


def _clamp(v) -> int:
    try:
        return max(1, min(5, int(v)))
    except (ValueError, TypeError):
        return 3


def traits_to_prompt(traits: Dict, base_prompt: str = "") -> str:
    """슬라이더 값 dict → 시스템 프롬프트."""
    f = _clamp(traits.get("friendliness", 3))
    k = _clamp(traits.get("knowledge", 3))
    e = _clamp(traits.get("empathy", 3))
    fm = _clamp(traits.get("formality", 3))

    lines = [
        "당신은 사람과 대화하는 AI 어시스턴트입니다.",
        _FRIENDLINESS[f],
        _KNOWLEDGE[k],
        _EMPATHY[e],
        _FORMALITY[fm],
        "질문자와 같은 언어로, 2~3문장 이내로 간결하게 대답합니다.",
    ]
    prompt = " ".join(lines)

    if base_prompt.strip():
        prompt = base_prompt.strip() + "\n\n" + prompt
    return prompt
