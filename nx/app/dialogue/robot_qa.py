import json

# 데이터 로딩
with open("config/robot_info.json", encoding="utf-8") as f:
    ROBOT_INFO = json.load(f)

with open("config/general_responses.json", encoding="utf-8") as f:
    GENERAL_RESPONSES = json.load(f)


def classify_robot_question(text: str) -> str:
    for category, entry in GENERAL_RESPONSES.items():
        if category in ["default", "greetings"]:
            continue
        keywords = entry.get("keywords", [])
        # 너무 짧은 키워드나 범용어 무시
        for kw in keywords:
            if len(kw) >= 2 and kw in text:
                return category
    return "fallback"


def answer_about_robot(user_input: str, persona="igris-C") -> str:
    model = ROBOT_INFO["models"].get(persona)
    if not model:
        return "__fallback__"

    question_type = classify_robot_question(user_input)

    # fallback 아니면 응답 반환, fallback이면 넘기기
    if question_type != "fallback" and question_type in GENERAL_RESPONSES:
        return GENERAL_RESPONSES[question_type]["response"]

    return "__fallback__"
