# general_chat.py
"""
Dummy version of general_chat module for public use.
This example shows how to structure a simple chatbot pipeline 
with FastAPI + LLM server integration and rule-based fallbacks.
"""

import os, json, requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from memory import memory

# ===== Load default responses (dummy) =====
CFG_PATH = os.path.join(os.path.dirname(__file__), "dummy_general_responses.json")
CFG_PATH = os.path.abspath(CFG_PATH)
GENERAL_RESPONSES = json.load(open(CFG_PATH, encoding="utf-8"))

# ===== Language detection hints (optional) =====
LANG_HINTS = {
    "ko": ["이", "너", "뭐", "왜", "있", "어디", "몇", "누구"],
    "en": ["what", "who", "is", "are", "do", "can", "where", "why"],
    "ja": ["何", "誰", "です", "する", "どこ", "なぜ"],
    "zh": ["什么", "谁", "是", "能", "为什么", "哪里"],
}

def infer_language(text: str) -> str:
    """
    Heuristic language detection based on presence of common keywords.
    Defaults to Korean ("ko") if uncertain.
    """
    scores = {lang: sum(1 for word in hints if word in text) for lang, hints in LANG_HINTS.items()}
    return max(scores, key=scores.get) if scores else "ko"

# ===== Server address: priority LLM_BASE_URL → fallback to VLM_BASE_URL → default local =====
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080")

def _make_session():
    """Create a requests session with retry policy."""
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"]
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

SESSION = _make_session()

def general_chat(user_input: str, model_type="dummy-model", session_id="default") -> dict:
    """
    Response strategy:
    1. Rule-based keyword responses
    2. Fallback responses
    3. Otherwise, query the LLM server (dummy endpoint)
    """
    user_input = user_input.strip()
    if len(user_input) < 2:
        return {"source": "rule", "text": GENERAL_RESPONSES.get("default", "Sorry.")}

    # Retrieve existing conversation history
    try:
        history = memory.get_history(session_id)
    except Exception:
        history = memory.get(session_id) if hasattr(memory, "get") else []
    messages = history + [{"role": "user", "content": user_input}]

    # === Robot-related questions first ===
    robot_response = answer_about_robot(user_input, model_type)
    if robot_response != "__fallback__":
        return {"source": "robot", "text": robot_response}

    # === Rule-based responses ===
    for category, entry in GENERAL_RESPONSES.items():
        if category in ["default", "greetings"]:
            continue
        if any(kw in user_input for kw in entry.get("keywords", [])):
            return {"source": "rule", "text": entry.get("response", GENERAL_RESPONSES.get("default", ""))}

    # === Build instruction-based prompt ===
    system_prompt = (
        "You are a humanoid AI robot that converses naturally with humans."
        "Respond politely and respectfully in the same language as the questioner, within 2 or 3 sentences."
    )
    formatted_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # === Call LLM API ===
    url = f"{LLM_BASE_URL.rstrip('/')}/chat"
    try:
        res = SESSION.post(
            url,
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
            print(f"[LLM API Error] {res.status_code} - {res.text[:200]}")
            llm_response = GENERAL_RESPONSES.get("default", "")
    except Exception as e:
        print(f"[❌ LLM HTTP Error]: {e}")
        llm_response = GENERAL_RESPONSES.get("default", "")

    # === If fallback phrases detected in LLM output, replace with robot response ===
    fallback_phrases = ["모르겠어요", "죄송", "잘 모르겠어요"]
    if any(phrase in llm_response for phrase in fallback_phrases):
        return {"source": "llm-fallback", "text": answer_about_robot(user_input, model_type)}

    return {"source": "llm", "text": llm_response}