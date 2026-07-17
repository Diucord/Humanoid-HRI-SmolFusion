# llm_chat.py
import os, requests
from dataclasses import dataclass
from app.config_loader import load_settings

settings = load_settings()
LLM_URL   = settings["llm"].get("url") or os.environ.get("LLM_URL", "http://127.0.0.1:8080")
MODEL_NAME= settings["llm"]["model"]
TEMP      = settings["llm"]["temperature"]
MAXTOK    = settings["llm"]["max_tokens"]

@dataclass
class ChatConfig:
    prompt: str
    session_id: str = "default"
    max_tokens: int = MAXTOK
    persona: str = "igris-C"
    language: str = "ko"

def generate_response_llm(config: ChatConfig) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role":"system","content": f"You are {config.persona}. Reply in {config.language}. Be concise."},
            {"role":"user","content": config.prompt}
        ],
        "max_tokens": int(config.max_tokens),
        "temperature": float(TEMP),
    }
    r = requests.post(f"{LLM_URL}/v1/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
