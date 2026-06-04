"""LLM 클라이언트.

기본: llama.cpp 서버(OpenAI 호환, 파인튜닝 Qwen3 GGUF).
폴백: 로컬 transformers 머지 모델.

기존 general_chat.py의 4단계 라우팅(robot_qa → 룰 → LLM)은
chat.py에서 조립하고, 여기서는 순수 LLM 호출만 담당한다.
"""
import requests
from typing import List, Dict

from core import settings

# transformers 폴백용 (지연 로딩)
_hf_model = None
_hf_tokenizer = None


def _build_messages(system_prompt: str, history: List[Dict], user_msg: str,
                    rag_context: str = "", vision_context: str = "") -> List[Dict]:
    sys = system_prompt.strip()
    if rag_context:
        sys += f"\n\n[관련 지식 베이스]\n{rag_context}"
    if vision_context:
        sys += f"\n\n[현재 카메라 시각 정보]\n{vision_context}"

    messages = [{"role": "system", "content": sys}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_msg})
    return messages


def chat_llm(system_prompt: str, history: List[Dict], user_msg: str,
             rag_context: str = "", vision_context: str = "") -> str:
    messages = _build_messages(system_prompt, history, user_msg, rag_context, vision_context)
    if settings.LLM_BACKEND == "transformers":
        return _chat_transformers(messages)
    return _chat_llamacpp(messages)


def _chat_llamacpp(messages: List[Dict]) -> str:
    """llama.cpp OpenAI 호환 엔드포인트."""
    try:
        res = requests.post(
            f"{settings.LLM_URL}/v1/chat/completions",
            json={
                "model": settings.LLM_MODEL,
                "messages": messages,
                "max_tokens": settings.LLM_MAX_TOKENS,
                "temperature": settings.LLM_TEMPERATURE,
            },
            timeout=60,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[LLM] llama.cpp 호출 오류: {e}")
        return ""


def _ensure_hf():
    global _hf_model, _hf_tokenizer
    if _hf_model is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = settings.resolve_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[LLM] transformers 로딩: {settings.LLM_LOCAL_PATH} on {device}")
    _hf_tokenizer = AutoTokenizer.from_pretrained(settings.LLM_LOCAL_PATH)
    _hf_model = AutoModelForCausalLM.from_pretrained(
        settings.LLM_LOCAL_PATH, torch_dtype=dtype, device_map=device
    )


def _chat_transformers(messages: List[Dict]) -> str:
    try:
        import torch
        _ensure_hf()
        prompt = _hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _hf_tokenizer(prompt, return_tensors="pt").to(_hf_model.device)
        with torch.no_grad():
            out = _hf_model.generate(
                **inputs,
                max_new_tokens=settings.LLM_MAX_TOKENS,
                temperature=settings.LLM_TEMPERATURE,
                do_sample=True,
            )
        text = _hf_tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return text.strip()
    except Exception as e:
        print(f"[LLM] transformers 호출 오류: {e}")
        return ""
