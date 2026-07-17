"""Qwen3-VL 기반 시각 분석 (llama.cpp HTTP).

llama.cpp 서버(OpenAI 호환, --mmproj로 비전 지원)에 이미지를 base64로 전송.
한 번의 추론으로 나이/성별/표정/장면을 JSON으로 받는다.

기존 nx/app/smolvlm_infer.py의 HTTP 패턴을 따르되, Qwen3-VL-4B GGUF를 사용한다.
"""
import io
import json
import re
import base64
import requests
from PIL import Image

from core import settings

_VLM_PROMPT = (
    "Analyze the people in this image. Respond with ONLY a JSON object, no extra text:\n"
    '{"has_person": true/false, '
    '"person_count": <integer number of people>, '
    '"age_group": "child|teenager|young adult|middle aged|elderly", '
    '"gender": "male|female|unknown", '
    '"is_smiling": true/false, '
    '"scene": "one short sentence describing the scene and the person\'s state"}\n'
    "age_group/gender refer to the most prominent person."
)


class QwenVLAnalyzer:
    """llama.cpp VLM 서버 HTTP 클라이언트."""

    def __init__(self):
        self._url = settings.VLM_URL
        self._model = settings.VLM_MODEL

    def available(self) -> bool:
        try:
            r = requests.get(f"{self._url}/health", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def describe_appearance(self, image: Image.Image) -> str:
        """외모/패션 묘사 (한국어). '나 어때 보여?' 류 질문 응답용."""
        try:
            data_url = _image_to_data_url(image)
            prompt = (
                "이 사람의 외모, 표정, 분위기를 따뜻하고 긍정적으로 한국어 2문장 이내로 묘사해줘. "
                "로봇이 사람에게 말하듯 친근하게."
            )
            payload = {
                "model": self._model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                "max_tokens": 100,
                "temperature": 0.6,
            }
            r = requests.post(f"{self._url}/v1/chat/completions", json=payload, timeout=60)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[QwenVL] 외모 묘사 오류: {e}")
            return "지금은 잘 보이지 않네요. 다시 한 번 봐주시겠어요?"

    def analyze(self, image: Image.Image) -> dict:
        """이미지 → {has_person, age_group, gender, is_smiling, scene}"""
        try:
            data_url = _image_to_data_url(image)
            payload = {
                "model": self._model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _VLM_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                "max_tokens": settings.VLM_MAX_TOKENS,
                "temperature": 0.2,
            }
            r = requests.post(
                f"{self._url}/v1/chat/completions", json=payload, timeout=60
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            return _parse_vlm_json(text)
        except Exception as e:
            print(f"[QwenVL] inference error (VLM 서버 미기동?): {e}")
            return _rule_based_fallback()


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=70)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return "data:image/jpeg;base64," + b64


def _parse_vlm_json(text: str) -> dict:
    """모델 출력에서 JSON 추출 (코드펜스/잡텍스트 방어)."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {"has_person": False, "scene": text.strip()[:120]}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"has_person": False, "scene": text.strip()[:120]}

    try:
        person_count = int(data.get("person_count", 0))
    except (ValueError, TypeError):
        person_count = 0

    return {
        "has_person": bool(data.get("has_person", False)),
        "person_count": max(person_count, 0),
        "age_group": _norm_age(data.get("age_group", "unknown")),
        "gender": data.get("gender") if data.get("gender") in ("male", "female") else "unknown",
        "is_smiling": bool(data.get("is_smiling", False)),
        "scene": str(data.get("scene", "")).strip(),
    }


def _norm_age(s: str) -> str:
    valid = {"child", "teenager", "young adult", "middle aged", "elderly"}
    s = (s or "").strip().lower()
    return s if s in valid else "unknown"


def _rule_based_fallback() -> dict:
    """VLM 서버 미기동 시 최소 동작 (얼굴 유무는 face_recognition이 별도 판단)."""
    return {
        "has_person": True,
        "person_count": 1,
        "age_group": "unknown",
        "gender": "unknown",
        "is_smiling": False,
        "scene": "환경을 분석할 수 없습니다 (VLM 서버 미기동).",
    }


# 싱글톤
analyzer = QwenVLAnalyzer()
