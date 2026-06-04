"""Qwen3-VL 기반 시각 분석.

기존 SmolVLM(개별 프롬프트 × N회 HTTP 호출)을 Qwen3-VL 단일 호출로 교체.
한 번의 추론으로 나이/성별/표정/장면을 JSON으로 받는다.
"""
import json
import re
import threading
from typing import Optional
from PIL import Image

from core import settings

_VLM_PROMPT = (
    "Analyze the person in this image. Respond with ONLY a JSON object, no extra text:\n"
    '{"has_person": true/false, '
    '"age_group": "child|teenager|young adult|middle aged|elderly", '
    '"gender": "male|female|unknown", '
    '"is_smiling": true/false, '
    '"scene": "one short sentence describing the scene and the person\'s state"}'
)


class QwenVLAnalyzer:
    """지연 로딩 + 스레드 안전 싱글톤."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None
        self._lock = threading.Lock()
        self._load_failed = False

    def _ensure_loaded(self):
        if self._model is not None or self._load_failed:
            return
        with self._lock:
            if self._model is not None or self._load_failed:
                return
            try:
                import torch
                from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

                self._device = settings.resolve_device()
                dtype = torch.float16 if self._device == "cuda" else torch.float32

                load_kwargs = {"torch_dtype": dtype, "device_map": self._device}

                # 4-bit / 8-bit 양자화 (4B 변형을 8GB에 올릴 때)
                if settings.VLM_QUANTIZE in ("4bit", "8bit") and self._device == "cuda":
                    from transformers import BitsAndBytesConfig
                    if settings.VLM_QUANTIZE == "4bit":
                        load_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                        )
                    else:
                        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                    load_kwargs.pop("torch_dtype", None)

                print(f"[QwenVL] loading {settings.VLM_MODEL_ID} on {self._device} "
                      f"(quantize={settings.VLM_QUANTIZE}) ...")
                self._processor = AutoProcessor.from_pretrained(settings.VLM_MODEL_ID)
                self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                    settings.VLM_MODEL_ID, **load_kwargs
                )
                print("[QwenVL] loaded.")
            except Exception as e:
                print(f"[QwenVL] load failed, falling back to rule-based: {e}")
                self._load_failed = True

    def available(self) -> bool:
        self._ensure_loaded()
        return self._model is not None

    def analyze(self, image: Image.Image) -> dict:
        """이미지 → {has_person, age_group, gender, is_smiling, scene}"""
        self._ensure_loaded()
        if self._model is None:
            return _rule_based_fallback(image)

        try:
            import torch
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _VLM_PROMPT},
                ],
            }]
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs.pop("token_type_ids", None)  # Qwen3-VL generate 호환
            inputs = inputs.to(self._device)

            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=settings.VLM_MAX_TOKENS, do_sample=False)
            trimmed = out[:, inputs["input_ids"].shape[1]:]
            text = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            return _parse_vlm_json(text)
        except Exception as e:
            print(f"[QwenVL] inference error: {e}")
            return _rule_based_fallback(image)


def _parse_vlm_json(text: str) -> dict:
    """모델 출력에서 JSON 추출 (코드펜스/잡텍스트 방어)."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {"has_person": False, "scene": text.strip()[:120]}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"has_person": False, "scene": text.strip()[:120]}

    return {
        "has_person": bool(data.get("has_person", False)),
        "age_group": _norm_age(data.get("age_group", "unknown")),
        "gender": data.get("gender") if data.get("gender") in ("male", "female") else "unknown",
        "is_smiling": bool(data.get("is_smiling", False)),
        "scene": str(data.get("scene", "")).strip(),
    }


def _norm_age(s: str) -> str:
    valid = {"child", "teenager", "young adult", "middle aged", "elderly"}
    s = (s or "").strip().lower()
    return s if s in valid else "unknown"


def _rule_based_fallback(image: Image.Image) -> dict:
    """VLM 로드 실패 시 최소 동작 (얼굴 유무는 face_recognition이 별도 판단)."""
    return {
        "has_person": True,
        "age_group": "unknown",
        "gender": "unknown",
        "is_smiling": False,
        "scene": "환경을 분석할 수 없습니다 (VLM 비활성).",
    }


# 싱글톤
analyzer = QwenVLAnalyzer()
