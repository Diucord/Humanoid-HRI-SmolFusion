"""얼굴 임베딩 + 매칭 + Qwen3-VL 분석 통합.

기존 nx/app/vlm/analyze_person.py 의 얼굴 매칭 로직을 유지하고,
나이/성별/표정 판단은 Qwen3-VL(vision/vlm.py)로 위임한다.
"""
import io
from typing import Optional
import numpy as np
from PIL import Image

from core import settings
from vision.vlm import analyzer

try:
    import face_recognition
    _FACE_OK = True
except Exception as e:  # dlib 미설치 등
    print(f"[analyze] face_recognition 사용 불가: {e}")
    _FACE_OK = False


def get_face_embedding(image: Image.Image) -> Optional[list]:
    """PIL 이미지 → 128D 얼굴 임베딩 (없으면 None)."""
    if not _FACE_OK:
        return None
    try:
        arr = np.array(image.convert("RGB"))
        encodings = face_recognition.face_encodings(arr)
        return encodings[0].tolist() if encodings else None
    except Exception as e:
        print(f"[analyze] 임베딩 추출 오류: {e}")
        return None


def is_same_person(new_vec, old_vec, threshold: float = None) -> bool:
    """코사인 유사도 기반 동일인 판단 (기존 로직과 동일)."""
    if threshold is None:
        threshold = settings.FACE_SIMILARITY_THRESHOLD
    if new_vec is None or old_vec is None:
        return False
    from scipy.spatial.distance import cosine
    sim = 1 - cosine(np.array(new_vec), np.array(old_vec))
    return sim > threshold


def analyze_person(image: Image.Image) -> dict:
    """통합 분석. 반환 형식은 기존 코드와 호환:
    {has_person, age_group, gender, is_smiling, scene, face_vector}
    """
    face_vector = get_face_embedding(image)

    # VLM 분석
    vlm_result = analyzer.analyze(image)

    # 얼굴 임베딩이 있으면 사람 존재로 확정 (face_recognition이 더 신뢰도 높음)
    has_person = vlm_result.get("has_person", False) or (face_vector is not None)

    return {
        "has_person": has_person,
        "person_count": vlm_result.get("person_count", 1 if has_person else 0),
        "age_group": vlm_result.get("age_group", "unknown"),
        "gender": vlm_result.get("gender", "unknown"),
        "is_smiling": vlm_result.get("is_smiling", False),
        "scene": vlm_result.get("scene", ""),
        "face_detected": face_vector is not None,
        "face_vector": face_vector,
    }


def decode_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")
