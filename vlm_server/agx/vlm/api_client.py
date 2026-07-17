# agx/vlm/api_client.py
import os
import requests
from io import BytesIO
from PIL import Image
from typing import Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# VLM 서버 주소: 환경변수 우선 (예: export VLM_BASE_URL=http://192.168.0.226:8000)
VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _make_session() -> requests.Session:
    """requests.Session with basic retry policy"""
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"],
        raise_on_status=False,
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


_SESSION = _make_session()


def run_smolvlm(
    image_path: Optional[str],
    prompt: str,
    decode: bool = True,
    session_id: str = "default",
    timeout: int = 8,
) -> dict:
    """
    이미지와 프롬프트를 받아 VLM API를 호출해 분석 결과를 반환합니다.

    Args:
        image_path: 업로드할 이미지 경로 (None이면 더미 1x1 이미지 업로드)
        prompt: 프롬프트 문자열
        decode: 현재 의미 없음(서버 호환용)
        session_id: 세션 식별자 (서버로 전달)
        timeout: 요청 타임아웃(초)

    Returns:
        dict: {"result": str}  (실패 시 {"result": ""})
    """
    if not decode:
        return {"error": "decode=False is not supported"}

    url = "{}/analyze".format(VLM_BASE_URL)
    data = {
        "prompt": prompt,
        "decode": str(decode).lower(),
        "session_id": session_id,
    }

    try:
        if image_path:
            if not os.path.exists(image_path):
                print("[CLIENT] 파일 없음: {}".format(image_path))
                return {"result": ""}
            size = os.path.getsize(image_path)

            with open(image_path, "rb") as f:
                # (filename, fileobj, content-type) 형태로 명시
                files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
                res = _SESSION.post(url, files=files, data=data, timeout=timeout)
        else:
            # dummy 1x1 이미지 업로드
            img = Image.new("RGB", (1, 1), color=(0, 0, 0))
            buffer = BytesIO()
            try:
                img.save(buffer, format="JPEG")
                buffer.seek(0)
                files = {"file": ("dummy.jpg", buffer, "image/jpeg")}
                res = _SESSION.post(url, files=files, data=data, timeout=timeout)
            finally:
                buffer.close()

        if res.status_code == 200:
            try:
                return res.json()
            except Exception as e:
                print("[CLIENT] JSON 파싱 실패: {}".format(e))
                return {"result": ""}
        else:
            snippet = (res.text or "")[:200]
            print("[CLIENT] API 오류: {} - {}".format(res.status_code, snippet))
            return {"result": ""}

    except Exception as e:
        print("[CLIENT] 예외: {}".format(e))
        return {"result": ""}
