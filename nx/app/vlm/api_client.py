# agx/vlm/api_client.py
import requests
from io import BytesIO
from PIL import Image

def run_smolvlm(image_path, prompt, decode=True):
    """
    이미지와 프롬프트를 받아 VLM API를 호출해 분석 결과를 반환합니다.
    Args:
        image_path (str): 분석할 이미지 경로
        prompt (str): 프롬프트 문장
        decode (bool): decode 옵션 (현재 사용하지 않음)
    Returns:
        dict: {"result": str}
    """
    if not decode:
        return {"error": "decode=False is not supported"}

    # 만약 image_path가 없으면 dummy 이미지 생성
    files = {}
    if image_path:
        files = {"file": open(image_path, "rb")}
    else:
        # dummy 1x1 이미지 생성
        img = Image.new("RGB", (1, 1), color=(0, 0, 0))
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        files = {"file": ("dummy.jpg", buffer, "image/jpeg")}

    data = {"prompt": prompt, "decode": str(decode).lower(), "session_id": "default"}

    try:
        res = requests.post("http://localhost:8000/analyze", files=files, data=data)
        if res.status_code == 200:
            return res.json()
        print(f"[API 오류] 상태코드: {res.status_code}")
    except Exception as e:
        print(f"[VLM 호출 실패]: {e}")

    return {"result": ""}