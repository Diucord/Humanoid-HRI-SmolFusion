# agx/vlm/api_client.py
import os, requests
from io import BytesIO
from PIL import Image
from typing import Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# VLM server address: environment variable takes priority
VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:8000").rstrip("/")


def _make_session() -> requests.Session:
    """Create a requests.Session with a basic retry policy."""
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


def run_vlm(
    image_path: Optional[str],
    prompt: str,
    decode: bool = True,
    session_id: str = "default",
    timeout: int = 8,
) -> dict:
    """
    Send an image and prompt to the VLM API and return the analysis result.

    Args:
        image_path: Path to the image to upload (if None, uploads a dummy 1x1 image)
        prompt: Prompt text
        decode: Currently unused (kept for server compatibility)
        session_id: Session identifier (passed to server)
        timeout: Request timeout in seconds

    Returns:
        dict: {"result": str}  (on failure, {"result": ""})
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
                print("[CLIENT] File not found: {}".format(image_path))
                return {"result": ""}
            size = os.path.getsize(image_path)

            with open(image_path, "rb") as f:
                # Explicitly set (filename, fileobj, content-type)
                files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
                res = _SESSION.post(url, files=files, data=data, timeout=timeout)
        else:
            # Upload a dummy 1x1 image
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
                print("[CLIENT] Failed to parse JSON: {}".format(e))
                return {"result": ""}
        else:
            snippet = (res.text or "")[:200]
            print("[CLIENT] API error: {} - {}".format(res.status_code, snippet))
            return {"result": ""}

    except Exception as e:
        print("[CLIENT] Exception: {}".format(e))
        return {"result": ""}