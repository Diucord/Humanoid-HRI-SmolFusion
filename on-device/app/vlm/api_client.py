# nx/vlm/api_client.py
import requests
from io import BytesIO
from PIL import Image

def run_vlm(image_path, prompt, decode=True):
    """
    Send an image and a prompt to the VLM API and return the analysis result.

    Args:
        image_path (str): Path to the image for analysis
        prompt (str): Prompt text

    Returns:
        dict: {"result": str}
    """
    if not decode:
        return {"error": "decode=False is not supported"}

    # If no image_path is provided, create a dummy image
    files = {}
    if image_path:
        files = {"file": open(image_path, "rb")}
    else:
        # Create a dummy 1x1 black image
        img = Image.new("RGB", (1, 1), color=(0, 0, 0))
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        files = {"file": ("dummy.jpg", buffer, "image/jpeg")}

    data = {"prompt": prompt, "decode": str(decode).lower(), "session_id": "default"}

    try:
        res = requests.post("http://localhost:8000/analyze", files=files, data=data)
        if res.status_code == 200:
            result = res.json()
            return result

        print(f"[API Error] Status Code: {res.status_code}")
    except Exception as e:
        print(f"[VLM Call Failed]: {e}")

    return {"result": ""}