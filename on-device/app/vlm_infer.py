# vlm_infer.py
import os, base64, requests
from app.config_loader import load_settings

# Load VLM (Vision-Language Model) configuration
settings   = load_settings()
VLM_URL    = settings["vlm"].get("url") or os.getenv("VLM_URL", "http://localhost:8081")
VLM_MODEL  = settings["vlm"]["model"]
VLM_TEMP   = settings["vlm"]["temperature"]
VLM_MAXTK  = settings["vlm"]["max_tokens"]

def run_vlm(image_path: str, prompt: str, decode: bool = True) -> dict:
    """
    Run inference with VLM using a local API endpoint.

    Args:
        image_path (str): Path to the input image
        prompt (str): Instruction or question for the model
        decode (bool): (Currently unused) option for controlling decoding

    Returns:
        dict: {
            "result": str,           # Generated text response
            "bounding_boxes": list   # Placeholder (currently empty)
        }
    """
    # Read image and encode as base64 data URL
    with open(image_path, "rb") as f:
        img = f.read()
    data_url = "data:image/jpeg;base64," + base64.b64encode(img).decode()

    # Construct request payload
    payload = {
        "model": VLM_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }],
        "max_tokens": int(VLM_MAXTK),
        "temperature": float(VLM_TEMP),
    }

    # Send request to VLM API
    r = requests.post(f"{VLM_URL}/v1/chat/completions", json=payload, timeout=60)
    r.raise_for_status()

    # Extract model output
    text = r.json()["choices"][0]["message"]["content"]

    return {"result": text, "bounding_boxes": []}