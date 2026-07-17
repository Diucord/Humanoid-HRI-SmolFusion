# smolvlm_infer.py

import os, base64, requests
from app.config_loader import load_settings

settings = load_settings()
VLM_URL   = settings["vlm"].get("url") or os.environ.get("VLM_URL", "http://127.0.0.1:8081")
VLM_MODEL = settings["vlm"]["model"]
VLM_TEMP  = settings["vlm"]["temperature"]
VLM_MAXTK = settings["vlm"]["max_tokens"]

def run_smolvlm(image_path, prompt, decode=True):
    with open(image_path, "rb") as f:
        img = f.read()
    data_url = "data:image/jpeg;base64," + base64.b64encode(img).decode()
    payload = {
      "model": VLM_MODEL,
      "messages": [{
        "role":"user",
        "content":[
          {"type":"text","text": prompt},
          {"type":"image_url","image_url":{"url": data_url}}
        ]
      }],
      "max_tokens": int(VLM_MAXTK),
      "temperature": float(VLM_TEMP),
    }
    r = requests.post(f"{VLM_URL}/v1/chat/completions", json=payload, timeout=60)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    return {"result": text, "bounding_boxes": []}
