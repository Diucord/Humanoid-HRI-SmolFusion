import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import json
import os

# 로컬 모델 경로
MODEL_PATH = "./smolvlm_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Processor + Model 로딩
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16
).to(DEVICE)
model.eval()

# chat_template 수동 로딩
chat_template_path = os.path.join(MODEL_PATH, "chat_template.json")
if os.path.exists(chat_template_path):
    with open(chat_template_path, "r", encoding="utf-8") as f:
        processor.chat_template = json.load(f)
else:
    print("[❌ 경고] chat_template.json 파일이 없습니다. SmolVLM은 이 파일이 필요합니다.")

# SmolVLM 추론 함수
def run_smolvlm(image_path, prompt, decode=True):
    image = Image.open(image_path).convert("RGB")

    # SmolVLM용 메시지 포맷 (공식 chat template 방식)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # 템플릿 적용
    prompt_text = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True,
        chat_template="chat_template"
    )

    # 입력 생성
    inputs = processor(
        text=prompt_text,
        images=[image],
        return_tensors="pt"
    ).to(DEVICE)

    # 생성
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)

    # 결과 디코딩
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0] if decode else outputs
    return {"result": result}