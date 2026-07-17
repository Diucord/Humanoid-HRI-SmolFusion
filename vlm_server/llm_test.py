from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import torch
import re

# ===== 경로 설정 =====
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "finetune" / "igris-tuned" / "merged-model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 모델 및 토크나이저 로딩 =====
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, 
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

# ===== Special token 명시적으로 추가 (필수!) =====
special_tokens = {"additional_special_tokens": ["<|user|>", "<|assistant|>"]}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

# ===== 불필요한 전처리 표현 제거 함수 =====
def clean_input(text):
    text = re.sub(r"(?i)^human:\s*", "", text.strip())  # 맨 앞의 'Human:' 또는 'human:' 제거
    text = re.sub(r"<\|/?(user|assistant)\|>", "", text)  # 토큰 중복 삽입 방지
    return text.strip()

# ===== 테스트 질문 목록 =====
test_data = [
    {
    "instruction": "당신은 휴머노이드 로봇 이그리스 C입니다. 사용자에게 자신의 정체를 공손하고 명확하게 소개하세요.",
    "input": "이름이 뭐야?",
    },
    {
    "instruction": "당신은 휴머노이드 로봇 이그리스 C입니다. 사용자에게 자신의 정체를 공손하고 명확하게 소개하세요.",
    "input": "누구야?",
    },
    {
    "instruction": "당신은 사람과 자연스럽게 대화하는 인공지능 휴머노이드 로봇 이그리스 C입니다. 사용자에게 자신의 정보를 공손하고 명확하게 소개하세요.",
    "input": "너는 왜 만들어졌어?",
    },
]

# ===== 응답 생성 함수 =====
def generate_response(instruction, input_text, max_new_tokens=128):
    cleaned_input = clean_input(input_text)
    user_message = f"{instruction}\n{cleaned_input}" if cleaned_input else instruction
    prompt = f"<|user|>\n{user_message}\n<|assistant|>\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    output_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    output_text = output_text.split("<|")[0].strip()  # post-processing
    return output_text

# ===== 실행 =====
for i, item in enumerate(test_data, 1):
    response = generate_response(item["instruction"], item["input"])
    print(f"[{i}] Q: {item['input'].strip()}")
    print(f"    A: {response}\n")
