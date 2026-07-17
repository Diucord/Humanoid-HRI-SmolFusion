from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import torch
import re

# ===== 모델 경로 및 디바이스 설정 =====
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "finetune" / "igris-tuned" / "merged-model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 토크나이저 및 모델 로딩 =====
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
    trust_remote_code=True,
    local_files_only=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

# ===== Special token 명시적으로 추가 (필수!) =====
special_tokens = {"additional_special_tokens": ["<|user|>", "<|assistant|>"]}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

# ===== 불필요한 전처리 표현 제거 =====
def clean_input(text):
    text = re.sub(r"(?i)^human:\s*", "", text.strip())
    text = re.sub(r"<\|/?(user|assistant)\|>", "", text)
    return text.strip()

# ===== Qwen 래퍼 클래스 =====
class QwenLLM:
    def __init__(self):
        self.tokenizer = tokenizer
        self.model = model
        self.device = DEVICE

    def create_chat_completion(self, instruction, input_text, max_tokens=128):
        cleaned_input = clean_input(input_text)
        user_message = f"{instruction}\n{cleaned_input}" if cleaned_input else instruction
        prompt = f"<|user|>\n{user_message}\n<|assistant|>\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        output_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return output_text.split("<|")[0].strip()


# ===== LLM 인스턴스 생성 =====
llm = QwenLLM()
@dataclass
class ChatConfig:
    prompt: str
    session_id: str = "default"
    max_tokens: int = 128
    persona: str = "igris-C"
    language: str = "ko"


# ===== 외부 호출용 함수 =====
def generate_response_llm(config: ChatConfig) -> str:

    # 이미 완성된 Qwen 형식의 prompt 문자열이므로 그대로 instruction에 사용
    response = llm.create_chat_completion(
        instruction=config.prompt,
        input_text="",  # 불필요하므로 빈 문자열
        max_tokens=config.max_tokens
    )
    return response