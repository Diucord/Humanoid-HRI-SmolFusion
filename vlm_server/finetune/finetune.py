from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from datasets import load_dataset
from pathlib import Path
import torch
import matplotlib.pyplot as plt


# ===== 경로 및 모델 설정 =====
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "qwen3-1.7b-instruct"
OUTPUT_DIR = BASE_DIR / "finetune" / "igris-tuned"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===== 토크나이저 및 모델 로딩 =====
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)


# pad_token이 없으면 eos_token으로 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# pad_token_id 명시적으로 설정
tokenizer.pad_token_id = tokenizer.pad_token_id


# 4비트 로딩 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)


# 모델 로딩
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id


# ===== LoRA 설정 =====
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)


# ===== 데이터 로딩 =====
data = load_dataset("json", data_files="train.json")["train"]


# ===== 전처리 함수 =====
def preprocess(example): 
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output = example.get("output", "").strip()

    user_message = f"{instruction}\n{input_text}" if input_text else instruction

    prompt_text = (
        "<|im_start|>system\n당신은 사람과 자연스럽게 대화하는 인공지능 휴머노이드 로봇 이그리스 C입니다.<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    full_text = prompt_text + output + "<|im_end|>"

    prompt_ids = tokenizer(
        prompt_text, 
        truncation=True, 
        max_length=256
    )["input_ids"]

    full_tokenized = tokenizer(
        full_text,
        padding="max_length",
        truncation=True,
        max_length=256,
    )

    labels = full_tokenized["input_ids"].copy()
    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)  # 프롬프트 부분은 마스킹
    full_tokenized["labels"] = labels
    return full_tokenized


# 데이터 전처리 및 분할
tokenized_data = data.map(preprocess, remove_columns=data.column_names)
split = tokenized_data.train_test_split(test_size=0.1, seed=42)
train_data = split["train"]
val_data = split["test"]


# ===== 학습 설정 =====
args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,     
    learning_rate=5e-5,              
    num_train_epochs=5,               
    logging_steps=50,
    save_strategy="steps",
    save_steps=100,                  
    eval_strategy="steps",
    eval_steps=100,                  
    weight_decay=0.01,
    warmup_ratio=0.05,
    load_best_model_at_end=True,
    save_total_limit=2,
    fp16=True,
    report_to="none"
)


# ===== loss 시각화 콜백 =====
class LossPlotCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.train_losses.append((state.global_step, logs["loss"]))
        if "eval_loss" in logs:
            self.eval_losses.append((state.global_step, logs["eval_loss"]))

    def on_train_end(self, args, state, control, **kwargs):
        steps_train, losses_train = zip(*self.train_losses) if self.train_losses else ([], [])
        steps_eval, losses_eval = zip(*self.eval_losses) if self.eval_losses else ([], [])

        plt.figure(figsize=(10, 5))
        if steps_train:
            plt.plot(steps_train, losses_train, label="Train Loss")
        if steps_eval:
            plt.plot(steps_eval, losses_eval, label="Eval Loss")
        plt.xlabel("Global Step")
        plt.ylabel("Loss")
        plt.title("Training & Evaluation Loss")
        plt.legend()
        plt.grid()
        plt.savefig(str(OUTPUT_DIR / "loss_plot.png"))
        plt.close()


# ===== Trainer 정의 =====
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=val_data,
    callbacks=[LossPlotCallback()]
)


# ===== 학습 실행 =====
trainer.train()


# ===== 평가 =====
metrics = trainer.evaluate()
print("Evaluation metrics:", metrics)


# ===== LoRA 모델 저장 (adapter) =====
final_model_dir = OUTPUT_DIR / "final-model"
model.save_pretrained(str(final_model_dir))
tokenizer.save_pretrained(str(final_model_dir))


# ===== 병합 및 저장 =====
print("[🔁 LoRA 병합 중...]")
peft_config = PeftConfig.from_pretrained(str(final_model_dir))
base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
lora_applied = PeftModel.from_pretrained(base_model, str(final_model_dir))
merged_model = lora_applied.merge_and_unload()


# ===== 병합 모델 저장 =====
merged_model_dir = OUTPUT_DIR / "merged-model"
merged_model.save_pretrained(str(merged_model_dir))
tokenizer.save_pretrained(str(merged_model_dir))
print(f"[✅ 병합 완료: {merged_model_dir}]")