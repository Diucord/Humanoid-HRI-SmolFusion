import io
import os
import gc
import re
import json
import time
import logging
import warnings
import h5py
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig

# === 설정 ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore", message="Using a slow image processor")

HDF5_PATH = "/home/robros4/Desktop/igris_smartmart/datasets/raw/IGRIS_B_BOX_episode_5.hdf5"
OUTPUT_JSON = "/home/robros4/Desktop/igris_smartmart/datasets/processed/generated_sequence_prompt.json"
LOG_FILE = "run.log"

MAX_FRAMES = 64
RESIZE_WIDTH = 224
REMOVE_SUBJECT = True

PROMPT_TEMPLATES = [
    "Given the full video sequence, describe the complete task the robot is accomplishing.",
    "Based on the robot's actions over time, summarize the overall goal of this episode.",
    "Observe the full sequence and generate a single high-level instruction that would achieve the observed behavior.",
    "From the start to end of the video, what task is the robot completing?",
    "Using all observed frames, provide a concise command that would guide the robot to perform the same behavior.",
    "Considering the sequence of movements, what is the robot trying to achieve?",
    "Summarize the robot's end-to-end behavior and express it as a task goal.",
    "What should be written as a single command to replicate this robot behavior in another environment?",
    "Describe the final objective the robot achieves by executing this sequence of actions.",
    "Translate the full video into a single sentence task directive."
]


logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# === 유틸 함수 ===

def remove_subject(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(the robot|it)\s+(is\s+|does\s+|can\s+|has\s+|will\s+)?", "", text, flags=re.IGNORECASE)
    return text[0].upper() + text[1:] if text else text

def dedup_sentences(text: str) -> str:
    seen = set()
    result = []
    for sent in re.split(r'[.?!]\s*', text):
        sent = sent.strip()
        if sent and sent.lower() not in seen:
            result.append(sent)
            seen.add(sent.lower())
    return '. '.join(result)

def select_best_prompt(prompt_dict: dict[str, str]) -> str:
    keywords = ["pick", "place", "put", "move", "grasp", "push", "pull", "navigate", "lift", "open", "close"]
    def score(prompt: str) -> float:
        lower = prompt.lower()
        return sum(kw in lower for kw in keywords) * 10 + len(lower.split()) / 10.0
    return max(prompt_dict.values(), key=score)

def load_video_and_state(h5_file, max_frames=MAX_FRAMES, resize_width=RESIZE_WIDTH):
    image_ds = h5_file["observation/image/head"]
    joint_ds = h5_file["observation/joint_pos/right"]
    hand_ds = h5_file["observation/hand_joint_pos/right"]
    xpos_ds = h5_file["observation/xpos/right"]

    total = image_ds.shape[0]
    indices = np.linspace(0, total - 1, min(total, max_frames)).astype(int)

    video_frames = []
    state_vectors = []

    for idx in indices:
        try:
            # === 영상 처리 ===
            jpeg = image_ds[idx].tobytes()
            img = Image.open(io.BytesIO(jpeg)).convert("RGB")
            w, h = img.size
            new_w = resize_width
            new_h = int(h * (resize_width / w))
            img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
            video_frames.append(np.array(img_resized))  # (H, W, C)

            # === 상태 벡터 구성 ===
            joint = joint_ds[idx]   # (6,)
            hand = hand_ds[idx]     # (6,)
            xpos = xpos_ds[idx]     # (3,)
            state = np.concatenate([joint, hand, xpos])  # (15,)
            state_vectors.append(state)

        except Exception as e:
            logging.warning(f"[프레임 로딩 실패] idx={idx} → {e}")
    
    if not video_frames:
        raise RuntimeError("프레임을 하나도 로드하지 못했습니다.")

    video = np.stack(video_frames)[:, np.newaxis, ...]  # (T, 1, H, W, C)
    state = np.stack(state_vectors)                     # (T, D)
    return video, state, video_frames


def generate_summary_prompt(images, templates, batch_size=8):
    prompt_dict = {}
    for template in templates:
        try:
            texts = []
            for i in range(0, len(images), batch_size):
                batch_imgs = images[i:i+batch_size]
                batch_inputs = blip_processor(batch_imgs, [template]*len(batch_imgs), return_tensors="pt", padding=True).to(blip_model.device)
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    output_ids = blip_model.generate(
                        **batch_inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        num_beams=1
                    )
                batch_texts = [blip_processor.decode(ids, skip_special_tokens=True) for ids in output_ids]
                texts.extend(batch_texts)
            full_text = " ".join(texts)
            prompt_dict[template] = dedup_sentences(full_text)
        except Exception as e:
            prompt_dict[template] = f"[BLIP 오류] {e}"
        torch.cuda.empty_cache()
        gc.collect()
    return prompt_dict


# === 실행 ===
if __name__ == "__main__":
    start_time = time.time()
    #print("[BLIP-2 모델 로딩 중...]")

    try:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl",
            quantization_config=bnb_config,
            device_map="auto"
        ).eval()
        print("[✅ 모델 로딩 완료]")
    except Exception as e:
        logging.exception(f"모델 로딩 실패: {e}")
        exit(1)

    try:
        with h5py.File(HDF5_PATH, "r") as h5:
            print(f"[데이터 로딩] {HDF5_PATH}")
            video, state, pil_images = load_video_and_state(h5)

            print("[프롬프트 생성 중...]")
            prompt_dict = generate_summary_prompt(pil_images, PROMPT_TEMPLATES)
            best_prompt = select_best_prompt(prompt_dict)

            if REMOVE_SUBJECT:
                best_prompt = remove_subject(best_prompt)

            result = {
                "video": video.tolist(),
                "state": state.tolist(),
                "language": best_prompt
            }

            os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"[✅ 저장 완료] {OUTPUT_JSON}")
            logging.info(f"[완료] 저장 경로: {OUTPUT_JSON} / 프레임 수: {len(video)}")

    except Exception as e:
        logging.exception(f"처리 실패: {e}")
        print(f"[❌ 오류 발생]: {e}")

    print(f"[총 소요 시간]: {time.time() - start_time:.2f}초")
