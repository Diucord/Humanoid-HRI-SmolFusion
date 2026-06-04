"""중앙 설정. 환경변수로 오버라이드 가능."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


# ===== 디바이스 =====
# auto: CUDA 있으면 cuda, 없으면 cpu
DEVICE = _env("DEVICE", "auto")

# ===== VLM (Qwen3-VL) =====
# 로컬 RTX 3070 8GB 기준.
#   2B-Instruct : FP16로 ~4-5GB, LLM과 동시 구동 안전
#   4B-Instruct : 4-bit 양자화 권장 (VLM_QUANTIZE=4bit), 정확도 손실 ~1%
VLM_MODEL_ID = _env("VLM_MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
VLM_MAX_TOKENS = int(_env("VLM_MAX_TOKENS", "64"))
VLM_ENABLED = _env("VLM_ENABLED", "true").lower() == "true"
# none | 4bit | 8bit  (bitsandbytes 필요)
VLM_QUANTIZE = _env("VLM_QUANTIZE", "none").lower()

# ===== LLM (파인튜닝 Qwen3) =====
# llama.cpp 서버(OpenAI 호환) URL. 없으면 로컬 transformers 폴백.
LLM_BACKEND = _env("LLM_BACKEND", "llamacpp")  # llamacpp | transformers
LLM_URL = _env("LLM_URL", "http://127.0.0.1:8080")
LLM_MODEL = _env("LLM_MODEL", "qwen3-igris-1.7b")
LLM_MAX_TOKENS = int(_env("LLM_MAX_TOKENS", "256"))
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0.7"))
# transformers 폴백용 로컬 머지 모델 경로
LLM_LOCAL_PATH = _env(
    "LLM_LOCAL_PATH",
    str(BASE_DIR.parent.parent / "vlm_server" / "finetune" / "igris-tuned" / "merged-model"),
)

# ===== RAG =====
EMBED_MODEL = _env("EMBED_MODEL", "BAAI/bge-m3")
RAG_TOP_K = int(_env("RAG_TOP_K", "3"))
RAG_CHUNK_SIZE = int(_env("RAG_CHUNK_SIZE", "300"))
RAG_CHUNK_OVERLAP = int(_env("RAG_CHUNK_OVERLAP", "50"))

# ===== 얼굴 매칭 =====
FACE_SIMILARITY_THRESHOLD = float(_env("FACE_SIMILARITY_THRESHOLD", "0.6"))

# ===== 서버 =====
HOST = _env("HOST", "0.0.0.0")
PORT = int(_env("PORT", "8000"))
CORS_ORIGINS = _env("CORS_ORIGINS", "http://localhost:3000").split(",")


def resolve_device() -> str:
    if DEVICE != "auto":
        return DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
