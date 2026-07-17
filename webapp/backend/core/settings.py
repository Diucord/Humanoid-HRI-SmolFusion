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

# ===== VLM (Qwen3-VL via llama.cpp) =====
# llama.cpp 서버(--mmproj 비전 지원)에 HTTP로 호출.
#   모델: Qwen3VL-4B-Instruct-Q4_K_M.gguf + mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf
VLM_URL = _env("VLM_URL", "http://127.0.0.1:8081")
VLM_MODEL = _env("VLM_MODEL", "Qwen3VL-4B-Instruct-Q4_K_M.gguf")
VLM_MAX_TOKENS = int(_env("VLM_MAX_TOKENS", "64"))
VLM_ENABLED = _env("VLM_ENABLED", "true").lower() == "true"

# ===== LLM =====
# 두 개의 llama.cpp 서버를 페르소나에 따라 라우팅:
#   - 파인튜닝 igris (8080): 이그리스 C 전용
#   - 일반 Qwen3-1.7B (8082): 커스텀 등 나머지 페르소나
LLM_FINETUNED_URL = _env("LLM_FINETUNED_URL", "http://127.0.0.1:8080")
LLM_FINETUNED_MODEL = _env("LLM_FINETUNED_MODEL", "qwen3-igris-1.7b")
LLM_GENERAL_URL = _env("LLM_GENERAL_URL", "http://127.0.0.1:8082")
LLM_GENERAL_MODEL = _env("LLM_GENERAL_MODEL", "Qwen3-1.7B-Q8_0.gguf")

LLM_MAX_TOKENS = int(_env("LLM_MAX_TOKENS", "256"))
# 일반 LLM은 CPU라 느리므로 응답을 더 짧게 제한
LLM_GENERAL_MAX_TOKENS = int(_env("LLM_GENERAL_MAX_TOKENS", "150"))
# 파인튜닝 igris: 낮은 temp로 정체성 일관성 유지 (환각 방지)
LLM_FINETUNED_TEMPERATURE = float(_env("LLM_FINETUNED_TEMPERATURE", "0.3"))
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0.7"))

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
