# Hera — Human-robot Engagement Responsive AI

실시간 멀티모달 HRI(Human-Robot Interaction) 웹 데모.
카메라로 사람을 인식하고, 음성으로 대화하며, 페르소나·RAG·파인튜닝 LLM을 결합합니다.

> 상세 설계·구현은 [ARCHITECTURE.md](ARCHITECTURE.md) 참고.

```
Frontend (Next.js, Vercel 대상)
   └ 웹캠 · 마이크(STT) · 채팅 · 페르소나 · RAG · 로봇 캐릭터
        │ HTTP
Backend (FastAPI, 로컬 RTX 3070)
   └ 얼굴매칭 · 대화 라우팅 · RAG · 페르소나 · TTS
        │ HTTP
llama.cpp × 3  (VLM 8081 · 파인튜닝 LLM 8080 · 일반 LLM 8082)
```

## 주요 기능
- 🎥 **실시간 시각 분석** — Qwen3-VL-4B + face_recognition (나이/성별/표정/인원/장면)
- 🆕 **자동 인사** — 새 사람 감지 시 연령·성별 맞춤 인사
- 🎙️ **음성 대화** — Web Speech API(STT, 연속) + edge-tts(TTS)
- 🤖 **페르소나** — 이그리스 C(파인튜닝) / 커스텀(슬라이더) / 사용자 직접 생성
- 📚 **RAG** — bge-m3 + ChromaDB, 문서 업로드 후 검색 증강
- 💬 **로봇 캐릭터** — 상태별(대기/듣기/말하기/생각/인사) 눈 변화

---

## 실행 방법

전제: **conda 환경 `smolfusion` (Python 3.10)**, **RTX 3070급 GPU**, Node 18+.

### 0) 모델 & llama.cpp 준비 (최초 1회)

`webapp/models/`에 GGUF 다운로드:
```powershell
# VLM
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct-GGUF Qwen3VL-4B-Instruct-Q4_K_M.gguf --local-dir models
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct-GGUF mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf --local-dir models
# 일반 LLM
huggingface-cli download Qwen/Qwen3-1.7B-GGUF Qwen3-1.7B-Q8_0.gguf --local-dir models
# 파인튜닝 igris (기존 자산): nx/models/qwen3-igris-1.7b.Q4_K_M.gguf
```

`webapp/llamacpp/`에 llama.cpp Windows CUDA 빌드 압축 해제
(https://github.com/ggml-org/llama.cpp/releases — `llama-*-bin-win-cuda-12.4-x64.zip` + `cudart-*`).

### 1) llama.cpp 서버 3개 기동

```powershell
$L = "webapp\llamacpp\llama-server.exe"
$M = "webapp\models"

# VLM (포트 8081, GPU)
& $L -m "$M\Qwen3VL-4B-Instruct-Q4_K_M.gguf" --mmproj "$M\mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf" -ngl 99 -c 4096 --port 8081

# 파인튜닝 이그리스 (포트 8080, GPU)
& $L -m "nx\models\qwen3-igris-1.7b.Q4_K_M.gguf" -ngl 99 -c 4096 --port 8080 --alias qwen3-igris-1.7b

# 일반 LLM (포트 8082, CPU — VRAM 절약)
& $L -m "$M\Qwen3-1.7B-Q8_0.gguf" -ngl 0 -c 4096 -t 8 --port 8082 --alias Qwen3-1.7B-Q8_0.gguf
```

### 2) 백엔드 (FastAPI)

```powershell
conda activate smolfusion
cd webapp\backend
copy .env.example .env          # 필요 시 수정
python app.py                   # → http://localhost:8000
```

의존성 최초 설치:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# face_recognition (dlib 소스빌드 회피)
pip install dlib-bin
pip install --no-deps face_recognition
pip install "git+https://github.com/ageitgey/face_recognition_models"
pip install "setuptools<81"
```

### 3) 프론트엔드 (Next.js)

```powershell
cd webapp\frontend
npm install
copy .env.local.example .env.local
npm run dev                     # → http://localhost:3000
```

브라우저에서 **http://localhost:3000** 접속 (Chrome/Edge 권장 — 음성 인식).

---

## RAG 테스트

`sample_rag/english_study.txt` + `sample_rag/README_영어튜터.md` 참고.
→ "영어 튜터" 페르소나 생성 후 문서 업로드, "핸드폰을 영어로?" 등 질문.

---

## 배포 (포폴 라이브 데모)

- **프론트**: Vercel (무료, 항상 켜짐)
- **백엔드**: 로컬 PC + Cloudflare Tunnel (파인튜닝 모델 그대로, GPU 활용)
- PC가 켜져 있을 때만 데모 작동 → 오프라인 폴백 안내 권장

---

## 환경 변수 (backend/.env)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `DEVICE` | auto | cuda/cpu 자동 |
| `VLM_URL` / `VLM_MODEL` | :8081 / Qwen3VL-4B | 비전 |
| `LLM_FINETUNED_URL` | :8080 | 이그리스 |
| `LLM_GENERAL_URL` | :8082 | 일반 |
| `LLM_FINETUNED_TEMPERATURE` | 0.3 | 정체성 일관성 |
| `EMBED_MODEL` | BAAI/bge-m3 | RAG 임베딩 |
| `CORS_ORIGINS` | localhost:3000 | (Vercel 도메인 추가) |

---

## 기존 코드와의 관계
- `nx/`, `vlm_server/agx/` — Jetson 온디바이스 원본 (보존)
- `qwen_trainer/`, `vlm_server/finetune/` — LLM 파인튜닝 파이프라인
- `webapp/` — 이 웹 데모 (신규)
