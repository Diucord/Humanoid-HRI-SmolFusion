# Humanoid HRI · SmolFusion — Web Demo

실시간 멀티모달 HRI(Human-Robot Interaction) 웹 데모.
음성으로 말하면 → 인식된 텍스트가 화면에 뜨고 → 페르소나에 맞춰 응답(텍스트+음성)합니다.
카메라를 켜면 Vision 모델이 사람을 분석해 대화 컨텍스트로 활용하고, 문서를 올리면 RAG로 지식을 보강합니다.

```
┌─ Frontend: Next.js (React) ──────────────────────┐
│  웹캠 · 🎤마이크(STT) · 채팅 · 페르소나 · 문서업로드   │
└──────────────────┬───────────────────────────────┘
                   │ HTTP
┌──────────────────▼───────────────────────────────┐
│  Backend: FastAPI (RTX 3070 로컬 추론)             │
│  /vision  face_recognition + Qwen3-VL (나이/성별/표정/장면)
│  /chat    페르소나 + RAG + 파인튜닝 Qwen3 LLM
│  /rag     문서 → bge-m3 임베딩 → ChromaDB
│  /tts     edge-tts (mp3)
└──────────────────────────────────────────────────┘
```

## 구성 요소

| 레이어 | 기술 | 비고 |
|--------|------|------|
| STT (음성→텍스트) | Web Speech API | 브라우저 내장, 한국어 지원 |
| VLM (시각 분석) | Qwen3-VL-2B-Instruct | 나이/성별/표정/장면 단일 호출 JSON |
| 얼굴 매칭 | face_recognition (128D) | 동일인 판단, 본인 파이프라인 |
| LLM (대화) | 파인튜닝 Qwen3-1.7B | llama.cpp(GGUF) 또는 transformers |
| RAG | bge-m3 + ChromaDB | 페르소나별 지식 베이스 |
| TTS (텍스트→음성) | edge-tts | 페르소나별 보이스 |

---

## 실행 방법

### 1) 백엔드 (FastAPI)

```powershell
# conda 환경 (Python 3.10 권장 — dlib 호환)
conda create -n smolfusion python=3.10 -y
conda activate smolfusion

cd webapp\backend

# PyTorch (CUDA 12.1 기준 — RTX 3070)
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

# 환경 설정
copy .env.example .env   # 필요 시 수정

# (옵션) LLM: 파인튜닝 GGUF를 llama.cpp 서버로 띄우기
#   llama-server -m qwen3-igris-1.7b.Q4_K_M.gguf -ngl 99 --host 0.0.0.0 --port 8080
#   또는 .env에서 LLM_BACKEND=transformers 로 로컬 머지모델 직접 추론

python app.py   # → http://localhost:8000
```

> **face_recognition (dlib)**: Windows에서 빌드가 까다로우면
> `pip install dlib-bin` 또는 미리 빌드된 휠을 사용하세요.
> 얼굴 매칭 없이도(임베딩 None) 나머지는 정상 동작합니다.

### 2) 프론트엔드 (Next.js)

```powershell
cd webapp\frontend
npm install
copy .env.local.example .env.local   # API URL 확인
npm run dev   # → http://localhost:3000
```

브라우저에서 `http://localhost:3000` 접속.

---

## 모델 선택 메모

- **VLM**: 2026년 기준 경량 VLM 중 Qwen3-VL이 한국어·OCR에서 우위. RTX 3070 8GB에 2B 변형이 적합.
  더 가볍게 하려면 `.env`의 `VLM_MODEL_ID`를 변경하거나 `VLM_ENABLED=false`로 끄면
  얼굴 매칭만 동작합니다.
- **LLM**: 기존 파인튜닝 자산(Qwen3-1.7B-igris)을 그대로 재활용.
  GGUF가 있으면 `llamacpp`(빠름), 없으면 `transformers`로 머지 모델 직접 로딩.
- **임베딩**: bge-m3는 멀티링구얼 RAG 표준급. 처음 실행 시 모델 다운로드가 있습니다.

## 배포 (나중에)

로컬 검증 후 Cloud Run + GPU(L4)로 백엔드 컨테이너를 올리고,
프론트는 Vercel 또는 Cloud Run으로 배포하면 "누구나 접속하는 라이브 데모"가 됩니다.
(Dockerfile은 추후 추가)

## 기존 코드와의 관계

- `nx/`, `vlm_server/agx/` : Jetson 온디바이스 원본 (참고/보존)
- `qwen_trainer/`, `vlm_server/finetune/` : LLM 파인튜닝 파이프라인 (재활용)
- `webapp/` : 이 웹 데모 (신규)
