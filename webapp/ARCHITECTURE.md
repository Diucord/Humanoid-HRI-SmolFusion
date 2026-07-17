# Hera — 아키텍처 & 구현 상세

> **Hera** (Human-robot Engagement Responsive AI)
> 실시간 멀티모달 HRI 시스템. 카메라로 사람을 인식하고, 음성으로 대화하며,
> 페르소나·RAG·파인튜닝 LLM을 결합한 풀스택 웹 애플리케이션.

---

## 1. 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend — Next.js 15 (React 19, TypeScript)                │
│  웹캠 · 마이크(STT) · 채팅 · 페르소나 · RAG 업로드 · 로봇 캐릭터  │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP (REST)
┌───────────────────────────▼─────────────────────────────────┐
│  Backend — FastAPI (Python 3.10, conda: smolfusion)          │
│  얼굴매칭 · 대화 라우팅 · RAG · 페르소나 관리 · TTS             │
└───────┬──────────────┬──────────────┬───────────────────────┘
        │ HTTP         │ HTTP         │ HTTP
┌───────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
│ llama.cpp    │ │ llama.cpp  │ │ llama.cpp  │   ← 모두 webapp/llamacpp
│ :8081 VLM    │ │ :8080 LLM  │ │ :8082 LLM  │
│ Qwen3-VL-4B  │ │ igris(FT)  │ │ Qwen3-1.7B │
│ (GPU)        │ │ (GPU)      │ │ (CPU)      │
└──────────────┘ └────────────┘ └────────────┘
        + bge-m3 임베딩 (RAG, GPU, 백엔드 프로세스 내)
        + face_recognition (얼굴 임베딩, CPU, 백엔드 내)
```

### 실행 중인 프로세스 (로컬 RTX 3070)
| 프로세스 | 포트 | 모델 | 디바이스 |
|---------|------|------|---------|
| FastAPI | 8000 | — | — |
| llama.cpp #1 (VLM) | 8081 | Qwen3-VL-4B-Instruct Q4_K_M + mmproj Q8 | GPU |
| llama.cpp #2 (이그리스) | 8080 | qwen3-igris-1.7b Q4_K_M (**파인튜닝**) | GPU |
| llama.cpp #3 (일반) | 8082 | Qwen3-1.7B Q8_0 | CPU |
| Next.js | 3000/3001 | — | — |

> **VRAM**: VLM(4B) + 이그리스(1.7B)를 GPU에 올려 약 7.6 / 8 GB 사용.
> 일반 LLM은 CPU(`-ngl 0`)로 분리해 OOM 회피.

---

## 2. 핵심 기능

### 2.1 실시간 시각 분석 (Vision)
- **입력**: 웹캠 스트림(1.5초 간격) 또는 "이미지로 테스트" 업로드
- **처리**:
  1. `face_recognition` → 128D 얼굴 임베딩 (동일인 판단)
  2. Qwen3-VL-4B → 한 번의 추론으로 JSON 반환:
     `{has_person, person_count, age_group, gender, is_smiling, scene}`
- **출력 값은 영어 원본**: `middle aged`, `male`, `smiling` 등 (prompt_map 기준)
- **새 사람 감지 → 자동 인사**: 연령/성별 기반 인사말 (greetings.json)
  - 얼굴 임베딩 매칭 / 데모그래픽 변화 / 수동 업로드(manual) 3중 판단

### 2.2 음성 (STT / TTS)
- **STT**: 브라우저 Web Speech API (continuous — 중단 버튼까지 계속 듣기)
- **TTS**: edge-tts → mp3 바이트, 페르소나별 보이스

### 2.3 대화 라우팅 (페르소나별)
| 페르소나 | LLM | 라우팅 |
|---------|-----|--------|
| **이그리스 C** | 파인튜닝(8080, temp 0.3) | RAG → robot_qa 룰 → 외모질문 분기 → LLM |
| **커스텀 / 사용자생성** | 일반(8082, temp 0.7) | 슬라이더(traits)→프롬프트 → RAG → LLM |

이그리스 대화 4단계:
1. **RAG** — 업로드 문서 있으면 최우선
2. **robot_qa** — 로봇 정보 키워드 룰 (긴 키워드 우선 매칭)
3. **외모 질문** — "나 어때 보여?" → 카메라 프레임 VLM 묘사
4. **파인튜닝 LLM** — 위에서 안 잡히면 (+ fallback 시 룰 default)

### 2.4 RAG (검색 증강)
- **임베딩**: bge-m3 (멀티링구얼)
- **벡터DB**: ChromaDB (인메모리, 페르소나별 컬렉션 분리)
- **청크**: 단어 300개 / 오버랩 50
- **파일 형식**: `.txt`, `.md`, `.pdf` (텍스트 PDF만)

### 2.5 페르소나 시스템
- **기본**: 이그리스 C(파인튜닝), 커스텀(즉석 슬라이더)
- **사용자 생성**: 모달에서 이름 + 프롬프트 + 슬라이더 + RAG → 영구 저장
  (`config/user_personas.json`)
- **슬라이더 4종**: 친절도 / 지식수준 / 공감능력 / 말투(격식) — 1~5 → 시스템 프롬프트 변환

---

## 3. API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/personas` | 페르소나 목록 |
| POST | `/personas` | 페르소나 생성 |
| DELETE | `/personas/{id}` | 페르소나 삭제 |
| POST | `/vision/analyze` | 프레임 분석 (얼굴매칭 + VLM + 인사) |
| POST | `/chat` | 대화 (페르소나 라우팅 + RAG + Vision 컨텍스트) |
| POST | `/rag/upload` | 문서 업로드 → 페르소나 지식베이스 |
| POST | `/rag/clear` | 지식베이스 초기화 |
| POST | `/tts` | 텍스트 → mp3 |
| POST | `/session/reset` | 세션 초기화 |
| GET | `/health` | 상태 (device 확인) |

---

## 4. 파일 구조

```
webapp/
├── backend/                      FastAPI
│   ├── app.py                    엔드포인트 + 세션 상태(얼굴/프레임/데모)
│   ├── core/
│   │   ├── settings.py           중앙 설정 (.env 오버라이드)
│   │   └── memory.py             세션별 대화 기록 (deque, max 20턴)
│   ├── vision/
│   │   ├── vlm.py                Qwen3-VL llama.cpp HTTP 클라이언트
│   │   └── analyze.py            face_recognition + VLM 통합
│   ├── dialogue/
│   │   ├── chat.py               대화 라우팅 (4단계)
│   │   ├── llm.py                llama.cpp LLM 호출 (finetuned/general)
│   │   ├── personas.py           페르소나 로딩/생성/저장
│   │   ├── traits.py             슬라이더 → 시스템 프롬프트
│   │   ├── greeting.py           연령/성별 인사말
│   │   ├── appearance.py         외모 질문 판별 (TF-IDF)
│   │   └── tts.py                edge-tts
│   ├── rag/
│   │   └── store.py              bge-m3 + ChromaDB
│   └── config/
│       ├── personas.json         기본 페르소나
│       ├── user_personas.json    사용자 생성 (런타임)
│       ├── general_responses.json 룰 응답 + 인사말
│       ├── robot_info.json       로봇 스펙
│       └── prompt_map.json       VLM 프롬프트
│
├── frontend/                     Next.js
│   ├── app/
│   │   ├── page.tsx              메인 화면 전체
│   │   ├── layout.tsx            폰트(Pretendard/Poppins)
│   │   └── globals.css           전체 스타일
│   ├── components/
│   │   ├── RobotFace.tsx         로봇 캐릭터 (상태별 눈)
│   │   ├── VisionPanel.tsx       실시간 분석 패널
│   │   ├── Camera.tsx            웹캠
│   │   ├── CreatePersonaModal.tsx 페르소나 생성 모달
│   │   └── Icons.tsx             SVG 아이콘
│   └── lib/
│       ├── api.ts                백엔드 통신
│       └── useSpeech.ts          Web Speech API (STT)
│
├── llamacpp/                     llama.cpp Windows CUDA 바이너리 (gitignore)
├── models/                       GGUF 모델 (gitignore)
└── sample_rag/                   RAG 테스트 파일
```

---

## 5. 기술 선택 근거

| 결정 | 이유 |
|------|------|
| **llama.cpp 통합** (VLM+LLM) | torch 추론 불필요, 양자화로 8GB에 VLM+LLM 동시 적재, 기존 HTTP 패턴 재활용 |
| **Qwen3-VL-4B Q4_K_M** | 경량 VLM 중 한국어·OCR 최강, 4bit 정확도 손실 ~1% |
| **파인튜닝 igris temp 0.3** | 1.7B 모델 환각 억제, 정체성 일관성 |
| **일반 LLM CPU 분리** | VRAM 부족(VLM+igris로 7.6GB) → 일반 LLM은 CPU |
| **bge-m3** | 멀티링구얼 RAG 표준급 |
| **Web Speech API (STT)** | 브라우저 내장, 설치 0, 기존 speech_recognition과 동일 엔진 |

---
## 6. 서버 주소
  Inspect     https://vercel.com/diucords-projects/frontend/Cd4ZZwDo1XW45tgcjoShyEmZxED2
▲ Production  https://frontend-hvcx6stle-diucords-projects.vercel.app
▲ Aliased     https://frontend-three-lac-47.vercel.app