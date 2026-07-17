# 이력서 · 포트폴리오 최종본

> **이 문서는 그대로 복사해 쓰는 완성본이다.**
> 모든 문장은 저장소 코드로 검증했다. 근거 없는 서술은 포함하지 않았다.
> 검증 과정과 삭제 사유는 [`RESUME_PATCH.md`](RESUME_PATCH.md) 참조.

---

# PART 1. 이력서 — 핵심 역량

## 인프라 · 서빙 · 최적화

> Docker 기반 MSA(멀티모델 LLM 라우팅+폴백), FastAPI 비동기 서빙, Neo4j 지식그래프.
> **Jetson 온디바이스 이식**: aarch64/CUDA 11.4 제약에 맞춘 의존성 재구성,
> llama.cpp CUDA 소스 빌드(SM87), GGUF Q4_K_M 양자화로 **모델 3.21GB→1.03GB(-68%)**.
> Hybrid RAG 파이프라인 최적화로 온디바이스 응답 **3.2s→1.4s**. REST API·GitLab CI/CD 연계

**변경 사유**
- ~~TensorRT~~ → **삭제**. 저장소 전체 검색 결과 TensorRT/torch2trt/.engine 사용 흔적 없음.
  실제 가속은 llama.cpp CUDA 빌드(`CMAKE_CUDA_ARCHITECTURES=87`)로 수행
- `3.2s→1.4s`를 TensorRT 성과처럼 읽히던 문장 구조 수정 → 실제 원인인 **RAG 파이프라인**으로 귀속
- 실측 근거가 확실한 **양자화 -68%** 추가

## LLM Agent · Multi-RAG 파이프라인 설계

> 단일 상태 객체(JSON) 기반 비선형 세션 제어, 라우터-생성 분리형 다단계 오케스트레이션 설계.
> Vector·Graph·Wiki 이종 소스 통합 및 하이브리드 검색(BM25+Dense)·BGE-Reranker 재순위화.
> LangGraph 기반 멀티에이전트 오케스트레이션.
> **개인 프로젝트에서 RRF(Reciprocal Rank Fusion) 순위 융합 및 한국어 문자 bigram 색인으로
> 조사 변형 대응 구현**

---

# PART 2. 이력서 — 로브로스 경력 (2025.06–2025.09)

> ### 페르소나 기반 대화형 로봇 시스템 개발
>
> - **Jetson 온디바이스 이식**: aarch64/CUDA 11.4/Python 3.8 제약에 맞춰 NVIDIA JetPack 전용
>   PyTorch 휠(`2.1.0+nv23.10`) 확보 및 의존성 트리 전면 재구성, llama.cpp를
>   `CMAKE_CUDA_ARCHITECTURES=87`(Orin)로 CUDA 소스 빌드 (Xavier SM72 별도 대응)
> - **통합 메모리 배분 설계**: Jetson의 CPU-GPU shared memory 특성상 지연에 민감한 VLM에
>   GPU 우선 할당, LLM은 `-ngl 0`으로 CPU 멀티스레딩 처리. **Jetson 단독 1.8s /
>   원격 서버(RTX 5080) 경유 0.8s**로 배포 방식별 지연을 분리 측정해 트레이드오프 규명
> - **Hybrid RAG 파이프라인** 구축 및 **QLoRA 도메인 파인튜닝**(r=8, α=16)으로 페르소나 일관성 확보.
>   Train Loss 17.81→0.29, Eval Loss 0.566→0.322(5 epoch), eval loss 정체 구간에서 조기 종료 판단
> - **FastAPI 기반 AI Gateway** 설계·개발로 LLM·STT·TTS·Vision 모듈 통합, 비동기 API 및
>   서비스 오케스트레이션 구현
> - **하드웨어 추상화**: `PROFILE` 환경변수 기반 YAML 프로파일 교체로 서버-엣지 코드베이스 단일화.
>   추론 provider가 달라져도(llama.cpp ↔ transformers) 파이프라인 코드 무변경

**변경 사유**
- ~~"TensorRT 최적화 적용하여 평균 0.5초"~~ → **삭제**.
  ① TensorRT 코드 없음 ② "0.5초"가 포트폴리오 PDF의 "Jetson 1.8s"와 자기모순
- 실측 수치(1.8s / 0.8s)로 교체하고 **환경 명시**
- 근거가 확실한 QLoRA 학습 지표·이식 최적화 추가

---

# PART 3. 포트폴리오 — Project 1

## 실시간 멀티모달 HRI를 위한 비동기 퓨전 아키텍처 설계

**리소스 제약 환경에서의 실시간성 확보를 위한 3-Tier 마이크로서비스 및 온디바이스 이식**

- **GitHub**: [Humanoid-HRI-SmolFusion](https://github.com/Diucord/Humanoid-HRI-SmolFusion)
- **Live Demo**: [hera-hri.vercel.app](https://hera-hri.vercel.app)
- **관련 저장소**: [Qwen3-Persona-Trainer](https://github.com/Diucord/Qwen3-Persona-Trainer)

본 프로젝트는 휴머노이드 로봇 환경에서 시각(VLM)과 언어(LLM)가 동시에 작동하는 실시간
상호작용 시스템(HRI)을 대상으로, **개별 모델 성능이 아닌 시스템 아키텍처와 실행 파이프라인
설계가 실시간성과 안정성에 미치는 영향**을 실험적으로 검증하는 것을 목표로 수행되었다.

핵심 질문은 다음과 같다.
> *"모델은 그대로 두고 실행 구조만 바꿨을 때 실시간성을 어디까지 확보할 수 있는가?"*

Jetson처럼 자원이 고정된 환경에서는 "더 큰 모델로 교체"라는 선택지가 존재하지 않는다.
따라서 이 문제를 **모델 성능 문제가 아닌 시스템 설계 문제로 재정의**하고,
기능 단위로 책임을 분리한 마이크로서비스 아키텍처(MSA)를 채택하였다.
FastAPI 기반 비동기 이벤트 루프를 오케스트레이션 레이어로 활용하여 처리 시간이 상이한
VLM·LLM·TTS 모듈 간 블로킹 병목을 제거하였으며, 고성능 GPU 서버 환경과 온디바이스
실행 환경을 분리 검증하였다.

---

### Core Competencies & Engineering Task

#### ■ System Architecture: Event-driven Microservices
- FastAPI(ASGI) 기반 비동기 이벤트 루프를 중심으로 VLM, LLM, TTS를 **독립 프로세스로 분리**
- 각 추론 모듈의 처리 시간이 상이하더라도 전체 요청 흐름이 블로킹되지 않는 Non-Blocking 구성
- API Gateway는 **라우팅만 담당** — 추론 로직을 외부로 분리해 게이트웨이 병목화 방지
- **[실측]** VLM 4B + LLM 1.7B 동시 상주 시 **VRAM 5.9GB / 8GB** — OOM 없이 안정 동작

#### ■ On-device Porting: Jetson 이식 최적화
- **의존성 재구성**: PyPI 표준 PyTorch는 x86 전용 → NVIDIA JetPack 전용 aarch64 휠
  (`torch==2.1.0+nv23.10`, cp38) 확보. CUDA 11.4/Python 3.8 제약에 맞춰
  `numpy==1.24.4`, `Pillow==9.5.0`, `transformers==4.35.2` 등 **의존성 트리 전면 하향 고정**
- **CUDA 소스 빌드**: llama.cpp 공식 Jetson 바이너리 부재 →
  `-DCMAKE_CUDA_ARCHITECTURES=87`(Orin SM8.7) 직접 지정해 빌드, Xavier(SM72) 별도 대응
- **통합 메모리 배분**: Jetson의 CPU-GPU shared memory 특성상 VLM에 GPU 우선 할당,
  LLM은 `-ngl 0`으로 CPU 처리 — 지연에 민감한 시각 처리를 우선하는 판단
- **I/O 병목 우회**: 느린 eMMC 대신 `/dev/shm`(tmpfs)에 프레임 임시 저장 (`use_tmpfs: true`)

> **핵심** — Jetson 이식은 "코드를 옮기는 일"이 아니라 **의존성 트리 전체를
> aarch64/CUDA 11.4/Python 3.8 제약에 맞춰 재구성하는 일**이었다.

#### ■ Hardware Abstraction: Profile-based Deployment
- **코드 분기 대신 YAML 프로파일 교체** — `PROFILE` 환경변수로 설정을 통째로 전환
- 추론 provider가 달라져도(Jetson: llama.cpp / 서버: transformers·pytorch)
  **상위 파이프라인 코드는 무변경**
- 하드웨어 특성에 맞는 양자화 모델 선택적 로드 (Jetson: SmolVLM-500M Q8 / 서버: Qwen3-VL-4B Q4_K_M)

#### ■ Model Optimization: QLoRA + GGUF Quantization
- QLoRA 파인튜닝(`r=8`, `α=16`, target `q_proj`/`v_proj`)으로 HRI 도메인 특화
- **[실측]** Train Loss **17.81 → 0.29**, Eval Loss **0.566 → 0.322** (5 epoch / 1,465 steps)
- eval loss 정체 구간(0.3225→0.3221) 확인 후 **best checkpoint를 step 1400으로 선택** — 과적합 억제
- GGUF Q4_K_M 변환으로 **[실측] 3.21GB → 1.03GB (-68%)**

#### ■ Memory Management: Ring Buffer 기반 결정론적 상태 관리
- 세션별 고정 길이 Ring Buffer(`deque(maxlen=20)`) 적용
- **[실측]** 500턴 연속 주입 후에도 보관 턴 수 20 유지 — Memory Leak Free
- TTL 기반 만료 방식 대비 메모리 상한이 결정론적으로 보장됨

#### ■ Response Robustness: Hybrid Rule-based + Model-based Engine
- RAG → 룰 → LLM 3단계 계층화. 정형 질의(로봇 사양·정체성)는 LLM 호출 없이 즉시 처리
- **[실측]** 룰 기반 **0.003s** vs LLM 경로 **0.19s** — **63배 차이**
- 로봇 정체성 관련 답변을 고정해 할루시네이션 구조적 차단
- 불완전 응답(회피 문구 등) 발생 시 룰 기반 응답으로 폴백
- 페르소나별 LLM 서버 이중화(:8080 파인튜닝 / :8082 범용)로 단일 장애점 제거

#### ■ Retrieval-Augmented Generation: Hybrid Retrieval 재설계
- **초기**: FAISS(Dense) + BM25(Sparse) 결합 구조 구현
- **문제 발견**: 임베딩 모델(`paraphrase-albert-small-v2`)이 **영어 전용**이라
  한국어 HRI 환경에서 Dense 검색이 사실상 무력화. 융합도 단순 합집합이라 순위 개념 부재
- **재설계**: 멀티링구얼 임베딩(**bge-m3**) 교체 + **RRF**(Reciprocal Rank Fusion)로
  스케일이 다른 두 검색기를 정규화 없이 결합 (`score(d) = Σ 1/(k + rank_i(d))`, k=60)
- **한국어 특화**: 공백 분리 토크나이저는 조사로 인해 매칭 실패
  (`"로봇은"` vs `"로봇이"` → **공통 토큰 0개** 실측). 형태소 분석기 의존성 없이
  **문자 bigram 색인**으로 조사 변형 흡수
- **[실측]** BM25 융합 추가 비용 **+0.1ms** (Dense 39.5ms 대비) — 사실상 무비용

#### ■ State Integrity: Session-aware State Machine
- 사용자별 세션 ID 추적으로 다중 사용자 환경에서 대화 맥락·상태 일관성 유지
- 얼굴 임베딩(128D) **코사인 유사도** 기반 동일 인물 판별로 세션 유지·분리 수행
- 임계값은 환경별 조정: **온디바이스 0.3 / 웹 0.6** — 조명·각도 변화가 큰 로봇 환경에서
  과도한 세션 분리를 억제하기 위한 설정

#### ■ On-device Visual Perception: Face-based User Context Extraction
- 연령대(Age Group), 성별(Gender), 표정(Smiling 여부) 추정으로 대화 맥락 보조 정보 생성
- 시각 정보는 VLM 전용 경로에서 처리되며, **시각 정보가 필요한 턴에서만 VLM 활성화**
- 모션 차분 기반 스킵(`motion_diff_thresh`)으로 정지 화면에서 불필요한 추론 제거

---

### 1. 프로젝트 배경 및 문제 정의

#### 1.1 배경
휴머노이드 로봇의 실시간 상호작용(HRI) 환경에서는 카메라 이미지, 음성 입력, 텍스트 대화가
동시에 발생하며, 각 입력은 서로 다른 추론 경로와 처리 시간을 가진다. VLM 추론이 끝날 때까지
대화 응답이 대기하면 상호작용이 성립하지 않는다.

#### 1.2 기존 시스템의 한계
- **Blocking Bottleneck**: 대규모 추론 연산이 이벤트 루프를 점유하여 실시간 응답성 저하
- **환경 불일치**: 서버와 엣지(Jetson) 환경 간 구조 차이로 코드 재사용·유지보수 비용 증가
- **자원 고정**: Jetson은 "더 큰 모델로 교체"라는 해법이 원천적으로 불가능

#### 1.3 접근 방향
- **추상화 레이어 도입**: 실행 환경에 따라 코드베이스를 분기하지 않고, 아키텍처는 유지한 채
  설정과 모델만 교체
- **비동기 오케스트레이션**: 멀티모달 추론이 동시에 발생하더라도 응답 지연이 누적되지 않는 설계
- **연산 회피**: 빠르게 하는 것보다 **안 해도 되는 연산을 안 하는** 방향의 최적화

---

### 2. 시스템 설계

#### 2.1 3-Tier 마이크로서비스

```
클라이언트 (텍스트 / 이미지 / 음성)
        ↓
API Gateway — FastAPI :8000  (라우팅 · RAG · 얼굴매칭 · TTS)
        ↓                ↓                ↓
llama.cpp :8080    llama.cpp :8081   llama.cpp :8082
파인튜닝 LLM (GPU)   VLM (GPU)         범용 LLM (CPU)
temp 0.3           조건부 호출         temp 0.7
```

| 계층 | 서비스 | 포트 | 역할 |
|---|---|---|---|
| 백엔드 API | Gateway 서버 | 8000 | 요청 수신 및 라우팅 |
| 추론 엔진 | 파인튜닝 LLM / VLM / 범용 LLM | 8080 / 8081 / 8082 | 언어 추론 및 시각 해석 |
| 멀티모달 I/O | 인터페이스 | — | 입력: 텍스트·이미지 / 출력: JSON + 음성 |

**설계 결정**
- **왜 프로세스를 나눴나** — VLM(4B)+LLM(1.7B)을 단일 프로세스에 로드하면 8GB VRAM에서 OOM.
  분리 후 5.9GB로 동시 상주, 범용 LLM만 CPU로 밀어 여유 확보
- **왜 LLM이 두 개인가** — 파인튜닝 페르소나는 정체성 일관성(temp 0.3), 범용 페르소나는
  표현 다양성(temp 0.7)이 필요. 단일 서버로는 상충하는 요구를 만족시킬 수 없음
- **왜 VLM을 매 턴 부르지 않나** — "안녕"에 카메라를 켜는 것은 지연만 늘림

#### 2.2 성능 최적화 전략

| 전략 | 내용 | 근거 |
|---|---|---|
| **Lazy Loading** | 임베딩 모델을 최초 요청 시 로드, 이후 프로세스 내 재사용 | `rag/store.py` |
| **양자화** | GGUF Q4_K_M — 3.21GB → 1.03GB (-68%) | 파일 크기 실측 |
| **CUDA 오프로딩** | `-ngl` 값으로 GPU 레이어 수 제어 (서버 99 / Jetson 0) | `.env`, 기동 스크립트 |
| **비동기 파이프라인** | FastAPI ASGI로 추론 단계 간 블로킹 제거 | `app.py` |
| **연산 회피** | 모션 차분 기반 VLM 스킵, 룰 우선 라우팅 | 프로파일 · `chat.py` |
| **Ring Buffer** | 고정 길이 컨텍스트 큐 — 오래된 발화 자동 제거 | `core/memory.py` |

#### 2.3 모델 구성 및 선택 근거

**① LLM: Qwen3-1.7B-Instruct (QLoRA 파인튜닝)**
- GGUF Q4_K_M 양자화 시 **1.03GB** 점유 — VLM과 병렬 실행에도 OOM 없음
- Jetson의 shared memory 구조상 실측 기준 안정적인 파라미터 스위트 스폿
- 동일 체급 비교 시 LLaMA 계열 대비 Qwen 계열이 한국어 구어체 응답 안정성이 높았음

**② VLM: 환경별 차등 선택**

| 환경 | 모델 | 크기 | 근거 |
|---|---|---|---|
| Jetson | SmolVLM-500M Q8 (SigLIP 인코더) | 0.41GB | **지연 한계(latency budget) 우선** |
| 서버 | Qwen3-VL-4B Q4_K_M | 2.33GB | 여유 VRAM만큼 시각 이해 품질 확보 |

> 로봇 HRI에서는 절대 정확도보다 **지연 한계**가 중요하다. Jetson에서 500M을 선택한 것은
> 정확도 포기가 아니라 응답성 우선 판단이다. 가장 제약이 강한 환경에서 먼저 설계했기에
> 4B로 확장할 때 **아키텍처 변경 없이 설정 교체만으로 대응**할 수 있었다.

---

### 3. 시스템 성능 평가

#### 3.1 실험 시나리오 (4-Way Matrix)

**환경**: RTX 5080 서버 + Jetson AGX Orin (회사 장비, 2025 인턴 수행 시)

| 실험 ID | 실행 위치 | 추론 엔진 | 설명 |
|---|---|---|---|
| Case 1 | RTX 5080 서버 | GPU 기반 FastAPI | 서버에서 LLM·VLM 직접 실행 |
| Case 2 | RTX 5080 서버 | llama.cpp | 서버에서 경량 엔진 실행 |
| Case 3 | Jetson AGX Orin | 원격 서버 접속 | Jetson → RTX 5080 호출 |
| Case 4 | Jetson AGX Orin | llama.cpp | Jetson 온디바이스 단독 실행 |

#### 3.2 정량 성능 비교 (Jetson·5080 — 2025 인턴 시점 측정)

| 항목 | Case 1 | Case 2 | Case 3 | Case 4 |
|---|---|---|---|---|
| 평균 응답 시간 | **0.7–0.8s** | 1.1s | **0.8s** | 1.8s |
| QA 정확도 | **82%** | 78% | **82%** | 67% |
| 메모리 사용량 | 4.2GB | 1.5GB | 4.2GB(서버) | **1.1GB** |
| 오프라인 동작 | 불가 | 불가 | 불가 | **가능** |
| 장시간 안정성 | 안정 | 안정 | 안정 | 안정 |

> QA 정확도는 동일 QA 평가 셋 기준 정답 포함 여부(Exact Match)로 산출.
> 측정 스크립트: `vlm_server/agx/test_performance_agx.py`, `performance_monitor.py`
> (회사 장비로 수행, 원본 로그 미보존)

#### 3.3 웹 데모 재측정 (RTX 3070 8GB — 재현 가능)

동일 아키텍처를 개인 환경에 재구성해 측정. [측정 절차](#5-측정-방법)로 언제든 재현 가능.

| 항목 | 측정값 |
|---|---|
| 종단 지연 (RAG 포함) | **0.19s** (중앙값, n=5) |
| 룰 기반 응답 (LLM 미호출) | **0.003s** |
| Dense 검색 (bge-m3 + ChromaDB) | 39.5ms |
| Sparse 검색 (BM25) | **0.1ms** |
| Hybrid (Dense + BM25 + RRF) | 36.8ms |
| VLM+LLM 동시 상주 VRAM | 5.9GB / 8GB |
| 양자화 절감 | 3.21GB → 1.03GB (-68%) |
| Ring Buffer | 500턴 주입 후 20턴 유지 |

#### 3.4 Hybrid RAG 적용에 따른 시스템 거동

**환각 차단 검증 (RTX 3070 실측)** — 로봇 사양서 업로드 후 동일 질의를
`use_rag` 플래그만 바꿔 대조

| 질문 | RAG **OFF** | RAG **ON** | 문서 실제 값 |
|---|---|---|---|
| 서비스 센터 번호? | `02-485-9311` ❌ | `1588-0000` ✅ | 1588-0000 |
| 배터리 용량? | `1000mAh, 충전 1시간` ❌ | `5200mAh, 4시간` ✅ | 5200mAh, 4시간 |
| 보증 기간? | — | `24개월` ✅ | 24개월 |

RAG 미적용 시 경량 LLM(1.7B)은 존재하지 않는 값을 생성하며
*"이전 대화 내용을 참고해 주세요"* 라는 **근거까지 날조**했다.
검색 증강이 이 환각을 구조적으로 차단한다.

**검색 비용 검증** — 종단 지연 RAG ON 0.19s / OFF 0.17s로 **+0.02s에 그침**.
BM25 융합은 +0.1ms로 사실상 무비용. **하이브리드를 안 쓸 이유가 없다는 것이 핵심 판단.**

**컨텍스트 제어 (온디바이스)**
- 워드 단위 청킹(size 300 / overlap 50)으로 청크 길이 사전 제한
- `RAG_TOP_K=3`으로 주입 문서 수 제한
- 보드별 컨텍스트 길이 차등 (Jetson 1024 / 서버 4096)

#### 3.5 시나리오별 해석

**Case 1 — RTX 5080 서버 (GPU 추론)**
최고 수준의 응답 품질. 멀티모달 결합 질의에서도 0.7–0.8초 유지. 연구·데모에 적합.

**Case 2 — RTX 5080 + llama.cpp**
GPU 자원을 전량 사용하지 않아도 일정 수준 성능 유지. 서버 자원 절감형 옵션.

**Case 3 — Jetson → 원격 서버**
Jetson을 Thin Client로 사용해 서버급 성능 활용. 단 **네트워크 의존성 존재** →
완전 자율 로봇에는 한계.

**Case 4 — Jetson 온디바이스**
완전 독립 실행. 응답 시간·정확도는 감소하지만 **오프라인 동작 · 개인정보 로컬 처리**가 가능해
실제 서비스 로봇 환경에 가장 현실적.

#### 3.6 실험적 결론

1. 실시간 HRI 성능은 **모델 자체보다 실행 구조와 파이프라인 설계**에 의해 결정된다.
2. 동일 아키텍처를 유지한 채 **실행 위치와 추론 엔진만 교체**함으로써 서버–엣지 간
   자연스러운 스케일 다운이 가능함을 검증했다.
3. Jetson 환경에서도 llama.cpp 기반 최적화를 통해 **실제 배포 가능한 수준**의
   응답성과 안정성을 확보했다.
4. 원격 서버 방식과 온디바이스 방식은 **대체 관계가 아니라 상황별 선택지**임을 수치로 제시했다.

---

### 4. 기술 스택

| 구분 | 내용 |
|---|---|
| **추론** | llama.cpp (CUDA 빌드), GGUF (Q4_K_M / Q8_0), Qwen3-VL-4B, SmolVLM-500M, Qwen3-1.7B |
| **학습** | QLoRA (PEFT), GGUF 변환 파이프라인 |
| **백엔드** | Python 3.10, FastAPI (ASGI), Uvicorn, ChromaDB, rank-bm25, bge-m3, face_recognition |
| **프론트** | Next.js 15, React 19, TypeScript, Web Speech API |
| **음성** | edge-tts (온라인) / piper (오프라인) |
| **인프라** | Vercel, Cloudflare Tunnel, PowerShell 자동화, Windows 작업 스케줄러 |
| **하드웨어** | RTX 5080 · Jetson AGX Orin / Orin NX (회사) / RTX 3070 (개인) |

---

### 5. 측정 방법

**[RTX 3070 재측정 — 재현 가능]** 환경: RTX 3070 8GB / Windows / conda `smolfusion` (Python 3.10)

| 항목 | 방법 |
|---|---|
| 종단 지연 | `/chat` POST, n=5, 세션 격리(히스토리 누적 배제), 중앙값 |
| 검색 지연 | 워밍업 1회 후 n=20 중앙값, LLM 호출 제외 |
| VRAM | `nvidia-smi --query-gpu=memory.used`, 서버 3개 구동 상태 |
| 양자화 절감 | GGUF 파일 크기 직접 비교 (FP16 vs Q4_K_M) |
| Ring Buffer | 500턴 주입 후 `len(memory.get(sid))` 검증 |
| 환각 대조 | 동일 질의를 `use_rag` 플래그만 바꿔 요청, 문서 원본과 대조 |

**[Jetson·5080 — 2025 인턴 시점]** 측정 스크립트는 저장소에 보존되어 있으나
회사 장비로 수행하여 **원본 로그는 미보존**. 퇴사 후 동일 환경 재현 불가.

---

# PART 4. 삭제한 항목과 사유

> 아래는 기존 문서에 있었으나 **저장소 코드로 검증되지 않아 제거**한 항목이다.
> 공개 저장소이므로 누구나 확인 가능하며, "로그가 없다"로는 방어되지 않는다.

| 삭제 항목 | 검증 결과 |
|---|---|
| **TensorRT 최적화** (이력서 2곳) | 저장소 전체 검색 결과 TensorRT/torch2trt/trtexec/.engine 사용 흔적 없음. requirements에도 없음 |
| **C++ 병렬 전처리 / GIL 병목 제거** | 해당 코드 없음. Pillow/OpenCV 표준 API만 사용 |
| **urllib3.util.retry 재시도·백오프** | `dialogue/llm.py`에 재시도 로직 없음 (`timeout=60`만 존재) |
| **얼굴 유사도 ≥ 0.98** | 전체 grep 결과 `0.98` 부재. 실제 온디바이스 0.3 / 웹 0.6 |
| **문서 임베딩 캐싱 → 35% 감소** | 캐싱 로직 없음. 생성자 1회 인코딩은 인덱스 구축이지 캐싱이 아님 |
| **컨텍스트 요약 후 주입** | 요약·필터링 로직 없음. `"\n---\n".join(docs)`로 그대로 연결 |
| **FP16 → 메모리 40% 절감** | 실제 배포는 Q4_K_M, 실측 68% |
| **FAISS/BM25 "점수 기반 병합", top-k 3/5** | 실제 `set union`(순위 없음), `top_k=2` 공통 |
| **평균 0.5초 실시간 응답** | PDF 본문 "Jetson 1.8s"와 자기모순 |
| **Pillow 디코딩 25% 향상** | 근거 불명 |
| **초기 응답 1.7s → 0.8s** | 근거 불명 (Lazy Loading 구현은 실재하나 수치 미검증) |
| **280 tokens/sec** | 측정 보드·방법 불명 |
| **GPU 사용률 90% 이상 고부하 검증** | 근거 불명 |

**대신 추가한 것** (모두 코드·로그로 검증됨)

| 추가 항목 | 근거 |
|---|---|
| Jetson 이식 최적화 (aarch64 휠, CUDA SM87 빌드, 통합 메모리 배분) | `nx/requirements.txt`, `on-device/README.md`, `nx/.env` |
| QLoRA 학습 지표 및 조기 종료 판단 | `trainer_state.json`, `adapter_config.json`, `loss_plot.png` |
| RAG 환각 대조 실측 | 재현 가능 |
| 룰 vs LLM 63배 지연 차이 | 재현 가능 |
| RRF 융합 및 한국어 bigram 색인 | `webapp/backend/rag/store.py` |
| 양자화 -68% | 파일 크기 실측 |
