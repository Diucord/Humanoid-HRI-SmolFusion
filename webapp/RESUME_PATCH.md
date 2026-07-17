# 이력서 · 포트폴리오 수정안 (검증 과정)

> 저장소 전수 조사 결과와 기존 문서(이력서 PDF / 포트폴리오 PDF)의 차이를 정리한다.
> 각 항목은 **코드로 검증**했으며, 근거 파일 경로를 함께 표기했다.
>
> **→ 바로 쓸 수 있는 완성본은 [`RESUME_FINAL.md`](RESUME_FINAL.md) 참조.**
> 이 문서는 "왜 그렇게 고쳤는가"의 근거 기록이다.

---

## 🔴 A-0. TensorRT — 저장소에 흔적 없음 (이력서 2곳)

**이력서 서술**
> (핵심 역량) on-device 양자화·**TensorRT**(Jetson, 지연 3.2s→1.4s)
> (로브로스 경력) Q4/Q8·FP16 양자화, **TensorRT 최적화**를 적용하여 평균 0.5초...

**검증 결과** — 저장소 전체 검색(`tensorrt`, `torch2trt`, `trtexec`, `.engine`)
결과 **사용 흔적 없음**. `requirements.txt`에도 TensorRT/ONNX 관련 패키지 부재.
(검색에 걸린 것은 GGUF 바이너리와 토크나이저 어휘뿐)

**실제 GPU 가속 경로** — llama.cpp CUDA 소스 빌드
```bash
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87   # Jetson Orin SM8.7
```

**→ 삭제.** 실제 수행한 가속으로 교체:
> llama.cpp CUDA 소스 빌드(`CMAKE_CUDA_ARCHITECTURES=87`, Xavier SM72 별도 대응),
> GGUF Q4_K_M 양자화로 모델 3.21GB→1.03GB(-68%)

> **면접 리스크 최상** — TensorRT는 구체적인 기술이라 "어떤 레이어를 변환했나요?",
> "FP16 캘리브레이션은?" 같은 후속 질문이 반드시 따라온다. 답할 수 없다.

---

## 요약

| 구분 | 개수 | 조치 |
|---|---|---|
| 🔴 코드에 없는 기능 | 5건 | **삭제 필수** — 공개 repo라 누구나 반박 가능 |
| 🟡 사실과 다른 수치 | 4건 | **수정** |
| 🟢 근거 있으나 문서에 없음 | 3건 | **추가하면 강해짐** |

---

## 🔴 A. 코드에 없는 기능 — 삭제 필수

이 항목들은 **repo가 공개되어 있어 누구나 확인할 수 있다.**
"로그가 없다"로는 방어되지 않는다. 면접에서 "그 코드 보여주세요" 한 마디면 끝난다.

### A-1. C++ 병렬 전처리 / GIL 병목 제거

**PDF 서술** (Core Competencies — Parallel Computing)
> 로봇 카메라 입력 이미지 전처리 과정에서 Python 루프를 배제하고
> 네이티브(C++/C++ backend) 병렬 처리 경로 활용
> GIL(Global Interpreter Lock) 병목을 제거하여 전처리 지연이 End-to-End Latency에
> 미치는 영향을 최소화

**검증 결과** — 해당 코드 없음. Pillow/OpenCV 표준 API만 사용.

**→ 삭제.** 대신 실제 수행한 최적화로 교체:
> - 모션 차분 기반 VLM 호출 스킵으로 불필요한 추론 제거 (`motion_diff_thresh`)
> - tmpfs(`/dev/shm`) 프레임 저장으로 Jetson eMMC I/O 병목 우회 (`use_tmpfs: true`)
> - 보드별 JPEG 품질 차등(Jetson 85 / 서버 90)으로 인코딩·전송 비용 조정

---

### A-2. urllib3.util.retry 재시도 / 지수 백오프

**PDF 서술** (2.5 하이브리드 응답 엔진 — LLM Inference)
> urllib3.util.retry 기반 재시도 및 지수 백오프 적용

**검증 결과** — [`dialogue/llm.py`](backend/dialogue/llm.py)에 재시도 로직 없음.
`timeout=60`만 존재.

**→ 삭제.** 실제 구현된 안정성 장치로 교체:
> - LLM이 빈 응답 또는 회피 문구("I don't know" 등) 반환 시 룰 기반 응답으로 폴백
> - 페르소나별 LLM 서버 이중화(:8080 파인튜닝 / :8082 범용)로 단일 장애점 제거

> **또는** 실제로 재시도를 구현하고 문구를 유지해도 된다 (구현 난도 낮음).

---

### A-3. 얼굴 유사도 `≥ 0.98`

**PDF 서술** (2.6절, 3.4.3절 — 두 번 등장)
> Similarity ≥ 0.98: 동일 사용자로 판단하여 기존 세션 유지

**검증 결과** — **저장소 전체 grep 결과 `0.98`이 존재하지 않음.**

| 버전 | 파일 | 실제 임계값 |
|---|---|---|
| 온디바이스 | `on-device/app/vlm/analyze_person.py` | **0.3** |
| 웹 | `webapp/backend/vision/analyze.py` | **0.6** |

두 버전 모두 코사인 유사도(`sim = 1 - cosine(...)`)가 맞지만 값이 다르다.
0.98은 거의 동일한 프레임만 통과하는 값이라 **실제 HRI에서 세션이 계속 끊어진다.**

**→ 수정.**
> 얼굴 임베딩(128D) 코사인 유사도 기반 동일인 판별로 세션 유지·분리 결정.
> 온디바이스 임계값 0.3, 웹 0.6 — 조명·각도 변화가 큰 로봇 환경에서
> 과도한 세션 분리를 억제하기 위한 조정.

> **면접 리스크 최상** — "0.98이면 거의 같은 프레임 아니면 다 새 사용자로 잡히는데
> 실제로 동작했나요?"라는 질문에 답할 수 없다.

---

### A-4. 문서 임베딩 캐싱 → 검색 지연 35% 감소

**PDF 서술** (3.4.2절)
> 자주 참조되는 문서 임베딩을 메모리에 캐싱하여, 반복 질의 시 FAISS 재검색 비용을 제거
> → 평균 검색 단계 지연 약 35% 감소

**검증 결과** — 캐싱 로직 없음. `rag_engine.py`는 생성자에서 1회 인코딩할 뿐이며,
그것은 "캐싱"이 아니라 인덱스 구축이다.

**→ 삭제.** 실제 측정된 검색 비용으로 교체:
> Dense(bge-m3) 39.5ms / Sparse(BM25) 0.1ms / Hybrid+RRF 36.8ms (n=20, 중앙값)
> → BM25 융합 추가 비용 +0.1ms로 사실상 무비용

---

### A-5. 컨텍스트 요약 후 주입

**PDF 서술** (2.5-1절, 3.4.2절)
> 검색된 문서를 그대로 LLM 입력에 포함하지 않고, 핵심 정보만 요약·필터링하여 주입
> 문서 길이 / 질의와의 의미적 유사도 / 중복 정보 여부를 기준으로 1차 필터링

**검증 결과** — 요약·필터링 로직 없음. 검색 결과를 `"\n---\n".join(docs)`로
그대로 연결한다.

**→ 삭제.** 실제 컨텍스트 제어 방식으로 교체:
> - 워드 단위 청킹(size 300 / overlap 50)으로 청크 길이 사전 제한
> - `RAG_TOP_K=3`으로 주입 문서 수 제한
> - 보드별 컨텍스트 길이 차등(Jetson 1024 / 서버 4096)

---

## 🟡 B. 사실과 다른 수치 — 수정

### B-1. FAISS + BM25 "점수 기반 병합", top-k 3/5

**PDF 서술**
> 두 검색 결과를 점수 기반으로 병합하여 상위 문서를 선정
> Dense Retrieval (FAISS): top-k = 3 / Sparse Retrieval (BM25): top-k = 5

**검증 결과** ([`server-based/scripts/rag_engine.py`](../server-based/scripts/rag_engine.py))
- 융합은 `list(set(faiss_indices[0]) | set(bm25_indices))` — **단순 합집합**. 순위 없음
- `def search(self, query, top_k: int = 2)` — **양쪽 공통 2**. 3/5 아님

**→ 재설계 서사로 전환** (이게 원래 서술보다 강하다)
> **Hybrid Retrieval 설계 및 재설계**
> - 초기: FAISS(Dense) + BM25(Sparse) 결합
> - **문제 발견**: 임베딩(`paraphrase-albert-small-v2`)이 영어 전용이라 한국어 HRI에서
>   Dense 검색이 사실상 무력화. 융합도 단순 합집합이라 순위 개념 부재
> - **재설계**: 멀티링구얼 임베딩(bge-m3) 교체 + **RRF**(Reciprocal Rank Fusion)로
>   스케일이 다른 두 검색기를 정규화 없이 결합 (k=60)
> - **한국어 특화**: 공백 분리 토크나이저는 조사로 매칭 실패(공통 토큰 0개 실측).
>   형태소 분석기 의존성 없이 **문자 bigram 색인**으로 해결
> - **비용 검증**: BM25 추가 +0.1ms (Dense 39.5ms 대비)

---

### B-2. FP16 추론 → 메모리 40% 절감

**PDF 서술** (2.2절)
> FP16(Half-precision) 추론 적용 → 메모리 사용량 약 40% 절감

**검증 결과** — 실제 배포는 **Q4_K_M**. 실측 절감률은 **68%**.

**→ 수정**
> GGUF Q4_K_M 양자화 적용 → **3.21GB → 1.03GB (-68%)**
> (`qwen3-igris-1.7b.gguf` → `qwen3-igris-1.7b.Q4_K_M.gguf`, 파일 크기 실측)

---

### B-3. 이력서 "평균 0.5초" vs PDF "Jetson 1.8s"

**이력서** (로브로스 경력)
> Jetson Orin NX에서 Q4/Q8·FP16 양자화, TensorRT 최적화를 적용하여
> **평균 0.5초 수준**의 실시간 대화 응답 구현

**포트폴리오 PDF** (3.4.1절)
> Case 4 (Jetson / llama.cpp) 평균 응답 시간 **1.8s**

**→ 자기모순.** 환경을 명시해 정리
> Llama.cpp 기반 온디바이스 LLM 환경 구축 및 Jetson Orin NX에서 Q4/Q8·FP16 양자화,
> TensorRT 최적화 적용. **Jetson 단독 1.8s / 원격 서버(RTX 5080) 경유 0.8s로
> 배포 방식별 지연을 분리 측정**하여 트레이드오프 규명

---

### B-4. 이력서 "지연 3.2s→1.4s (TensorRT)"

**이력서** (핵심 역량 — 인프라·서빙·최적화)
> on-device 양자화·TensorRT(Jetson, **지연 3.2s→1.4s**)

**문제** — `3.2s→1.4s`는 PDF에서 **"LLM 단독 → Jetson Hybrid RAG"** 비교 수치다.
TensorRT 효과가 아니라 **RAG 파이프라인 효과**인데 TensorRT 성과로 읽힌다.

**→ 수정**
> on-device 양자화·TensorRT(Jetson). **GGUF Q4_K_M으로 모델 3.21GB→1.03GB(-68%)**,
> RAG 파이프라인 최적화로 응답 3.2s→1.4s

---

## 🟢 C. 근거 있으나 문서에 없음 — 추가하면 강해짐

### C-1. QLoRA 학습 기록 ⭐

**근거 보존됨** — [`trainer_state.json`](../vlm_server/finetune/igris-tuned/checkpoint-1465/trainer_state.json),
[`adapter_config.json`](../vlm_server/finetune/igris-tuned/checkpoint-1465/adapter_config.json),
[`loss_plot.png`](../vlm_server/finetune/igris-tuned/loss_plot.png)

| 설정 | 값 |
|---|---|
| LoRA | `r=8`, `alpha=16`, `dropout=0.05` |
| Target | `q_proj`, `v_proj` |
| 학습 | 5 epoch / 1,465 steps / batch 4 / 2.54e16 FLOPs |

| 구간 | Train Loss | Eval Loss |
|---|---|---|
| step 50 | 17.81 | — |
| step 100 | 3.85 | 0.566 |
| **step 1400 (best)** | 0.290 | **0.322** |

**왜 강한가** — `best_checkpoint`가 마지막(1465)이 아니라 **1400**이다.
eval loss 정체(0.3225→0.3221)를 보고 **과적합 직전에 멈춘 판단의 증거**.
"QLoRA 썼다"가 아니라 "학습 곡선 보고 언제 멈출지 판단했다"가 된다.

---

### C-2. Jetson 이식 최적화 ⭐

**근거 전부 보존됨** — 상세: [`JETSON_PORTING.md`](JETSON_PORTING.md)

| 최적화 | 근거 |
|---|---|
| aarch64 PyTorch 휠 (`2.1.0+nv23.10`) + 의존성 트리 하향 고정 | [`nx/requirements.txt`](../nx/requirements.txt) |
| llama.cpp CUDA 소스 빌드 `CMAKE_CUDA_ARCHITECTURES=87` | [`on-device/README.md`](../on-device/README.md) |
| Xavier(SM72) 별도 대응 | [`start_llm_xavier.sh`](../nx/scripts/start_llm_xavier.sh) |
| 통합 메모리 배분 (`-ngl 0` LLM→CPU, GPU는 VLM 우선) | [`nx/.env`](../nx/.env) |
| PROFILE 기반 YAML 교체 (provider까지 다름) | [`config_loader.py`](../nx/app/config_loader.py) |

**왜 강한가** — 기존 PDF엔 "환경변수로 지원"이라고만 쓰여 있어 이 강점이 안 보인다.
실제로는 **`pip install torch` 한 줄이 안 되는 환경**에서 의존성 트리 전체를
재구성한 작업이다. 해본 사람만 안다.

---

### C-3. RAG 환각 대조 실측 ⭐

동일 질의를 `use_rag` 플래그만 바꿔 요청한 결과:

| 질문 | RAG OFF | RAG ON | 문서 실제 값 |
|---|---|---|---|
| 서비스 센터 번호? | `02-485-9311` | `1588-0000` | 1588-0000 |
| 배터리 용량? | `1000mAh, 충전 1시간` | `5200mAh, 4시간` | 5200mAh, 4시간 |

RAG OFF에서는 존재하지 않는 값을 생성하며 *"이전 대화 내용을 참고해 주세요"* 라는
**근거까지 날조**했다.

**추가로** — 룰 기반 0.003s vs LLM 0.19s = **63배**. 계층형 파이프라인의 정량 근거.

---

## D. 하드웨어 환경 (확정)

| 환경 | 소속 | 시점 | 재현 |
|---|---|---|---|
| RTX 5080 서버 | 회사(로브로스) | 2025 인턴 | 불가 (장비 접근 종료) |
| Jetson AGX Orin / Orin NX | 회사 | 2025 인턴 | 불가 |
| RTX 3070 8GB | 개인 PC | 2026 확장 | **가능** |

→ 4-Way Matrix(RTX 5080 / Jetson)는 **회사 장비 기준 사실**.
개인 프로젝트 수치(RTX 3070)와 **환경만 명시해 분리**하면 모순이 해소된다.

---

## E. 남은 확인 항목

| 항목 | 상태 |
|---|---|
| **280 tokens/sec** (2.3절) | 보드(AGX? NX?)·측정법 불명. 1.7B Q4_K_M을 Jetson에서 280 t/s는 높은 편 — llama.cpp `llama-bench` 로그가 있으면 보드명과 함께, 없으면 삭제 |
| **1.7초 → 0.8초** (Lazy Loading) | 근거 있으면 "콜드 스타트 기준" 명시 |
| **Pillow 디코딩 25% 향상** | 근거 불명 |
| **GPU 사용률 90% 이상** (3.4.3절) | 근거 불명 |

---

## F. 적용 우선순위

| 순위 | 항목 | 이유 |
|---|---|---|
| **1** | 🔴 A-1~A-5 삭제 | 공개 repo로 반박 가능. 신뢰도 붕괴 위험 |
| **2** | 🟡 B-3 자기모순 (0.5s vs 1.8s) | 같은 지원서 내 충돌 |
| **3** | 🟢 C-1, C-2 추가 | 근거 확실 + 차별화 포인트 |
| **4** | 🟡 B-1 RAG 재설계 서사 | 원래 서술보다 강함 |
| **5** | 🟢 C-3 환각 대조 | 설득력 |
| **6** | 🟡 E절 검증 | 근거 없으면 삭제 |
