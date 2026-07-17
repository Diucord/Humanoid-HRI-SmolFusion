# Jetson 이식 최적화 기록

> 서버(RTX)에서 동작하던 시스템을 Jetson 온디바이스로 이식하며 수행한 최적화.
> **모든 항목은 저장소에 코드/설정으로 남아 있다** — 성능 로그는 유실됐으나
> *무엇을 어떻게 바꿨는지*는 파일로 검증 가능하다.

---

## 1. 왜 이식이 어려웠나

Jetson은 x86 서버와 **하드웨어 계약 자체가 다르다.**

| 축 | x86 + RTX | Jetson Orin |
|---|---|---|
| CPU 아키텍처 | x86_64 | **aarch64 (ARM)** |
| GPU 아키텍처 | SM 86/89 | **SM 87 (Orin) / SM 72 (Xavier)** |
| 메모리 | VRAM 독립 | **CPU-GPU 통합(shared) 메모리** |
| PyTorch | pip 표준 휠 | **NVIDIA JetPack 전용 휠 필요** |
| CUDA | 12.x | **11.4 (JetPack 5.x 고정)** |

→ pip install torch 한 줄이 안 된다. 바이너리 배포판이 존재하지 않는다.

---

## 2. 수행한 최적화

### 2.1 aarch64 전용 PyTorch 휠 확보

**문제** — PyPI의 PyTorch는 x86_64 전용. Jetson에서 `pip install torch`는 실패하거나
CUDA 없는 CPU 빌드가 설치된다.

**해결** — NVIDIA가 배포하는 JetPack 전용 휠을 직접 지정
([`nx/requirements.txt`](../nx/requirements.txt))

```bash
# JetPack 5.x + CUDA 11.4
wget https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/\
torch-2.1.0+nv23.10-cp38-cp38-linux_aarch64.whl
pip install torch-2.1.0+nv23.10-cp38-cp38-linux_aarch64.whl
```

```
torch==2.1.0+nv23.10        # NVIDIA JetPack 5.x용 PyTorch wheel
torchvision==0.16.0+nv23.10
torchaudio==2.1.0+nv23.10
```

- `+nv23.10` 접미사 = NVIDIA 빌드 (표준 `2.1.0`과 다름)
- `cp38` = Python 3.8 고정 (JetPack 5.x 시스템 Python)
- 이에 맞춰 **전체 의존성 버전을 하향 고정**: `numpy==1.24.4`, `Pillow==9.5.0`,
  `scipy==1.10.1`, `transformers==4.35.2`, `pydantic==1.10.14`

> **핵심** — Jetson 이식은 "코드를 옮기는 일"이 아니라 **의존성 트리 전체를
> aarch64/CUDA 11.4/Python 3.8 제약에 맞춰 재구성하는 일**이었다.

---

### 2.2 llama.cpp CUDA 소스 빌드 (아키텍처 직접 지정)

**문제** — llama.cpp 공식 릴리스에 Jetson용 CUDA 바이너리가 없다.

**해결** — 타겟 GPU 아키텍처를 지정해 소스 빌드 ([`on-device/README.md`](../on-device/README.md))

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc

cmake .. \
 -DGGML_CUDA=ON \
 -DCMAKE_CUDA_ARCHITECTURES=87 \    # ← Jetson Orin = SM 8.7
 -DLLAMA_BUILD_TESTS=OFF \          # ← 빌드 시간·용량 절감
 -DLLAMA_BUILD_EXAMPLES=OFF
make -j$(nproc)
```

| 보드 | Compute Capability | 비고 |
|---|---|---|
| Jetson Orin (AGX/NX) | **87** | `-DCMAKE_CUDA_ARCHITECTURES=87` |
| Jetson Xavier | **72** | [`start_llm_xavier.sh`](../nx/scripts/start_llm_xavier.sh)에 별도 대응 |

- 아키텍처를 잘못 주면 **컴파일은 되지만 런타임에 커널이 실행되지 않는다.**
- `TESTS`/`EXAMPLES` 제외 — Jetson의 느린 스토리지·CPU에서 빌드 시간을 줄이기 위함.

---

### 2.3 보드별 실행 파라미터 분리

동일 llama.cpp 바이너리에 **보드 특성에 맞는 인자만 교체**한다.

| 파라미터 | Orin NX ([`nx/.env`](../nx/.env)) | Xavier ([`start_llm_xavier.sh`](../nx/scripts/start_llm_xavier.sh)) | 근거 |
|---|---|---|---|
| `-ngl` (GPU 레이어) | **0** (CPU) | **9999** (전량 GPU) | Orin NX는 shared memory 압박 → VLM에 GPU 양보 |
| `-c` (컨텍스트) | **1024** | 4096 | 메모리 점유 최소화 |
| `-t` (스레드) | 6 | 8 | 코어 수 차이 |

> **설계 판단** — Orin NX에서 LLM을 CPU(`-ngl 0`)로 내린 것은 성능 포기가 아니라
> **통합 메모리 배분 결정**이다. VLM(시각 토큰 생성)이 지연에 더 민감하므로
> GPU를 VLM에 우선 할당하고, LLM은 CPU 멀티스레딩으로 처리했다.

---

### 2.4 실행 프로파일 추상화 (코드 분기 제거)

**문제** — 서버/엣지 코드가 `if platform == "jetson"` 식으로 분기되면
유지보수 비용이 이중화된다.

**해결** — `PROFILE` 환경변수로 YAML을 통째로 교체
([`config_loader.py`](../nx/app/config_loader.py))

```python
def load_settings():
    profile = os.environ.get("PROFILE", "nx")
    cfg_path = os.path.join(..., "config", f"app.{profile}.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    # .env의 ${VAR} 치환
    return _walk(cfg)
```

**프로파일 비교** — 바뀌는 것은 설정값뿐, 파이프라인 코드는 동일

| 항목 | [`app.nx.yaml`](../nx/app/config/app.nx.yaml) (Jetson) | [`app.5080.yaml`](../nx/app/config/app.5080.yaml) (서버) |
|---|---|---|
| LLM provider | **llama.cpp** (HTTP) | **transformers** (로컬 가속) |
| VLM provider | **llama.cpp** (mmproj) | **pytorch** (:9000 별도 서버) |
| LLM max_tokens | 128 | 256 |
| VLM max_tokens | 64 | 96 |
| 분석 주기 | 1.5s | 1.0s |
| JPEG 품질 | 85 | 90 |
| `use_tmpfs` | **true** (`/dev/shm`) | false |
| TTS | **piper** (오프라인) | edge-tts (온라인) |

> **provider 자체가 다르다** — 서버는 transformers/pytorch로 직접 로드,
> Jetson은 llama.cpp HTTP 서버. 그런데도 **상위 파이프라인 코드는 한 줄도 안 바뀐다.**
> 이게 추상화의 실제 효과다.

---

### 2.5 I/O 병목 제거 — tmpfs 활용

**문제** — Jetson의 eMMC/SD 스토리지는 x86 NVMe 대비 현저히 느리다.
카메라 프레임을 매번 디스크에 쓰면 그 자체가 병목이 된다.

**해결** — `use_tmpfs: true`로 **`/dev/shm`(RAM 디스크)** 에 프레임 임시 저장
- Jetson: `use_tmpfs: true` — 디스크 쓰기 왕복 제거
- 서버: `use_tmpfs: false` — NVMe라 불필요

---

### 2.6 불필요한 추론 스킵 — 모션 감지

**문제** — 1.5초마다 VLM을 호출하면 정지 화면에서도 GPU를 계속 태운다.

**해결** — 프레임 차분 기반 스킵 (프로파일 파라미터)

| 파라미터 | Jetson | 서버 | 의미 |
|---|---|---|---|
| `analysis_interval` | 1.5s | 1.0s | 분석 주기 |
| `motion_diff_thresh` | 3.0 | 3.0 | 이 값 이하 변화면 스킵 |
| `still_frame_min` | 0.7 | 0.5 | 정지 판정 최소 시간 |
| `jpeg_quality` | 85 | 90 | 인코딩 품질 ↓ = 전송·디코딩 비용 ↓ |

→ **연산을 빠르게 하는 대신, 안 해도 되는 연산을 안 하는 방향**의 최적화.

---

### 2.7 모델 경량화 — QLoRA + GGUF 양자화

**파이프라인**
```
Qwen3-1.7B-Instruct (FP16)
  → QLoRA 파인튜닝 (r=8, α=16, q_proj/v_proj)
  → 어댑터 머지
  → GGUF 변환 (convert_hf_to_gguf.py / convert_lora_to_gguf.py)
  → Q4_K_M 양자화
```

| 단계 | 크기 |
|---|---|
| FP16 GGUF (`qwen3-igris-1.7b.gguf`) | **3.21GB** |
| Q4_K_M (`qwen3-igris-1.7b.Q4_K_M.gguf`) | **1.03GB** |
| **절감** | **-68%** |

- 변환 스크립트: [`robros/convert/`](../robros/convert/)
  (`convert_hf_to_gguf.py`, `convert_lora_to_gguf.py`, `convert_llama_ggml_to_gguf.py`)
- 학습 근거: [`trainer_state.json`](../vlm_server/finetune/igris-tuned/checkpoint-1465/trainer_state.json)

**VLM도 동일 전략** — SmolVLM-500M Q8_0 (0.41GB) + mmproj Q8_0 (0.11GB)
→ 서버에서 쓰던 Qwen3-VL-4B(2.33GB)를 Jetson에 올릴 수 없으므로
**지연 한계(latency budget) 기준으로 500M 선택**.

---

## 3. 이식 결과

| 항목 | 달성 |
|---|---|
| 오프라인 동작 | **가능** — 네트워크 없이 단독 실행 |
| 개인정보 | **로컬 처리** — 영상 외부 전송 없음 |
| 메모리 | 1.1GB (Jetson llama.cpp 구성) |
| 코드베이스 | **단일 유지** — 프로파일 교체만으로 서버/엣지 전환 |

**아키텍처는 그대로, 설정만 교체.** 가장 제약이 강한 환경(Jetson/500M)에서 먼저
설계했기 때문에 서버(RTX/4B)로 확장할 때 구조 변경이 필요 없었다.

---

## 4. 근거 파일 목록

| 최적화 | 파일 |
|---|---|
| aarch64 PyTorch 휠 | [`nx/requirements.txt`](../nx/requirements.txt) |
| CUDA 아키텍처 빌드 (SM87) | [`on-device/README.md`](../on-device/README.md) |
| Xavier 대응 (SM72) | [`nx/scripts/start_llm_xavier.sh`](../nx/scripts/start_llm_xavier.sh) |
| 보드별 실행 파라미터 | [`nx/.env`](../nx/.env) |
| 프로파일 추상화 | [`nx/app/config_loader.py`](../nx/app/config_loader.py) |
| Jetson 프로파일 | [`nx/app/config/app.nx.yaml`](../nx/app/config/app.nx.yaml) |
| 서버 프로파일 | [`nx/app/config/app.5080.yaml`](../nx/app/config/app.5080.yaml) |
| 온디바이스 프로파일 | [`on-device/app/config/app.on-device.yaml`](../on-device/app/config/app.on-device.yaml) |
| GPU 기동 + CUDA 검증 | [`on-device/scripts/start_all_gpu.sh`](../on-device/scripts/start_all_gpu.sh) |
| CPU 폴백 기동 | [`nx/scripts/start_all_cpu.sh`](../nx/scripts/start_all_cpu.sh) |
| GGUF 변환 도구 | [`robros/convert/`](../robros/convert/) |
| 파인튜닝 학습 기록 | [`vlm_server/finetune/igris-tuned/`](../vlm_server/finetune/igris-tuned/) |
| 성능 측정 스크립트 | [`vlm_server/agx/test_performance_agx.py`](../vlm_server/agx/test_performance_agx.py) |

---

## 5. 포트폴리오 기재 문구 (제안)

> **Jetson 온디바이스 이식 및 최적화**
>
> - **의존성 재구성**: aarch64/CUDA 11.4/Python 3.8 제약에 맞춰 NVIDIA JetPack 전용
>   PyTorch 휠(`2.1.0+nv23.10`) 확보 및 전체 의존성 트리 하향 고정
> - **llama.cpp CUDA 소스 빌드**: 공식 Jetson 바이너리 부재로 `CMAKE_CUDA_ARCHITECTURES=87`(Orin)
>   직접 지정해 빌드, Xavier(SM72) 별도 대응
> - **통합 메모리 배분 설계**: Jetson의 CPU-GPU shared memory 특성상 VLM에 GPU 우선 할당,
>   LLM은 `-ngl 0`으로 CPU 멀티스레딩 처리 — 지연에 민감한 시각 처리를 우선하는 판단
> - **실행 프로파일 추상화**: `PROFILE` 환경변수로 YAML 교체(`config_loader.py`),
>   provider(llama.cpp ↔ transformers)까지 달라지는데도 파이프라인 코드는 단일 유지
> - **I/O 병목 제거**: 느린 eMMC 대신 `/dev/shm`(tmpfs)에 프레임 임시 저장
> - **불필요 추론 제거**: 모션 차분 기반 VLM 호출 스킵(`motion_diff_thresh`),
>   JPEG 품질·컨텍스트 길이를 보드별 차등 적용
> - **모델 경량화**: QLoRA 파인튜닝 → GGUF 변환 → Q4_K_M 양자화로 **3.21GB → 1.03GB (-68%)**
