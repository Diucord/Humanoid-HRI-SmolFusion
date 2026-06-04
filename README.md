# SmolFusion: A Real-Time Vision-Language Fusion Architecture for Humanoid HRI
This project introduces **SmolFusion**, a novel, lightweight multimodal system that achieves **real-time Human-Robot Interaction (HRI)** by fusing a **fine-tuned VLM and LLM** running concurrently. Optimized for both server-based and on-device deployment, this architecture delivers low-latency, context-aware intelligence crucial for embodied agents.

The project supports both **"server-based"** and **"on-device"** environments, targeting real-time performance for **Human-Robot Interaction (HRI)**.

---
## 1. Overview
- Traditional HRI has been limited to **command-based interactions**, which struggle with complex or unstructured situations.
- This project proposes a **multimodal architecture** that integrates **text, image, and speech inputs** to enable **real-time, natural interactions**.

**Key Research Highlights**
- Unified architecture combining **VLM + lightweight, fine-tuned LLM**
- Dual-path development: **high-performance server-based** and **lightweight on-device**
- **Session-based memory management** for contextual and long-term conversations
- Lightweight deployment suitable for embedded devices

---
## 2. Architecture
SmolFusion maximizes real-time responsiveness through modular separation of concerns.

### 2.1. Server-based Environment
- **Stack**: FastAPI + High-performance GPU Server (RTX 5080)
- **Features**: High-performance inference and real-time response, scalable to multiple users, suitable for research and service demos.

### 2.2. On-device Environment (Primary Target)
- **Stack**: FastAPI + `llama.cpp` (GGUF models)
- **Features**: Minimal reliance on network connectivity, independent operation in low-power environments, strong privacy protection, and real-time inference on embedded devices like **NVIDIA Jetson AGX Orin/NX**.

---
## 3. Models & Optimization (The Core of SmolFusion)
The SmolFusion architecture utilizes two core models and advanced optimization strategies to maximize **real-time responsiveness (low-latency)** for Humanoid HRI.

### 3.1. Vision-Language Model (VLM)
* **Role**: Responsible for real-time visual perception, including face/object detection, expression analysis (see `analyze_person.py`), and extraction of visual features specialized for HRI.
* **Base Model**: **SmolVLM (Quantized Version)**
* **Key Optimization**:
    * **CUDA Optimization**: CUDA-based inference optimization is applied to accelerate VLM inference speed in the **Jetson environment**.
    * **Pre-processing Optimization**: Image pre-processing utilizes **C++-based logic** instead of Python to reduce GIL overhead. Image encoding is **parallelized across separate threads** to ensure visual token generation is complete before the prompt is handed to the LLM.

### 3.2. Large Language Model (LLM)
* **Role**: Performs high-level reasoning based on VLM's visual analysis and the user's conversational context, generating robotic action commands.
* **Base Model**: **Qwen3-1.7B-Instruct (Fine-Tuned)**
* **Key Optimization**:
    * **Custom Quantization Pipeline**: We implemented a complete deployment pipeline by **directly fine-tuning Qwen3-1.7B-Instruct** on HRI data, followed by **custom quantization and GGUF conversion**. This process ensures maximal efficiency, enables advanced **Quantization** (e.g., Q4\_K), and minimizes model memory footprint for real-time inference on low-power **Edge Devices**.
    * **HRI Fine-Tuning**: The model is fine-tuned on HRI-specific dialogue data to improve conversational responsiveness, robot persona consistency, and action instruction accuracy.

### 3.3. Real-Time Fusion Strategy (SmolFusion)
The VLM and LLM operate via an **asynchronous pipeline**. The VLM's analysis runs concurrently on its dedicated server. The VLM’s **compressed visual tokens** are injected into the LLM's prompt in real-time. This strategy **minimizes latency degradation** while enabling rich multimodal context understanding.

---
## 4. Modules & Core Logic

- `/chat` : Text-based dialogue query and response module (see `dialogue/general_chat.py`).
- `/analyze` : Image-based analysis and multimodal reasoning module (see `vlm/analyze_person.py`).

All core modules are accessible via the **FastAPI Gateway (Port 8000)**.

| File/Module | Role | Technical Feature |
| :--- | :--- | :--- |
| `scripts/test.py` | **SmolFusion Core Pipeline** | Contains the core asynchronous logic for VLM/LLM API calls, camera input handling, and VLM analysis skipping based on motion detection for real-time efficiency. |
| `scripts/memory.py` | **Session-Based Memory** | Manages conversation history using a `defaultdict(deque)` with a maximum length to ensure context retention while automatically pruning old turns for memory efficiency. |
| `scripts/performance_monitor.py` | **Performance Measurement** | Includes logic for monitoring CPU/GPU resource usage and recording API call times to validate the system's **Real-Time Responsiveness (Latency)**. |

---
## 5. Dependencies & Execution

### 5.1. Environment Setup
As listed in `requirements.txt`, this project is specialized for the **NVIDIA Jetson AGX Orin/NX** environment, requiring specific dependency versions and custom **Jetson-specific Wheels** for installation.

### 5.2. Quick Start (Execution)
The system is activated by simultaneously running the LLM server, VLM server, and the FastAPI application.

1.  **Model Preparation**: Create a `models/` directory and place the necessary GGUF files (Qwen3-1.7B, SmolVLM, MM Projection) inside.
2.  **`llama.cpp` Build**: Build the `llama-server` binary from the `llama.cpp` repository.
3.  **Environment Variables**: Set environment variables to match your model filenames:
    ```bash
    export LLM_MODEL="Qwen3-1.7B-Instruct.gguf"
    export VLM_MODEL="SmolVLM.gguf"
    export VLM_MMPROJ="mmproj.bin"
    ```
4.  **Integrated Execution (`start.sh` logic)**:
    ```bash
    # === Launch LLM Server (8080): Qwen3-1.7B-Instruct GGUF ===
    ./llama.cpp/build/bin/llama-server \
       --model models/"$LLM_MODEL" \
       --host 0.0.0.0 \
       --port 8080 \
       --ctx-size 2048 \
       --threads 8 \
       --batch-size 256 \
       --n-gpu-layers 20 > logs/llm.log 2>&1 &
    
    # === Launch VLM Server (8081): SmolVLM Quantized ===
    ./llama.cpp/build/bin/llama-server \
       --model models/"$VLM_MODEL" \
       --mmproj models/"$VLM_MMPROJ" \
       --host 0.0.0.0 \
       --port 8081 \
       --ctx-size 2048 \
       --threads 8 \
       --batch-size 256 \
       --n-gpu-layers 999 > logs/vlm.log 2>&1 &
    
    # === Launch FastAPI Server (8000): SmolFusion API Gateway ===
    uvicorn app.server:app --host 0.0.0.0 --port 8000 > logs/api.log 2>&1 &
    ```
    This script demonstrates the **3-Tier Server Architecture** of SmolFusion, optimally utilizing on-device GPU resources by dedicating layers to the VLM (`--n-gpu-layers 999`) and allocating necessary resources to the LLM (`--n-gpu-layers 20`).
    
---
## 6. Training & External Resources
The code for the model fine-tuning and custom quantization pipeline is maintained in a separate repository to ensure the modularity of the main deployment repository.
- **Fine-Tuning Repository**: **[https://github.com/Diucord/QWEN3-Finetuning.git]**

---
## Author
- Seyoon Oh
- Korea University : School of Industrial & Management Engineering
- Email : osy7336@korea.ac.kr
