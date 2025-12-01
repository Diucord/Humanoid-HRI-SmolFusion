# SmolFusion: A Real-Time Vision-Language Fusion Architecture for Humanoid HRI

This project introduces **SmolFusion**, a novel, lightweight multimodal system that achieves **real-time Human-Robot Interaction (HRI)** by fusing a **fine-tuned VLM and LLM** running concurrently. Optimized for both server-based and on-device deployment, this architecture delivers low-latency, context-aware intelligence crucial for embodied agents.
The project supports both **"server-based"** and **"on-device"** environments, targeting real-time performance for **Human-Robot Interaction (HRI)**.

--- 
## Overview

- Traditional HRI has been limited to **command-based interactions**, which struggle with complex or unstructured situations.
- This project proposes a **multimodal architecture** that integrates **text, image, and speech inputs** to enable **real-time, natural interactions**.

**Key Research Highlights**

- Unified architecture combining **VLM + lightweight, fine-tuned LLM**
- Dual-path development : **high-performance server-based** and **lightweight on-device**
- **Session-based memory management** for contextual and long-term conversations
- Lightweight deployment suitable for embedded devices

---
## Architecture

### 1. Server-based 

- **Stack** : FastAPI + GPU Server (RTX 5080)
- **Features** :
    - High-performance inference with real-time response
    - Scalable to multiple users
    - Suitable for research and service demos

### 2. On-device 

- **Stack** : FastAPI + llama.cpp (lightweight GGUF models)
- **Features** :
    - Minimal reliance on network connectivity
    - Independent operation in low-power environments
    - Stronger privacy protection
    - Real-time inference on embedded devices (Jetson AGX Orin/NX)

---
## Modules

- **/chat** : Text-based dialogue queries
- **/analyze** : Image-based analysis and multimodal reasoning

All core modules are accessible via the **FastAPI Gateway (Port 8000)**

---
## Project Structure
```bash 
.
├── llama.cpp/        # External dependency (manual download required)
├── server-based/     # Server-based implementation & configs
├── on-device/        # On-device implementation & configs
├── requirements.txt  # Python dependencies
├── README.md         # Project documentation
└── .gitignore
```

---
## Future Work

- Advanced model compression for smaller devices
- Large-scale user testing for UX validation
- Extension to additional modules (multi-camera input, extended LLM integration)

---
## Author

- Seyoon Oh
- Korea University : School of Industrial & Management Engineering
- Email : osy7336@korea.ac.kr
