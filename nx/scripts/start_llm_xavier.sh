#!/bin/bash
set -a
source .env
set +a

# Xavier GPU 최적화 (Jetson Xavier는 SM72 아키텍처)
./llama.cpp/build/bin/llama-server \
  -m models/"$LLM_MODEL".gguf \
  -t 8 \
  -c 4096 \
  -ngl 9999 \
  --host 0.0.0.0 \
  --port 8080
