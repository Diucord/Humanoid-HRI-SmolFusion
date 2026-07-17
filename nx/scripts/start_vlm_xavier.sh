#!/bin/bash
set -a
source .env
set +a

# Xavier GPU 전용 VLM 실행
python3 -m vlm_server.main \
    --model-path models/"$VLM_MODEL" \
    --device cuda \
    --host 0.0.0.0 \
    --port 8081
