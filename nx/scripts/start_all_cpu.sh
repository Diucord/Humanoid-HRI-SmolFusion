#!/bin/bash
set -a
source .env
set +a

check_port() {
    local port=$1
    local name=$2
    if lsof -i:$port -sTCP:LISTEN >/dev/null; then
        echo "[✅] $name 서버가 포트 $port에서 실행 중"
    else
        echo "[❌] $name 서버가 포트 $port에서 실행되지 않음"
        exit 1
    fi
}

echo "=== LLM 서버 (8080) 시작 ==="
./llama.cpp/build/bin/llama-server \
    --model models/"$LLM_MODEL" \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 4096 \
    --threads 8 \
    --log-level error > /dev/null 2>&1 &
LLM_PID=$!
sleep 2
check_port 8080 "LLM"

echo "=== VLM 서버 (8081) 시작 ==="
./llama.cpp/build/bin/llama-server \
    --model models/"$VLM_MODEL" \
    --mmproj models/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf \
    --host 0.0.0.0 \
    --port 8081 \
    --ctx-size 4096 \
    --threads 8 \
    --log-level error > /dev/null 2>&1 &
VLM_PID=$!
sleep 2
check_port 8081 "VLM"

echo "=== FastAPI 서버 (8000) 시작 ==="
uvicorn app.server:app --host 0.0.0.0 --port 8000 &
API_PID=$!
sleep 2
check_port 8000 "FastAPI"

echo "[INFO] 모든 서버 실행 완료"
echo "PID 목록: LLM=$LLM_PID, VLM=$VLM_PID, API=$API_PID"

wait