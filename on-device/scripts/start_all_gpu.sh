#!/bin/bash
set -a
source .env
set +a

check_port() {
    local port=$1
    local name=$2
    if lsof -i:$port -sTCP:LISTEN >/dev/null; then
        echo "[✅] $name server is running on port $port"
    else
        echo "[❌] $name server is NOT running on port $port"
        exit 1
    fi
}

check_cuda() {
    echo "[INFO] Checking CUDA availability..."
    if command -v nvcc &>/dev/null; then
        nvcc --version
    else
        echo "[WARN] Could not find nvcc command (this may be normal on Jetson)"
    fi

    # PyTorch CUDA check (for VLM/LLM server)
    if python3 -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)"; then
        echo "[✅] CUDA available (torch.cuda is enabled)"
    else
        echo "[❌] CUDA not available! Please check driver, CUDA, and PyTorch settings."
        exit 1
    fi
}

./llama.cpp/build/bin/llama-server \
    --model models/"$LLM_MODEL" \
    --host 0.0.0.0 \
    --port $LLM_PORT \
    --ctx-size $CTX \
    --threads $THREADS \
    --batch-size $BATCH \
    --n-gpu-layers $NGL > logs/llm.log 2>&1 &
LLM_PID=$!
sleep 1
check_port $LLM_PORT "LLM"

./llama.cpp/build/bin/llama-server \
    --model models/"$VLM_MODEL" \
    --mmproj models/"$VLM_MMPROJ" \
    --host 0.0.0.0 \
    --port $VLM_PORT \
    --ctx-size $CTX \
    --threads $THREADS \
    --batch-size $BATCH \
    --n-gpu-layers $NGL > logs/vlm.log 2>&1 &
VLM_PID=$!
sleep 1
check_port $VLM_PORT "VLM"

uvicorn app.server:app --host 0.0.0.0 --port $GATEWAY_PORT > logs/api.log 2>&1 &
API_PID=$!
sleep 1
check_port $GATEWAY_PORT "FastAPI"

echo "[INFO] All servers have started successfully"
echo "PID list: LLM=$LLM_PID, VLM=$VLM_PID, API=$API_PID"
wait