#!/bin/bash
# RunPod startup script for Triton + GLiNER Guard.
# Runs automatically when the pod starts.

set -e

# SSH (RunPod injects public key via PUBLIC_KEY env var)
mkdir -p /root/.ssh
if [ -n "$PUBLIC_KEY" ]; then
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi
chmod 700 /root/.ssh
service ssh start
echo "[start.sh] SSH started"

# Clone / update model_repository from git
REPO_URL="https://github.com/annareshetnyak799-netizen/TRITON.git"
REPO_DIR="/workspace/TRITON"
BRANCH="orchestrator"

if [ ! -d "$REPO_DIR/.git" ]; then
    echo "[start.sh] Cloning repo..."
    git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$REPO_DIR"
else
    echo "[start.sh] Updating repo..."
    git -C "$REPO_DIR" pull origin "$BRANCH"
fi

MODEL_REPO="$REPO_DIR/TRITON/triton-serve/model_repository"
echo "[start.sh] Model repository: $MODEL_REPO"

# Switch config to GPU mode
sed -i 's/KIND_CPU/KIND_GPU/g'  "$MODEL_REPO/gliner_guard/config.pbtxt"
sed -i 's/count: 1/count: 4/g' "$MODEL_REPO/gliner_guard/config.pbtxt"
echo "[start.sh] Config switched to GPU (count: 4)"

# Start Triton (only load gliner_guard Python backend, skip encoder)
echo "[start.sh] Starting tritonserver..."
tritonserver \
    --model-repository="$MODEL_REPO" \
    --model-control-mode=explicit \
    --load-model=gliner_guard \
    --http-port=8000 \
    --grpc-port=8001 \
    --metrics-port=8002 \
    --log-verbose=1
