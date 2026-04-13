#!/bin/bash
# RunPod A100 setup script for Triton GLiNER Guard
# Run this ONCE after creating the pod:
#   bash runpod_setup.sh
#
# Prerequisites (RunPod provides these automatically):
#   - NVIDIA GPU (A100 recommended)
#   - Docker + NVIDIA Container Toolkit already installed
#   - Internet access (nvcr.io reachable from RunPod)

set -e
echo "=== Triton GLiNER Guard — RunPod Setup ==="

# ── 1. Clone or update repo ───────────────────────────────────────────────────
REPO_DIR="$HOME/TRITON"
if [ -d "$REPO_DIR/.git" ]; then
    echo "[1/5] Updating repo..."
    git -C "$REPO_DIR" pull
else
    echo "[1/5] Cloning repo..."
    # Replace with your actual repo URL
    git clone https://github.com/YOUR_USERNAME/TRITON.git "$REPO_DIR"
fi
cd "$REPO_DIR/triton-serve"

# ── 2. Verify GPU ─────────────────────────────────────────────────────────────
echo "[2/5] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi -L

# ── 3. Pull Triton image (~8GB, ~2 min on RunPod) ─────────────────────────────
echo "[3/5] Pulling Triton image..."
docker pull --platform linux/amd64 nvcr.io/nvidia/tritonserver:25.01-py3

# ── 4. Build custom image with gliner2 deps ───────────────────────────────────
echo "[4/5] Building custom image..."
# Switch config to GPU mode
sed -i 's/KIND_CPU/KIND_GPU/g'   model_repository/gliner_guard/config.pbtxt
sed -i 's/count: 1/count: 4/g'  model_repository/gliner_guard/config.pbtxt
# Uncomment warmup block
sed -i 's/^# model_warmup/model_warmup/' model_repository/gliner_guard/config.pbtxt
sed -i 's/^#   {/  {/'            model_repository/gliner_guard/config.pbtxt
sed -i 's/^#     /    /g'         model_repository/gliner_guard/config.pbtxt

docker compose build triton

# ── 5. Start Triton ───────────────────────────────────────────────────────────
echo "[5/5] Starting Triton server..."
# GPU mode: uncomment deploy section in docker-compose.yml
docker compose up -d triton

echo ""
echo "Waiting for Triton (model download + warmup ~60s)..."
until curl -sf http://localhost:8000/v2/health/live > /dev/null 2>&1; do
    printf '.'
    sleep 5
done
echo ""
echo "=== Triton is LIVE ==="
curl -s http://localhost:8000/v2/models/gliner_guard/ready | python3 -m json.tool

echo ""
echo "Next steps:"
echo "  make smoke       — one test request"
echo "  make bench-rest  — REST benchmark"
echo "  make bench-grpc  — gRPC benchmark"
echo "  make bench-all   — full comparison"
