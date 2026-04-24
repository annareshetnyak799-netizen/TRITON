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

# Switch configs to GPU mode
sed -i 's/KIND_CPU/KIND_GPU/g'  "$MODEL_REPO/gliner_guard/config.pbtxt"
sed -i 's/count: 1/count: 4/g' "$MODEL_REPO/gliner_guard/config.pbtxt"
echo "[start.sh] Configs switched to GPU (gliner_guard: count=4)"

# Используем тот же Python, что использовал pip во время сборки образа.
# sys.executable возвращает абсолютный путь к текущему интерпретатору — без угадывания.
PYTHON_RUNTIME=$(python3 -c "import sys; print(sys.executable)")
SITE_PKG=$(python3 -c "import site; print(':'.join(site.getsitepackages()))")
echo "[start.sh] Python runtime : $PYTHON_RUNTIME"
echo "[start.sh] Site packages  : $SITE_PKG"

# Передаём site-packages дочернему subprocess stub-а через PYTHONPATH
export PYTHONPATH="${SITE_PKG}${PYTHONPATH:+:$PYTHONPATH}"
echo "[start.sh] PYTHONPATH=$PYTHONPATH"

# Проверяем импорты до запуска Triton — быстрый fail с понятным сообщением
echo "[start.sh] Verifying imports..."
python3 -c "import torch; print('torch', torch.__version__, 'at', torch.__file__)" \
    || { echo "FATAL: torch not importable via $PYTHON_RUNTIME"; exit 1; }
python3 -c "from gliner2 import GLiNER2; print('gliner2 ok')" \
    || { echo "FATAL: gliner2 not importable"; exit 1; }

# Start Triton: gliner_guard (Python backend, full pipeline)
#              + gliner_ensemble (Python pre/encoder/postprocessor)
echo "[start.sh] Starting tritonserver..."
tritonserver \
    --model-repository="$MODEL_REPO" \
    --model-control-mode=explicit \
    --load-model=gliner_guard \
    --load-model=gliner_preprocessor \
    --load-model=gliner_encoder_py \
    --load-model=gliner_postprocessor \
    --load-model=gliner_ensemble \
    --backend-config=python,python-runtime-path="$PYTHON_RUNTIME" \
    --http-port=8000 \
    --grpc-port=8001 \
    --metrics-port=8002 \
    --log-verbose=1
