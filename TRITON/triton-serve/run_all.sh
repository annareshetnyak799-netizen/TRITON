#!/bin/bash
# ============================================================
# run_all.sh — полный прогон на RunPod A100
# Запускать ОДИН РАЗ после git clone:
#   bash triton-serve/run_all.sh 2>&1 | tee /tmp/triton_run.log
#
# Время: ~20-30 мин (pull 8GB + build + warmup + benchmarks)
# Результат: /tmp/triton_results.csv + лог в /tmp/triton_run.log
# ============================================================

set -euo pipefail
LOG=/tmp/triton_run.log
CSV=/tmp/triton_results.csv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Цвета ─────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
step() { echo -e "\n${GREEN}[$(date +%H:%M:%S)] $*${NC}"; }
warn() { echo -e "${YELLOW}WARN: $*${NC}"; }
die()  { echo -e "${RED}ERROR: $*${NC}"; exit 1; }

# ── Зависимости ──────────────────────────────────────────────────────────────
step "0/7  Проверка окружения"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  || die "GPU не найдена. Нужен RunPod с NVIDIA GPU."
docker info > /dev/null 2>&1 || die "Docker не запущен."
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi -L \
  || die "NVIDIA Container Toolkit не настроен."
echo "Python: $(python3 --version)"
pip install -q tritonclient[grpc] httpx 2>/dev/null || warn "pip install failed, продолжаем"

# ── GPU режим в конфигах ──────────────────────────────────────────────────────
step "1/7  Переключение на GPU режим"
# config.pbtxt: CPU → GPU, count: 1 → 4
sed -i 's/KIND_CPU/KIND_GPU/g'  model_repository/gliner_guard/config.pbtxt
sed -i 's/count: 1/count: 4/g' model_repository/gliner_guard/config.pbtxt
# docker-compose.yml: раскомментируем deploy секцию
sed -i 's/#     deploy:/    deploy:/g'                     docker-compose.yml
sed -i 's/#       resources:/      resources:/g'           docker-compose.yml
sed -i 's/#         reservations:/        reservations:/g' docker-compose.yml
sed -i 's/#           devices:/          devices:/g'       docker-compose.yml
sed -i 's/#             - driver:/            - driver:/g' docker-compose.yml
sed -i 's/#               count:/              count:/g'   docker-compose.yml
sed -i 's/#               capabilities:/              capabilities:/g' docker-compose.yml
echo "Конфиги переключены на GPU."

# ── Pull + Build ──────────────────────────────────────────────────────────────
step "2/7  Pull Triton image (~2 мин на RunPod)"
docker pull nvcr.io/nvidia/tritonserver:25.01-py3

step "3/7  Build custom image с gliner2 deps (~5 мин)"
docker compose build triton

# ── Start Triton ──────────────────────────────────────────────────────────────
step "4/7  Запуск Triton сервера"
docker compose down 2>/dev/null || true
docker compose up -d triton

echo "Ожидание старта (model load + warmup, до 3 мин)..."
TIMEOUT=180
ELAPSED=0
until curl -sf http://localhost:8000/v2/health/live > /dev/null 2>&1; do
  sleep 5; ELAPSED=$((ELAPSED+5))
  printf "  %ds elapsed...\n" $ELAPSED
  [ $ELAPSED -ge $TIMEOUT ] && die "Triton не поднялся за ${TIMEOUT}s. Лог: docker compose logs triton"
done
echo "Triton LIVE!"

curl -sf http://localhost:8000/v2/models/gliner_guard/ready \
  || die "Модель не готова. Лог: docker compose logs triton"
echo "Модель готова."

# ── Smoke test ────────────────────────────────────────────────────────────────
step "5/7  Smoke test"
SMOKE=$(curl -sf -X POST http://localhost:8000/v2/models/gliner_guard/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs":[{"name":"text","shape":[1,1],"datatype":"BYTES","data":["Hi Anna, email anna@test.com"]}],"outputs":[{"name":"result"}]}')
echo "Ответ: $SMOKE" | python3 -m json.tool
echo "$SMOKE" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'outputs' in d" \
  || die "Smoke test провалился"
echo "Smoke test OK"

# ── REST Benchmark ────────────────────────────────────────────────────────────
step "6/7  REST Benchmark (128 запросов, concurrency=128)"
REST_OUT=$(python3 bench/bench_rest.py --num-requests 512 --concurrency 128 2>&1)
echo "$REST_OUT"

# Парсим CSV строку из вывода bench_rest.py
REST_CSV=$(echo "$REST_OUT" | grep "^CSV: triton" | tail -1 | sed 's/^CSV: //')

# ── gRPC Benchmark ────────────────────────────────────────────────────────────
step "7/7  gRPC Benchmark (512 запросов, concurrency=32)"
GRPC_OUT=$(python3 bench/bench_grpc.py --num-requests 512 --concurrency 32 2>&1)
echo "$GRPC_OUT"
GRPC_CSV=$(echo "$GRPC_OUT" | grep "^CSV: triton" | tail -1 | sed 's/^CSV: //')

# ── Результаты ───────────────────────────────────────────────────────────────
step "Итоговые результаты"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

{
  echo "backend,protocol,rps,p50_ms,p95_ms,p99_ms,errors,gpu"
  # LitServe baseline (из anna-notes / results/)
  echo "litserve,REST,148.2,570,1500,,0,A100"
  # Наши результаты
  echo "${REST_CSV},${GPU_NAME}"
  echo "${GRPC_CSV},${GPU_NAME}"
} > "$CSV"

echo ""
echo "========================================"
echo "  GPU: $GPU_NAME"
echo "========================================"
python3 - <<PYEOF
import csv, sys
rows = list(csv.DictReader(open("$CSV")))
print(f"{'Backend':<20} {'Proto':<6} {'RPS':>8} {'P50ms':>8} {'P95ms':>8} {'Errors':>8}")
print("-" * 60)
for r in rows:
    print(f"{r['backend']:<20} {r['protocol']:<6} {r['rps']:>8} {r['p50_ms']:>8} {r['p95_ms']:>8} {r.get('errors',''):>8}")
PYEOF

echo ""
echo "CSV сохранён: $CSV"
echo "Полный лог:   $LOG"
echo ""
echo "Скопируй CSV и отправь в чат с Claude для анализа:"
echo "  cat $CSV"
