# Makefile for Triton GLiNER Guard serving
# Usage: make <target>

.PHONY: help start stop logs bench-rest bench-grpc bench-all bench-locust \
        smoke health check-model

TRITON_REST  := http://localhost:8000
TRITON_GRPC  := localhost:8001
MODEL        := gliner_guard
NUM_REQUESTS := 128
CONCURRENCY  := 128

help:          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*##"}{printf "  %-20s %s\n",$$1,$$2}'

# ─── Build ───────────────────────────────────────────────────────────────────

build:         ## Build custom Triton image with gliner2 deps (~5-10 min first time)
	docker compose build triton

# ─── Server lifecycle ─────────────────────────────────────────────────────────

start:         ## Build (if needed) and start Triton server
	docker compose up -d --build triton
	@echo "Waiting for Triton to be ready (may take 3-5 min on CPU)..."
	@until curl -sf $(TRITON_REST)/v2/health/live > /dev/null 2>&1; do \
	  printf '.'; sleep 5; \
	done
	@echo ""
	@echo "Triton is live at $(TRITON_REST)"

stop:          ## Stop Triton server
	docker compose down

logs:          ## Tail Triton logs
	docker compose logs -f triton

restart:       ## Restart Triton (picks up model_repository changes)
	docker compose restart triton

# ─── Health checks ────────────────────────────────────────────────────────────

health:        ## Check Triton server health (REST)
	@curl -s $(TRITON_REST)/v2/health/live  | python3 -m json.tool || echo "server not live"
	@curl -s $(TRITON_REST)/v2/health/ready | python3 -m json.tool || echo "server not ready"

check-model:   ## Check model is loaded and ready
	@curl -s $(TRITON_REST)/v2/models/$(MODEL)/ready | python3 -m json.tool

smoke:         ## Send one REST request and print result
	@curl -sf -X POST $(TRITON_REST)/v2/models/$(MODEL)/infer \
	  -H "Content-Type: application/json" \
	  -d '{"inputs":[{"name":"text","shape":[1,1],"datatype":"BYTES","data":["Hi, my name is John and my email is john@test.com"]}],"outputs":[{"name":"result"}]}' \
	  | python3 -m json.tool

# ─── Benchmarks ──────────────────────────────────────────────────────────────

bench-rest:    ## REST benchmark (128 concurrent requests)
	cd bench && python bench_rest.py \
	  --url $(TRITON_REST)/v2/models/$(MODEL)/infer \
	  --num-requests $(NUM_REQUESTS) \
	  --concurrency $(CONCURRENCY)

bench-grpc:    ## gRPC benchmark (128 requests, 32 threads)
	cd bench && python bench_grpc.py \
	  --url $(TRITON_GRPC) \
	  --num-requests $(NUM_REQUESTS) \
	  --concurrency 32

bench-litserve: ## REST benchmark against LitServe baseline (must be running on :8000)
	cd bench && python bench_rest.py \
	  --litserve \
	  --url http://localhost:8000/predict \
	  --num-requests $(NUM_REQUESTS) \
	  --concurrency $(CONCURRENCY)

bench-all:     ## Run REST + gRPC benchmarks and print comparison
	@echo "=== REST ===" && $(MAKE) bench-rest
	@echo ""
	@echo "=== gRPC ===" && $(MAKE) bench-grpc

bench-locust:  ## Run Locust load test (15 min, 100 users)
	docker compose --profile bench up locust

# ─── Results ─────────────────────────────────────────────────────────────────

bench-table:   ## Generate markdown table from CSV results
	cd ../gliner-guard-serve && python scripts/gen-benchmark-table.py
