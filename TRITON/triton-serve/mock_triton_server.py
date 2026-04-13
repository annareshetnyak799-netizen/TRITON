"""
Mock Triton Inference Server (HTTP v2 protocol).

Implements the subset of Triton REST API used by bench_rest.py:
  GET  /v2/health/live
  GET  /v2/health/ready
  GET  /v2/models/{model}/ready
  POST /v2/models/{model}/infer

Also implements dynamic batching:
  Requests that arrive within BATCH_TIMEOUT_MS are grouped into one
  batch_extract() call — same as Triton dynamic_batching config.

This mock lets you run bench_rest.py locally without nvcr.io access.
On A100, swap this for real Triton — bench_rest.py doesn't change.

Usage:
  cd triton-serve
  python mock_triton_server.py          # port 8000
  python mock_triton_server.py --port 8003

Then in another terminal:
  python bench/bench_rest.py
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mock-triton")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME    = "gliner_guard"
MAX_BATCH     = 64          # mirrors config.pbtxt max_batch_size
BATCH_TIMEOUT = 0.05        # 50ms — mirrors max_queue_delay_microseconds: 50000
PII_LABELS    = ["person", "address", "email", "phone"]
SAFETY_LABELS = ["safe", "unsafe"]

# ── State shared across requests ──────────────────────────────────────────────
class AppState:
    model     = None
    schema    = None
    # Dynamic batching queue
    queue: asyncio.Queue = None
    batcher_task: asyncio.Task = None

state = AppState()


# ── Dynamic batcher ───────────────────────────────────────────────────────────

async def dynamic_batcher():
    """
    Collects requests from the queue for up to BATCH_TIMEOUT seconds,
    then runs one batch_extract() call for the whole batch.

    This mirrors Triton's dynamic_batching behaviour:
      preferred_batch_size: [1, 8, 16, 32, 64]
      max_queue_delay_microseconds: 50000
    """
    log.info("Dynamic batcher started (max_batch=%d, timeout=%.0fms)",
             MAX_BATCH, BATCH_TIMEOUT * 1000)

    while True:
        # Wait for the first item
        try:
            first = await state.queue.get()
        except asyncio.CancelledError:
            break

        batch = [first]
        deadline = time.monotonic() + BATCH_TIMEOUT

        # Drain the queue until timeout or max batch
        while len(batch) < MAX_BATCH:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(state.queue.get(), timeout=remaining)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        texts   = [item["text"]   for item in batch]
        futures = [item["future"] for item in batch]

        log.info("batch size=%d", len(batch))
        t0 = time.perf_counter()

        try:
            # Run batch_extract in a thread pool so it doesn't block the
            # asyncio event loop — batch_extract is CPU/GPU bound (synchronous)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: state.model.batch_extract(
                    texts=texts,
                    schemas=state.schema,
                    batch_size=len(texts),
                )
            )
            elapsed = (time.perf_counter() - t0) * 1000
            log.info("batch_extract done in %.0fms", elapsed)

            for fut, result in zip(futures, results):
                if not fut.done():
                    fut.set_result(result)

        except Exception as e:
            log.error("batch_extract error: %s", e)
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading GLiNER Guard model...")
    from gliner2 import GLiNER2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state.model = GLiNER2.from_pretrained("hivetrace/gliner-guard-uniencoder")
    state.model.to(device).to(torch.float16 if device == "cuda" else torch.float32)
    state.model.eval()

    state.schema = (
        state.model.create_schema()
        .entities(entity_types=PII_LABELS, threshold=0.4)
        .classification(task="safety", labels=SAFETY_LABELS)
    )
    log.info("Model ready on %s", device)

    state.queue = asyncio.Queue()
    state.batcher_task = asyncio.create_task(dynamic_batcher())

    yield  # server runs

    state.batcher_task.cancel()
    try:
        await state.batcher_task
    except asyncio.CancelledError:
        pass
    log.info("Shutdown complete")


app = FastAPI(title="Mock Triton Server", lifespan=lifespan)


# ── Health endpoints (Triton v2) ──────────────────────────────────────────────

@app.get("/v2/health/live")
async def health_live():
    return {"live": True}

@app.get("/v2/health/ready")
async def health_ready():
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"ready": True}

@app.get("/v2/models/{model_name}/ready")
async def model_ready(model_name: str):
    if model_name != MODEL_NAME or state.model is None:
        raise HTTPException(status_code=404, detail=f"Model {model_name!r} not found")
    return {"name": model_name, "ready": True}


# ── Inference endpoint (Triton v2) ────────────────────────────────────────────

@app.post("/v2/models/{model_name}/infer")
async def infer(model_name: str, request: Request):
    """
    Triton HTTP Inference Protocol v2.

    Expected request body:
    {
      "inputs": [{
        "name": "text",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": ["some text here"]
      }],
      "outputs": [{"name": "result"}]
    }

    Response:
    {
      "model_name": "gliner_guard",
      "outputs": [{
        "name": "result",
        "shape": [1, 1],
        "datatype": "BYTES",
        "data": ["{\"entities\": ..., \"safety\": ...}"]
      }]
    }
    """
    if model_name != MODEL_NAME:
        raise HTTPException(status_code=404, detail=f"Model {model_name!r} not found")

    body = await request.json()

    # Parse input tensor
    try:
        inputs = {inp["name"]: inp for inp in body.get("inputs", [])}
        text_input = inputs["text"]
        # data is list of strings (Triton BYTES encoding for HTTP)
        text = text_input["data"][0]
        if isinstance(text, bytes):
            text = text.decode("utf-8")
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=400, detail=f"Bad input: {e}")

    # Enqueue for dynamic batching
    loop = asyncio.get_event_loop()
    future: asyncio.Future = loop.create_future()
    await state.queue.put({"text": text, "future": future})

    # Wait for batcher to process this request
    try:
        result = await asyncio.wait_for(future, timeout=120.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")

    # Build Triton v2 response
    result_str = json.dumps(result, ensure_ascii=False)
    return {
        "model_name": MODEL_NAME,
        "model_version": "1",
        "outputs": [
            {
                "name":     "result",
                "shape":    [1, 1],
                "datatype": "BYTES",
                "data":     [result_str],
            }
        ],
    }


# ── LitServe-compatible endpoint (for bench_rest.py --litserve) ──────────────

@app.post("/predict")
async def predict_litserve(request: Request):
    """LitServe-compatible endpoint for comparison benchmarks."""
    body = await request.json()
    text = body.get("text", "")

    loop = asyncio.get_event_loop()
    future: asyncio.Future = loop.create_future()
    await state.queue.put({"text": text, "future": future})

    try:
        result = await asyncio.wait_for(future, timeout=120.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock Triton HTTP v2 server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--workers", type=int, default=1,
                        help="Uvicorn workers (keep 1 — model lives in main process)")
    args = parser.parse_args()

    log.info("Starting Mock Triton server on %s:%d", args.host, args.port)
    uvicorn.run(
        "mock_triton_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="warning",   # uvicorn access log quieter, our log handles info
    )
