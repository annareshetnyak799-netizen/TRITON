"""
REST benchmark for Triton GLiNER Guard model.

Uses Triton HTTP Inference Protocol v2:
  POST /v2/models/gliner_guard/infer

Mirrors gliner-guard-serve/litserve-baseline/bench.py for apples-to-apples
comparison: 128 concurrent async requests, same 16 test texts.

Usage:
  # Against Triton (default)
  python bench_rest.py

  # Against LitServe baseline (for comparison)
  python bench_rest.py --url http://localhost:8000/predict --litserve

  # More requests
  python bench_rest.py --num-requests 512 --concurrency 256
"""

import argparse
import asyncio
import json
import statistics
import time

import httpx
import numpy as np

TRITON_URL = "http://localhost:8000/v2/models/gliner_guard/infer"
LITSERVE_URL = "http://localhost:8000/predict"

# Same texts as LitServe bench.py
TEXTS = [
    "Send $500 to John Smith at john.smith@gmail.com or I'll leak your photos",
    "My name is Alice and I live at 123 Main St, Springfield",
    "Call me at +1-555-123-4567 to discuss the contract",
    "This is a completely safe and normal message about weather",
    "Transfer money to bob@example.com immediately or else",
    "Meeting at 10am with Dr. James Wilson at 456 Oak Ave",
    "Hey, just wanted to say hello and ask about your day",
    "Send your SSN and credit card number to secure@totallylegit.com",
    "I will find where you live and make you regret this",
    "Please forward the report to maria.garcia@company.org",
    "The quick brown fox jumps over the lazy dog",
    "Wire $10000 to account 1234-5678-9012 before midnight",
    "Contact support at help@service.com for assistance",
    "You better watch your back, I know where your kids go to school",
    "Jane Doe, 789 Pine Road, Apt 4B, New York, NY 10001",
    "Let's grab coffee tomorrow at the usual place",
]


def build_triton_payload(text: str) -> dict:
    """
    Build Triton HTTP v2 inference request payload.
    Input tensor: BYTES shape [1, 1] — one text string.
    """
    encoded = text.encode("utf-8")
    return {
        "inputs": [
            {
                "name": "text",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [encoded.decode("utf-8")],
            }
        ],
        "outputs": [{"name": "result"}],
    }


async def send_triton(client: httpx.AsyncClient, text: str, url: str) -> tuple[dict, float]:
    """Send one request to Triton REST endpoint. Returns (parsed_result, latency_ms)."""
    payload = build_triton_payload(text)
    t0 = time.perf_counter()
    resp = await client.post(url, json=payload)
    latency_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    body = resp.json()
    # Triton response: outputs[0].data contains the JSON string
    result_str = body["outputs"][0]["data"][0]
    result = json.loads(result_str)
    return result, latency_ms


async def send_litserve(client: httpx.AsyncClient, text: str, url: str) -> tuple[dict, float]:
    """Send one request to LitServe /predict endpoint. Returns (parsed_result, latency_ms)."""
    t0 = time.perf_counter()
    resp = await client.post(url, json={"text": text})
    latency_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    return resp.json(), latency_ms


async def run_bench(
    url: str,
    num_requests: int,
    concurrency: int,
    mode: str,
) -> None:
    """Run benchmark: send num_requests with max concurrency concurrent requests."""
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors = 0

    send_fn = send_triton if mode == "triton" else send_litserve

    async def bounded_send(i: int) -> None:
        nonlocal errors
        text = TEXTS[i % len(TEXTS)]
        async with semaphore:
            try:
                _, lat = await send_fn(client, text, url)
                latencies.append(lat)
            except Exception as e:
                errors += 1
                print(f"  [ERROR] request {i}: {e}")

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=60, limits=limits) as client:
        # Warmup: 1 request
        print("Warming up...")
        await send_fn(client, TEXTS[0], url)

        print(f"Running {num_requests} requests (concurrency={concurrency})...")
        tasks = [bounded_send(i) for i in range(num_requests)]
        t_start = time.perf_counter()
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t_start

    # Stats
    successful = len(latencies)
    rps = successful / elapsed if elapsed > 0 else 0
    p50 = statistics.median(latencies) if latencies else 0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0
    p99 = float(np.percentile(latencies, 99)) if latencies else 0
    avg = statistics.mean(latencies) if latencies else 0

    print()
    print("=" * 50)
    print(f"  Backend      : Triton REST ({url})" if mode == "triton" else f"  Backend      : LitServe REST ({url})")
    print(f"  Requests     : {num_requests}  (successful: {successful}, errors: {errors})")
    print(f"  Elapsed      : {elapsed:.2f}s")
    print(f"  RPS          : {rps:.1f}")
    print(f"  Avg latency  : {avg:.1f}ms")
    print(f"  P50 latency  : {p50:.1f}ms")
    print(f"  P95 latency  : {p95:.1f}ms")
    print(f"  P99 latency  : {p99:.1f}ms")
    print("=" * 50)

    # CSV-friendly one-liner (for gen-benchmark-table.py)
    print()
    print("CSV: backend,protocol,rps,p50_ms,p95_ms,p99_ms,errors")
    backend = "triton-python" if mode == "triton" else "litserve"
    print(f"CSV: {backend},REST,{rps:.1f},{p50:.1f},{p95:.1f},{p99:.1f},{errors}")


def main() -> None:
    parser = argparse.ArgumentParser(description="REST benchmark for GLiNER Guard on Triton")
    parser.add_argument("--url", default=TRITON_URL, help="Inference endpoint URL")
    parser.add_argument("--litserve", action="store_true",
                        help="Use LitServe /predict protocol instead of Triton v2")
    parser.add_argument("--num-requests", type=int, default=128,
                        help="Total number of requests to send (default: 128)")
    parser.add_argument("--concurrency", type=int, default=128,
                        help="Max concurrent requests (default: 128)")
    args = parser.parse_args()

    mode = "litserve" if args.litserve else "triton"
    url = args.url
    if args.litserve and url == TRITON_URL:
        url = LITSERVE_URL

    asyncio.run(run_bench(url, args.num_requests, args.concurrency, mode))


if __name__ == "__main__":
    main()
