"""
gRPC benchmark for Triton GLiNER Guard model.

Uses tritonclient.grpc (Triton gRPC Inference Protocol v2).
Runs against port 8001 (Triton default gRPC port).

Compare results with bench_rest.py to quantify REST vs gRPC overhead.
Expected: gRPC ~15-30% faster due to binary Protobuf serialization.

Usage:
  python bench_grpc.py

  # More requests / higher concurrency
  python bench_grpc.py --num-requests 512 --concurrency 256

Dependencies (install inside Triton client container or locally):
  pip install tritonclient[grpc]
"""

import argparse
import json
import queue
import statistics
import threading
import time

import numpy as np

try:
    import tritonclient.grpc as grpcclient
    from tritonclient.grpc import InferenceServerException
except ImportError:
    raise SystemExit(
        "tritonclient[grpc] is not installed.\n"
        "Run: pip install tritonclient[grpc]"
    )

TRITON_GRPC_URL = "localhost:8001"
MODEL_NAME = "gliner_guard"

# Same texts as bench_rest.py / litserve bench.py
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


def build_grpc_inputs(text: str) -> list:
    """Build Triton gRPC InferInput for one text string."""
    # BYTES tensor shape [1, 1]
    input_tensor = grpcclient.InferInput("text", [1, 1], "BYTES")
    # numpy array of bytes objects, shape [1, 1]
    data = np.array([[text.encode("utf-8")]], dtype=object)
    input_tensor.set_data_from_numpy(data)
    return [input_tensor]


def build_grpc_outputs() -> list:
    """Request the 'result' output tensor."""
    return [grpcclient.InferRequestedOutput("result")]


def send_sync(client: grpcclient.InferenceServerClient, text: str) -> tuple[dict, float]:
    """Send one synchronous gRPC inference request. Returns (result, latency_ms)."""
    inputs = build_grpc_inputs(text)
    outputs = build_grpc_outputs()

    t0 = time.perf_counter()
    response = client.infer(
        model_name=MODEL_NAME,
        inputs=inputs,
        outputs=outputs,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    result_bytes = response.as_numpy("result")  # shape [1, 1], dtype object
    result = json.loads(result_bytes[0][0].decode("utf-8"))
    return result, latency_ms


def run_bench_threaded(
    url: str,
    num_requests: int,
    concurrency: int,
) -> None:
    """
    Thread-pool benchmark using tritonclient.grpc synchronous calls.

    gRPC connections are not async-friendly in tritonclient, so we use
    a thread pool — each thread holds its own client connection.
    """
    latencies: list[float] = []
    errors: list[str] = []
    latency_lock = threading.Lock()
    task_queue: queue.Queue[int] = queue.Queue()

    for i in range(num_requests):
        task_queue.put(i)

    def worker() -> None:
        # Each thread creates its own gRPC client (connection reuse within thread)
        client = grpcclient.InferenceServerClient(url=url, verbose=False)
        while True:
            try:
                i = task_queue.get_nowait()
            except queue.Empty:
                break
            text = TEXTS[i % len(TEXTS)]
            try:
                _, lat = send_sync(client, text)
                with latency_lock:
                    latencies.append(lat)
            except Exception as e:
                with latency_lock:
                    errors.append(str(e))
            finally:
                task_queue.task_done()

    # Warmup
    print("Warming up (1 request)...")
    warmup_client = grpcclient.InferenceServerClient(url=url, verbose=False)
    send_sync(warmup_client, TEXTS[0])

    print(f"Running {num_requests} requests (concurrency={concurrency} threads)...")
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]

    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
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
    print(f"  Backend      : Triton gRPC ({url})")
    print(f"  Model        : {MODEL_NAME}")
    print(f"  Requests     : {num_requests}  (successful: {successful}, errors: {len(errors)})")
    print(f"  Elapsed      : {elapsed:.2f}s")
    print(f"  RPS          : {rps:.1f}")
    print(f"  Avg latency  : {avg:.1f}ms")
    print(f"  P50 latency  : {p50:.1f}ms")
    print(f"  P95 latency  : {p95:.1f}ms")
    print(f"  P99 latency  : {p99:.1f}ms")
    print("=" * 50)

    if errors:
        print(f"\n  First 5 errors:")
        for e in errors[:5]:
            print(f"    {e}")

    # CSV-friendly one-liner
    print()
    print("CSV: backend,protocol,rps,p50_ms,p95_ms,p99_ms,errors")
    print(f"CSV: triton-python,gRPC,{rps:.1f},{p50:.1f},{p95:.1f},{p99:.1f},{len(errors)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="gRPC benchmark for GLiNER Guard on Triton")
    parser.add_argument("--url", default=TRITON_GRPC_URL,
                        help=f"Triton gRPC server (default: {TRITON_GRPC_URL})")
    parser.add_argument("--num-requests", type=int, default=128,
                        help="Total number of requests (default: 128)")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="Number of parallel threads (default: 32)")
    args = parser.parse_args()

    # Verify server is up
    try:
        client = grpcclient.InferenceServerClient(url=args.url, verbose=False)
        if not client.is_server_live():
            raise SystemExit(f"Triton server at {args.url} is not live")
        if not client.is_model_ready(MODEL_NAME):
            raise SystemExit(f"Model '{MODEL_NAME}' is not ready on {args.url}")
    except Exception as e:
        raise SystemExit(f"Cannot connect to Triton at {args.url}: {e}")

    run_bench_threaded(args.url, args.num_requests, args.concurrency)


if __name__ == "__main__":
    main()
