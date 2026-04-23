"""
Locust gRPC load test for Triton GLiNER Guard — matches REST baseline conditions.

Replicates locust_triton.py but uses Triton gRPC protocol (port 8001).
Results are directly comparable with locust_triton.py REST results.

Conditions identical to REST Locust test:
  - 100 users, 15 min, constant_throughput(5)
  - Same prompts.csv / responses.csv texts
  - External client via RunPod TCP proxy

Usage (from repo root):
  locust -f triton-serve/bench/locust_triton_grpc.py \\
      --host 213.173.102.4:12417 \\
      --users 100 --spawn-rate 10 --run-time 15m --headless \\
      --csv triton-serve/results/locust-triton-grpc-15min

Set CSV path via env var if prompts/responses are not in default location:
  CSV_DIR=/path/to/gliner-guard-serve/test-script locust -f ...
"""

import os
import time

import numpy as np
import pandas as pd
from locust import User, task, constant_throughput

# gRPC uses C-level networking — not compatible with gevent monkey-patching.
# Solution: run each gRPC call in a real OS thread via gevent's threadpool.
# get_hub().threadpool.apply() blocks the current greenlet (not the whole loop)
# while the OS thread does the blocking I/O — correct pattern for gevent + gRPC.
from gevent.hub import get_hub
from locust import events

# gRPC uses C-level networking — not compatible with gevent monkey-patching.
# Solution: run each gRPC call in a real OS thread via gevent's threadpool.
# Default threadpool size is ~10; with 100 Locust users this creates a queue
# (100 users / 10 threads = 10x latency multiplier). Set to 200 before test starts.
@events.init.add_listener
def set_threadpool_size(environment, **kwargs):
    pool = get_hub().threadpool
    pool.maxsize = 200
    pool.size = 200

try:
    import tritonclient.grpc as grpcclient
except ImportError:
    raise SystemExit('tritonclient[grpc] not installed. Run: pip install "tritonclient[grpc]"')

_here = os.path.dirname(os.path.abspath(__file__))
_default_csv_dir = os.path.join(_here, "..", "..", "gliner-guard-serve", "test-script")
CSV_DIR = os.getenv("CSV_DIR", _default_csv_dir)

prompts = pd.read_csv(os.path.join(CSV_DIR, "prompts.csv"))
responses = pd.read_csv(os.path.join(CSV_DIR, "responses.csv"))

MODEL_NAME = "gliner_guard"
REQUEST_NAME = "/v2/models/gliner_guard/infer"


def _build_inputs(text: str) -> list:
    inp = grpcclient.InferInput("text", [1, 1], "BYTES")
    inp.set_data_from_numpy(np.array([[text.encode("utf-8")]], dtype=object))
    return [inp]


def _build_outputs() -> list:
    return [grpcclient.InferRequestedOutput("result")]


class TritonGrpcUser(User):
    """
    One Locust user = one persistent gRPC channel to Triton.
    constant_throughput(5): each user targets 5 req/s → 100 users = 500 req/s demand.
    """
    host = os.getenv("GLINER_HOST", "localhost:8001")
    wait_time = constant_throughput(5)

    def on_start(self):
        self._client = grpcclient.InferenceServerClient(
            url=self.host, verbose=False
        )

    def on_stop(self):
        self._client.close()

    def _blocking_infer(self, text: str) -> None:
        """Runs in a real OS thread — safe for gRPC's C-level networking."""
        self._client.infer(
            model_name=MODEL_NAME,
            inputs=_build_inputs(text),
            outputs=_build_outputs(),
        )

    def _infer(self, text: str) -> None:
        t0 = time.perf_counter()
        exc = None
        try:
            # Run blocking gRPC call in OS thread; yields greenlet, not event loop.
            get_hub().threadpool.apply(self._blocking_infer, (text,))
        except Exception as e:
            exc = e
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self.environment.events.request.fire(
            request_type="gRPC",
            name=REQUEST_NAME,
            response_time=elapsed_ms,
            response_length=0,
            exception=exc,
            context=self.context(),
        )

    @task
    def predict_prompt(self):
        text = prompts.sample(n=1).iloc[0]["user_msg"]
        self._infer(text)

    @task
    def predict_response(self):
        text = responses.sample(n=1).iloc[0]["assistant_msg"]
        self._infer(text)
