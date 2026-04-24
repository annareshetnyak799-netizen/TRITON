"""
Locust load test for Triton GLiNER Guard — matches LitServe baseline conditions exactly.

Replicates gliner-guard-serve/test-script/test-gliner.py but uses Triton v2 HTTP protocol:
  POST /v2/models/gliner_guard/infer

Conditions identical to baseline:
  - 100 users, 15 min, constant_throughput(5)
  - Same prompts.csv / responses.csv texts
  - External client via RunPod proxy

Usage (run from repo root or triton-serve/bench/):
  locust -f triton-serve/bench/locust_triton.py \\
      --host http://<RUNPOD_HOST>:<PORT> \\
      --users 100 --spawn-rate 10 --run-time 15m --headless \\
      --csv triton-serve/results/locust-triton

  # If running from gliner-guard-serve/test-script/ (CSVs in same dir):
  locust -f ../../triton-serve/bench/locust_triton.py --host http://...

Set CSV path via env var if prompts/responses are not in cwd:
  CSV_DIR=/path/to/gliner-guard-serve/test-script locust -f ...
"""

import json
import os
import random


import pandas as pd
from locust import FastHttpUser, task, constant_throughput

# Allow overriding CSV location via env (default: same dir as this file → fallback: test-script)
_here = os.path.dirname(os.path.abspath(__file__))
_default_csv_dir = os.path.join(_here, "..", "..", "gliner-guard-serve", "test-script")
CSV_DIR = os.getenv("CSV_DIR", _default_csv_dir)

prompts = pd.read_csv(os.path.join(CSV_DIR, "prompts.csv"))
responses = pd.read_csv(os.path.join(CSV_DIR, "responses.csv"))

_MODEL_NAME = os.getenv("MODEL_NAME", "gliner_guard")
TRITON_ENDPOINT = f"/v2/models/{_MODEL_NAME}/infer"


def _triton_payload(text: str) -> dict:
    return {
        "inputs": [
            {
                "name": "text",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [text],
            }
        ],
        "outputs": [{"name": "result"}],
    }


class TritonUser(FastHttpUser):
    host = os.getenv("GLINER_HOST", "http://localhost:8000")
    wait_time = constant_throughput(5)

    @task
    def predict_prompt(self):
        text = prompts.sample(n=1).iloc[0]["user_msg"]
        self.client.post(
            TRITON_ENDPOINT,
            json=_triton_payload(text),
            headers={"Content-Type": "application/json"},
        )

    @task
    def predict_response(self):
        text = responses.sample(n=1).iloc[0]["assistant_msg"]
        self.client.post(
            TRITON_ENDPOINT,
            json=_triton_payload(text),
            headers={"Content-Type": "application/json"},
        )
