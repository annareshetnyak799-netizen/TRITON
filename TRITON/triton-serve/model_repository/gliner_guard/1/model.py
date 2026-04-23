"""
Triton Python Backend for GLiNER Guard model.

Mirrors the logic of gliner-guard-serve/litserve-baseline/main.py
but runs inside NVIDIA Triton Inference Server Python backend.

Protocol:
  Input:  BYTES tensor [batch_size, 1] — UTF-8 encoded text strings
  Output: BYTES tensor [batch_size, 1] — JSON-encoded result per sample

Triton Python backend contract:
  - TritonPythonModel.initialize(args) — called once on load
  - TritonPythonModel.execute(requests)  — called per batch
  - TritonPythonModel.finalize()         — called on shutdown
"""

import sys
import glob
# Triton Python backend stub uses isolated env — add all system dist-packages
# (torch may be in python3.10, gliner2 in python3.12 — add both)
for _p in sorted(glob.glob('/usr/local/lib/python3.*/dist-packages'), reverse=True):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
import logging
import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from gliner2 import GLiNER2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PII_LABELS = ["person", "address", "email", "phone"]
SAFETY_LABELS = ["safe", "unsafe"]
PII_THRESHOLD = 0.4
MODEL_ID = "hivetrace/gliner-guard-uniencoder"


class TritonPythonModel:
    """GLiNER Guard Triton Python Backend."""

    def initialize(self, args: dict) -> None:
        """Load model and build schema. Called once when Triton loads the model."""
        logger.info("Initializing GLiNER Guard Python backend...")

        self.model_config = json.loads(args["model_config"])

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", device)

        # Load model — same as LitServe baseline
        self.model = GLiNER2.from_pretrained(MODEL_ID)
        self.model.to(device).to(torch.float16)
        self.model.eval()

        # Build schema once — same labels as LitServe baseline
        self.schema = (
            self.model.create_schema()
            .entities(entity_types=PII_LABELS, threshold=PII_THRESHOLD)
            .classification(task="safety", labels=SAFETY_LABELS)
        )

        logger.info("GLiNER Guard backend ready on %s", device)

    def execute(self, requests: list) -> list:
        """
        Process a batch of Triton requests.

        Triton dynamic batching may merge multiple client requests into one
        execute() call. We flatten all texts, run batch_extract once,
        then split results back per original request.
        """
        responses = []

        # --- Collect all texts from all requests in this batch ---
        all_texts: list[str] = []
        request_sizes: list[int] = []  # how many texts each request contributed

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            # Shape: [N, 1], dtype BYTES — each element is a UTF-8 encoded string
            raw_bytes = input_tensor.as_numpy()
            texts = [b[0].decode("utf-8") if isinstance(b[0], bytes) else str(b[0])
                     for b in raw_bytes]
            all_texts.extend(texts)
            request_sizes.append(len(texts))

        logger.info("execute: total_texts=%d across %d requests",
                    len(all_texts), len(requests))

        # --- Single batch inference (same as LitServe predict()) ---
        try:
            results = self.model.batch_extract(
                texts=all_texts,
                schemas=self.schema,
                batch_size=len(all_texts),
            )
        except Exception as e:
            logger.error("batch_extract failed: %s", e)
            # Return error for every request
            error = pb_utils.TritonError(str(e))
            return [pb_utils.InferenceResponse(error=error) for _ in requests]

        # --- Split results back and build Triton responses ---
        offset = 0
        for request, size in zip(requests, request_sizes):
            batch_results = results[offset: offset + size]
            offset += size

            # Encode each result as JSON bytes, shape [N, 1]
            output_bytes = np.array(
                [[json.dumps(r, ensure_ascii=False).encode("utf-8")]
                 for r in batch_results],
                dtype=object,
            )

            out_tensor = pb_utils.Tensor("result", output_bytes)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self) -> None:
        """Clean up resources. Called once when Triton unloads the model."""
        logger.info("Finalizing GLiNER Guard Python backend.")
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
