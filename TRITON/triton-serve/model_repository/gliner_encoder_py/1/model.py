"""
Triton Python Backend: GLiNER2 encoder (ModernBERT), Python backend.

Runs only the encoder forward pass on GPU. Pre/postprocessing handled
by gliner_preprocessor and gliner_postprocessor.

TorchScript export (PyTorch backend) is not feasible for ModernBERT:
create_bidirectional_mask uses dynamic Python ops incompatible with
torch.jit.trace. Python backend achieves the same GPU execution.

Part of gliner_ensemble:
  gliner_preprocessor → [this model] → gliner_postprocessor

Protocol:
  Input:
    input_ids       INT64  [N, seq_len]
    attention_mask  FP32   [N, seq_len]
  Output:
    last_hidden_state  FP32  [N, seq_len, 384]
"""

import sys
import glob
import json

try:
    with open('/python_site_packages.txt') as _f:
        for _p in json.load(_f):
            if _p not in sys.path:
                sys.path.insert(0, _p)
except Exception:
    for _pat in [
        '/opt/conda/lib/python3.*/site-packages',
        '/usr/local/lib/python3.*/dist-packages',
        '/usr/local/lib/python3.*/site-packages',
    ]:
        for _p in sorted(glob.glob(_pat), reverse=True):
            if _p not in sys.path:
                sys.path.insert(0, _p)

import logging
import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from gliner2 import GLiNER2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "hivetrace/gliner-guard-uniencoder"


class TritonPythonModel:
    """Encoder: input_ids + attention_mask → last_hidden_state."""

    def initialize(self, args: dict) -> None:
        logger.info("Initializing GLiNER2 encoder (Python backend)...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model = GLiNER2.from_pretrained(MODEL_ID)
        self.encoder = model.encoder.to(device=self.device, dtype=torch.float32)
        self.encoder.eval()

        del model

        logger.info("GLiNER2 encoder ready on %s", self.device)

    def execute(self, requests: list) -> list:
        responses = []

        for request in requests:
            ids_np   = pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy()
            mask_np  = pb_utils.get_input_tensor_by_name(request, "attention_mask").as_numpy()

            try:
                input_ids      = torch.from_numpy(ids_np.copy()).to(self.device)       # [N, seq]
                attention_mask = torch.from_numpy(mask_np.copy()).to(self.device)      # [N, seq]

                with torch.no_grad():
                    outputs = self.encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                lhs = outputs.last_hidden_state.cpu().numpy().astype(np.float32)  # [N, seq, 384]

                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("last_hidden_state", lhs)]
                ))

            except Exception as e:
                logger.error("Encoder error: %s", e)
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                ))

        return responses

    def finalize(self) -> None:
        logger.info("Finalizing GLiNER2 encoder.")
        del self.encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
