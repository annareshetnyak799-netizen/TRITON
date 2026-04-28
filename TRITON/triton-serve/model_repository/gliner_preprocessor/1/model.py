"""
Triton Python Backend: GLiNER2 Preprocessor.

Tokenizes raw text strings using GLiNER2's SchemaTransformer (ExtractorCollator)
with a fixed PII + safety schema.

Part of gliner_ensemble:
  [this model] → input_ids, attention_mask → gliner_guard_encoder (PyTorch backend)
              → text_word_indices, meta    → gliner_postprocessor (bypasses encoder)

Protocol:
  Input:  text   BYTES  [N, 1]  — UTF-8 text strings
  Output:
    input_ids          INT64  [N, seq_len]   — token IDs (text + schema tokens)
    attention_mask     FP32   [N, seq_len]
    text_word_indices  INT64  [N, max_words] — positions of text words in full seq
    meta               BYTES  [N, 1]         — JSON with per-sample batch metadata
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
import triton_python_backend_utils as pb_utils

from gliner2 import GLiNER2
from gliner2.training.trainer import ExtractorCollator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PII_LABELS = ["person", "address", "email", "phone"]
SAFETY_LABELS = ["safe", "unsafe"]
PII_THRESHOLD = 0.4
MODEL_ID = "hivetrace/gliner-guard-uniencoder"


class TritonPythonModel:
    """Preprocessor: text → token tensors + per-sample metadata."""

    def initialize(self, args: dict) -> None:
        logger.info("Initializing GLiNER2 preprocessor...")

        model = GLiNER2.from_pretrained(MODEL_ID)
        model.eval()

        self.processor = model.processor
        self.collator = ExtractorCollator(model.processor, is_training=False)

        # Build fixed schema dict once
        schema = (
            model.create_schema()
            .entities(entity_types=PII_LABELS, threshold=PII_THRESHOLD)
            .classification(task="safety", labels=SAFETY_LABELS)
        )
        self.schema_dict = schema.build()
        for cls_config in self.schema_dict.get("classifications", []):
            cls_config.setdefault("true_label", ["N/A"])

        logger.info("GLiNER2 preprocessor ready (CPU)")

    def execute(self, requests: list) -> list:
        responses = []

        for request in requests:
            raw = pb_utils.get_input_tensor_by_name(request, "text").as_numpy()
            texts = [
                b[0].decode("utf-8") if isinstance(b[0], bytes) else str(b[0])
                for b in raw
            ]
            n = len(texts)

            try:
                dataset = [(text, self.schema_dict) for text in texts]
                batch = self.collator(dataset)

                input_ids_np = batch.input_ids.numpy().astype(np.int64)       # [N, seq_len]
                attn_mask_np = batch.attention_mask.numpy().astype(np.float32) # [N, seq_len]

                if batch.text_word_indices is not None:
                    twi_np = batch.text_word_indices.numpy().astype(np.int64)  # [N, max_words]
                else:
                    twi_np = np.zeros((n, 1), dtype=np.int64)

                # Serialize per-sample metadata to JSON BYTES [N, 1]
                meta_list = []
                for i in range(n):
                    schema_special = batch.schema_special_indices[i] if batch.schema_special_indices else []
                    m = {
                        "text_word_counts":     (batch.text_word_counts[i] if batch.text_word_counts else 0),
                        "schema_counts":        batch.schema_counts[i],
                        "schema_special_indices": schema_special,
                        "task_types":           batch.task_types[i],
                        "schema_tokens_list":   batch.schema_tokens_list[i],
                        "text_tokens":          batch.text_tokens[i],
                        "original_text":        batch.original_texts[i],
                        "original_schema":      batch.original_schemas[i],
                        "start_mappings":       batch.start_mappings[i],
                        "end_mappings":         batch.end_mappings[i],
                    }
                    meta_list.append([json.dumps(m, ensure_ascii=False).encode("utf-8")])

                meta_np = np.array(meta_list, dtype=object)  # [N, 1]

                out_input_ids = pb_utils.Tensor("input_ids", input_ids_np)
                out_attn_mask = pb_utils.Tensor("attention_mask", attn_mask_np)
                out_twi       = pb_utils.Tensor("text_word_indices", twi_np)
                out_meta      = pb_utils.Tensor("meta", meta_np)

                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[out_input_ids, out_attn_mask, out_twi, out_meta]
                ))

            except Exception as e:
                logger.error("Preprocessor error: %s", e)
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                ))

        return responses

    def finalize(self) -> None:
        logger.info("Finalizing GLiNER2 preprocessor.")
