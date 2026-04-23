"""
Triton Python Backend: GLiNER2 Postprocessor.

Receives last_hidden_state from the native PyTorch encoder backend,
reconstructs batch metadata from meta JSON, extracts span embeddings,
runs classification and span extraction, returns JSON results.

Part of gliner_ensemble:
  gliner_guard_encoder (PyTorch) → last_hidden_state ──► [this model] → result
  gliner_preprocessor  (Python)  → text_word_indices, meta ──────────► [this model]

Protocol:
  Input:
    last_hidden_state  FP32  [N, seq_len, 384]
    text_word_indices  INT64 [N, max_words]
    meta               BYTES [N, 1]           — JSON from preprocessor
  Output:
    result             BYTES [N, 1]           — JSON extraction result per sample
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
from gliner2.processor import PreprocessedBatch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PII_LABELS = ["person", "address", "email", "phone"]
SAFETY_LABELS = ["safe", "unsafe"]
PII_THRESHOLD = 0.4
MODEL_ID = "hivetrace/gliner-guard-uniencoder"

# Fixed metadata for format_results (schema is always PII + safety classification)
_FIXED_METADATA = {
    "field_metadata": {},
    "entity_metadata": {},
    "relation_metadata": {},
    "field_orders": {},
    "entity_order": [],
    "relation_order": [],
    "classification_tasks": ["safety"],
}


class TritonPythonModel:
    """Postprocessor: last_hidden_state + meta → JSON results."""

    def initialize(self, args: dict) -> None:
        logger.info("Initializing GLiNER2 postprocessor...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load model for span_rep MLP + extraction logic (not the encoder)
        self.model = GLiNER2.from_pretrained(MODEL_ID)
        # Move only non-encoder components to GPU (span_rep MLP, classifiers)
        # The encoder weights are not needed here — gliner_guard_encoder handles that.
        self.model.to(device=device, dtype=torch.float32)
        self.model.eval()

        # Build schema once — same as preprocessor
        self.schema = (
            self.model.create_schema()
            .entities(entity_types=PII_LABELS, threshold=PII_THRESHOLD)
            .classification(task="safety", labels=SAFETY_LABELS)
        )
        self.schema_dict = self.schema.build()
        for cls_config in self.schema_dict.get("classifications", []):
            cls_config.setdefault("true_label", ["N/A"])

        logger.info("GLiNER2 postprocessor ready on %s", device)

    def _reconstruct_batch(
        self,
        meta_list: list,
        twi_np: np.ndarray,
    ) -> PreprocessedBatch:
        """Reconstruct PreprocessedBatch from per-sample JSON metadata."""
        n = len(meta_list)
        twi_tensor = torch.from_numpy(twi_np.copy()).to(device=self.device)  # [N, max_words]

        # Dummy input_ids tensor just to satisfy PreprocessedBatch.__len__
        dummy_ids = torch.zeros(n, 1, dtype=torch.int64)

        return PreprocessedBatch(
            input_ids=dummy_ids,
            attention_mask=torch.zeros(n, 1),
            mapped_indices=[[] for _ in meta_list],
            schema_counts=[m["schema_counts"] for m in meta_list],
            original_lengths=[0] * n,
            structure_labels=[[] for _ in meta_list],
            task_types=[m["task_types"] for m in meta_list],
            text_tokens=[m["text_tokens"] for m in meta_list],
            schema_tokens_list=[m["schema_tokens_list"] for m in meta_list],
            start_mappings=[m["start_mappings"] for m in meta_list],
            end_mappings=[m["end_mappings"] for m in meta_list],
            original_texts=[m["original_text"] for m in meta_list],
            original_schemas=[m["original_schema"] for m in meta_list],
            text_word_indices=twi_tensor,
            text_word_counts=[m["text_word_counts"] for m in meta_list],
            schema_special_indices=[m["schema_special_indices"] for m in meta_list],
        )

    def execute(self, requests: list) -> list:
        responses = []

        for request in requests:
            lhs_np  = pb_utils.get_input_tensor_by_name(request, "last_hidden_state").as_numpy()
            twi_np  = pb_utils.get_input_tensor_by_name(request, "text_word_indices").as_numpy()
            meta_np = pb_utils.get_input_tensor_by_name(request, "meta").as_numpy()

            n = lhs_np.shape[0]

            try:
                meta_list = [
                    json.loads(
                        meta_np[i, 0].decode("utf-8")
                        if isinstance(meta_np[i, 0], bytes)
                        else meta_np[i, 0]
                    )
                    for i in range(n)
                ]

                lhs_tensor = torch.from_numpy(lhs_np.copy()).to(
                    device=self.device, dtype=torch.float32
                )  # [N, seq_len, 384]

                batch = self._reconstruct_batch(meta_list, twi_np)

                # Extract token + schema embeddings using precomputed last_hidden_state
                all_token_embs, all_schema_embs = (
                    self.model.processor.extract_embeddings_from_batch(
                        lhs_tensor,
                        batch.input_ids,  # dummy, only used if not fast path
                        batch,
                    )
                )

                # Batch span rep for samples that need it
                span_samples = [
                    i for i in range(n)
                    if any(t != "classifications" for t in batch.task_types[i])
                    and all_token_embs[i].numel() > 0
                ]
                all_span_info: list = [None] * n
                if span_samples:
                    span_embs = [all_token_embs[i] for i in span_samples]
                    span_results = self.model.compute_span_rep_batched(span_embs)
                    for idx, si in zip(span_samples, span_results):
                        all_span_info[idx] = si

                # Extract + format results per sample
                results = []
                metadata_list = [_FIXED_METADATA] * n
                for i in range(n):
                    try:
                        raw = self.model._extract_sample(
                            token_embs=all_token_embs[i],
                            schema_embs=all_schema_embs[i],
                            schema_tokens_list=batch.schema_tokens_list[i],
                            task_types=batch.task_types[i],
                            text_tokens=batch.text_tokens[i],
                            original_text=batch.original_texts[i],
                            schema=batch.original_schemas[i],
                            start_mapping=batch.start_mappings[i],
                            end_mapping=batch.end_mappings[i],
                            threshold=0.5,
                            metadata=metadata_list[i],
                            include_confidence=False,
                            include_spans=False,
                            span_info=all_span_info[i],
                        )
                        formatted = self.model.format_results(
                            raw,
                            include_confidence=False,
                            requested_relations=[],
                            classification_tasks=["safety"],
                        )
                        results.append(formatted)
                    except Exception as e:
                        logger.error("Sample %d extraction error: %s", i, e)
                        results.append({})

                output_bytes = np.array(
                    [[json.dumps(r, ensure_ascii=False).encode("utf-8")] for r in results],
                    dtype=object,
                )
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("result", output_bytes)]
                ))

            except Exception as e:
                logger.error("Postprocessor error: %s", e)
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                ))

        return responses

    def finalize(self) -> None:
        logger.info("Finalizing GLiNER2 postprocessor.")
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
