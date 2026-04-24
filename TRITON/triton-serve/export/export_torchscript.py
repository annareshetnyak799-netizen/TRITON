"""
Фаза 1: Экспорт энкодера GLiNER2 в TorchScript.

Зачем это нужно:
  Python backend (model.py) каждый раз загружает модель через Python.
  Triton PyTorch backend (nvcr.io/nvidia/tritonserver) принимает model.pt —
  C++ runtime грузит TorchScript напрямую, без Python overhead (~10-15% быстрее).

Что экспортируем:
  Только encoder (DeBERTa/mmBERT) — он занимает 95% времени inference.
  Вход:  input_ids [B, S], attention_mask [B, S]
  Выход: last_hidden_state [B, S, H]

  Постпроцессинг (extract_embeddings, classify, span extraction) остаётся
  в Python backend — там слишком много динамики для TorchScript.

Использование:
  python export/export_torchscript.py
  python export/export_torchscript.py --verify  # проверить совпадение выходов

Результат:
  model_repository/gliner_guard_encoder/1/model.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]          # TRITON/
REPO = ROOT / "repo_gliner2"
OUT  = ROOT / "triton-serve" / "model_repository" / "gliner_guard_encoder" / "1"
sys.path.insert(0, str(REPO))


# ── Encoder wrapper ───────────────────────────────────────────────────────────

class EncoderWrapper(nn.Module):
    """
    TorchScript-совместимый wrapper вокруг DeBERTa/mmBERT энкодера.

    Принимает только тензоры (TorchScript не поддерживает dataclass с листами).
    Возвращает last_hidden_state — сырые контекстные эмбеддинги токенов.

    Это Triton PyTorch backend: принимает input_ids + attention_mask,
    возвращает token_embeddings. Остальное — в Python backend.
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        input_ids: torch.Tensor,       # [batch, seq_len]  int64
        attention_mask: torch.Tensor,  # [batch, seq_len]  float16/float32
    ) -> torch.Tensor:                 # [batch, seq_len, hidden_size]
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state


# ── Export ────────────────────────────────────────────────────────────────────

def export(verify: bool = False) -> Path:
    from gliner2 import GLiNER2

    print("Loading GLiNER2 model...")
    model = GLiNER2.from_pretrained("hivetrace/gliner-guard-uniencoder")
    model.eval()

    # GLiNER2 inherits from Extractor directly — encoder is on the top-level object
    encoder = model.encoder         # nn.Module (DeBERTa / mmBERT)
    hidden_size = encoder.config.hidden_size
    print(f"  Encoder: {encoder.__class__.__name__}  hidden={hidden_size}")

    # ModernBERT's SDPA/Flash attention uses dynamic Python ops in masking_utils.py
    # that are incompatible with torch.jit.trace (fail at runtime during tracing).
    # Force "eager" attention so the tracer sees only simple matmul-based attention.
    if hasattr(encoder, 'config'):
        encoder.config._attn_implementation = "eager"
        print(f"  Attention impl forced to: eager (for TorchScript compatibility)")

    wrapper = EncoderWrapper(encoder).eval()

    # ── Dummy inputs for tracing ──────────────────────────────────────────────
    # Shape [2, 64]: batch=2 so tracer exercises the batch dimension
    batch, seq_len = 2, 64
    dummy_input_ids      = torch.randint(0, 1000, (batch, seq_len), dtype=torch.int64)
    dummy_attention_mask = torch.ones(batch, seq_len, dtype=torch.float32)

    # ── TorchScript trace ─────────────────────────────────────────────────────
    print("Tracing encoder...")
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (dummy_input_ids, dummy_attention_mask),
            strict=False,   # ModernBERT uses dict outputs — strict=False needed
        )

    # ── Verify trace outputs match original ──────────────────────────────────
    if verify:
        print("Verifying outputs match original model...")
        with torch.no_grad():
            out_original = wrapper(dummy_input_ids, dummy_attention_mask)
            out_traced   = traced(dummy_input_ids, dummy_attention_mask)

        max_diff = (out_original - out_traced).abs().max().item()
        print(f"  Max absolute diff: {max_diff:.2e}")
        assert max_diff < 1e-4, f"Outputs diverged! max_diff={max_diff}"
        print("  Outputs match ✓")

        # Test with different shape (dynamic axes)
        print("  Testing with different batch/seq sizes...")
        for b, s in [(1, 32), (4, 128), (1, 256)]:
            ids  = torch.randint(0, 1000, (b, s), dtype=torch.int64)
            mask = torch.ones(b, s, dtype=torch.float32)
            out  = traced(ids, mask)
            assert out.shape == (b, s, hidden_size), f"Wrong shape: {out.shape}"
            print(f"    [{b}, {s}] → {out.shape} ✓")

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "model.pt"
    traced.save(str(out_path))

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nSaved: {out_path}  ({size_mb:.1f} MB)")
    print("\nNext: create model_repository/gliner_guard_encoder/config.pbtxt")
    print("      (see comments at bottom of this file)")

    return out_path


# ── config.pbtxt template (printed for reference) ────────────────────────────

CONFIG_TEMPLATE = """
# model_repository/gliner_guard_encoder/config.pbtxt
#
# This is the PyTorch backend config for the exported encoder.
# Use alongside gliner_guard (Python backend) in an ensemble.
#
name: "gliner_guard_encoder"
backend: "pytorch"
max_batch_size: 64

dynamic_batching {{
  preferred_batch_size: [ 1, 8, 16, 32, 64 ]
  max_queue_delay_microseconds: 50000
}}

input [
  {{
    name:      "input_ids"
    data_type: TYPE_INT64
    dims:      [ -1 ]          # variable seq_len
  }},
  {{
    name:      "attention_mask"
    data_type: TYPE_FP16
    dims:      [ -1 ]
  }}
]
output [
  {{
    name:      "last_hidden_state"
    data_type: TYPE_FP16
    dims:      [ -1, {hidden} ]  # [seq_len, hidden_size]
  }}
]

instance_group [{{ kind: KIND_GPU, count: 4 }}]
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export GLiNER2 encoder to TorchScript")
    parser.add_argument("--verify", action="store_true",
                        help="Verify traced outputs match original model")
    parser.add_argument("--print-config", action="store_true",
                        help="Print config.pbtxt template and exit")
    args = parser.parse_args()

    if args.print_config:
        print(CONFIG_TEMPLATE.format(hidden=768))
        sys.exit(0)

    out = export(verify=args.verify)
    print("\nDone. Run with --verify to check output correctness.")
