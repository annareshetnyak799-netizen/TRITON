"""
Фаза 1b: Экспорт энкодера GLiNER2 в ONNX для Triton ONNX Runtime backend.

Почему ONNX вместо TorchScript:
  ModernBERT вызывает create_bidirectional_mask() с 0-dim тензором seq_len,
  что несовместимо с torch.jit.trace (IndexError: tuple index out of range).
  torch.onnx.export с dynamo=True использует torch.export (AOT компиляция)
  вместо трейсинга — не имеет этой проблемы.

Результат:
  model_repository/gliner_guard_encoder_onnx/1/model.onnx

Использование:
  python export/export_onnx.py
  python export/export_onnx.py --verify
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]          # TRITON/
OUT  = ROOT / "triton-serve" / "model_repository" / "gliner_guard_encoder_onnx" / "1"


class EncoderWrapper(nn.Module):
    """Thin wrapper: exposes only input_ids + attention_mask → last_hidden_state."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        input_ids: torch.Tensor,       # [batch, seq_len]  int64
        attention_mask: torch.Tensor,  # [batch, seq_len]  float32
    ) -> torch.Tensor:                 # [batch, seq_len, hidden_size]
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


def _patch_sdpa_mask() -> bool:
    """
    Monkey-patch masking_utils.sdpa_mask to handle 0-dim tensors.

    ModernBERT internally computes q_length = torch.tensor(seq_len) — a 0-dim
    scalar. sdpa_mask expects at least 1-dim. This patch reshapes it before the
    function runs, so the ONNX graph captures correct attention ops.
    """
    try:
        import transformers.masking_utils as _mu
        _orig = _mu.sdpa_mask

        def _patched(*args, **kwargs):
            if 'q_length' in kwargs:
                q = kwargs['q_length']
                if isinstance(q, torch.Tensor) and q.ndim == 0:
                    kwargs['q_length'] = q.unsqueeze(0)
            return _orig(*args, **kwargs)

        _mu.sdpa_mask = _patched
        return True
    except Exception as e:
        print(f"  Warning: could not patch sdpa_mask: {e}")
        return False


def export(verify: bool = False) -> Path:
    from gliner2 import GLiNER2

    print("Loading GLiNER2 model...")
    model = GLiNER2.from_pretrained("hivetrace/gliner-guard-uniencoder")
    model.eval()

    encoder = model.encoder
    hidden_size = encoder.config.hidden_size
    print(f"  Encoder: {encoder.__class__.__name__}  hidden={hidden_size}")

    # Force eager attention — SDPA path has dynamic Python ops harder to export
    if hasattr(encoder, 'config'):
        encoder.config._attn_implementation = "eager"
        print("  Attention impl: eager")

    # Patch sdpa_mask before any tracing/export
    patched = _patch_sdpa_mask()
    print(f"  sdpa_mask patch: {'ok' if patched else 'skipped'}")

    wrapper = EncoderWrapper(encoder).eval()  # fp32 (fp16 causes ONNX Runtime queue stall)

    batch, seq_len = 2, 64
    dummy_ids  = torch.randint(0, 1000, (batch, seq_len), dtype=torch.int64)
    dummy_mask = torch.ones(batch, seq_len, dtype=torch.float32)  # fp32

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "model.onnx"

    # ── Strategy 1: dynamo export (PyTorch 2.x AOT, no tracing issues) ────────
    try:
        print("\nTrying dynamo ONNX export (torch.onnx.export dynamo=True)...")
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_ids, dummy_mask),
                str(out_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["last_hidden_state"],
                dynamic_axes={
                    "input_ids":         {0: "batch_size", 1: "seq_len"},
                    "attention_mask":    {0: "batch_size", 1: "seq_len"},
                    "last_hidden_state": {0: "batch_size", 1: "seq_len"},
                },
                opset_version=17,
                dynamo=True,
            )
        print("  dynamo export succeeded")
    except Exception as e:
        print(f"  dynamo export failed: {e}")

        # ── Strategy 2: classic trace export with sdpa_mask patch ──────────
        print("\nFalling back to classic torch.onnx.export (trace mode)...")
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_ids, dummy_mask),
                str(out_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["last_hidden_state"],
                dynamic_axes={
                    "input_ids":         {0: "batch_size", 1: "seq_len"},
                    "attention_mask":    {0: "batch_size", 1: "seq_len"},
                    "last_hidden_state": {0: "batch_size", 1: "seq_len"},
                },
                opset_version=17,
                do_constant_folding=True,
            )
        print("  classic export succeeded")

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nSaved: {out_path}  ({size_mb:.1f} MB)")

    # ── Verify with onnxruntime ──────────────────────────────────────────────
    if verify:
        print("\nVerifying with onnxruntime...")
        import onnxruntime as ort
        import numpy as np

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = ort.InferenceSession(str(out_path), providers=providers)

        with torch.no_grad():
            ref = wrapper(dummy_ids, dummy_mask).numpy()

        ort_out = sess.run(
            ["last_hidden_state"],
            {"input_ids": dummy_ids.numpy(), "attention_mask": dummy_mask.numpy()},
        )[0]

        max_diff = abs(ref - ort_out).max()
        print(f"  Max abs diff (torch vs onnxruntime): {max_diff:.2e}")
        assert max_diff < 1e-3, f"Outputs diverged: {max_diff}"
        print("  Outputs match ✓")

        # Test different shapes
        for b, s in [(1, 32), (4, 128)]:
            ids  = torch.randint(0, 1000, (b, s), dtype=torch.int64)
            mask = torch.ones(b, s, dtype=torch.float32)
            out  = sess.run(["last_hidden_state"],
                            {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})[0]
            assert out.shape == (b, s, hidden_size), f"Wrong shape: {out.shape}"
            print(f"  [{b}, {s}] → {out.shape} ✓")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    export(verify=args.verify)
    print("\nDone.")
