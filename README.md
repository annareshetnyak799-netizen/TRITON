# Triton Inference Serving for a GLiNER2-Based Safety Classifier

Production-grade inference infrastructure for an open-source guard model (PII detection + safety classification) built on the [GLiNER2](https://github.com/fastino-ai/GLiNER2) architecture. The project migrates the model from a single-process Python serving baseline to **NVIDIA Triton Inference Server** with multiple backend strategies, ensemble pipelines, and reproducible benchmarking.

The work covers the full lifecycle: model export, backend selection, ensemble decomposition, batch tuning, and end-to-end load testing on production-like hardware.

---

## Why this project

LLM safety guards are increasingly deployed in front of production language model traffic. Their inference latency directly shapes user-perceived response times, and their throughput defines the cost ceiling of any system that uses them. Moving a guard model from a research-grade Python wrapper to a production inference server is a non-trivial engineering exercise — and a useful one to study, because the same trade-offs apply to any small encoder-based classifier in the same role.

This repository is a hands-on study of those trade-offs on **NVIDIA Triton**, a widely used inference server in the industry.

---

## What was built

- **Triton model repository** with multiple coexisting backends for the same model:
  - Python backend (full pipeline in a single process)
  - Native ONNX Runtime backend for the encoder
  - TorchScript backend (attempted; see "Lessons learned")
  - Triton **ensemble** that composes preprocessor → ONNX encoder → postprocessor as separate, independently scalable units
- **Reproducible Docker setup** for both local development and RunPod GPU deployment
- **Model export pipeline** with `torch.onnx.export(dynamo=True)` and a custom encoder wrapper to bridge a `dataclass`-based forward signature with tensor-only backends
- **Benchmark harness** with both `Locust` (sustained load, 15-minute runs at fixed throughput) and lightweight micro-benchmarks for `REST` vs `gRPC` comparison
- **Operational tooling** for RunPod: startup scripts, model export automation, health-gated container orchestration

---

## Architectural decisions

A few decisions worth highlighting:

### Ensemble over monolith

Splitting the pipeline into preprocessor / encoder / postprocessor as separate Triton models — composed via an ensemble — exposes each component as an independently tunable unit. This makes it possible to scale the bottleneck stage without paying for unused parallelism elsewhere, and to swap the encoder backend (Python vs ONNX vs TorchScript) without touching the surrounding logic.

### ONNX Runtime over TorchScript for the encoder

The original plan used Triton's PyTorch backend with TorchScript. The encoder turned out to be incompatible with `torch.jit.trace` due to a known interaction between modern transformer attention masking utilities and the tracing flow. The project pivoted to **ONNX Runtime via `torch.onnx.export(dynamo=True)`**, which uses ahead-of-time graph compilation rather than tracing. This is the same path used by the major production inference stacks (HuggingFace Optimum, vLLM's CPU paths) for transformer encoders.

### Real load shapes, not micro-benchmarks alone

Sustained 15-minute Locust runs at a fixed request rate were used as the primary measurement, with micro-benchmarks reserved for protocol-level (REST vs gRPC) comparison. This caught several issues that short-burst benchmarks would have missed (Python event-loop blocking, asyncio task starvation, GPU model placement bugs, queue saturation under realistic concurrency).

### gRPC vs REST

Both protocols were measured under identical load. For inference-heavy workloads where model time dominates over wire overhead, the protocols are roughly equivalent on average throughput — but **gRPC consistently wins on tail latency** thanks to HTTP/2 multiplexing eliminating head-of-line blocking. This matters for guard models, where p99 latency directly bounds end-to-end response time.

---

## Lessons learned

- **TorchScript export is not free for modern transformer encoders.** Modern attention masking implementations in `transformers` rely on Python-level control flow that doesn't trace cleanly. ONNX Runtime with dynamo-based export is the more reliable production path.
- **GPU placement bugs hide behind dtype calls.** A `.to(torch.float16)` after model load looks like quantization, but it doesn't move the model to GPU. The model silently runs on CPU. Always: `.to(device).to(dtype)`, in that order.
- **Bottlenecks shift as you scale.** Scaling Python backends in an ensemble revealed that the encoder became the new ceiling. Iterative `count` tuning per stage matters more than tuning any single component to its maximum.
- **Reproducible benchmarks need fixed methodology.** Same client, same hardware, same load profile, same duration — across all variants. Results that are not directly comparable to the baseline are not useful for decisions.

---

## Repository structure

```
triton-serve/                        Main project: Triton model repository, Docker setup,
                                     benchmarks, export scripts, RunPod orchestration
gliner-guard-serve/  (submodule)     External baseline serving implementation used as
                                     a reference for performance comparison
repo_gliner2/        (submodule)     External GLiNER2 reference implementation
repo_guardrails/     (submodule)     External guard-model usage notebook
```

The submodules are pinned external references — they are not part of this work and are included only for reproducibility of the inference setup.

### Inside `triton-serve/`

- `model_repository/` — Triton model configurations and Python backends
  - `gliner_guard/` — full pipeline as a single Python backend
  - `gliner_guard_encoder/`, `gliner_guard_encoder_onnx/` — encoder-only backends (TorchScript, ONNX)
  - `gliner_preprocessor/`, `gliner_postprocessor/` — split-pipeline components
  - `gliner_ensemble/` — composition of the above into a single served model
- `export/` — model export scripts (`export_onnx.py`, `export_torchscript.py`)
- `bench/` — Locust scripts and micro-benchmarks for REST and gRPC
- `results/` — raw benchmark output as CSV
- `Dockerfile`, `Dockerfile.runpod`, `docker-compose.yml`, `start_runpod.sh`, `run_all.sh` — runtime and orchestration

---

## Status

The Triton serving setup is complete and was benchmarked end-to-end on a single-GPU production-like instance (NVIDIA A100). The repository is functional but not packaged for general use — it is shared primarily as a record of the engineering process.

Detailed benchmark numbers are not included in this README. They will be published separately.
