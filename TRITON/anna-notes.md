# Triton Inference Server — Дневник экспериментов

## Цель проекта

Реализовать NVIDIA Triton Inference Server для GLiNER2 guard-модели
(`hivetrace/gliner-guard-uniencoder`) поверх существующего LitServe baseline.

Требования: подготовка модели · dynamic batching · REST vs gRPC · нативный PyTorch backend

---

## Модель

**hivetrace/gliner-guard-uniencoder** — GLiNER2, ModernBERT encoder, hidden=384
- Задача: PII extraction (person, address, email, phone) + safety classification (safe/unsafe)
- Размер: ~400MB, fp16 на GPU

---

## Инфраструктура

- GPU: RunPod A100 80GB
- Образ: `annaresh2024/triton-gliner:latest` (Docker Hub)
- Triton: `nvcr.io/nvidia/tritonserver:25.01-py3`
- Клиент: локальный MacBook Air (Locust) / локальные bench-скрипты

---

## Baseline — LitServe (до проекта)

| Метрика | Значение |
|---|---|
| RPS | 185.3 |
| P50 | 500ms |
| P95 | 1500ms |
| P99 | 1700ms |
| Errors | 0 |
| Условия | Locust 100 users, 15 min, constant_throughput(5), RunPod A100 |

---

## Эксперимент 1 — Triton Python Backend (REST)

**Дата:** 23 апреля 2026

**Конфигурация:**
- Backend: `gliner_guard` (Triton Python backend)
- instance_group: KIND_GPU, count: 4
- dynamic batching: max_batch_size=64, preferred=[1,8,16,32,64], timeout=50ms
- Протокол: REST, endpoint `/v2/models/gliner_guard/infer`

**Условия теста (совпадают с LitServe baseline):**
- Инструмент: Locust 2.43.4
- Пользователи: 100, spawn_rate=10, constant_throughput(5)
- Длительность: 15 мин
- Клиент: локальный Mac через RunPod TCP proxy (213.173.102.4:12416)
- Файл: `triton-serve/bench/locust_triton.py`

**Результаты (финальный чистый прогон):**

| Метрика | Triton REST | LitServe baseline | Δ |
|---|---|---|---|
| RPS | **146.8** | 185.3 | -21% |
| P50 | **600ms** | 500ms | +20% |
| P95 | **1200ms** | 1500ms | **-20% ✅** |
| P99 | **1900ms** | 1700ms | +12% |
| Errors | **0** | 0 | = |
| Всего запросов | 132,076 | — | — |

**Вывод:** Triton Python backend проигрывает по throughput (-21%), но выигрывает
по P95 (-20%) — dynamic batching сглаживает хвостовые задержки.

**Файлы результатов:**
- `triton-serve/results/locust-triton-rest_stats.csv`
- `triton-serve/results/a100-benchmark.csv`

---

## Эксперимент 2 — Triton Python Backend (gRPC)

**Дата:** 23 апреля 2026

**Условия:**
- Инструмент: `bench_grpc.py` (512 запросов, concurrency=32)
- Сервер: тот же `gliner_guard` Python backend
- Протокол: gRPC, порт 8001 (TCP 213.173.102.4:12417)
- Цель: сравнение REST vs gRPC при одинаковом backend

**Результаты (bench_grpc.py vs bench_rest.py, одинаковые параметры: 512 req, concurrency=32):**

| Метрика | REST | gRPC |
|---|---|---|
| RPS | 34.6 | 132.8 |
| P50 | 195ms | 200ms |
| P95 | 467ms | 367ms |
| P99 | 1022ms | 372ms |
| Errors | 0 | 0 |

**Анализ:**
- P50 REST ≈ gRPC (195 vs 200ms) — время inference на сервере одинаковое, протокол не влияет
- Разница в RPS — клиентский артефакт: bench_rest.py использует asyncio, которое создаёт bottleneck на Mac при concurrency=32 (ожидалось ~160 RPS при P50=195ms, получили 34)
- gRPC выигрывает по P99 (372 vs 1022ms) — HTTP/2 мультиплексирование убирает head-of-line blocking
- Надёжный REST RPS из Locust: **147 RPS** — для gRPC аналогичный Locust тест не проводился

**Вывод:** для inference-heavy задач (модель занимает ~200ms) REST ≈ gRPC по latency. gRPC преимущество проявляется на хвостах (P99) и при очень высокой конкурентности.

---

## Эксперимент 3 — Ensemble (нативный PyTorch backend)

**Дата:** планируется после rebuild образа

**Архитектура:**
```
text → [gliner_preprocessor] (Python, CPU)
     → input_ids, attention_mask → [gliner_guard_encoder] (PyTorch, model.pt, GPU)
     → last_hidden_state → [gliner_postprocessor] (Python, GPU)
     → meta (bypass encoder) ──────────────────────► [gliner_postprocessor]
     → result
```

**Ключевое отличие от экспериментов 1-2:**
- Encoder запускается как **нативный TorchScript в C++ libtorch** (не через Python)
- model.pt генерируется при первом старте через `export_torchscript.py`
- Preprocessor/Postprocessor остаются Python backend (preprocessing нельзя TorchScript-ировать из-за Python-списков в PreprocessedBatch)

**Условия теста:** планируется — Locust 100 users, 15 min

**Результаты:**
> TODO: заполнить после деплоя и прогона

---

## Баги и исправления

| Баг | Причина | Решение |
|---|---|---|
| `No module named 'torch'` | Triton 25.01 имеет libtorch (C++) но не Python torch | Явная установка torch в Dockerfile через pip |
| hash mismatch при скачивании torch | CDN хеш не совпадал при резюме загрузки | Убрать pin версии, добавить BuildKit cache mount |
| `No module named 'onnxruntime'` | gliner зависит от onnxruntime, пропущен с --no-deps | Добавить onnxruntime-gpu в Dockerfile |
| `No module named 'requests'` | gliner2 импортирует requests | Добавить requests в Dockerfile |
| 0.5 RPS вместо 145 RPS | `.to(torch.float16)` без `.to(device)` — модель оставалась на CPU | Исправить порядок: сначала to(device), потом to(dtype) |
| Неправильный REST benchmark | concurrency=128 перегружало очередь batching | Использовать Locust вместо bench_rest.py для нагрузочного теста |

---

## Ключевые файлы

| Файл | Назначение |
|---|---|
| `triton-serve/Dockerfile.runpod` | Docker образ для RunPod |
| `triton-serve/start_runpod.sh` | Startup: git pull, export model.pt, запуск tritonserver |
| `triton-serve/model_repository/gliner_guard/` | Python backend (полный pipeline) |
| `triton-serve/model_repository/gliner_guard_encoder/` | PyTorch backend (TorchScript encoder) |
| `triton-serve/model_repository/gliner_preprocessor/` | Python backend (text → tokens) |
| `triton-serve/model_repository/gliner_postprocessor/` | Python backend (embeddings → results) |
| `triton-serve/model_repository/gliner_ensemble/` | Ensemble config |
| `triton-serve/export/export_torchscript.py` | Экспорт encoder в TorchScript |
| `triton-serve/bench/locust_triton.py` | Locust тест для Triton REST |
| `triton-serve/bench/bench_grpc.py` | gRPC микробенчмарк |
| `triton-serve/results/a100-benchmark.csv` | Сводная таблица всех результатов |
