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

**Результаты — Locust 100 users, 15 min (честное сравнение с REST):**

| Метрика | REST | gRPC | Δ |
|---|---|---|---|
| RPS | 146.8 | **149.1** | +1.5% |
| P50 | 600ms | **600ms** | = |
| P95 | 1200ms | **1100ms** | -8% ✅ |
| P99 | 1900ms | **1500ms** | -21% ✅ |
| Errors | 0 | 0 | = |

**Micro-benchmark (bench_grpc.py vs bench_rest.py, 512 req concurrency=32):**
- REST: 34.6 RPS / P50=195ms (клиентский артефакт: asyncio bottleneck на Mac)
- gRPC: 132.8 RPS / P50=200ms (threading, корректно)

**Анализ:**
- Throughput: REST ≈ gRPC — сервер является узким местом, не протокол
- P50 одинаковый: inference time (~200ms) доминирует над protocol overhead
- gRPC выигрывает на хвостах (P99 -21%): HTTP/2 мультиплексирование убирает head-of-line blocking
- Ошибок нет ни в одном протоколе

**Технические проблемы при настройке Locust gRPC:**
- gRPC использует C-level networking — несовместим с gevent monkey-patching
- Решение: `get_hub().threadpool.apply()` — запуск в реальном OS потоке
- По умолчанию threadpool.maxsize=10 → при 100 users создаёт очередь → P50=1900ms
- Финальное решение: `pool.maxsize = pool.size = 200` в `@events.init`

**Вывод:** для inference-heavy задач REST ≈ gRPC по throughput, gRPC лучше по tail latency.

---

## Эксперимент 3 — Ensemble (нативный backend)

**Дата:** 24 апреля 2026

**Архитектура:**
```
text → [gliner_preprocessor] (Python, CPU)
     → input_ids, attention_mask → [gliner_guard_encoder_onnx] (ONNX Runtime, GPU)
     → last_hidden_state → [gliner_postprocessor] (Python, GPU)
     → meta (bypass encoder) ─────────────────────────────────► [gliner_postprocessor]
     → result
```

### Почему не TorchScript (изначальный план)

Планировалось использовать Triton **PyTorch backend** с `model.pt` (TorchScript, C++ libtorch).
Экспорт через `torch.jit.trace` падал с `IndexError: tuple index out of range` внутри
`transformers/masking_utils.py:sdpa_mask`.

**Корневая причина:** ModernBERT всегда вызывает `create_bidirectional_mask()` в своём
`forward()`, независимо от переданного `attention_mask`. Внутри функции вычисляется
`q_length = torch.tensor(seq_len)` — **0-dim скалярный тензор**. Следующая строка:
```python
q_length, q_offset = q_length.shape[0], q_length[0].to(device)
# shape[0] на 0-dim тензоре → IndexError: tuple index out of range
```
`torch.jit.trace` запускает Python-код буквально во время трейсинга — ошибка возникает
до того, как граф успевает быть записан. Это баг совместимости в библиотеке `transformers`:
ModernBERT не поддерживает экспорт в TorchScript.

Попытки обхода:
- `encoder.config._attn_implementation = "eager"` — `eager_mask` всё равно вызывает `sdpa_mask`
- `attention_mask=None` в `EncoderWrapper.forward` — не помогает, маска вычисляется внутри всегда

**Решение: ONNX Runtime backend** (нативный C++, без Python-интерпретатора).
`torch.onnx.export` с `dynamo=True` использует AOT-компиляцию вместо трейсинга.
ONNX Runtime — стандарт для продакшн-инференса трансформеров (HuggingFace Optimum).

**Файлы:**
- `triton-serve/export/export_onnx.py` — экспорт в ONNX (dynamo + fallback)
- `triton-serve/model_repository/gliner_guard_encoder_onnx/config.pbtxt` — `backend: "onnxruntime"`

**Условия теста:** Locust 100 users, spawn_rate=10, constant_throughput(5), 15 min, RunPod A100 через TCP proxy (195.26.233.96)

### Результаты

**REST (gliner_ensemble):**

| Метрика | Ensemble REST | gliner_guard REST | Δ |
|---|---|---|---|
| RPS | **79.9** | 146.8 | -46% |
| P50 | **1200ms** | 600ms | +100% |
| P95 | **1300ms** | 1200ms | +8% |
| P99 | **1400ms** | 1900ms | **-26% ✅** |
| Max | 24948ms | 146074ms | **-83% ✅** |
| Errors | **0** | 0 | = |

**gRPC (gliner_ensemble):**

| Метрика | Ensemble gRPC | Ensemble REST | Δ |
|---|---|---|---|
| RPS | **78.9** | 79.9 | -1% |
| P50 | **1300ms** | 1200ms | +8% |
| P95 | **1300ms** | 1300ms | = |
| P99 | **1400ms** | 1400ms | = |
| Max | **1638ms** | 24948ms | **-93% ✅** |
| Errors | **0** | 0 | = |

### Анализ: почему ensemble медленнее gliner_guard

**Bottleneck — Python backend count=1 для preprocessor и postprocessor.**

`gliner_guard` (Python backend, один монолитный Python-процесс) запускается с `count: 4` (start_runpod.sh делает `sed s/KIND_CPU/KIND_GPU/; s/count: 1/count: 4/`).
Это 4 параллельных Python-процесса, каждый обрабатывает батч независимо.

Ensemble же состоит из трёх отдельных моделей с `count: 1` каждая:
- `gliner_preprocessor` — Python, CPU, count=1 → **последовательный bottleneck**
- `gliner_guard_encoder_onnx` — ONNX Runtime, GPU → быстрый
- `gliner_postprocessor` — Python, GPU, count=1 → **последовательный bottleneck**

С 1 экземпляром постпроцессора все запросы выстраиваются в очередь.
`~80 RPS ≈ 148 RPS / ~2` — как раз соответствует 2 Python-bottleneck вместо 4 параллельных воркеров.

**REST vs gRPC для ensemble:** протокол не играет роли — bottleneck внутри Triton.
Единственное преимущество gRPC: Max=1638ms против REST Max=24948ms — gRPC HTTP/2
устраняет экстремальные выбросы (tail spikes от HTTP/1.1 head-of-line blocking).

### Как улучшить ensemble

Увеличить `count` для постпроцессора:
```
instance_group [{ kind: KIND_GPU, count: 2 }]  # postprocessor
```
Каждый экземпляр загружает GLiNER2 (~400MB GPU) → при count=4 потребуется ~1.6GB.
Ожидаемый прирост: 80 RPS → ~160 RPS.

**Файлы результатов:**
- `triton-serve/results/locust-ensemble-rest_stats.csv`
- `triton-serve/results/locust-ensemble-grpc_stats.csv`
- `triton-serve/results/a100-benchmark.csv`

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
| `sdpa_mask() got multiple values for argument 'batch_size'` | Патч ONNX-экспорта захватывал `q_length` позиционно, затем передавал его и позиционно и через kwargs | Изменить сигнатуру патча на `*args, **kwargs`, модифицировать только `kwargs['q_length']` |
| `No module named 'onnx'` | `torch.onnx.export(dynamo=True)` требует пакет `onnx` отдельно от `onnxruntime-gpu` | Добавить `onnx` в Dockerfile |
| `TYPE_BYTES` rejected by Triton | `data_type: TYPE_BYTES` не принимается Triton 25.01 protobuf-парсером в config.pbtxt | Использовать числовой код `data_type: 13` |
| Ensemble routing bug | `value: "INPUT_TEXT"` в ensemble шаге не совпадал с именем входа `name: "text"` | Использовать точные имена тензоров: key=имя модели, value=имя в ensemble pipeline |
| Entities всегда `{}` | `_FIXED_METADATA["entity_order"] = []` → `metadata.get("entity_order", default)` возвращает `[]`, цикл не выполняется | Удалить ключ `entity_order` из metadata — тогда `.get()` возвращает fallback `entity_names` |
| `No module named 'packaging'` при рестарте | tritonserver запущен с `/usr/bin/python3` (системный, без pip-пакетов) вместо `/usr/local/bin/python3` | Явно передавать `--backend-config=python,python-runtime-path=/usr/local/bin/python3` |
| Пустой body на `/v2/health/ready` | KFServing v2 spec: Triton возвращает HTTP 200 с пустым телом | `json.loads(body) if body else {}` |

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
