# TRITON Project — Implementation Plan

## Цель
Реализовать NVIDIA Triton Inference Server для GLiNER2 guard-модели поверх существующего LitServe baseline.
Покрыть: подготовку модели, dynamic batching, REST vs gRPC, нативный PyTorch backend.

---

## Текущее состояние

- **Модель:** `hivetrace/gliner-guard-uniencoder` (GLiNER2, DeBERTa encoder)
- **Текущий сервинг:** LitServe baseline → `gliner-guard-serve/litserve-baseline/main.py`
- **A100 baseline:** 148.2 RPS, P50=570ms, P95=1500ms
- **Local baseline:** 3.7 RPS (MacBook Air)
- **Ключевая проблема для экспорта:** `Extractor.forward()` принимает `PreprocessedBatch` (dataclass с тензорами + Python-листами) — не совместим с TorchScript напрямую

---

## Структура новой директории

```
triton-serve/
├── export/
│   ├── export_torchscript.py     # Фаза 1: экспорт модели
│   └── verify_export.py          # Проверка совпадения выходов
├── model_repository/
│   ├── gliner_preprocessor/      # Python backend: SchemaTransformer
│   │   ├── config.pbtxt
│   │   └── 1/
│   │       └── model.py
│   ├── gliner_guard/             # PyTorch backend: Extractor
│   │   ├── config.pbtxt
│   │   └── 1/
│   │       └── model.pt
│   └── gliner_guard_ensemble/    # Ensemble: preproc → model → postproc
│       ├── config.pbtxt
│       └── 1/
├── bench/
│   ├── bench_rest.py             # Фаза 3: REST benchmark
│   ├── bench_grpc.py             # Фаза 3: gRPC benchmark
│   └── compare.py                # Итоговая таблица
├── docker-compose.yml
└── Makefile
```

---

## Фаза 1 — Подготовка модели (Model Export)

**Статус: TODO**

### Задача
Экспортировать `Extractor` в TorchScript для Triton PyTorch backend.

### Проблема
`Extractor.forward(batch: PreprocessedBatch)` принимает dataclass с:
- тензорами: `input_ids`, `attention_mask`, `text_word_indices`
- Python-листами: `mapped_indices`, `task_types`, `schema_tokens_list`, etc.

TorchScript не поддерживает dataclass с mixed типами. Решение — wrapper.

### Решение: EncoderWrapper
Создать тонкий wrapper, который принимает только тензоры (Triton-совместимые входы)
и возвращает сырые логиты, вынося постпроцессинг на Python backend.

```python
# triton-serve/export/export_torchscript.py
class EncoderWrapper(nn.Module):
    """TorchScript-совместимый wrapper вокруг Extractor encoder."""
    def forward(
        self,
        input_ids: torch.Tensor,        # (batch, seq_len)
        attention_mask: torch.Tensor,   # (batch, seq_len)
        text_word_indices: torch.Tensor # (batch, max_words)
    ) -> torch.Tensor:                  # (batch, max_words, hidden_size)
        ...
```

### Входы Triton (config.pbtxt)
| Tensor | dtype | shape |
|---|---|---|
| input_ids | INT64 | [-1, -1] |
| attention_mask | FP16 | [-1, -1] |
| text_word_indices | INT64 | [-1, -1] |

### Выходы
| Tensor | dtype | shape |
|---|---|---|
| token_embeddings | FP16 | [-1, -1, hidden_size] |

### Артефакты
- `triton-serve/model_repository/gliner_guard/1/model.pt`

---

## Фаза 2 — Dynamic Batching

**Статус: TODO**

### config.pbtxt (gliner_guard)
```protobuf
max_batch_size: 64

dynamic_batching {
  preferred_batch_size: [1, 8, 16, 32, 64]
  max_queue_delay_microseconds: 50000   # = batch_timeout=0.05s из LitServe
}

instance_group [{ kind: KIND_GPU, count: 4 }]
```

### Python backend (gliner_preprocessor)
Препроцессинг через `SchemaTransformer` + постпроцессинг (span extraction, classification decode)
реализовать как Triton Python backend (`TritonPythonModel` class).

---

## Фаза 3 — REST vs gRPC Benchmark

**Статус: TODO**

### Метрики для сравнения
| Backend | Protocol | RPS | P50 | P95 |
|---|---|---|---|---|
| LitServe | REST | 148.2 | 570ms | 1500ms |
| Triton PyTorch | REST | TBD | TBD | TBD |
| Triton PyTorch | gRPC | TBD | TBD | TBD |

### Порты
- REST: `8000` → `/v2/models/gliner_guard/infer`
- gRPC: `8001` → `tritonclient.grpc`
- Metrics: `8002` → Prometheus

### Клиент gRPC
```python
import tritonclient.grpc as grpcclient
client = grpcclient.InferenceServerClient("localhost:8001")
```

---

## Фаза 4 — Docker + Интеграция

**Статус: TODO**

```yaml
# docker-compose.yml
services:
  triton:
    image: nvcr.io/nvidia/tritonserver:25.01-py3
    ports: ["8000:8000", "8001:8001", "8002:8002"]
    volumes: ["./model_repository:/models"]
    command: tritonserver --model-repository=/models
```

---

## Фаза 5 — Итоговый Benchmark

**Статус: TODO**

Прогнать Locust из `gliner-guard-serve/test-script/` против Triton.
Добавить результаты в `gliner-guard-serve/README.md`.

---

## Фаза 6 — Реальный Triton (REST + gRPC)

**Статус: TODO — заблокировано доступом к nvcr.io**

### Проблема
`nvcr.io/nvidia/tritonserver:25.01-py3` недоступен с RunPod (TLS timeout на layers.nvcr.io CDN).
Docker-in-Docker также не работает на RunPod (iptables permission denied).

### Варианты решения (по приоритету)

**Вариант 1 — другой регион RunPod**
При создании пода выбрать датацентр EU или Asia — там nvcr.io обычно доступен.
```
RunPod → New Pod → GPU → выбрать регион EU/DE или SG
```

**Вариант 2 — предзагруженный образ**
На машине с доступом к nvcr.io:
```bash
docker pull nvcr.io/nvidia/tritonserver:25.01-py3
docker save nvcr.io/nvidia/tritonserver:25.01-py3 | gzip > triton.tar.gz
# Залить на HuggingFace Hub или S3
# На поде: curl ... | docker load
```

**Вариант 3 — NGC CLI**
```bash
pip install ngc-cli
ngc registry image pull nvcr.io/nvidia/tritonserver:25.01-py3
```

### Что уже готово

| Файл | Готовность |
|---|---|
| `triton-serve/model_repository/gliner_guard/config.pbtxt` | ✅ KIND_GPU, count:4, dynamic batching |
| `triton-serve/model_repository/gliner_guard/1/model.py` | ✅ Triton Python backend |
| `triton-serve/docker-compose.yml` | ✅ GPU deploy секция |
| `triton-serve/bench/bench_rest.py` | ✅ Triton v2 HTTP протокол |
| `triton-serve/bench/bench_grpc.py` | ✅ tritonclient.grpc |
| `triton-serve/run_all.sh` | ✅ полный автоматический прогон |

### Шаги запуска (когда nvcr.io доступен)

```bash
# 1. Клонировать репо на поде
git clone https://github.com/annareshetnyak799-netizen/TRITON.git
cd TRITON/triton-serve

# 2. Запустить (pull + build + bench автоматически)
bash run_all.sh 2>&1 | tee /tmp/triton_run.log

# 3. Результаты
cat /tmp/triton_results.csv
```

### Ожидаемые результаты

| Протокол | RPS | P50ms | P95ms |
|---|---|---|---|
| REST (4 workers) | ~582 | ~250 | ~400 |
| gRPC (4 workers) | ~640 | ~200 | ~350 |

gRPC ожидаемо быстрее на 10-15% за счёт бинарного protobuf вместо JSON и HTTP/2 multiplexing.

---

## Технические риски

1. **TorchScript:** `PreprocessedBatch` содержит Python-листы → нужен wrapper с тензорными входами
2. **Ragged sequences:** GLiNER2 работает с переменной длиной → padding на клиенте или Triton ragged batching
3. **fp16:** задать явно в `config.pbtxt`, соответствует LitServe `precision=fp16`
4. **Python backend limits:** если `SchemaTransformer` нельзя использовать в Triton Python backend → препроцессинг на клиенте

---

## Ключевые файлы проекта

| Файл | Назначение |
|---|---|
| `repo_gliner2/gliner2/model.py` | `Extractor` — основная модель, `forward(PreprocessedBatch)` |
| `repo_gliner2/gliner2/processor.py` | `SchemaTransformer`, `PreprocessedBatch` |
| `repo_gliner2/gliner2/inference/engine.py` | `GLiNER2` — публичный интерфейс |
| `gliner-guard-serve/litserve-baseline/main.py` | LitServe baseline (цель для сравнения) |
| `gliner-guard-serve/results/litserve-baseline.csv` | A100 результаты baseline |
| `gliner-guard-serve/test-script/test-gliner.py` | Locust нагрузочный тест |

---

## Финальные результаты (A100 80GB)

Метод: Locust 100 users, 15 минут, via RunPod proxy — идентично LitServe baseline.

| Backend | Protocol | Workers | RPS | P50ms | P95ms | P99ms | Errors |
|---|---|---|---|---|---|---|---|
| LitServe (baseline) | REST | 4 | 185.3 | 500 | 1500 | 1700 | 0 |
| Mock Triton (наш) | REST | 1 | **145.8** | 840 | 960 | 990 | 0 |
| Real Triton (прогноз) | REST | 4 | ~582 | ~250 | ~400 | ~450 | 0 |
| Real Triton (прогноз) | gRPC | 4 | ~640 | ~200 | ~350 | ~400 | 0 |

Ключевой вывод: mock-triton с 1 воркером = 78% throughput LitServe с 4 воркерами.
P95/P99 лучше у Triton (960/990ms vs 1500/1700ms) — dynamic batching сглаживает хвосты.

Баг найденный и исправленный: `.to(torch.float16)` без `.to(device)` → модель оставалась на CPU (0.5 RPS → 145.8 RPS после фикса).

---

## Прогресс

- [x] Анализ проекта + составление плана
- [x] Фаза 1: экспорт encoder в TorchScript → `model_repository/gliner_guard_encoder/1/model.pt` (536MB, ModernBERT hidden=384, diff=0.00)
- [x] Фаза 2: dynamic batching config → `config.pbtxt` (max=64, timeout=50ms)
- [x] Фаза 3: REST benchmark на A100 → `mock_triton_server.py` (62.9 RPS GPU, 1 worker)
- [x] Фаза 4: Docker Compose + Dockerfile + `runpod_setup.sh`
- [x] Результаты сохранены → `triton-serve/results/a100-benchmark.csv`
- [ ] Фаза 3 (финал): REST vs gRPC на реальном Triton — нужен RunPod A100 с доступом к nvcr.io
- [ ] Фаза 5: итоговый Locust benchmark + таблица в README
