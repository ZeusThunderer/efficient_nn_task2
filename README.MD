# Воспроизведение Rethinking the Value of Network Pruning

Liu et al., ICLR 2019, arXiv:1810.05270. Сравнение fine-tune и обучения прунинговой архитектуры с нуля на CIFAR. Числа отчёта берутся из CSV, не из этого файла.

## Окружение

Нужен Python 3.11 и CUDA, в списке архитектур которой есть `sm_120`. Системный Python 3.13 и debian-сборку PyTorch не использовать. Видеокарта для обучения — RTX 5070 Ti. RTX 2070 в этой машине есть, но код её не выбирает: `CUDA_VISIBLE_DEVICES=0` в порядке fastest-first — это 5070 Ti. В `nvidia-smi` у неё индекс 1.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock
.venv/bin/python scripts/env_check.py
```

`scripts/env_check.py` проверяет карту, bf16-свёртку, md5 CIFAR и запись в локальный Aim (каталог `.aim`).

## Прогоны

```bash
.venv/bin/python scripts/smoke.py          # G1, budget/f0.json
.venv/bin/python scripts/pipeline.py       # dev, заморозка, канарейка, три фазы
.venv/bin/python scripts/latency.py        # после обучения, на свободной 5070 Ti
.venv/bin/python -m src.report             # report/report.pdf и report/slides.pdf
```

Очередь — DAG. Повторный запуск пропускает каталоги, где уже есть `result.json`. Два падения одного прогона останавливают очередь. Логи: `outputs/<run_id>/train.log`.

Aim локальный. Если репозиторий Aim занят, прогон всё равно пишет `result.json`, и отчёт читает только его.

## Что нельзя менять после тега hparams-v1

`conf/frozen_hparams.json`, бюджеты 160 и 40 эпох, потолок Scratch-B и правило «в отчёт идёт последняя эпоха». Правка, которая меняет числа, требует нового хэша и перезапуска затронутых прогонов. Инварианты для правок кода — в `AGENTS.md`.
