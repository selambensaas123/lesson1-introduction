# Урок 1 — Домашнее задание: Основы PyTorch

Решение составлено по [`HOMEWORK.md`](HOMEWORK.md) преподавателя. Оригинальные примеры сохранены в `tensors_basics/`.

## Структура решения

- `homework_tensors.py` — пункты 1.1–1.4: создание, матричные операции, индексация и reshape.
- `homework_autograd.py` — пункты 2.1–2.3: производные, MSE и цепное правило, проверка аналитически.
- `homework_performance.py` — пункты 3.1–3.4: большие тензоры, `time.time`, `torch.cuda.Event`, сравнение CPU/GPU.
- `performance_results.csv` — фактические результаты запуска на доступном оборудовании.
- `PERFORMANCE_REPORT.md` — методика измерения, анализ и ограничения.
- `test_homework.py` — автоматическая проверка основных результатов.

## Запуск

Установить PyTorch (CPU или CUDA в зависимости от оборудования):

```bash
python -m pip install -r requirements.txt
```

Выполнить все задания и тесты:

```bash
python homework_tensors.py
python homework_autograd.py
python homework_performance.py
python -m unittest -v test_homework.py
```

Быстрая проверка без больших 3D аллокаций:

```bash
python homework_performance.py --quick
```

**Без видеокарты NVIDIA / CUDA:** программа автоматически измеряет только CPU и выводит `N/A` для GPU без вымышленных результатов. GPU-числа зависят от устройства и должны измеряться на реальном GPU.

**Честное ограничение эксперимента:** большие тензоры создаются, но для сравнения операций используются срезы 2D, а не полный 3D батч. Это обеспечивает запуск на обычном компьютере при разумной нагрузке на CPU.
