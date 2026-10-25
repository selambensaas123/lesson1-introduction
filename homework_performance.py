"""Урок 1, задание 3: подготовка больших тензоров и CPU/CUDA benchmark.

Запуск: python homework_performance.py
Без CUDA сценарий честно выводит N/A, не подменяя GPU измерения.
"""

from __future__ import annotations

import argparse
import csv
import gc
import platform
import time
from pathlib import Path
from typing import Callable

import torch

BASE_DIR = Path(__file__).resolve().parent
LARGE_SHAPES = (
    (64, 1024, 1024),
    (128, 512, 512),
    (256, 256, 256),
)


def validate_matrix_size(size: int) -> None:
    """Проверить разумный размер 2D матрицы для демонстрации."""
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Matrix size must be a positive integer")


def cpu_time_ms(operation: Callable[[], torch.Tensor], repeats: int = 3) -> float:
    """Среднее время CPU операции в миллисекундах через time.time()."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    operation()  # warm-up
    times: list[float] = []
    for _ in range(repeats):
        start = time.time()
        operation()
        times.append((time.time() - start) * 1000)
    return sum(times) / len(times)


def cuda_time_ms(operation: Callable[[], torch.Tensor], repeats: int = 3) -> float:
    """Среднее CUDA время через torch.cuda.Event с синхронизацией."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this computer")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    operation()  # warm-up
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)


def benchmark_operations(
    a: torch.Tensor, b: torch.Tensor, repeats: int = 3
) -> list[dict[str, object]]:
    """Замерить пять операций на одинаковых CPU и CUDA 2D матрицах.

    Времена включают только операции, но не передачу матриц на GPU.
    transpose().contiguous() измеряет реальное копирование элементов,
    поскольку обычный transpose возвращает представление без копирования.
    """
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("Expected PyTorch tensors")
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ValueError("Expected two same-shaped square matrices")
    if a.device.type != "cpu" or b.device.type != "cpu":
        raise ValueError("Input tensors must be CPU tensors")

    operations = {
        "matmul": lambda x, y: torch.matmul(x, y),
        "add": lambda x, y: x + y,
        "multiply": lambda x, y: x * y,
        "transpose_contiguous": lambda x, y: x.transpose(0, 1).contiguous(),
        "sum": lambda x, y: torch.sum(x),
    }

    gpu_available = torch.cuda.is_available()
    if gpu_available:
        a_cuda = a.to("cuda")
        b_cuda = b.to("cuda")
    else:
        a_cuda = b_cuda = None

    rows: list[dict[str, object]] = []
    for name, operation in operations.items():
        cpu_ms = cpu_time_ms(lambda op=operation: op(a, b), repeats)
        gpu_ms = (
            cuda_time_ms(lambda op=operation: op(a_cuda, b_cuda), repeats)
            if gpu_available
            else None
        )
        rows.append(
            {
                "operation": name,
                "cpu_ms": round(cpu_ms, 5),
                "gpu_ms": round(gpu_ms, 5) if gpu_ms is not None else None,
                "speedup": round(cpu_ms / gpu_ms, 3) if gpu_ms and gpu_ms > 0 else None,
            }
        )
    return rows


def measure_gpu_transfers(a: torch.Tensor, repeats: int = 3) -> dict[str, float] | None:
    """Замерить CPU→GPU и GPU→CPU передачу отдельно от вычислений."""
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    to_gpu_ms = []
    to_cpu_ms = []
    for _ in range(repeats):
        start = time.time()
        on_cuda = a.to("cuda")
        torch.cuda.synchronize()
        to_gpu_ms.append((time.time() - start) * 1000)
        start = time.time()
        on_cpu = on_cuda.to("cpu")
        torch.cuda.synchronize()
        to_cpu_ms.append((time.time() - start) * 1000)
        del on_cuda, on_cpu
    return {
        "cpu_to_gpu_ms": sum(to_gpu_ms) / len(to_gpu_ms),
        "gpu_to_cpu_ms": sum(to_cpu_ms) / len(to_cpu_ms),
    }


def fmt(value: object, suffix: str = "") -> str:
    """Напечатать N/A вместо несуществующего GPU измерения."""
    return f"{value:.3f}{suffix}" if isinstance(value, (int, float)) else "N/A"


def run_benchmark(
    full_allocation: bool = True,
    repeats: int = 3,
    output_dir: Path | None = None,
) -> list[dict[str, object]]:
    """Создать требуемые большие тензоры по очереди и замерить 2D срезы.

    Для экономии памяти одновременно хранится только один большой тензор.
    Матрицы 2D берутся из первого слоя каждой 3D формы. Если выделить
    память для полного тензора невозможно, пропускаем конфигурацию.
    full_allocation=False полезен для быстрого запуска/тестирования.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    folder = output_dir or BASE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "not available"
    print(f"PyTorch: {torch.__version__} | CPU: {platform.processor() or platform.machine()}")
    print(f"CUDA: {has_cuda} | GPU: {gpu_name}")
    results: list[dict[str, object]] = []
    transfer_results: list[dict[str, object]] = []

    for shape in LARGE_SHAPES:
        shape_name = "x".join(map(str, shape))
        # Оригинальные формы огромны. В тестовом режиме используем лишь 2D срез.
        try:
            if full_allocation:
                big = torch.rand(shape, dtype=torch.float32)
                assert tuple(big.shape) == shape
                a = big[0].clone()
                b = big[1].clone()
                print(f"\nCreated full 3D tensor {shape_name}; {big.numel() * 4 / 2**20:.1f} MiB")
                del big  # no need to keep the 3D tensor during benchmarks
            else:
                size = shape[-1]
                validate_matrix_size(size)
                a = torch.rand((size, size))
                b = torch.rand((size, size))
                print(f"\nQuick benchmark 2D shape {size}x{size} (3D allocation skipped)")
            size = a.shape[0]
            transfer = measure_gpu_transfers(a, repeats)
            if transfer:
                transfer_results.append({"source_shape": shape_name, **transfer})
            for result in benchmark_operations(a, b, repeats):
                row = {"source_shape": shape_name, "matrix_size": size, **result}
                results.append(row)
                print(
                    f"{result['operation']:<22} CPU {fmt(result['cpu_ms']):>10} ms | "
                    f"CUDA {fmt(result['gpu_ms']):>10} ms | "
                    f"speedup {fmt(result['speedup'], 'x'):>10}"
                )
            del a, b
            gc.collect()
            if has_cuda:
                torch.cuda.empty_cache()
        except (RuntimeError, MemoryError) as exc:
            print(f"Skipped {shape_name}: {type(exc).__name__}: {exc}")
            continue

    with (folder / "performance_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["source_shape", "matrix_size", "operation", "cpu_ms", "gpu_ms", "speedup"],
        )
        writer.writeheader()
        writer.writerows(results)

    table = [
        "| Исходный тензор | 2D матрица | Операция | CPU (мс) | CUDA (мс) | Ускорение |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in results:
        table.append(
            f"| {r['source_shape']} | {r['matrix_size']}×{r['matrix_size']} "
            f"| {r['operation']} | {fmt(r['cpu_ms'])} | {fmt(r['gpu_ms'])} | {fmt(r['speedup'], 'x')} |"
        )

    transfer_lines = (
        "\n".join(
            f"- {t['source_shape']}: CPU→GPU {t['cpu_to_gpu_ms']:.3f} мс, "
            f"GPU→CPU {t['gpu_to_cpu_ms']:.3f} мс"
            for t in transfer_results
        )
        if transfer_results
        else "CUDA недоступна: измерения передачи данных и ускорения GPU отсутствуют."
    )
    report = f"""# Урок 1 — Задание 3: CPU vs CUDA

## Окружение и методика

- Python: {platform.python_version()}; PyTorch: {torch.__version__}
- Платформа: {platform.platform()}
- CPU: {platform.processor() or platform.machine()}
- GPU CUDA: {gpu_name}
- Seed: 42; повторы каждого измерения: {repeats}
- Подготовка больших тензоров: {"выполнена для каждой доступной формы" if full_allocation else "пропущена (--quick)"}.
- Запрошенные формы: 64×1024×1024, 128×512×512, 256×256×256.
- **Важное ограничение:** для операций используется один 2D срез соответствующего большого
  3D тензора (1024×1024, 512×512, 256×256), а не полный пакет из 64/128/256 матриц.
  Это ограничивает время выполнения на обычном CPU и уменьшает расход памяти.
- Время CPU измеряется `time.time()`, CUDA — `torch.cuda.Event()`.
  Время передачи данных измеряется отдельно и не входит в ускорение.
- Для транспонирования вызывается `.transpose(...).contiguous()`, чтобы
  учесть физическое копирование, поскольку чистый `transpose` создаёт view.
- Числа зависят от оборудования, прогрева, фоновой нагрузки и версии PyTorch.

## Замеры

{"\n".join(table)}

## Передача данных CPU ↔ GPU

{transfer_lines}

## Анализ результатов

1. **Наиболее быстрые на GPU операции.** Обычно выигрыш заметен в крупных
   матричных умножениях: параллельные вычисления позволяют загрузить GPU.
   Конкретный лидер определяется по таблице замеров выше; при отсутствии CUDA
   фактическое ускорение утверждать нельзя.
2. **Почему GPU бывает медленнее.** Запуск CUDA kernels и передача данных
   имеют накладные расходы. Для небольших матриц CPU иногда быстрее.
3. **Как влияет размер.** При увеличении размерности работы становится больше;
   в матричном умножении число арифметических операций растёт примерно как n³,
   и GPU обычно раскрывает преимущества сильнее.
4. **Передача CPU ↔ GPU.** Копирование по шине занимает время. Поэтому ускорение,
   полученное только для операции на GPU, не равно ускорению полного процесса.
   Частая передача небольших тензоров может полностью уничтожить выигрыш.
5. **Ограничения.** Время транспонирования измеряет также материализацию view,
   замеры проведены для 2D срезов, а не для всех 3D batched операций.
   Для строгого исследования нужны дополнительные повторы и отдельные батч-замеры.

## Воспроизведение

```bash
python homework_performance.py
```

Для сокращённого запуска без выделения полных 3D тензоров:

```bash
python homework_performance.py --quick
```
"""
    (folder / "PERFORMANCE_REPORT.md").write_text(report, encoding="utf-8")
    print("\nSaved performance_results.csv and PERFORMANCE_REPORT.md")
    return results


def main() -> None:
    """CLI точка входа."""
    parser = argparse.ArgumentParser(description="Lesson 1 CPU/CUDA performance experiment")
    parser.add_argument("--quick", action="store_true", help="Skip full 3D allocations")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per operation")
    args = parser.parse_args()
    run_benchmark(full_allocation=not args.quick, repeats=args.repeats)


if __name__ == "__main__":
    main()
