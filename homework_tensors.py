"""Урок 1, задание 1: создание тензоров, операции, индексация и формы."""

from __future__ import annotations

import torch


def make_tensors(seed: int = 42) -> dict[str, torch.Tensor]:
    """Создать четыре тензора из пункта 1.1 с воспроизводимым генератором."""
    generator = torch.Generator().manual_seed(seed)
    result = {
        "random_3x4": torch.rand((3, 4), generator=generator),
        "zeros_2x3x4": torch.zeros((2, 3, 4)),
        "ones_5x5": torch.ones((5, 5)),
        "numbers_4x4": torch.arange(16).reshape(4, 4),
    }
    return result


def matrix_operations(a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    """Операции пункта 1.2; A должна быть 3×4, B — 4×3."""
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("A and B must be torch.Tensor objects")
    if a.shape != (3, 4) or b.shape != (4, 3):
        raise ValueError("Expected A shape (3, 4) and B shape (4, 3)")
    return {
        "a_transposed": a.T,
        "matrix_product": torch.matmul(a, b),
        "elementwise_product": a * b.T,
        "sum_of_a": a.sum(),
    }


def extract_slices(tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Извлечь строки/столбцы/центр и чётные индексы 5×5×5 тензора.

    Первая строка и последний столбец берутся на каждом из 5 слоёв.
    Центральная подматрица 2×2: индексы [2:4, 2:4] каждого слоя.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    if tensor.shape != (5, 5, 5):
        raise ValueError("Expected tensor shape (5, 5, 5)")
    return {
        "first_row_each_layer": tensor[:, 0, :],  # 5×5
        "last_column_each_layer": tensor[:, :, -1],  # 5×5
        "central_2x2_each_layer": tensor[:, 2:4, 2:4],  # 5×2×2
        "even_indices_all_axes": tensor[::2, ::2, ::2],  # 3×3×3
    }


def reshape_24() -> dict[str, torch.Tensor]:
    """Представить 24 последовательных элемента в 5 разных формах."""
    source = torch.arange(24)
    shapes = [(2, 12), (3, 8), (4, 6), (2, 3, 4), (2, 2, 2, 3)]
    return {"x".join(map(str, shape)): source.reshape(shape) for shape in shapes}


def main() -> None:
    """Вывести результаты всех четырёх подпунктов."""
    torch.set_printoptions(precision=4)
    print("=== 1.1 Создание тензоров ===")
    for name, tensor in make_tensors().items():
        print(f"{name}: shape={tuple(tensor.shape)}\n{tensor}\n")

    print("=== 1.2 Операции с тензорами ===")
    a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    b = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    for name, value in matrix_operations(a, b).items():
        print(f"{name}:\n{value}\n")

    print("=== 1.3 Индексация и срезы ===")
    tensor = torch.arange(125).reshape(5, 5, 5)
    for name, value in extract_slices(tensor).items():
        print(f"{name}: shape={tuple(value.shape)}\n{value}\n")

    print("=== 1.4 Изменение формы ===")
    for name, value in reshape_24().items():
        print(f"{name}: shape={tuple(value.shape)}\n{value}\n")


if __name__ == "__main__":
    main()
