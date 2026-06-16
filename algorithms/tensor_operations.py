# algorithms/tensor_operations.py

"""
Базовые операции с TT-тензорами.

Все операции работают напрямую с TT-ядрами,
не восстанавливая полный тензор.

Содержит:
    - tt_add:         поэлементное сложение
    - tt_scalar_mul:  умножение на скаляр
    - tt_hadamard:    поэлементное произведение (Адамар)
    - tt_dot:         скалярное произведение <A, B>
    - tt_norm:        Фробениусова норма
    - tt_diff_norm:   ||A - B||_F без восстановления полных тензоров

Все операции через backend.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface

Number = int | float


def tt_add(
        tt1: TTTensor,
        tt2: TTTensor,
        backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного сложения двух TT-тензоров.

    Сложение двух TT-тензоров выполняется путем конкатенации ядер:
    Для каждого ядра k создается блочная матрица:
        core_k = [[core1_k, 0],
                  [0, core2_k]]
    с соответствующими рангами: r_k = r1_k + r2_k

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError("Формы тензоров должны совпадать")

    d = tt1.order
    new_cores = []

    for k in range(d):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]

        r1_prev, n1, r1_next = core1.shape
        r2_prev, n2, r2_next = core2.shape

        # Проверяем, что размерности мод совпадают
        if n1 != n2:
            raise ValueError(f"Размерность моды {k} не совпадает: {n1} vs {n2}")
        n = n1

        # Новые ранги: сумма рангов
        r_prev = r1_prev + r2_prev
        r_next = r1_next + r2_next

        # Создаем новое ядро размером (r_prev, n, r_next)
        new_core_data = []
        for i in range(r_prev):
            for j in range(n):
                for k_idx in range(r_next):
                    if i < r1_prev and k_idx < r1_next:
                        # Берем из первого ядра
                        val = core1.data[i * n * r1_next + j * r1_next + k_idx]
                    elif i >= r1_prev and k_idx >= r1_next:
                        # Берем из второго ядра
                        i2 = i - r1_prev
                        k2 = k_idx - r1_next
                        val = core2.data[i2 * n * r2_next + j * r2_next + k2]
                    else:
                        val = 0.0
                    new_core_data.append(float(val))

        new_cores.append(DenseTensor((r_prev, n, r_next), data=new_core_data))

    return TTTensor(new_cores)


def tt_scalar_mul(
        tt: TTTensor,
        alpha: Number,
        backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат умножения TT-тензора на скаляр.
    Модифицируем только первое ядро.

    Args:
        tt:      TTTensor
        alpha:   число
        backend: интерфейс backend
    """
    if not isinstance(alpha, (int, float)):
        raise TypeError("alpha должно быть числом")

    # Создаем копию тензора
    new_cores = [core.copy() for core in tt.cores]

    # Умножаем первое ядро на скаляр
    r_prev, n, r_next = new_cores[0].shape
    new_data = [float(alpha) * val for val in new_cores[0].data]
    new_cores[0] = DenseTensor((r_prev, n, r_next), data=new_data)

    return TTTensor(new_cores)


def tt_hadamard(
        tt1: TTTensor,
        tt2: TTTensor,
        backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного произведения (произведения Адамара).

    Произведение Адамара двух TT-тензоров выполняется путем кронекеровского
    произведения ядер:
        core_k = core1_k ⊗ core2_k
    с рангами: r_k = r1_k * r2_k

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError("Формы тензоров должны совпадать")

    d = tt1.order
    new_cores = []

    for k in range(d):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]

        r1_prev, n1, r1_next = core1.shape
        r2_prev, n2, r2_next = core2.shape

        if n1 != n2:
            raise ValueError(f"Размерность моды {k} не совпадает: {n1} vs {n2}")
        n = n1

        # Новые ранги: произведение рангов
        r_prev = r1_prev * r2_prev
        r_next = r1_next * r2_next

        # Кронекеровское произведение ядер
        new_core_data = []
        for i1 in range(r1_prev):
            for i2 in range(r2_prev):
                for j in range(n):
                    for k1 in range(r1_next):
                        for k2 in range(r2_next):
                            val1 = core1.data[i1 * n * r1_next + j * r1_next + k1]
                            val2 = core2.data[i2 * n * r2_next + j * r2_next + k2]
                            new_core_data.append(float(val1 * val2))

        new_cores.append(DenseTensor((r_prev, n, r_next), data=new_core_data))

    return TTTensor(new_cores)


def tt_dot(
        tt1: TTTensor,
        tt2: TTTensor,
        backend: BackendInterface
) -> Number:
    """
    Возвращает скалярное произведение двух TT-тензоров: <tt1, tt2>.

    Вычисляется как сумма по всем индексам произведения соответствующих элементов.
    Используется эффективный алгоритм через последовательное сжатие ядер.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError("Формы тензоров должны совпадать")

    d = tt1.order

    # Инициализируем матрицу размера (r1_0 * r2_0, r1_0 * r2_0)
    # r1_0 = r2_0 = 1, поэтому это скаляр
    result = DenseTensor((1, 1), data=[1.0])

    for k in range(d):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]

        r1_prev, n, r1_next = core1.shape
        r2_prev, _, r2_next = core2.shape

        # Вычисляем скалярное произведение ядер с учетом предыдущего результата
        new_size = r1_next * r2_next
        new_data = [0.0] * (new_size * new_size)

        for i1 in range(r1_prev):
            for i2 in range(r2_prev):
                for j in range(n):
                    for k1 in range(r1_next):
                        for k2 in range(r2_next):
                            val1 = core1.data[i1 * n * r1_next + j * r1_next + k1]
                            val2 = core2.data[i2 * n * r2_next + j * r2_next + k2]

                            # Умножаем на предыдущий результат
                            prev_val = result.data[0]
                            idx_out = k1 * r2_next + k2
                            new_data[idx_out * new_size + idx_out] += float(prev_val * val1 * val2)

        result = DenseTensor((new_size, new_size), data=new_data)

    # Возвращаем единственный элемент
    return result.data[0]


def tt_norm(
        tt: TTTensor,
        backend: BackendInterface
) -> float:
    """
    Возвращает Фробениусову норму TT-тензора.

    Используется tt_dot(tt, tt).

    Args:
        tt:      TTTensor
        backend: интерфейс backend
    """
    dot_product = tt_dot(tt, tt, backend)
    return math.sqrt(float(dot_product))


def tt_diff_norm(
        tt1: TTTensor,
        tt2: TTTensor,
        backend: BackendInterface
) -> float:
    """
    Возвращает норму разности: ||tt1 - tt2||_F.
    Вычисляется без восстановления полных тензоров

    Args:
        tt1, tt2: TTTensor
        backend:  интерфейс backend
    """
    norm1_sq = tt_dot(tt1, tt1, backend)
    norm2_sq = tt_dot(tt2, tt2, backend)
    dot = tt_dot(tt1, tt2, backend)

    diff_sq = float(norm1_sq - 2 * dot + norm2_sq)
    if diff_sq < 0 and diff_sq > -1e-10:
        diff_sq = 0.0

    return math.sqrt(max(0.0, diff_sq))