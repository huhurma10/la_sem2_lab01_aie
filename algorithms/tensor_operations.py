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

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    new_cores = []
    for core1, core2 in zip(tt1.cores, tt2.cores):
        core_sum_data = [a + b for a, b in zip(core1.data, core2.data)]
        new_core = DenseTensor(core1.shape, data=core_sum_data)
        new_cores.append(new_core)
    return TTTensor(new_cores)
    pass


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
    new_cores = []
    for i, core in enumerate(tt.cores):
        if i == 0:
            new_data = [a * alpha for a in core.data]
            new_core = DenseTensor(core.shape, data=new_data)
        else:
            new_core = core.copy()
        new_cores.append(new_core)
    return TTTensor(new_cores)
    pass


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного произведения (произведения Адамара).

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    new_cores = []
    for core1, core2 in zip(tt1.cores, tt2.cores):
        # Умножение ядра поэлементно
        data = [a * b for a, b in zip(core1.data, core2.data)]
        new_core = DenseTensor(core1.shape, data=data)
        new_cores.append(new_core)
    return TTTensor(new_cores)
    pass


def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    """
    Возвращает скалярное произведение двух TT-тензоров: <tt1, tt2>.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    result = 1.0
    for core1, core2 in zip(tt1.cores, tt2.cores):
        # Векторизуем по рангу
        sum_core = 0.0
        for r_prev in range(core1.shape[0]):
            for r_next in range(core1.shape[2]):
                # Складываем по r
                # создаем временные списки для r
                val1 = core1.data[r_prev * core1.shape[1] * core1.shape[2] + r_next]
                val2 = core2.data[r_prev * core2.shape[1] * core2.shape[2] + r_next]
                sum_core += val1 * val2
        result *= sum_core
    return result
    pass


def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает Фробениусову норму TT-тензора.

    Args:
        tt:      TTTensor
        backend: интерфейс backend
    """
    return math.sqrt(tt_dot(tt, tt, backend))
    pass


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает норму разности: ||tt1 - tt2||_F.
    Вычисляется без восстановления полных тензоров:

    Args:
        tt1, tt2: TTTensor
        backend:  интерфейс backend
    """
    norm_a = tt_norm(tt1, backend)
    norm_b = tt_norm(tt2, backend)
    inner_ab = tt_dot(tt1, tt2, backend)
    return math.sqrt(norm_a ** 2 + norm_b ** 2 - 2 * inner_ab)
    pass