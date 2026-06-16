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
    if tt1.shape != tt2.shape:
        raise ValueError()

    new_cores = []

    for k in range(tt1.order):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]

        r_prev1, n, r_next1 = core1.shape
        r_prev2, _, r_next2 = core2.shape

        new_r_prev = r_prev1 * r_prev2
        new_r_next = r_next1 * r_next2

        new_data = []

        for rp1 in range(r_prev1):
            for rp2 in range(r_prev2):
                for i in range(n):
                    for rn1 in range(r_next1):
                        for rn2 in range(r_next2):
                            idx1 = rp1 * n * r_next1 + i * r_next1 + rn1
                            idx2 = rp2 * n * r_next2 + i * r_next2 + rn2
                            new_data.append(core1.data[idx1] * core2.data[idx2])

        new_core = DenseTensor((new_r_prev, n, new_r_next), data=new_data)
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
    result = backend.eye(1)

    for k in range(tt1.order):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]

        mat1_data = []
        mat2_data = []

        r_prev, n, r_next = core1.shape

        for i in range(r_prev * n):
            r = i // n
            ni = i % n
            for j in range(r_next):
                mat1_data.append(core1.data[r * n * r_next + ni * r_next + j])
                mat2_data.append(core2.data[r * n * r_next + ni * r_next + j])

        mat1 = DenseTensor((r_prev * n, r_next), data=mat1_data)
        mat2 = DenseTensor((r_prev * n, r_next), data=mat2_data)

        product = [[0.0] * r_next for _ in range(r_next)]

        for i in range(r_next):
            for j in range(r_next):
                s = 0.0
                for k_idx in range(r_prev * n):
                    s += mat1.data[k_idx * r_next + i] * mat2.data[k_idx * r_next + j]
                product[i][j] = s

        new_result = [[0.0] * r_next for _ in range(len(result))]

        for i in range(len(result)):
            for j in range(r_next):
                s = 0.0
                for k_idx in range(len(result[0])):
                    s += result[i][k_idx] * product[k_idx][j]
                new_result[i][j] = s

        result = new_result
    return result[0][0]
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