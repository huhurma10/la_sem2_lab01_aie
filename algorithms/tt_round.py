# algorithms/tt_round.py

"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор с уменьшенными рангами

    Args:
        tt:       исходный тензор
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    cores = [core.copy() for core in tt.cores]
    d = tt.order

    tt_tmp = TTTensor(cores)
    right_canonicalize(tt_tmp, backend)

    new_cores = []
    for k in range(d - 1):
        core = tt_tmp.cores[k]
        core_mat = DenseTensor(
            (core.shape[0], core.shape[1] * core.shape[2]),
            data=[core.data[r * core.shape[1] * core.shape[2] + j] for r in range(core.shape[0]) for j in range(core.shape[1] * core.shape[2])]
        )

        U, S, VT = backend.svd(core_mat)
        rank = _compute_rank(S, eps, max_rank)
        U_truncated = _truncate_columns(U, rank, backend)
        S_truncated = _truncate_vector(S, rank, backend)
        VT_truncated = _truncate_rows(VT, rank, backend)
        new_core_data = []
        for r in range(rank):
            for i in range(cores[k+1].shape[1]):
                new_core_data.append(S_truncated.data[r] * VT_truncated.data[r * cores[k+1].shape[1] + i])
        new_core = DenseTensor((rank, cores[k+1].shape[1], cores[k+1].shape[2]), data=new_core_data)
        new_cores.append(_multiply_diag_matrix(U_truncated, S_truncated, rank, backend))
        cores[k+1] = new_core

    new_cores.append(cores[-1])
    return TTTensor(new_cores)
    pass


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает int ранг усечения по вектору сингулярных значений.

    Args:
        S:        одномерный тензор формы (k,) — сингулярные значения
                  в порядке убывания
        delta:    абсолютный порог усечения (0 — без усечения по delta)
        max_rank: максимально допустимый ранг (None = без ограничения)
    """
    for r in range(len(S.data)):
        if S.data[r] <= delta:
            return r
        if max_rank is not None and r + 1 >= max_rank:
            return r + 1
    return len(S.data)
    pass


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    return _truncate_vector(DenseTensor((matrix.shape[1],), data=matrix.data), rank, backend).reshape((matrix.shape[0], rank))
    pass


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (k, n)
        rank:    число сохраняемых строк
        backend: интерфейс backend
    """
    return _truncate_vector(DenseTensor((matrix.shape[0],), data=matrix.data), rank, backend).reshape((rank, matrix.shape[1]))
    pass


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.

    Args:
        vector:  одномерный тензор формы (k,)
        rank:    число сохраняемых элементов
        backend: интерфейс backend
    """
    if len(vector.shape) != 1:
        raise ValueError("vector должна быть одномерной")
    return DenseTensor((rank,), data=vector.data[:rank])
    pass


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает произведение диагональной матрицы на обычную матрицу:
        diag(diag_vec) @ matrix

    Args:
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        matrix:   двумерный тензор формы (rank, n)
        rank:     число строк матрицы и длина диагонального вектора
        backend:  интерфейс backend
    """
    data = []
    for r in range(rank):
        scalar = diag_vec.data[r]
        for j in range(matrix.shape[1]):
            data.append(scalar * matrix.data[r * matrix.shape[1] + j])
    return DenseTensor((rank, matrix.shape[1]), data=data)
    pass