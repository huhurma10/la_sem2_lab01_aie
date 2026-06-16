# algorithms/tt_svd.py

"""
TT-SVD алгоритм: разложение плотного тензора в TT-формат.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def tt_svd(
    tensor: DenseTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — тензор в TT-формате.

    Args:
        tensor:   DenseTensor с shape (n_0, n_1, ..., n_{d-1})
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    shape = tensor.shape
    d = len(shape)
    cores = []
    residual = tensor
    ranks = [1]

    for k in range(d - 1):
        m, n = residual.shape
        residual = residual.reshape((m, n))
        U, S, Vt = backend.svd(residual)
        rank = _compute_truncated_rank(S, eps, max_rank)
        U_trunc = _truncate_columns(U, rank, backend)
        S_trunc = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)

        # Создаем ядро
        core_data = []
        for r in range(rank):
            for i in range(shape[k]):
                core_data.append(U_trunc.data[r * shape[k] + i])
        core = DenseTensor((ranks[-1], shape[k], rank), data=core_data)
        cores.append(core)
        # Обновляем residual
        residual_data = []
        for r_idx in range(rank):
            for j in range(Vt_trunc.shape[1]):
                residual_data.append(S_trunc.data[r_idx] * Vt_trunc.data[r_idx * Vt_trunc.shape[1] + j])
        residual = DenseTensor((rank, residual.shape[1]), data=residual_data)
        ranks.append(rank)

    # Последнее ядро
    last_core = DenseTensor((ranks[-1], shape[-1], 1), data=[])
    for r in range(ranks[-1]):
        for i in range(shape[-1]):
            last_core.data.append(1.0)
    cores.append(last_core)

    return TTTensor(cores)
    pass


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_truncated_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает ранг усечения по сингулярным значениям.

    Args:
        S:        DenseTensor (k,) — сингулярные значения по убыванию
        delta:    порог усечения
        max_rank: максимальный ранг (None = без ограничения)
    """
    rank = 0
    total = sum(S.data)
    accum = 0.0
    threshold = delta * total
    for r, s in enumerate(S.data):
        accum += s
        if s <= delta or (max_rank is not None and r + 1 >= max_rank):
            rank = r + 1
            break
    else:
        rank = len(S.data)
    return rank
    pass


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Используется после SVD для усечения матрицы левых сингулярных векторов:
        U in R^{m x n} -> U_trunc in R^{m x rank}

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    if len(matrix.shape) != 2:
        raise ValueError("matrix должна быть двумерной")
    data = matrix.data[:matrix.shape[0] * rank]
    return DenseTensor((matrix.shape[0], rank), data=data)
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
    if len(matrix.shape) != 2:
        raise ValueError()
    data = matrix.data[:rank * matrix.shape[1]]
    return DenseTensor((rank, matrix.shape[1]), data=data)
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
        raise ValueError()
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