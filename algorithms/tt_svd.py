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
    residual = DenseTensor(shape, data=tensor.data.copy())
    ranks = [1]

    for k in range(d - 1):
        left_size = ranks[-1] * shape[k]
        right_size = math.prod(shape[k+1:])
        residual = residual.reshape((left_size, right_size))
        U, S, Vt = backend.svd(residual)
        rank = _compute_truncated_rank(S, eps, max_rank)
        U_trunc = _truncate_columns(U, rank, backend)
        S_trunc = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)

        # Создаем ядро
        core_shape = (ranks[-1], shape[k], rank)
        core_data = []
        for i in range(ranks[-1]):
            for j in range(shape[k]):
                for s in range(rank):
                    idx = i * shape[k] + j
                    core_data.append(U_trunc.data[idx * rank + s])

        core = DenseTensor(core_shape, data=core_data)
        cores.append(core)
        # Обновляем residual
        residual = _multiply_diag_matrix(S_trunc, Vt_trunc, rank, backend)
        ranks.append(rank)

    # Последнее ядро
    last_core_shape = (ranks[-1], shape[-1], 1)
    last_core_data = []

    for i in range(ranks[-1]):
        for j in range(shape[-1]):
            idx = i * shape[-1] + j
            last_core_data.append(residual.data[idx])

    last_core = DenseTensor(last_core_shape, data=last_core_data)
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
    n = len(S.data)
    if n == 0:
        return 0

    total_sq = sum(s * s for s in S.data)
    if total_sq == 0:
        return 1
    threshold_sq = (delta * delta) * total_sq
    cumsum_sq = 0.0
    rank = n

    for r in range(n - 1, -1, -1):
        cumsum_sq += S.data[r] * S.data[r]
        if cumsum_sq > threshold_sq:
            rank = r + 1
            break
    else:
        rank = 1

    if max_rank is not None:
        rank = min(rank, max_rank)
    return max(1, rank)

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
    m, n = matrix.shape
    data = []
    for i in range(m):
        for j in range(rank):
            data.append(matrix.data[i * n + j])
    return DenseTensor((m, rank), data=data)

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
        raise ValueError("matrix должна быть двумерной")
    k, n = matrix.shape
    data = matrix.data[:rank * n]
    return DenseTensor((rank, n), data=data)


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
    if len(diag_vec.shape) != 1 or len(matrix.shape) != 2:
        raise ValueError("Неверные размерности")

    if diag_vec.shape[0] != rank or matrix.shape[0] != rank:
        raise ValueError("Несовместимые размеры")

    data = []
    for r in range(rank):
        scalar = diag_vec.data[r]
        for j in range(matrix.shape[1]):
            data.append(scalar * matrix.data[r * matrix.shape[1] + j])

    return DenseTensor((rank, matrix.shape[1]), data=data)