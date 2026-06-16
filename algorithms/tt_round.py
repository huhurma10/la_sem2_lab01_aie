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
    tt_tmp = TTTensor([core.copy() for core in tt.cores])
    right_canonicalize(tt_tmp, backend)

    new_cores = []

    for k in range(tt.order - 1):
        core = tt_tmp.cores[k]
        next_core = tt_tmp.cores[k + 1]

        r_prev, n, r_next = core.shape
        mat_data = []
        for i in range(r_prev * n):
            r = i // n
            ni = i % n
            for j in range(r_next):
                mat_data.append(core.data[r * n * r_next + ni * r_next + j])

        core_mat = DenseTensor((core.shape[0] * core.shape[1], core.shape[2]), data=mat_data)

        U, S, VT = backend.svd(core_mat)

        norm = sum(s * s for s in S.data) ** 0.5
        threshold = eps * norm if norm > 0 else 0
        rank = 0
        for r, s in enumerate(S.data):
            if s <= threshold or (max_rank is not None and r + 1 >= max_rank):
                rank = r + 1 if s <= threshold else max_rank
                break
        else:
            rank = len(S.data)

        if rank == 0:
            rank = 1

        U_data = []
        for i in range(core.shape[0] * core.shape[1]):
            for j in range(rank):
                U_data.append(U.data[i * U.shape[1] + j])
        U_trunc = DenseTensor((core.shape[0] * core.shape[1], rank), data=U_data)

        S_data = S.data[:rank]
        VT_data = []
        for i in range(rank):
            for j in range(VT.shape[1]):
                VT_data.append(VT.data[i * VT.shape[1] + j])
        VT_trunc = DenseTensor((rank, VT.shape[1]), data=VT_data)

        new_core_data = []
        for r in range(core.shape[0]):
            for n in range(core.shape[1]):
                for s in range(rank):
                    new_core_data.append(U_trunc.data[r * core.shape[1] * rank + n * rank + s])
        new_cores.append(DenseTensor((core.shape[0], core.shape[1], rank), data=new_core_data))

        sv_data = []
        for r in range(r_prev):
            for ni in range(n):
                for s in range(rank):
                    idx = r * n * rank + ni * rank + s
                    new_core_data.append(U_trunc.data[idx])
        new_cores.append(DenseTensor((r_prev, n, rank), data=new_core_data))

        sv_data = []
        for r in range(rank):
            for j in range(r_next):
                sv_data.append(S_data[r] * VT_trunc.data[r * r_next + j])

        r_next, n_next, r_next2 = next_core.shape
        result_data = []
        for r1 in range(rank):
            for n in range(n_next):
                for r2 in range(r_next2):
                    val = 0.0
                    for t in range(r_next):
                        val += sv_data[r1 * r_next + t] * next_core.data[t * n_next * r_next2 + ni * r_next2 + r2]
                    result_data.append(val)

        tt_tmp.cores[k + 1] = DenseTensor((rank, next_core.shape[1], next_core.shape[2]), data=result_data)

    new_cores.append(tt_tmp.cores[-1])
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
    norm = sum(s * s for s in S.data) ** 0.5
    if norm == 0:
        return len(S.data)

    threshold = delta * norm
    rank = 0

    for r, s in enumerate(S.data):
        if s <= threshold:
            rank = r + 1
            break
        if max_rank is not None and r + 1 >= max_rank:
            rank = max_rank
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
        raise ValueError("matrix должна быть двумерной")

    k, n = matrix.shape
    data = []
    for i in range(rank):
        for j in range(n):
            data.append(matrix.data[i * n + j])

    return DenseTensor((rank, n), data=data)
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