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
    tt_tmp = right_canonicalize(tt, backend)
    new_cores = []

    for k in range(tt.order - 1):
        core = tt_tmp.cores[k]
        next_core = tt_tmp.cores[k + 1]
        r_prev, n, r_next = core.shape
        core_mat_data = []
        for r_idx in range(r_prev):
            for n_idx in range(n):
                for r_next_idx in range(r_next):
                    flat_index = (r_idx * n + n_idx) * r_next + r_next_idx
                    core_mat_data.append(core.data[flat_index])
        core_mat = DenseTensor((r_prev * n, r_next), data=core_mat_data)

        # Выполняем SVD на этой матрице
        U, S, VT = backend.svd(core_mat)
        norm_sq = sum(s * s for s in S.data)
        norm = norm_sq ** 0.5

        threshold = eps * norm if norm > 0 else 0.0
        rank = 0
        for r, s in enumerate(S.data):
            if s <= threshold:
                rank = r
                if rank == 0:
                    rank = 1
                break
        else:
            rank = len(S.data)

        if max_rank is not None:
            rank = min(rank, max_rank)

        if rank == 0 and len(S.data) > 0:
            rank = 1
        elif len(S.data) == 0:
            rank = 0

        U_trunc_data = []
        for i in range(r_prev * n):
            for j in range(rank):
                U_trunc_data.append(U.data[i * U.shape[1] + j])
        U_trunc = DenseTensor((r_prev * n, rank), data=U_trunc_data)

        # Усекаем S
        S_trunc = DenseTensor((rank,), data=S.data[:rank])

        # Усекаем VT до размера rank x r_next
        VT_trunc_data = []
        for i in range(rank):
            for j in range(r_next):
                VT_trunc_data.append(VT.data[i * VT.shape[1] + j])
        VT_trunc = DenseTensor((rank, r_next), data=VT_trunc_data)

        new_core_data = []

        for r in range(r_prev):
            for ni in range(n):
                for s in range(rank):
                    idx = r * n * rank + ni * rank + s
                    new_core_data.append(U_trunc.data[idx])
        new_cores.append(DenseTensor((r_prev, n, rank), data=new_core_data))

        # Умножаем S на VT
        sv_data = []
        for r in range(rank):
            for j in range(r_next):
                sv_data.append(S_trunc.data[r] * VT_trunc.data[r * r_next + j])

        # Умножаем полученную матрицу на следующий core
        r_next, n_next, r_next2 = next_core.shape
        result_data = []
        for r1 in range(rank):
            for ni in range(n_next):
                for r2 in range(r_next2):
                    val = 0.0
                    for t in range(r_next):
                        val += sv_data[r1 * r_next + t] * next_core.data[t * n_next * r_next2 + ni * r_next2 + r2]
                    result_data.append(val)
        tt_tmp.cores[k + 1] = DenseTensor((rank, n_next, r_next2), data=result_data)
    new_cores.append(tt_tmp.cores[-1])

    return TTTensor(new_cores)

def _compute_rank(
        S: DenseTensor,
        delta: float,
        max_rank: int | None
) -> int:
    """
    Возвращает int ранг усечения по вектору сингулярных значений.
    """

    threshold = delta
    rank = 0

    for r, s in enumerate(S.data):
        if s <= threshold:
            rank = r
            break
    else:
        rank = len(S.data)

    if max_rank is not None:
        rank = min(rank, max_rank)

    if rank == 0 and len(S.data) > 0:
        rank = 1

    return rank

def _truncate_columns(
        matrix: DenseTensor,
        rank: int,
        backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.
    """
    if len(matrix.shape) != 2:
        raise ValueError("matrix должна быть двумерной")

    m, n = matrix.shape
    if rank > n:
        raise ValueError(f"rank ({rank}) не может быть больше количества столбцов ({n})")

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
    """
    if len(matrix.shape) != 2:
        raise ValueError("matrix должна быть двумерной")

    k, n = matrix.shape
    if rank > k:
        raise ValueError(f"rank ({rank}) не может быть больше количества строк ({k})")

    data = []
    for i in range(rank):
        for j in range(n):
            data.append(matrix.data[i * n + j])

    return DenseTensor((rank, n), data=data)


def _truncate_vector(
        vector: DenseTensor,
        rank: int,
        backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.
    """
    if len(vector.shape) != 1:
        raise ValueError("vector должна быть одномерной")

    if rank > len(vector.data):
        raise ValueError(f"rank ({rank}) не может быть больше длины вектора ({len(vector.data)})")

    return DenseTensor((rank,), data=vector.data[:rank])


def _multiply_diag_matrix(
        diag_vec: DenseTensor,
        matrix: DenseTensor,
        rank: int,
        backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает произведение диагональной матрицы A @ matrix,
    где A = diag(diag_vec).
    A имеет форму (rank, rank). matrix может быть (rank, n) или (m, rank).

    Args:
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        matrix:   двумерный тензор формы (rank, n) или (m, rank)
        rank:     число строк/столбцов диагональной матрицы.
        backend:  интерфейс backend
    """
    if len(diag_vec.shape) != 1 or diag_vec.shape[0] != rank:
        raise ValueError(f"diag_vec должен быть одномерным формы ({rank},)")
    if (len(matrix.shape) != 2) or (matrix.shape[0] != rank and matrix.shape[1] != rank):
        raise ValueError(f"matrix должна быть двумерной с одной из размерностей равной {rank}")
    if diag_vec.shape[0] != rank or matrix.shape[0] != rank:
        raise ValueError("Несовпадение размерностей")

    data = []
    for r in range(rank):
        scalar = diag_vec.data[r]
        for j in range(matrix.shape[1]):
            data.append(scalar * matrix.data[r * matrix.shape[1] + j])

    return DenseTensor((rank, matrix.shape[1]), data=data)