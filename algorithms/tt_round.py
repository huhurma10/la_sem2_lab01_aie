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
        r_prev, n, r_next_old = core.shape
        core_mat_data = []
        for r_idx in range(r_prev):
            for n_idx in range(n):
                for r_next_idx in range(r_next_old):
                    flat_index = (r_idx * n + n_idx) * r_next_old + r_next_idx
                    core_mat_data.append(core.data[flat_index])
        core_mat = DenseTensor((r_prev * n, r_next_old), data=core_mat_data)

        # Выполняем SVD на этой матрице
        U, S, VT = backend.svd(
            core_mat)
        norm_sq = sum(s * s for s in S.data)
        norm = norm_sq ** 0.5

        threshold = eps * norm if norm > 0 else 0.0
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
        elif len(S.data) == 0:
            rank = 0

        if rank == 0:
            zero_cores = [DenseTensor(core.shape, data=[0.0] * core.size) for core in tt.cores]
            return TTTensor(zero_cores)

        U_trunc_cols = U.shape[1]
        if U_trunc_cols < rank:
            pass

        S_trunc = _truncate_vector(S, rank, backend)
        VT_trunc = _truncate_rows(VT, rank, backend)
        U_for_core = _truncate_columns(U, rank, backend)

        new_core_k_data = []
        for r_prev_idx in range(r_prev):
            for n_idx in range(n):
                for rank_idx in range(rank):
                    current_u_part = _multiply_diag_matrix(
                        S_trunc,
                        U_for_core,
                        rank,
                        backend
                    )

                    flat_idx_in_u = (r_prev_idx * n + n_idx) * rank + rank_idx
                    new_core_k_data.append(current_u_part.data[flat_idx_in_u])

        new_cores.append(DenseTensor((r_prev, n, rank), data=new_core_k_data))

        processed_vt = _multiply_diag_matrix(
            S_trunc,
            VT_trunc,
            rank,
            backend
        )

        r_next_old_actual, n_next, r_next2 = next_core.shape
        next_core_reshaped_data = []
        for t_idx in range(r_next_old_actual):
            for j_idx in range(n_next):
                for l_idx in range(r_next2):
                    flat_index_next = t_idx * (n_next * r_next2) + j_idx * r_next2 + l_idx
                    next_core_reshaped_data.append(next_core.data[flat_index_next])
        next_core_reshaped = DenseTensor((r_next_old_actual, n_next * r_next2), data=next_core_reshaped_data)
        new_next_core_flat = backend.matmul(processed_vt, next_core_reshaped)

        new_next_core_data = []
        for rank_idx in range(rank):
            for j_idx in range(n_next):
                for r2_idx in range(r_next2):
                    flat_index_result = rank_idx * (n_next * r_next2) + j_idx * r_next2 + r2_idx
                    new_next_core_data.append(new_next_core_flat.data[flat_index_result])

        tt_tmp.cores[k + 1] = DenseTensor((rank, n_next, r_next2), data=new_next_core_data)
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
            rank = r + 1
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
    if len(matrix.shape) != 2 or matrix.shape[0] != rank and matrix.shape[1] != rank:
        raise ValueError(f"matrix должна быть двумерной с одной из размерностей равной {rank}")

    result_data = []

    if matrix.shape[0] == rank:
        m, n = matrix.shape
        for r in range(rank):
            scalar = diag_vec.data[r]
            for col_idx in range(n):
                result_data.append(scalar * matrix.data[r * n + col_idx])
        return DenseTensor((rank, n), data=result_data)
    else:
        raise ValueError