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

    for k in range(d - 1, 0, -1):
        core = cores[k]
        r_prev, n, r_next = core.shape

        # Развернуть ядро в матрицу (r_prev) x (n * r_next)
        A_data = []
        for i in range(r_prev):
            for j in range(n * r_next):
                A_data.append(core.data[i * n * r_next + j])
        A = DenseTensor((r_prev, n * r_next), data=A_data)

        # SVD
        U, S, VT = backend.svd(A)

        # Новый ранг – минимальный из возможных для ортогонализации, но с учётом max_rank
        r_new = min(r_prev, n * r_next)
        if max_rank is not None:
            r_new = min(r_new, max_rank)
        if r_new == 0:
            r_new = 1

        # Q = V^T (первые r_new строк) – ортонормированные строки
        Q_data = []
        for i in range(r_new):
            for j in range(n * r_next):
                Q_data.append(VT.data[i * (n * r_next) + j])
        Q = DenseTensor((r_new, n * r_next), data=Q_data)

        # R = U * S (первые r_new столбцов U и первые r_new сингулярных значений)
        R_data = []
        for i in range(r_prev):
            for j in range(r_new):
                val = 0.0
                for s in range(r_new):
                    val += U.data[i * U.shape[1] + s] * S.data[s]
                R_data.append(val)
        R = DenseTensor((r_prev, r_new), data=R_data)

        # Свернуть Q в ядро (r_new, n, r_next)
        new_core_data = []
        for i in range(r_new):
            for j in range(n):
                for l in range(r_next):
                    idx = i * (n * r_next) + j * r_next + l
                    new_core_data.append(Q.data[idx])
        cores[k] = DenseTensor((r_new, n, r_next), data=new_core_data)

        # Поглотить R в предыдущее ядро
        prev_core = cores[k - 1]
        r_prev_prev, n_prev, _ = prev_core.shape

        # prev_core как матрица (r_prev_prev * n_prev) x r_prev
        prev_mat_data = []
        for i in range(r_prev_prev * n_prev):
            for j in range(r_prev):
                prev_mat_data.append(prev_core.data[i * r_prev + j])
        prev_mat = DenseTensor((r_prev_prev * n_prev, r_prev), data=prev_mat_data)

        # Умножить prev_mat на R
        result_data = []
        for i in range(r_prev_prev * n_prev):
            for j in range(r_new):
                val = 0.0
                for t in range(r_prev):
                    val += prev_mat.data[i * r_prev + t] * R.data[t * r_new + j]
                result_data.append(val)

        cores[k - 1] = DenseTensor(
            (r_prev_prev, n_prev, r_new),
            data=result_data
        )

    # --- Левый проход с SVD-усечением (стандартный TT-rounding) ---
    new_cores = []

    for k in range(d - 1):
        core = cores[k]
        next_core = cores[k + 1]
        r_prev, n, r_next = core.shape

        # Развернуть ядро в матрицу (r_prev * n) x r_next
        A_data = []
        for i in range(r_prev * n):
            for j in range(r_next):
                A_data.append(core.data[i * r_next + j])
        A = DenseTensor((r_prev * n, r_next), data=A_data)

        # SVD
        U, S, VT = backend.svd(A)

        # Вычисляем порог и новый ранг
        norm_sq = sum(s * s for s in S.data)
        norm = norm_sq ** 0.5 if norm_sq > 0 else 0.0
        threshold = eps * norm if norm > 0 else 0.0

        r_new = len(S.data)
        for i, s in enumerate(S.data):
            if s <= threshold:
                r_new = i
                if r_new == 0:
                    r_new = 1
                break
        if max_rank is not None:
            r_new = min(r_new, max_rank)
        if r_new == 0:
            r_new = 1

        # Усекаем U
        U_trunc_data = []
        for i in range(r_prev * n):
            for j in range(r_new):
                U_trunc_data.append(U.data[i * U.shape[1] + j])
        U_trunc = DenseTensor((r_prev * n, r_new), data=U_trunc_data)

        S_trunc = DenseTensor((r_new,), data=S.data[:r_new])

        VT_trunc_data = []
        for i in range(r_new):
            for j in range(r_next):
                VT_trunc_data.append(VT.data[i * VT.shape[1] + j])
        VT_trunc = DenseTensor((r_new, r_next), data=VT_trunc_data)

        # Формируем новое ядро
        new_core_data = []
        for r in range(r_prev):
            for ni in range(n):
                for s in range(r_new):
                    idx = r * n * r_new + ni * r_new + s
                    new_core_data.append(U_trunc.data[idx])
        new_cores.append(DenseTensor((r_prev, n, r_new), data=new_core_data))

        # Обновляем следующее ядро: R = S * V^T
        sv_data = []
        for i in range(r_new):
            for j in range(r_next):
                val = 0.0
                for s in range(r_new):
                    val += S_trunc.data[s] * VT_trunc.data[s * r_next + j]
                sv_data.append(val)

        r_next_old, n_next, r_next2 = next_core.shape
        result_data = []
        for i in range(r_new):
            for ni in range(n_next):
                for j in range(r_next2):
                    val = 0.0
                    for t in range(r_next_old):
                        val += sv_data[i * r_next_old + t] * next_core.data[t * n_next * r_next2 + ni * r_next2 + j]
                    result_data.append(val)
        cores[k + 1] = DenseTensor((r_new, n_next, r_next2), data=result_data)

    # Добавляем последнее ядро
    new_cores.append(cores[-1])

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