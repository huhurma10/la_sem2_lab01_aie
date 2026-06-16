# algorithms/canonical_form.py

"""
Приведение TT-тензора в канонические формы (полная правая и
левая ортогонализация ядер).
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в лево-канонической форме.

    Лево-каноническая форма: все ядра, кроме последнего, лево-ортогональны.
    Для каждого ядра k < d-1: core_k^T @ core_k = I (по первым двум размерностям)

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    tt_copy = TTTensor([core.copy() for core in tt.cores])
    d = tt_copy.order

    for k in range(d - 1):
        core = tt_copy.cores[k]
        r_prev, n, r_next = core.shape

        core_mat_data = []
        for i in range(r_prev * n):
            for j in range(r_next):
                core_mat_data.append(core.data[i * r_next + j])
        core_mat = DenseTensor((r_prev * n, r_next), data=core_mat_data)

        U, S, VT = backend.svd(core_mat)

        rank = len(S.data)

        Q_data = []
        for i in range(r_prev * n):
            for j in range(rank):
                Q_data.append(U.data[i * U.shape[1] + j])
        Q = DenseTensor((r_prev * n, rank), data=Q_data)

        R_data = []
        for i in range(rank):
            for j in range(r_next):
                val = 0.0
                for s in range(rank):
                    val += S.data[s] * VT.data[s * r_next + j]
                R_data.append(val)
        R = DenseTensor((rank, r_next), data=R_data)

        new_core_data = []
        for r in range(r_prev):
            for ni in range(n):
                for s in range(rank):
                    idx = r * n * rank + ni * rank + s
                    new_core_data.append(Q.data[idx])
        tt_copy.cores[k] = DenseTensor((r_prev, n, rank), data=new_core_data)

        # Умножаем R на следующее ядро
        if k + 1 < d:
            next_core = tt_copy.cores[k + 1]
            r_next_next = next_core.shape[2]

            result_data = []
            for r1 in range(rank):
                for ni in range(next_core.shape[1]):
                    for r2 in range(r_next_next):
                        val = 0.0
                        for t in range(r_next):
                            val += R.data[r1 * r_next + t] * next_core.data[
                                t * next_core.shape[1] * r_next_next + ni * r_next_next + r2]
                        result_data.append(val)

            tt_copy.cores[k + 1] = DenseTensor((rank, next_core.shape[1], r_next_next), data=result_data)

    return tt_copy


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Право-каноническая форма: все ядра, кроме первого, право-ортогональны.
    Для каждого ядра k > 0: core_k @ core_k^T = I (по последним двум размерностям)

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    # Создаем копию тензора
    tt_copy = TTTensor([core.copy() for core in tt.cores])
    d = tt_copy.order

    for k in range(d - 1, 0, -1):
        core = tt_copy.cores[k]
        r_prev, n, r_next = core.shape

        core_mat_data = []
        for i in range(r_prev):
            for j in range(n * r_next):
                core_mat_data.append(core.data[i * n * r_next + j])
        core_mat = DenseTensor((r_prev, n * r_next), data=core_mat_data)

        # SVD разложение
        U, S, VT = backend.svd(core_mat)

        rank = len(S.data)

        Q_data = []
        for i in range(r_prev):
            for j in range(rank):
                Q_data.append(U.data[i * U.shape[1] + j])
        Q = DenseTensor((r_prev, rank), data=Q_data)
        R_data = []
        for i in range(rank):
            for j in range(n * r_next):
                val = 0.0
                for s in range(rank):
                    val += S.data[s] * VT.data[s * (n * r_next) + j]
                R_data.append(val)
        R = DenseTensor((rank, n * r_next), data=R_data)

        new_core_data = []
        for r in range(r_prev):
            for ni in range(n):
                for s in range(rank):
                    # Q - это матрица (r_prev, rank)
                    new_core_data.append(Q.data[r * rank + s])
        tt_copy.cores[k] = DenseTensor((r_prev, n, rank), data=new_core_data)

        # Умножаем предыдущее ядро на R
        if k - 1 >= 0:
            prev_core = tt_copy.cores[k - 1]
            r_prev_prev = prev_core.shape[0]
            n_prev = prev_core.shape[1]

            r_core_data = []
            for i in range(rank):
                for ni in range(n):
                    for j in range(r_next):
                        idx = i * n * r_next + ni * r_next + j
                        r_core_data.append(R.data[idx])
            R_core = DenseTensor((rank, n, r_next), data=r_core_data)

            prev_mat_data = []
            for i in range(r_prev_prev * n_prev):
                for j in range(r_prev):
                    prev_mat_data.append(prev_core.data[i * r_prev + j])
            prev_mat = DenseTensor((r_prev_prev * n_prev, r_prev), data=prev_mat_data)
            result_mat_data = []
            for i in range(r_prev_prev * n_prev):
                for j in range(n * r_next):
                    val = 0.0
                    for t in range(r_prev):
                        val += prev_mat.data[i * r_prev + t] * R.data[t * (n * r_next) + j]
                    result_mat_data.append(val)
            result_mat = DenseTensor((r_prev_prev * n_prev, n * r_next), data=result_mat_data)
            new_prev_data = []
            for i in range(r_prev_prev * n_prev):
                for j in range(rank):
                    new_prev_data.append(result_mat.data[i * (n * r_next) + j])

            tt_copy.cores[k - 1] = DenseTensor((r_prev_prev, n_prev, rank), data=new_prev_data)

    return tt_copy


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _numerical_rank(
        S: DenseTensor,
        rel_tol: float = 1e-8,
        abs_tol: float = 1e-12
) -> int:
    """
    Возвращает числовой ранг матрицы по вектору сингулярных значений.

    Сингулярное число \sigma_i считаем ненулевым, если:
        |\sigma_i| > max(abs_tol, rel_tol * max(\sigma_1, ..., \sigma_n))

    Args:
        S:       одномерный тензор формы (k,) — сингулярные значения
                 в порядке убывания
        rel_tol: относительный допуск (по умолчанию 1e-8)
        abs_tol: абсолютный допуск (по умолчанию 1e-12)
    """
    if len(S.data) == 0:
        return 0

    max_s = max(abs(s) for s in S.data)
    if max_s == 0:
        return 0

    threshold = max(abs_tol, rel_tol * max_s)
    rank = 0
    for s in S.data:
        if abs(s) > threshold:
            rank += 1
        else:
            break
    return rank


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

    if rank <= 0:
        raise ValueError("rank должен быть положительным")

    m, n = matrix.shape
    if rank > n:
        rank = n

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

    if rank <= 0:
        raise ValueError("rank должен быть положительным")

    k, n = matrix.shape
    if rank > k:
        rank = k

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

    Args:
        vector:  одномерный тензор формы (k,)
        rank:    число сохраняемых элементов
        backend: интерфейс backend
    """
    if len(vector.shape) != 1:
        raise ValueError("vector должна быть одномерной")

    if rank <= 0:
        raise ValueError("rank должен быть положительным")

    k = vector.shape[0]
    if rank > k:
        rank = k

    return DenseTensor((rank,), data=vector.data[:rank])


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
        rank:     длина диагонального вектора
        backend:  интерфейс backend
    """
    if len(diag_vec.shape) != 1:
        raise ValueError("diag_vec должна быть одномерной")

    if len(matrix.shape) != 2:
        raise ValueError("matrix должна быть двумерной")

    if diag_vec.shape[0] != rank or matrix.shape[0] != rank:
        raise ValueError("Несовпадение размерностей")

    data = []
    for r in range(rank):
        scalar = diag_vec.data[r]
        for j in range(matrix.shape[1]):
            data.append(scalar * matrix.data[r * matrix.shape[1] + j])

    return DenseTensor((rank, matrix.shape[1]), data=data)


def _multiply_columns_by_diag(
        matrix: DenseTensor,
        diag_vec: DenseTensor,
        backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает результат произведения обычной матрицы на диагональную:
        matrix @ diag(diag_vec)

    Args:
        matrix:   двумерный тензор формы (m, n)
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        backend:  интерфейс backend
    """
    if len(matrix.shape) != 2:
        raise ValueError("matrix должна быть двумерной")

    if len(diag_vec.shape) != 1:
        raise ValueError("diag_vec должна быть одномерной")

    m, n = matrix.shape
    rank = diag_vec.shape[0]

    if rank != n:
        raise ValueError("Размерность diag_vec должна совпадать с числом столбцов matrix")

    data = []
    for i in range(m):
        for j in range(rank):
            data.append(matrix.data[i * n + j] * diag_vec.data[j])

    return DenseTensor((m, rank), data=data)