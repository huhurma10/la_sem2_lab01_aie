# algorithms/canonical_form.py

"""
Приведение TT-тензора в канонические формы (полная правая и
левая ортогонализация ядер).
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def transpose_matrix(mat: DenseTensor) -> DenseTensor:
    """
    Транспонирует двумерную матрицу.
    """
    if len(mat.shape) != 2:
        raise ValueError()

    m, n = mat.shape
    data = []
    for j in range(n):
        for i in range(m):
            data.append(mat.data[i * n + j])

    return DenseTensor((n, m), data=data)

def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в лево-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [core.copy() for core in tt.cores]
    d = tt.order

    for k in range(d - 1):
        core = cores[k]
        r_prev, n, r_next = core.shape
        mat_data = []
        for i in range(r_prev * n):
            r = i // n
            ni = i % n
            for j in range(r_next):
                mat_data.append(core.data[r * n * r_next + ni * r_next + j])

        mat = DenseTensor((core.shape[0], core.shape[1] * core.shape[2]),
                          data=core.data)
        if mat.shape[0] < mat.shape[1]:
            mat_T = transpose_matrix(mat)
            Q, R = backend.qr(mat_T)
            R_T = transpose_matrix(R)
            rank = min(R_T.shape[1], R_T.shape[0])
            Q_new_data = []
            for i in range(R_T.shape[0]):
                for j in range(rank):
                    Q_new_data.append(R_T.data[i * R_T.shape[1] + j])
            Q_new = DenseTensor((R_T.shape[0], rank), data=Q_new_data)
            Q = Q_new
        else:
            Q, R = backend.qr(mat)
        rank = Q.shape[1]
        new_core_data = []
        for r in range(r_prev):
            for ni in range(n):
                for s in range(rank):
                    idx = r * n * rank + ni * rank + s
                    new_core_data.append(Q.data[idx])

        new_core = DenseTensor((r_prev, n, rank), data=new_core_data)
        cores[k] = new_core

        next_core = cores[k + 1]
        r_next, n_next, r_next2 = next_core.shape

        new_next_data = []
        for r in range(rank):
            for ni in range(n_next):
                for r2 in range(r_next2):
                    val = 0.0
                    for s in range(r_next):
                        val += R.data[r * r_next + s] * next_core.data[s * n_next * r_next2 + ni * r_next2 + r2]
                    new_next_data.append(val)

        cores[k + 1] = DenseTensor((rank, n_next, r_next2), data=new_next_data)

    return TTTensor(cores)

    pass


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [core.copy() for core in tt.cores]
    d = tt.order

    for k in reversed(range(1, d)):
        core = cores[k]
        r_prev, n, r_next = core.shape

        mat_data = []
        for i in range(r_prev * n):
            r = i // n
            ni = i % n
            for j in range(r_next):
                mat_data.append(core.data[r * n * r_next + ni * r_next + j])

        mat = DenseTensor((r_prev * n, r_next), data=mat_data)

        mat_T = transpose_matrix(mat)
        Q, R = backend.qr(mat_T)

        rank = min(Q.shape[0], Q.shape[1])

        Q_T = transpose_matrix(Q)
        Q_new_data = []
        for i in range(Q_T.shape[0]):
            for j in range(rank):
                Q_new_data.append(Q_T.data[i * Q_T.shape[1] + j])
        Q_new = DenseTensor((Q_T.shape[0], rank), data=Q_new_data)

        new_core_data = []
        for r in range(r_prev):
            for ni in range(n):
                for s in range(rank):
                    idx = r * n * rank + ni * rank + s
                    new_core_data.append(Q_new.data[idx])

        new_core = DenseTensor((r_prev, n, rank), data=new_core_data)
        cores[k] = new_core

        prev_core = cores[k - 1]
        r_prev_prev, n_prev, r_prev_next = prev_core.shape

        new_rank = rank
        new_prev_data = []
        for rp in range(r_prev_prev):
            for ni in range(n_prev):
                for rn in range(new_rank):
                    val = 0.0
                    for s in range(r_prev_next):
                        val += prev_core.data[rp * n_prev * r_prev_next + ni * r_prev_next + s] * R.data[
                            s * new_rank + rn]
                    new_prev_data.append(val)

        cores[k - 1] = DenseTensor((r_prev_prev, n_prev, new_rank), data=new_prev_data)

    return TTTensor(cores)
    pass


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
    if S.shape[0] == 0:
        return 0

    sigma_max = float(S[0])
    threshold = max(abs_tol, rel_tol * sigma_max)

    rank = 0
    for i in range(S.shape[0]):
        if abs(float(S[i])) > threshold:
            rank += 1
        else:
            break
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
    if rank >= matrix.shape[1]:
        return matrix.copy()

    result_shape = (matrix.shape[0], rank)
    result = DenseTensor(result_shape)

    for i in range(matrix.shape[0]):
        for j in range(rank):
            result[i, j] = matrix[i, j]

    return result
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
    if rank >= matrix.shape[0]:
        return matrix.copy()

    result_shape = (rank, matrix.shape[1])
    result = DenseTensor(result_shape)

    for i in range(rank):
        for j in range(matrix.shape[1]):
            result[i, j] = matrix[i, j]

    return result
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
    if rank >= vector.shape[0]:
        return vector.copy()

    result = DenseTensor((rank,))
    for i in range(rank):
        result[i] = vector[i]

    return result
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
        rank:     длина диагонального вектора
        backend:  интерфейс backend
    """
    result_shape = (rank, matrix.shape[1])
    result = DenseTensor(result_shape)

    for i in range(rank):
        diag_val = float(diag_vec[i])
        for j in range(matrix.shape[1]):
            result[i, j] = diag_val * matrix[i, j]

    return result
    pass


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
    rank = diag_vec.shape[0]
    result_shape = (matrix.shape[0], rank)
    result = DenseTensor(result_shape)

    for i in range(matrix.shape[0]):
        for j in range(rank):
            result[i, j] = matrix[i, j] * float(diag_vec[j])

    return result
    pass