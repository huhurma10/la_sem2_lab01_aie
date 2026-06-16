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

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [core.copy() for core in tt.cores]
    d = tt.order

    for k in range(d - 1):
        core = cores[k]
        mat = DenseTensor((core.shape[0], core.shape[1] * core.shape[2]),
                          data=core.data)
        Q, R = backend.qr(mat)
        new_core = DenseTensor((core.shape[0], core.shape[1], Q.shape[1]), data=Q.data)
        cores[k] = new_core
        next_core = cores[k + 1]
        R_dense = R
        new_data = []
        for r in range(R_dense.shape[0]):
            for j in range(next_core.shape[1]):
                val = 0.0
                for s in range(R_dense.shape[1]):
                    val += R_dense.data[r * R_dense.shape[1] + s] * next_core.data[
                        s * next_core.shape[1] * next_core.shape[2] + j * next_core.shape[2]]
                new_data.append(val)
        cores[k + 1] = DenseTensor((R_dense.shape[0], next_core.shape[1], next_core.shape[2]), data=new_data)

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
        mat = DenseTensor((core.shape[0], core.shape[1] * core.shape[2]),
                          data=core.data)
        Q, R = backend.qr(mat.transpose())
        Q = Q.transpose()
        R = R.transpose()
        # обновляем ядро
        new_core = DenseTensor((core.shape[0], core.shape[1], Q.shape[1]), data=Q.data)
        cores[k] = new_core
        prev_core = cores[k - 1]
        R_dense = R
        new_data = []
        for r in range(prev_core.shape[0]):
            for j in range(R_dense.shape[1]):
                val = 0.0
                for s in range(prev_core.shape[2]):
                    val += prev_core.data[r * prev_core.shape[1] * prev_core.shape[2] + j * prev_core.shape[2] + s] * \
                           R_dense.data[s * R_dense.shape[1] + j]
                new_data.append(val)
        cores[k - 1] = DenseTensor((prev_core.shape[0], prev_core.shape[1], R_dense.shape[1]), data=new_data)

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