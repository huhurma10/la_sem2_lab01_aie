# core/tt_tensor.py

"""
Тензор в TT-формате (Tensor Train).

TT-тензор порядка d с shape (n_0, n_1, ..., n_{d-1}) хранится как
список d ядер (cores), где k-е ядро — это 3D DenseTensor с shape:
    (r_k, n_k, r_{k+1})

Граничные условия: r_0 = r_d = 1.

TT-ранги: (r_0, r_1, ..., r_d) = (1, r_1, ..., r_{d-1}, 1).
"""

from __future__ import annotations

from core.dense_tensor import DenseTensor
from core.utils import validate_shape, compute_size
import random


class TTTensor:
    """
    Тензор в TT-формате.

    Атрибуты:
        cores:  список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        order:  порядок тензора d (число мод)
        shape:  кортеж (n_0, n_1, ..., n_{d-1})
        ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d), r_0 = r_d = 1
    """

    __slots__ = ('cores', 'order', 'shape', 'ranks')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(self, cores: list[DenseTensor]) -> None:
        """
        Создаёт TT-тензор из списка ядер.

        Args:
            cores: список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        """
        if not isinstance(cores, list) or len(cores) == 0:
            raise ValueError("cores должен быть непустым списком DenseTensor.")

        self.cores = cores
        self.order = len(cores)

        # Проверяем, что все ядра трехмерные
        for i, core in enumerate(cores):
            if len(core.shape) != 3:
                raise ValueError(
                    f"Ядро {i} должно иметь форму (r_{i}, n_{i}, r_{i + 1}), "
                    f"получено {core.shape}"
                )

        # Проверяем согласованность рангов
        ranks = [cores[0].shape[0]]  # r_0
        for i, core in enumerate(cores):
            r_prev, n, r_next = core.shape
            if r_prev != ranks[-1]:
                raise ValueError(
                    f"Несоответствие рангов: r_{i} = {r_prev}, "
                    f"ожидается {ranks[-1]}"
                )
            ranks.append(r_next)

        # Проверяем, что r_0 = r_d = 1
        if ranks[0] != 1:
            raise ValueError(f"r_0 должен быть равен 1, получено {ranks[0]}")
        if ranks[-1] != 1:
            raise ValueError(f"r_d должен быть равен 1, получено {ranks[-1]}")

        self.ranks = tuple(ranks)
        self.shape = tuple(core.shape[1] for core in cores)

    @staticmethod
    def random(shape, ranks, seed=None):
        """
        Создаёт случайный TT-тензор с заданными рангами.

        Args:
            shape:  кортеж размеров мод (n_0, ..., n_{d-1})
            ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d)
                    или список внутренних рангов (r_1, ..., r_{d-1})
            seed:   seed для воспроизводимости

        NB: это отладочная функция, она не проверяется тестами
        """
        if seed is not None:
            random.seed(seed)

        shape_validated = validate_shape(shape)
        d = len(shape_validated)

        if not isinstance(ranks, (list, tuple)):
            raise TypeError("ranks должен быть списком или кортежем.")

        ranks_list = list(ranks)

        # Если передан список внутренних рангов, дополняем граничными
        if len(ranks_list) == d - 1:
            ranks_list = [1] + ranks_list + [1]
        elif len(ranks_list) == d + 1:
            if ranks_list[0] != 1 or ranks_list[-1] != 1:
                raise ValueError("Ранги r_0 и r_d должны быть равны 1")
        else:
            raise ValueError(
                f"ranks должен иметь длину {d - 1} (внутренние ранги) "
                f"или {d + 1} (полные ранги), получено {len(ranks_list)}"
            )

        cores = []
        for i in range(d):
            r_prev = ranks_list[i]
            r_next = ranks_list[i + 1]
            n_i = shape_validated[i]

            # Создаем ядро со случайными значениями в диапазоне [-1, 1]
            data = [random.uniform(-1, 1) for _ in range(r_prev * n_i * r_next)]
            core = DenseTensor((r_prev, n_i, r_next), data=data)
            cores.append(core)

        return TTTensor(cores)

    # ────────────────────────────────────────────
    # Доступ к элементам
    # ────────────────────────────────────────────

    def get_element(
            self,
            indices: tuple[int, ...] | list[int]
    ) -> float:
        """
        Возвращает элемент TT-тензора по его мультииндексу.

        Args:
            indices: кортеж/список длины d
        """
        if len(indices) != self.order:
            raise IndexError(
                f"Длина индексов {len(indices)} должна совпадать с порядком тензора {self.order}"
            )

        # Начинаем с вектора размера r_0 = 1
        result = [1.0]

        for k in range(self.order):
            i_k = indices[k]
            core = self.cores[k]
            r_prev, n_k, r_next = core.shape

            # Умножаем текущий вектор на ядро
            new_result = [0.0] * r_next

            for alpha in range(r_prev):
                for beta in range(r_next):
                    # Индекс в плоском массиве ядра
                    idx = alpha * n_k * r_next + i_k * r_next + beta
                    new_result[beta] += result[alpha] * core.data[idx]

            result = new_result

        # В конце должен быть вектор размера r_d = 1
        return result[0]

    # ────────────────────────────────────────────
    # Восстановление полного тензора
    # ────────────────────────────────────────────

    def full(self) -> DenseTensor:
        """Возвращает полный DenseTensor из его TT-формата."""
        if self.order == 0:
            return DenseTensor((), data=[1.0])

        # Начинаем с первого ядра
        core = self.cores[0]
        r_0, n_0, r_1 = core.shape

        # r_0 должно быть 1
        if r_0 != 1:
            raise ValueError(f"r_0 должен быть 1, получено {r_0}")

        # Результат после первого ядра: матрица (n_0, r_1)
        result = []
        for i in range(n_0):
            for j in range(r_1):
                idx = i * r_1 + j
                result.append(core.data[idx])

        # Последовательно умножаем на остальные ядра
        for k in range(1, self.order):
            core = self.cores[k]
            r_prev, n_k, r_next = core.shape

            # Текущий результат: матрица (prev_shape, r_prev)
            prev_size = len(result) // r_prev
            new_result = []

            for prev_idx in range(prev_size):
                for i_k in range(n_k):
                    for r_next_idx in range(r_next):
                        val = 0.0
                        for r_prev_idx in range(r_prev):
                            prev_val = result[prev_idx * r_prev + r_prev_idx]
                            core_idx = r_prev_idx * n_k * r_next + i_k * r_next + r_next_idx
                            core_val = core.data[core_idx]
                            val += prev_val * core_val
                        new_result.append(val)

            result = new_result

        # В конце r_d должно быть 1
        if self.ranks[-1] != 1:
            raise ValueError(f"r_d должен быть 1, получено {self.ranks[-1]}")

        # Формируем финальную форму
        return DenseTensor(self.shape, data=result)

    # ────────────────────────────────────────────
    # Информация и отладка
    # ────────────────────────────────────────────

    def core_sizes(self):
        """Возвращает размеры всех ядер."""
        return [core.shape for core in self.cores]

    def total_storage(self) -> int:
        """
        Возвращает общее число элементов во всех ядрах.
        Это то, сколько памяти реально занимает TT-тензор.
        """
        return sum(core.size for core in self.cores)

    def compression_ratio(self) -> float:
        """
        Возвращает отношение числа элементов полного тензора к числу
        элементов TT-тензора. Показывает, насколько TT-формат компактнее.
        """
        full_size = compute_size(self.shape)
        total = self.total_storage()
        if total == 0:
            return float('inf')
        return full_size / total

    def copy(self) -> TTTensor:
        """Возвращает глубокую копию TT-тензора."""
        cores_copy = [core.copy() for core in self.cores]
        return TTTensor(cores_copy)

    def __repr__(self) -> str:
        """
        Возвращает строковое представление TT-тензора для отладки.
        """
        return (
            f"TTTensor(order={self.order}, shape={self.shape}, "
            f"ranks={self.ranks}, total_storage={self.total_storage()})"
        )

    def __str__(self) -> str:
        """Возвращает строковое представление TT-тензора."""
        return self.__repr__()