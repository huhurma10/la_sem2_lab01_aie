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
import itertools


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
        ranks = [cores[0].shape[0]]  # r_0
        for i, core in enumerate(cores):
            if len(core.shape) != 3:
                raise ValueError(f"Ядро {i} должно иметь форму (r_{i}, n_{i}, r_{i + 1})")
            r_prev, n, r_next = core.shape
            if r_prev != ranks[-1]:
                raise ValueError(f"Несоответствие рангов в ядрах: r_{i} != r_{i - 1}")
            ranks.append(r_next)
        self.ranks = tuple(ranks)
        self.shape = tuple(core.shape[1] for core in cores)
        pass


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
            import random
            random.seed(seed)
        shape_validated = validate_shape(shape)
        d = len(shape_validated)
        if isinstance(ranks, (list, tuple)):
            ranks_list = list(ranks)
        else:
            raise TypeError("ranks должен быть списком или кортежем.")
        if ranks_list[0] != 1 or ranks_list[-1] != 1:
            raise ValueError("Ранги r_0 и r_d должны быть равны 1")
        if len(ranks_list) != d + 1:
            raise ValueError()
        cores = []
        for i in range(d):
            r_prev = ranks_list[i]
            r_next = ranks_list[i + 1]
            n_i = shape_validated[i]
            # ядро с случайными значениями
            data = [random.uniform(-1, 1) for _ in range(r_prev * n_i * r_next)]
            core = DenseTensor((r_prev, n_i, r_next), data=data)
            cores.append(core)
        return TTTensor(cores)
        pass

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
            raise IndexError("длина индексов должна совпадать с порядком тензора.")
        current = [1.0]
        for k in range(self.order):
            i = indices[k]
            next_current = []
            core = self.cores[k]
            for r_prev_idx, r_prev in enumerate(range(self.ranks[k])):
                for r_next_idx, r_next in enumerate(range(self.ranks[k+1])):
                    val = core.data[r_prev * core.shape[1] * core.shape[2] + i * core.shape[2] + r_next]
                    for val_in in current[r_prev_idx]:
                        next_current.append(val_in * val)
            current = next_current
        return current[0] if len(current) == 1 else sum(current)
        pass

    # ────────────────────────────────────────────
    # Восстановление полного тензора
    # ────────────────────────────────────────────

    def full(self) -> DenseTensor:
        """Возвращает полный DenseTensor из его TT-формата."""
        result_data = [1.0]
        result_shape = []

        for k in range(self.order):
            core = self.cores[k]
            r_prev, n_k, r_next = core.shape
            new_data = []
            for prev_idx, i_k in itertools.product(range(len(result_data) // r_prev), range(n_k)):
                for r_idx in range(r_prev):
                    val = 0.0
                    for r_next_idx in range(r_next):
                        # индекс ядра
                        core_idx = r_idx * n_k * r_next + i_k * r_next + r_next_idx
                        val += result_data[prev_idx * r_prev + r_idx] * core.data[core_idx]
                    new_data.append(val)
            result_data = new_data
            result_shape.append(n_k)

        return DenseTensor(tuple(result_shape), data=result_data)
        pass

    # ────────────────────────────────────────────
    # Информация и отладка
    # ────────────────────────────────────────────

    def core_sizes(self):
        """Возвращает размеры всех ядер."""
        return [core.shape for core in self.cores]
        pass

    def total_storage(self) -> int:
        """
        Возвращает общее число элементов во всех ядрах.
        Это то, сколько памяти реально занимает TT-тензор.
        """
        return sum(core.size for core in self.cores)
        pass

    def compression_ratio(self) -> float:
        """
        Возвращает отношение числа элементов полного тензора к числу
        элементов TT-тензора. Показывает, насколько TT-формат компактнее.
        """
        full_size = compute_size(self.shape)
        return full_size / self.total_storage()
        pass

    def copy(self) -> TTTensor:
        """Возвращает глубокую копию TT-тензора."""
        cores_copy = [core.copy() for core in self.cores]
        return TTTensor(cores_copy)
        pass

    def __repr__(self) -> str:
        """
        Возвращает строковое представление TT-тензора для отладки.

        Формирует многострочную строку с основной служебной информацией
        об объекте:
            - порядок тензора (order),
            - исходная форма (shape),
            - TT-ранги (ranks),
            - размеры TT-ядер (cores),
            - суммарный объём хранения в элементах.

        NB: это отладочная функция, которая не покрывается тестами
        """
        return (
            f"TTTensor(order={self.order}, shape={self.shape}, "
            f"ranks={self.ranks}, total_storage={self.total_storage()})"
        )
        pass

    def __str__(self) -> str:
        """
        Возвращает строковое представление TT-тензора.

        Делегирует работу методу __repr__, обеспечивая единый формат
        отображения при вызове.
        """
        return self.__repr__()