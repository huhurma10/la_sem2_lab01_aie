# core/dense_tensor.py

"""Функции для работы с тензорами в стандартной плотной форме."""


from __future__ import annotations

import random
import math
from typing import List

from core.utils import (
    validate_shape,
    compute_size,
    compute_strides,
    multi_index_to_flat,
    flat_to_multi_index,
check_shapes_match,
)


class DenseTensor:
    """
    Плотный тензор произвольного порядка.

    Атрибуты:
        shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        ndim:    порядок тензора (число мод)
        size:    общее число элементов
        data:    плоский список значений (row-major / C-order)
        strides: шаги для перевода мультииндекса в плоский индекс
    """

    __slots__ = ('shape', 'ndim', 'size', 'data', 'strides')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(
        self,
        shape: tuple[int, ...] | list[int],
        data: list[float] | None = None,
        fill: float = 0.0
    ) -> None:
        """
        Создаёт тензор заданной формы.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            data:  плоский список значений (если None — заполняется fill)
            fill:  значение для заполнения (по умолчанию 0.0)
        """
        shape_validated = validate_shape(shape)
        self.shape = shape_validated
        self.ndim = len(shape_validated)
        self.size = compute_size(shape_validated)
        self.strides = compute_strides(shape_validated)
        if data is None:
            self.data = [fill] * self.size
        else:
            if len(data) != self.size:
                raise ValueError()
            self.data = data
        pass

    @staticmethod
    def zeros(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный нулями.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, data=None, fill=0.0)
        pass

    @staticmethod
    def ones(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный единицами.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        size = compute_size(validate_shape(shape))
        data = [1.0] * size
        return DenseTensor(shape, data=data)
        pass

    @staticmethod
    def random(
        shape: tuple[int, ...] | list[int],
        low: int = -5,
        high: int = 5,
        integer: bool = True,
        seed: int | None = None
    ) -> DenseTensor:
        """
        Возвращает тензор со случайными значениями.

        Args:
            shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            low:     нижняя граница значений тензора
            high:    верхняя граница значений тензора
            integer: True — целые числа, False — вещественные
            seed:    seed для воспроизводимости (None — без фиксации)

        NB: эта функция не тестируется, ее можно использовать для отладки
        """
        if seed is not None:
            random.seed(seed)
        shape_validated = validate_shape(shape)
        size = compute_size(shape_validated)
        if integer:
            data = [random.randint(low, high) for _ in range(size)]
        else:
            data = [random.uniform(low, high) for _ in range(size)]
        return DenseTensor(shape_validated, data=data)
        pass

    @staticmethod
    def from_nested_list(nested: list) -> DenseTensor:
        """
        Создаёт тензор из вложенного списка Python.
        Автоматически определяет shape.

        Args:
            nested: список
        """
        def get_shape(lst: list) -> list:
            shape = []
            while isinstance(lst, list):
                shape.append(len(lst))
                if len(lst) == 0:
                    break
                lst = lst[0]
            return shape

        def flatten(nested_list: list) -> List[float]:
            if not isinstance(nested_list, list):
                return [nested_list]
            result = []
            for item in nested_list:
                result.extend(flatten(item))
            return result

        shape = get_shape(nested)
        data = flatten(nested)
        tensor = DenseTensor(shape, data)
        # Проверка согласованности данных с формой
        if len(data) != compute_size(tensor.shape):
            raise ValueError("Длина данных не совпадает с вычисленной размерностью формы.")
        return tensor
        pass

    # ────────────────────────────────────────────
    # Индексация
    # ────────────────────────────────────────────

    def _validate_index(
        self,
        multi_index: tuple[int, ...] | int
    ) -> tuple[int, ...]:
        """
        Возвращает нормализованный мультииндекс в виде кортежа.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        if isinstance(multi_index, int):
            if multi_index < 0 or multi_index >= self.size:
                raise IndexError("Индекс вне диапазона.")
            return flat_to_multi_index(multi_index, self.shape)
        elif isinstance(multi_index, tuple):
            if len(multi_index) != self.ndim:
                raise IndexError()
            for idx, dim in zip(multi_index, self.shape):
                if not isinstance(idx, int):
                    raise TypeError("Индексы должны быть целыми числами.")
                if idx < 0 or idx >= dim:
                    raise IndexError("Индекс выходит за границы формы.")
            return multi_index
        else:
            raise TypeError("Индекс должен быть кортежем или целым числом.")
        pass

    def __getitem__(self, multi_index: tuple[int, ...] | int) -> float:
        """
        Возвращает значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        indices = self._validate_index(multi_index)
        flat_idx = multi_index_to_flat(indices, self.strides)
        return self.data[flat_idx]
        pass

    def __setitem__(
        self,
        multi_index: tuple[int, ...] | int,
        value: float
    ) -> None:
        """
        Устанавливает новое значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
            value:       новое значение (число)
        """
        indices = self._validate_index(multi_index)
        flat_idx = multi_index_to_flat(indices, self.strides)
        self.data[flat_idx] = value
        pass

    # ────────────────────────────────────────────
    # Преобразования формы
    # ────────────────────────────────────────────

    def reshape(self, new_shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает новый объект тензора с новой формой и скопированными данными.

        Args:
            new_shape: кортеж новых размеров (n'_0, n'_1, ..., n'_{k-1})
        """
        new_shape_validated = validate_shape(new_shape)
        new_size = compute_size(new_shape_validated)
        if new_size != self.size:
            raise ValueError()
        new_tensor = DenseTensor(new_shape_validated)
        new_tensor.data = self.data.copy()
        return new_tensor
        pass

    def unfolding(self, mode: int) -> DenseTensor:
        """
        Возвращает матрицу — развертку тензора по моде n.

        Args:
            mode: номер моды (0 ≤ mode < ndim), которая становится индексом строк
        """
        if mode < 0 or mode >= self.ndim:
            raise IndexError("Недопустимый номер моды")
        other_dims = self.shape[:mode] + self.shape[mode+1:]
        rows = self.shape[mode]
        cols = compute_size(other_dims)
        result_data = []
        for i in range(rows):
            for j in range(cols):
                indices = list(flat_to_multi_index(i, self.shape))
                indices[mode] = flat_to_multi_index(j, other_dims)
                flat_idx = multi_index_to_flat(tuple(indices), self.strides)
                result_data.append(self.data[flat_idx])
        return DenseTensor((rows, cols), data=result_data)
        pass

    def left_unfolding(self, k: int) -> DenseTensor:
        """
        Возвращает матрицу — "левую развертку" тензора для TT-SVD.

        Args:
            k: номер границы разбиения (0 ≤ k < ndim - 1)
        """
        if k < 0 or k >= self.ndim - 1:
            raise IndexError("Недопустимый номер границы.")
        left_dims = self.shape[:k+1]
        right_dims = self.shape[k+1:]
        rows = compute_size(left_dims)
        cols = compute_size(right_dims)
        result_data = []
        for i in range(rows):
            for j in range(cols):
                indices = list(flat_to_multi_index(i, self.shape))
                left_indices = indices[:k+1]
                right_indices = indices[k+1:]
                flat_idx = multi_index_to_flat(tuple(indices), self.strides)
                result_data.append(self.data[flat_idx])
        return DenseTensor((rows, cols), data=result_data)
        pass

    # ────────────────────────────────────────────
    # Копирование
    # ────────────────────────────────────────────

    def copy(self) -> DenseTensor:
        """Возвращает глубокую копию тензора."""
        return DenseTensor(self.shape, data=self.data.copy())
        pass

    # ────────────────────────────────────────────
    # Арифметика
    # ────────────────────────────────────────────

    def norm(self) -> float:
        """Возвращает Фробениусову норму тензора."""
        return math.sqrt(sum(x * x for x in self.data))
        pass


    def __add__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного сложения: t1 + t2.

        Args:
            other: t2
        """
        check_shapes_match(self.shape, other.shape)
        new_data = [a + b for a, b in zip(self.data, other.data)]
        return DenseTensor(self.shape, data=new_data)
        pass

    def __sub__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного вычитания: t1 - t2.

        Args:
            other: t2
        """
        check_shapes_match(self.shape, other.shape)
        new_data = [a - b for a, b in zip(self.data, other.data)]
        return DenseTensor(self.shape, data=new_data)
        pass

    def __mul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: t1 * scalar.

        Args:
            scalar: число
        """
        new_data = [a * scalar for a in self.data]
        return DenseTensor(self.shape, data=new_data)
        pass

    def __rmul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: scalar * t1.

        Args:
            scalar: число, на которое умножаем
        """
        return self.__mul__(scalar)
        pass

    def __neg__(self) -> DenseTensor:
        """Возвращает тензор — результат умножения тензора на -1."""
        new_data = [-a for a in self.data]
        return DenseTensor(self.shape, data=new_data)
        pass

    # ────────────────────────────────────────────
    # Сравнение и отладка
    # ────────────────────────────────────────────

    def allclose(
        self,
        other: DenseTensor,
        atol: float = 1e-8,
        rtol: float = 1e-5
    ) -> bool:
        """
        Возвращает True, если тензоры равны с заданной точностью.

        Условие равенства: shape равны и для каждой пары элементов
        тензоров с равными индексами выполняется:
            |a - b| <= atol + rtol * max(|a|, |b|)


        Args:
            other: DenseTensor для сравнения
            atol:  абсолютная погрешность (по умолчанию 1e-8)
            rtol:  относительная погрешность (по умолчанию 1e-5)
        """
        if self.shape != other.shape:
            return False
        for a, b in zip(self.data, other.data):
            if abs(a - b) > atol + rtol * max(abs(a), abs(b)):
                return False
        return True
        pass

    def to_nested_list(self) -> list:
        """Возвращает тензор в формате вложенного списка."""

        def build_list(data: List[float], shape: List[int]) -> list:
            if len(shape) == 0:
                return data[0]
            size = shape[0]
            chunk_size = compute_size(shape[1:])
            return [build_list(data[i * chunk_size:(i + 1) * chunk_size], shape[1:]) for i in range(size)]

        return build_list(self.data, list(self.shape))
        pass

    def __repr__(self) -> str:
        """
        Возвращает строковое представление тензора для отладки.

        NB: эта функция не проверяется тестами, ее реализация может быть произвольной
        """
        return f"DenseTensor(shape={self.shape}, data_sample={self.data[:10]}{'...' if self.size > 10 else ''})"

    def __str__(self) -> str:
        """Возвращает строковое представление тензора для отладки."""
        return self.__repr__()