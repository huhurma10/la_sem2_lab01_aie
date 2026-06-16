# core/utils.py

"""Вспомогательные функции для работы с тензорами."""


def validate_shape(
        shape: tuple[int, ...] | list[int]
) -> tuple[int, ...]:
    """
    Проверяет корректность формы тензора и приводит её к стандартному виду.

    Убеждается, что shape является последовательностью положительных целых
    чисел. Преобразует список в кортеж для единообразия.

    Args:
        shape: кортеж или список размеров тензора по каждой моде

    Returns:
        tuple: проверенный кортеж положительных целых чисел

    Raises:
        TypeError:  если shape не является tuple или list
        ValueError: если хотя бы один элемент shape не является
                    положительным целым числом
    """
    if not isinstance(shape, (tuple, list)):
        raise TypeError(f"shape должен быть tuple или list, получен {type(shape)}")

    # Преобразование списка в кортеж
    shape_tuple = tuple(shape)

    for dim in shape_tuple:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(
                f"Каждое измерение формы должно быть положительным целым числом, получено {dim}"
            )

    return shape_tuple


def compute_size(shape: tuple[int, ...]) -> int:
    """
    Возвращает общее число элементов тензора заданной формы.

    Args:
        shape: кортеж размеров тензора (n_0, n_1, ..., n_{d-1})
    """
    size = 1
    for dim in shape:
        size *= dim
    return size


def compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Возвращает кортеж strides, содержащий для каждой моды k свой strides[k].

    Stride по моде k — это число элементов в плоском списке, на которое
    нужно сдвинуться, чтобы перейти к следующему элементу вдоль моды k.

    Args:
        shape: кортеж размеров тензора (n_0, n_1, ..., n_{d-1})
    """
    if not shape:
        return ()

    strides = []
    stride = 1
    for size in reversed(shape):
        strides.insert(0, stride)
        stride *= size
    return tuple(strides)


def multi_index_to_flat(
        multi_index: tuple[int, ...],
        strides: tuple[int, ...]
) -> int:
    """
    Возвращает позицию элемента в плоском списке данных по его
    многомерным координатам и заранее вычисленным strides.

    Args:
        multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1})
        strides:     кортеж шагов   (s_0, s_1, ..., s_{d-1})
    """
    if len(multi_index) != len(strides):
        raise ValueError(
            f"Длина multi_index ({len(multi_index)}) и strides ({len(strides)}) должна совпадать."
        )

    flat_index = 0
    for idx, stride in zip(multi_index, strides):
        if not isinstance(idx, int):
            raise TypeError(f"Индексы должны быть целыми числами, получен {type(idx)}")
        flat_index += idx * stride

    return flat_index


def flat_to_multi_index(
        flat_index: int,
        shape: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Возвращает мультииндекс на основе плоского индекса.

    Args:
        flat_index: плоский индекс в списке данных
        shape:      кортеж размеров тензора (n_0, n_1, ..., n_{d-1})
    """
    if not isinstance(flat_index, int):
        raise TypeError(f"Плоский индекс должен быть целым числом, получен {type(flat_index)}")

    total_size = compute_size(shape)
    if flat_index < 0 or flat_index >= total_size:
        raise IndexError(f"Плоский индекс {flat_index} вне диапазона [0, {total_size - 1}]")

    multi_index = []
    for size in reversed(shape):
        multi_index.insert(0, flat_index % size)
        flat_index //= size

    return tuple(multi_index)


def check_shapes_match(
        shape1: tuple[int, ...],
        shape2: tuple[int, ...]
) -> None:
    """
    Проверяет совпадение форм двух тензоров.

    Используется перед поэлементными операциями (сложение, вычитание),
    чтобы гарантировать совместимость тензоров.

    Args:
        shape1: кортеж размеров первого тензора
        shape2: кортеж размеров второго тензора

    Raises:
        ValueError: если формы не совпадают
    """
    if shape1 != shape2:
        raise ValueError(f"Формы тензоров не совпадают: {shape1} != {shape2}")