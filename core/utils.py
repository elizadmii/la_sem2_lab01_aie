"""Вспомогательные функции для работы с тензорами."""


def validate_shape(
    shape: tuple[int, ...] | list[int]
) -> tuple[int, ...]:
    """
    Проверяет корректность формы тензора и приводит её к стандартному виду.
    """
    if not isinstance(shape, (tuple, list)):
        raise TypeError("shape must be a tuple or a list")

    if len(shape) == 0:
        raise ValueError("shape must not be empty")

    result = []

    for dim in shape:
        if not isinstance(dim, int):
            raise ValueError("all shape elements must be integers")
        if dim <= 0:
            raise ValueError("all shape elements must be positive")

        result.append(dim)

    return tuple(result)


def compute_size(shape: tuple[int, ...]) -> int:
    """
    Возвращает общее число элементов тензора заданной формы.
    """
    size = 1

    for dim in shape:
        size *= dim

    return size


def compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Возвращает strides для C-order.
    Последний индекс меняется быстрее всего.
    """
    strides = []
    current_stride = 1

    for dim in reversed(shape):
        strides.append(current_stride)
        current_stride *= dim

    strides.reverse()

    return tuple(strides)


def multi_index_to_flat(
    multi_index: tuple[int, ...],
    strides: tuple[int, ...]
) -> int:
    """
    Возвращает позицию элемента в плоском списке данных.
    """
    if len(multi_index) != len(strides):
        raise ValueError("multi_index and strides must have the same length")

    flat_index = 0

    for index, stride in zip(multi_index, strides):
        if not isinstance(index, int):
            raise TypeError("all indices must be integers")
        if index < 0:
            raise ValueError("indices must be non-negative")

        flat_index += index * stride

    return flat_index


def flat_to_multi_index(
    flat_index: int,
    shape: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Возвращает мультииндекс на основе плоского индекса.
    """
    if not isinstance(flat_index, int):
        raise TypeError("flat_index must be an integer")

    size = compute_size(shape)

    if flat_index < 0 or flat_index >= size:
        raise ValueError("flat_index is out of tensor bounds")

    strides = compute_strides(shape)
    multi_index = []

    rest = flat_index

    for stride, dim in zip(strides, shape):
        index = rest // stride
        rest = rest % stride

        if index >= dim:
            raise ValueError("flat_index is out of tensor bounds")

        multi_index.append(index)

    return tuple(multi_index)


def check_shapes_match(
    shape1: tuple[int, ...],
    shape2: tuple[int, ...]
) -> None:
    """
    Проверяет совпадение форм двух тензоров.
    """
    if shape1 != shape2:
        raise ValueError(f"shapes do not match: {shape1} != {shape2}") 