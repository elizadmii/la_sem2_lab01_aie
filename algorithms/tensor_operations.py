# algorithms/tensor_operations.py

"""
Базовые операции с TT-тензорами.

Все операции работают напрямую с TT-ядрами,
не восстанавливая полный тензор.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


Number = int | float


def _check_same_shape(tt1: TTTensor, tt2: TTTensor) -> None:
    if tt1.shape != tt2.shape:
        raise ValueError("TT-tensors must have the same shape")


def tt_add(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного сложения двух TT-тензоров.
    """
    _ = backend
    _check_same_shape(tt1, tt2)

    if tt1.order != tt2.order:
        raise ValueError("TT-tensors must have the same order")

    d = tt1.order
    new_cores = []

    # Отдельный простой случай: тензор порядка 1
    if d == 1:
        core1 = tt1.cores[0]
        core2 = tt2.cores[0]

        data = []
        for a, b in zip(core1.data, core2.data):
            data.append(a + b)

        return TTTensor([DenseTensor(core1.shape, data)])

    for k in range(d):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]

        r1_left, n, r1_right = core1.shape
        r2_left, _, r2_right = core2.shape

        if k == 0:
            # Первое ядро: горизонтальное склеивание [G1 G2]
            new_core = DenseTensor.zeros((1, n, r1_right + r2_right))

            for i in range(n):
                for b in range(r1_right):
                    new_core[0, i, b] = core1[0, i, b]

                for b in range(r2_right):
                    new_core[0, i, r1_right + b] = core2[0, i, b]

        elif k == d - 1:
            # Последнее ядро: вертикальное склеивание
            new_core = DenseTensor.zeros((r1_left + r2_left, n, 1))

            for a in range(r1_left):
                for i in range(n):
                    new_core[a, i, 0] = core1[a, i, 0]

            for a in range(r2_left):
                for i in range(n):
                    new_core[r1_left + a, i, 0] = core2[a, i, 0]

        else:
            # Средние ядра: блочно-диагональная матрица для каждого i
            new_core = DenseTensor.zeros(
                (r1_left + r2_left, n, r1_right + r2_right)
            )

            for a in range(r1_left):
                for i in range(n):
                    for b in range(r1_right):
                        new_core[a, i, b] = core1[a, i, b]

            for a in range(r2_left):
                for i in range(n):
                    for b in range(r2_right):
                        new_core[r1_left + a, i, r1_right + b] = core2[a, i, b]

        new_cores.append(new_core)

    return TTTensor(new_cores)


def tt_scalar_mul(
    tt: TTTensor,
    alpha: Number,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат умножения TT-тензора на скаляр.
    Модифицируем только первое ядро.
    """
    _ = backend

    if not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a number")

    new_tt = tt.copy()

    first_core = new_tt.cores[0]

    for i in range(first_core.size):
        first_core.data[i] *= alpha

    return new_tt


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного произведения Адамара.
    """
    _ = backend
    _check_same_shape(tt1, tt2)

    if tt1.order != tt2.order:
        raise ValueError("TT-tensors must have the same order")

    new_cores = []

    for k in range(tt1.order):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]

        r1_left, n, r1_right = core1.shape
        r2_left, _, r2_right = core2.shape

        new_shape = (
            r1_left * r2_left,
            n,
            r1_right * r2_right
        )

        new_core = DenseTensor.zeros(new_shape)

        # Для каждого i_k берём кронекерово произведение срезов
        for i in range(n):
            for a1 in range(r1_left):
                for a2 in range(r2_left):
                    new_left = a1 * r2_left + a2

                    for b1 in range(r1_right):
                        for b2 in range(r2_right):
                            new_right = b1 * r2_right + b2

                            new_core[new_left, i, new_right] = (
                                core1[a1, i, b1] * core2[a2, i, b2]
                            )

        new_cores.append(new_core)

    return TTTensor(new_cores)


def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    """
    Возвращает скалярное произведение двух TT-тензоров: <tt1, tt2>.
    """
    _ = backend
    _check_same_shape(tt1, tt2)

    if tt1.order != tt2.order:
        raise ValueError("TT-tensors must have the same order")

    # Z в начале имеет размер 1 x 1
    z = [[1.0]]

    for k in range(tt1.order):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]

        r1_left, n, r1_right = core1.shape
        r2_left, _, r2_right = core2.shape

        new_z = []

        for a_right in range(r1_right):
            row = []

            for b_right in range(r2_right):
                value = 0.0

                for i in range(n):
                    for a_left in range(r1_left):
                        for b_left in range(r2_left):
                            value += (
                                core1[a_left, i, a_right]
                                * z[a_left][b_left]
                                * core2[b_left, i, b_right]
                            )

                row.append(value)

            new_z.append(row)

        z = new_z

    # В конце из-за граничных рангов получаем матрицу 1 x 1
    return z[0][0]


def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает Фробениусову норму TT-тензора.
    """
    dot = tt_dot(tt, tt, backend)

    # max нужен из-за возможной маленькой численной ошибки типа -1e-15
    return math.sqrt(max(0.0, dot))


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает норму разности: ||tt1 - tt2||_F.
    """
    _check_same_shape(tt1, tt2)

    dot11 = tt_dot(tt1, tt1, backend)
    dot22 = tt_dot(tt2, tt2, backend)
    dot12 = tt_dot(tt1, tt2, backend)

    value = dot11 + dot22 - 2.0 * dot12

    return math.sqrt(max(0.0, value)) 