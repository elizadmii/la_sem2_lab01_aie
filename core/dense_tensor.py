# core/dense_tensor.py

"""Функции для работы с тензорами в стандартной плотной форме."""

from __future__ import annotations

import random
import math

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
    """

    __slots__ = ("shape", "ndim", "size", "data", "strides")

    def __init__(
        self,
        shape: tuple[int, ...] | list[int],
        data: list[float] | None = None,
        fill: float = 0.0
    ) -> None:
        self.shape = validate_shape(shape)
        self.ndim = len(self.shape)
        self.size = compute_size(self.shape)
        self.strides = compute_strides(self.shape)

        if data is None:
            self.data = [float(fill) for _ in range(self.size)]
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) != self.size:
                raise ValueError("data length does not match tensor shape")

            self.data = [float(x) for x in data]

    @staticmethod
    def zeros(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        return DenseTensor(shape, fill=0.0)

    @staticmethod
    def ones(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        return DenseTensor(shape, fill=1.0)

    @staticmethod
    def random(
        shape: tuple[int, ...] | list[int],
        low: int = -5,
        high: int = 5,
        integer: bool = True,
        seed: int | None = None
    ) -> DenseTensor:
        shape = validate_shape(shape)
        size = compute_size(shape)

        rng = random.Random(seed)
        data = []

        for _ in range(size):
            if integer:
                data.append(float(rng.randint(low, high)))
            else:
                data.append(rng.uniform(low, high))

        return DenseTensor(shape, data)

    @staticmethod
    def from_nested_list(nested: list) -> DenseTensor:
        def get_shape(obj: list) -> tuple[int, ...]:
            if not isinstance(obj, list):
                return ()

            if len(obj) == 0:
                raise ValueError("nested list must not be empty")

            first_shape = get_shape(obj[0])

            for item in obj:
                if get_shape(item) != first_shape:
                    raise ValueError("nested list must be rectangular")

            return (len(obj),) + first_shape

        def flatten(obj: list, result: list[float]) -> None:
            if isinstance(obj, list):
                for item in obj:
                    flatten(item, result)
            else:
                result.append(float(obj))

        if not isinstance(nested, list):
            raise TypeError("nested must be a list")

        shape = get_shape(nested)
        data = []
        flatten(nested, data)

        return DenseTensor(shape, data)

    def _validate_index(
        self,
        multi_index: tuple[int, ...] | int
    ) -> tuple[int, ...]:
        if isinstance(multi_index, int):
            if self.ndim == 1:
                multi_index = (multi_index,)
            else:
                return flat_to_multi_index(multi_index, self.shape)

        if not isinstance(multi_index, tuple):
            raise TypeError("index must be an int or a tuple")

        if len(multi_index) != self.ndim:
            raise ValueError("wrong number of indices")

        for index, dim in zip(multi_index, self.shape):
            if not isinstance(index, int):
                raise TypeError("all indices must be integers")

            if index < 0 or index >= dim:
                raise IndexError("tensor index out of range")

        return multi_index

    def __getitem__(self, multi_index: tuple[int, ...] | int) -> float:
        multi_index = self._validate_index(multi_index)
        flat_index = multi_index_to_flat(multi_index, self.strides)
        return self.data[flat_index]

    def __setitem__(
        self,
        multi_index: tuple[int, ...] | int,
        value: float
    ) -> None:
        multi_index = self._validate_index(multi_index)
        flat_index = multi_index_to_flat(multi_index, self.strides)
        self.data[flat_index] = float(value)

    def reshape(self, new_shape: tuple[int, ...] | list[int]) -> DenseTensor:
        new_shape = validate_shape(new_shape)
        new_size = compute_size(new_shape)

        if new_size != self.size:
            raise ValueError("new shape must have the same size")

        return DenseTensor(new_shape, self.data.copy())

    def unfolding(self, mode: int) -> DenseTensor:
        if not isinstance(mode, int):
            raise TypeError("mode must be an integer")

        if mode < 0 or mode >= self.ndim:
            raise ValueError("mode is out of range")

        row_count = self.shape[mode]
        col_count = self.size // row_count

        result = DenseTensor.zeros((row_count, col_count))

        other_shape = self.shape[:mode] + self.shape[mode + 1:]
        other_strides = compute_strides(other_shape)

        for flat_index in range(self.size):
            multi_index = flat_to_multi_index(flat_index, self.shape)

            row = multi_index[mode]
            other_index = multi_index[:mode] + multi_index[mode + 1:]
            col = multi_index_to_flat(other_index, other_strides)

            result[row, col] = self.data[flat_index]

        return result

    def left_unfolding(self, k: int) -> DenseTensor:
        if not isinstance(k, int):
            raise TypeError("k must be an integer")

        if k < 0 or k >= self.ndim - 1:
            raise ValueError("k is out of range")

        left_shape = self.shape[:k + 1]
        right_shape = self.shape[k + 1:]

        left_size = compute_size(left_shape)
        right_size = compute_size(right_shape)

        return DenseTensor((left_size, right_size), self.data.copy())

    def copy(self) -> DenseTensor:
        return DenseTensor(self.shape, self.data.copy())

    def norm(self) -> float:
        result = 0.0

        for value in self.data:
            result += value * value

        return math.sqrt(result)

    def __add__(self, other: DenseTensor) -> DenseTensor:
        if not isinstance(other, DenseTensor):
            return NotImplemented

        check_shapes_match(self.shape, other.shape)

        data = []

        for a, b in zip(self.data, other.data):
            data.append(a + b)

        return DenseTensor(self.shape, data)

    def __sub__(self, other: DenseTensor) -> DenseTensor:
        if not isinstance(other, DenseTensor):
            return NotImplemented

        check_shapes_match(self.shape, other.shape)

        data = []

        for a, b in zip(self.data, other.data):
            data.append(a - b)

        return DenseTensor(self.shape, data)

    def __mul__(self, scalar: float | int) -> DenseTensor:
        if not isinstance(scalar, (int, float)):
            return NotImplemented

        data = []

        for value in self.data:
            data.append(value * scalar)

        return DenseTensor(self.shape, data)

    def __rmul__(self, scalar: float | int) -> DenseTensor:
        return self.__mul__(scalar)

    def __neg__(self) -> DenseTensor:
        return self * (-1)

    def allclose(
        self,
        other: DenseTensor,
        atol: float = 1e-8,
        rtol: float = 1e-5
    ) -> bool:
        if not isinstance(other, DenseTensor):
            return False

        if self.shape != other.shape:
            return False

        for a, b in zip(self.data, other.data):
            diff = abs(a - b)
            limit = atol + rtol * max(abs(a), abs(b))

            if diff > limit:
                return False

        return True

    def to_nested_list(self) -> list:
        def build(shape: tuple[int, ...], start: int) -> list:
            if len(shape) == 1:
                return self.data[start:start + shape[0]]

            step = compute_size(shape[1:])
            result = []

            for i in range(shape[0]):
                result.append(build(shape[1:], start + i * step))

            return result

        return build(self.shape, 0)

    def __repr__(self) -> str:
        return (
            f"DenseTensor(shape={self.shape}, "
            f"data={self.to_nested_list()})"
        )

    def __str__(self) -> str:
        return self.__repr__() 