# core/tt_tensor.py

"""
Тензор в TT-формате (Tensor Train).
"""

from __future__ import annotations

import random

from core.dense_tensor import DenseTensor
from core.utils import validate_shape, compute_size, flat_to_multi_index


class TTTensor:
    """
    Тензор в TT-формате.
    """

    __slots__ = ("cores", "order", "shape", "ranks")

    def __init__(self, cores: list[DenseTensor]) -> None:
        """
        Создаёт TT-тензор из списка ядер.
        Каждое ядро имеет форму (r_left, n, r_right).
        """
        if not isinstance(cores, list):
            raise TypeError("cores must be a list")

        if len(cores) == 0:
            raise ValueError("TT-tensor must have at least one core")

        for core in cores:
            if not isinstance(core, DenseTensor):
                raise TypeError("all cores must be DenseTensor objects")

            if core.ndim != 3:
                raise ValueError("each TT-core must be a 3D tensor")

        if cores[0].shape[0] != 1:
            raise ValueError("first TT-rank must be equal to 1")

        if cores[-1].shape[2] != 1:
            raise ValueError("last TT-rank must be equal to 1")

        for i in range(len(cores) - 1):
            right_rank = cores[i].shape[2]
            next_left_rank = cores[i + 1].shape[0]

            if right_rank != next_left_rank:
                raise ValueError("neighbour TT-ranks do not match")

        self.cores = cores
        self.order = len(cores)

        shape = []
        for core in cores:
            shape.append(core.shape[1])

        self.shape = tuple(shape)

        ranks = [cores[0].shape[0]]
        for core in cores:
            ranks.append(core.shape[2])

        self.ranks = tuple(ranks)

    @staticmethod
    def random(shape, ranks, seed=None):
        """
        Создаёт случайный TT-тензор с заданными рангами.
        Это функция для отладки.
        """
        shape = validate_shape(shape)
        order = len(shape)

        if not isinstance(ranks, (tuple, list)):
            raise TypeError("ranks must be a tuple or a list")

        ranks = tuple(ranks)

        # Можно передать либо полные ранги (1, r1, ..., 1),
        # либо только внутренние (r1, ..., r_{d-1})
        if len(ranks) == order + 1:
            full_ranks = ranks
        elif len(ranks) == order - 1:
            full_ranks = (1,) + ranks + (1,)
        else:
            raise ValueError("wrong number of TT-ranks")

        if full_ranks[0] != 1 or full_ranks[-1] != 1:
            raise ValueError("boundary TT-ranks must be equal to 1")

        for rank in full_ranks:
            if not isinstance(rank, int) or rank <= 0:
                raise ValueError("all TT-ranks must be positive integers")

        rng = random.Random(seed)
        cores = []

        for k in range(order):
            core_shape = (full_ranks[k], shape[k], full_ranks[k + 1])
            size = compute_size(core_shape)

            data = []
            for _ in range(size):
                data.append(rng.uniform(-1.0, 1.0))

            cores.append(DenseTensor(core_shape, data))

        return TTTensor(cores)

    def get_element(
        self,
        indices: tuple[int, ...] | list[int]
    ) -> float:
        """
        Возвращает элемент TT-тензора по мультииндексу.
        """
        if not isinstance(indices, (tuple, list)):
            raise TypeError("indices must be a tuple or a list")

        if len(indices) != self.order:
            raise ValueError("wrong number of indices")

        indices = tuple(indices)

        for index, dim in zip(indices, self.shape):
            if not isinstance(index, int):
                raise TypeError("all indices must be integers")

            if index < 0 or index >= dim:
                raise IndexError("tensor index out of range")

        # Начинаем с вектора длины 1, потому что r_0 = 1
        current = [1.0]

        for k in range(self.order):
            core = self.cores[k]
            index = indices[k]

            left_rank = core.shape[0]
            right_rank = core.shape[2]

            new_current = [0.0 for _ in range(right_rank)]

            # Умножаем текущий вектор на срез ядра G_k[:, index, :]
            for a in range(left_rank):
                for b in range(right_rank):
                    new_current[b] += current[a] * core[a, index, b]

            current = new_current

        # В конце должен остаться один элемент, потому что r_d = 1
        return current[0]

    def full(self) -> DenseTensor:
        """
        Восстанавливает полный плотный тензор из TT-формата.
        """
        size = compute_size(self.shape)
        data = []

        for flat_index in range(size):
            multi_index = flat_to_multi_index(flat_index, self.shape)
            data.append(self.get_element(multi_index))

        return DenseTensor(self.shape, data)

    def core_sizes(self) -> list[tuple[int, ...]]:
        """
        Возвращает размеры всех ядер.
        """
        result = []

        for core in self.cores:
            result.append(core.shape)

        return result

    def total_storage(self) -> int:
        """
        Возвращает общее число элементов во всех TT-ядрах.
        """
        total = 0

        for core in self.cores:
            total += core.size

        return total

    def compression_ratio(self) -> float:
        """
        Во сколько раз полный тензор больше TT-представления.
        """
        full_size = compute_size(self.shape)
        tt_size = self.total_storage()

        return full_size / tt_size

    def copy(self) -> TTTensor:
        """
        Возвращает глубокую копию TT-тензора.
        """
        new_cores = []

        for core in self.cores:
            new_cores.append(core.copy())

        return TTTensor(new_cores)

    def __repr__(self) -> str:
        return (
            "TTTensor(\n"
            f"  order={self.order},\n"
            f"  shape={self.shape},\n"
            f"  ranks={self.ranks},\n"
            f"  core_sizes={self.core_sizes()},\n"
            f"  total_storage={self.total_storage()}\n"
            ")"
        )

    def __str__(self) -> str:
        return self.__repr__() 