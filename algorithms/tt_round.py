# algorithms/tt_round.py

"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает новый TT-тензор с уменьшенными рангами.
    """
    if not isinstance(tt, TTTensor):
        raise TypeError("tt must be TTTensor")

    if max_rank is not None and max_rank < 1:
        raise ValueError("max_rank must be positive")

    if eps < 0:
        raise ValueError("eps must be non-negative")

    if tt.order == 1:
        return tt.copy()

    # 1. Сначала приводим к правой канонической форме
    rounded_tt = right_canonicalize(tt, backend)
    cores = rounded_tt.cores
    d = rounded_tt.order

    first_norm = backend.norm(cores[0])

    if first_norm > 1e-30:
        delta = eps * first_norm / math.sqrt(d - 1)
    else:
        delta = 0.0

    # 2. Идём слева направо и делаем SVD-усечение
    for k in range(d - 1):
        core = cores[k]
        r_left, n, r_right = core.shape

        matrix = backend.reshape(core, (r_left * n, r_right))

        U, S, Vt = backend.svd(matrix, full_matrices=False)

        rank = _compute_rank(S, delta, max_rank)

        U_trunc = _truncate_columns(U, rank, backend)
        S_trunc = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)

        cores[k] = backend.reshape(U_trunc, (r_left, n, rank))

        # Остаток diag(S) @ Vt поглощаем в следующее ядро
        rest = _multiply_diag_matrix(S_trunc, Vt_trunc, rank, backend)

        next_core = cores[k + 1]
        old_rank, next_n, next_r = next_core.shape

        new_next_core = DenseTensor.zeros((rank, next_n, next_r))

        for i in range(next_n):
            for a in range(rank):
                for b in range(next_r):
                    value = 0.0

                    for c in range(old_rank):
                        value += rest[a, c] * next_core[c, i, b]

                    new_next_core[a, i, b] = value

        cores[k + 1] = new_next_core

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает ранг усечения по сингулярным значениям.
    """
    if S.ndim != 1:
        raise ValueError("S must be a vector")

    if S.size == 0:
        return 1

    max_sigma = max(abs(x) for x in S.data)
    threshold = max(1e-12, 1e-8 * max_sigma)

    numerical_rank = 0

    for value in S.data:
        if abs(value) > threshold:
            numerical_rank += 1

    numerical_rank = max(1, numerical_rank)

    rank = numerical_rank

    if delta > 0:
        limit = delta * delta

        for candidate in range(1, numerical_rank + 1):
            tail_sum = 0.0

            for i in range(candidate, numerical_rank):
                tail_sum += S[i] * S[i]

            if tail_sum <= limit:
                rank = candidate
                break

    if max_rank is not None:
        rank = min(rank, max_rank)

    rank = max(1, min(rank, S.size))

    return rank


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу из первых rank столбцов.
    """
    _ = backend

    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")

    rows, cols = matrix.shape

    if rank < 0 or rank > cols:
        raise ValueError("wrong rank")

    result = DenseTensor.zeros((rows, rank))

    for i in range(rows):
        for j in range(rank):
            result[i, j] = matrix[i, j]

    return result


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу из первых rank строк.
    """
    _ = backend

    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")

    rows, cols = matrix.shape

    if rank < 0 or rank > rows:
        raise ValueError("wrong rank")

    result = DenseTensor.zeros((rank, cols))

    for i in range(rank):
        for j in range(cols):
            result[i, j] = matrix[i, j]

    return result


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает вектор из первых rank элементов.
    """
    _ = backend

    if vector.ndim != 1:
        raise ValueError("vector must be 1D")

    if rank < 0 or rank > vector.size:
        raise ValueError("wrong rank")

    result = DenseTensor.zeros((rank,))

    for i in range(rank):
        result[i] = vector[i]

    return result


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает diag(diag_vec) @ matrix.
    """
    _ = backend

    if diag_vec.ndim != 1:
        raise ValueError("diag_vec must be a vector")

    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")

    rows, cols = matrix.shape

    if rank > diag_vec.size or rank > rows:
        raise ValueError("wrong rank")

    result = DenseTensor.zeros((rank, cols))

    for i in range(rank):
        for j in range(cols):
            result[i, j] = diag_vec[i] * matrix[i, j]

    return result