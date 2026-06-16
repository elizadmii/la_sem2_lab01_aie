# algorithms/canonical_form.py

"""
Приведение TT-тензора в канонические формы.
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает новый TT-тензор в лево-канонической форме.
    """
    new_tt = tt.copy()
    cores = new_tt.cores

    if new_tt.order == 1:
        return new_tt

    for k in range(new_tt.order - 1):
        core = cores[k]
        r_left, n, r_right = core.shape

        # Разворачиваем ядро в матрицу (r_left * n) x r_right
        matrix = backend.reshape(core, (r_left * n, r_right))

        # QR: matrix = Q @ R
        Q, R = backend.qr(matrix)

        new_rank = Q.shape[1]

        # Q обратно превращаем в TT-ядро
        cores[k] = backend.reshape(Q, (r_left, n, new_rank))

        # R поглощаем в следующее ядро
        next_core = cores[k + 1]
        old_rank, next_n, next_r = next_core.shape

        new_next_core = DenseTensor.zeros((new_rank, next_n, next_r))

        for i in range(next_n):
            for a in range(new_rank):
                for b in range(next_r):
                    value = 0.0

                    for c in range(old_rank):
                        value += R[a, c] * next_core[c, i, b]

                    new_next_core[a, i, b] = value

        cores[k + 1] = new_next_core

    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает новый TT-тензор в право-канонической форме.
    """
    new_tt = tt.copy()
    cores = new_tt.cores

    if new_tt.order == 1:
        return new_tt

    for k in range(new_tt.order - 1, 0, -1):
        core = cores[k]
        r_left, n, r_right = core.shape

        # Для правой ортогонализации нужна RQ.
        # Делаем её через QR от транспонированной матрицы.
        matrix = backend.reshape(core, (r_left, n * r_right))
        matrix_t = backend.transpose(matrix)

        Q_t, R_t = backend.qr(matrix_t)

        Q = backend.transpose(Q_t)
        R = backend.transpose(R_t)

        new_rank = Q.shape[0]

        # Q обратно превращаем в TT-ядро
        cores[k] = backend.reshape(Q, (new_rank, n, r_right))

        # R поглощаем в предыдущее ядро
        prev_core = cores[k - 1]
        prev_r_left, prev_n, old_rank = prev_core.shape

        new_prev_core = DenseTensor.zeros((prev_r_left, prev_n, new_rank))

        for i in range(prev_n):
            for a in range(prev_r_left):
                for b in range(new_rank):
                    value = 0.0

                    for c in range(old_rank):
                        value += prev_core[a, i, c] * R[c, b]

                    new_prev_core[a, i, b] = value

        cores[k - 1] = new_prev_core

    return TTTensor(cores)


def _numerical_rank(
    S: DenseTensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12
) -> int:
    """
    Возвращает числовой ранг по сингулярным значениям.
    """
    if S.ndim != 1:
        raise ValueError("S must be a vector")

    if S.size == 0:
        return 0

    max_sigma = max(abs(x) for x in S.data)
    threshold = max(abs_tol, rel_tol * max_sigma)

    rank = 0

    for value in S.data:
        if abs(value) > threshold:
            rank += 1

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
    Считает diag(diag_vec) @ matrix.
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


def _multiply_columns_by_diag(
    matrix: DenseTensor,
    diag_vec: DenseTensor,
    backend: BackendInterface
) -> DenseTensor:
    """
    Считает matrix @ diag(diag_vec).
    """
    _ = backend

    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")

    if diag_vec.ndim != 1:
        raise ValueError("diag_vec must be a vector")

    rows, cols = matrix.shape

    if diag_vec.size > cols:
        raise ValueError("diag_vec is too long")

    result = DenseTensor.zeros((rows, diag_vec.size))

    for i in range(rows):
        for j in range(diag_vec.size):
            result[i, j] = matrix[i, j] * diag_vec[j]

    return result