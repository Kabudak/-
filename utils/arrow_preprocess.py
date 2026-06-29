from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def _combine_if_needed(array: pa.Array | pa.ChunkedArray) -> pa.Array:
    if isinstance(array, pa.ChunkedArray):
        return array.combine_chunks()
    return array


def _is_list_type(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
    )


def _flatten_with_parent(
    array: pa.Array | pa.ChunkedArray,
    batch_size: int,
) -> tuple[pa.Array, np.ndarray]:
    """Flatten nested Arrow list arrays while keeping the original row index.

    Production parquet columns are commonly stored as list<int64> or
    list<list<int64>>.  This function flattens every list level in Arrow's C++
    kernels and returns one primitive value array plus a NumPy parent index
    array that maps each primitive value back to its top-level batch row.
    """
    current = _combine_if_needed(array)
    parent: np.ndarray | None = None

    while _is_list_type(current.type):
        level_parent = pc.list_parent_indices(current).to_numpy(zero_copy_only=False)
        current = pc.list_flatten(current)
        if parent is None:
            parent = level_parent.astype(np.int64, copy=False)
        else:
            parent = parent[level_parent]

    if parent is None:
        parent = np.arange(batch_size, dtype=np.int64)
    return current, parent.astype(np.int64, copy=False)


def _values_to_numpy(values: pa.Array, dtype: Any) -> np.ndarray:
    if pa.types.is_null(values.type):
        return np.zeros(len(values), dtype=dtype)

    if np.dtype(dtype).kind in {"f"}:
        casted = pc.cast(pc.fill_null(values, 0), pa.float64(), safe=False)
        out = casted.to_numpy(zero_copy_only=False).astype(dtype, copy=False)
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    casted = pc.cast(pc.fill_null(values, 0), pa.int64(), safe=False)
    return casted.to_numpy(zero_copy_only=False).astype(dtype, copy=False)


def _row_counts(parent: np.ndarray, batch_size: int) -> np.ndarray:
    if parent.size == 0:
        return np.zeros(batch_size, dtype=np.int64)
    return np.bincount(parent, minlength=batch_size).astype(np.int64, copy=False)


def _row_positions(parent: np.ndarray, counts: np.ndarray) -> np.ndarray:
    if parent.size == 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.empty_like(counts)
    starts[0] = 0
    if counts.size > 1:
        np.cumsum(counts[:-1], out=starts[1:])
    return np.arange(parent.size, dtype=np.int64) - starts[parent]


def signed_log1p_array(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return (np.sign(values) * np.log1p(np.abs(values))).astype(np.float32, copy=False)


def bucket_array(raw: np.ndarray, bucket_size: int) -> np.ndarray:
    raw = raw.astype(np.int64, copy=False)
    return np.where(raw == 0, 0, np.abs(raw) % int(bucket_size) + 1).astype(np.int32)


def column_to_scalar(
    array: pa.Array | pa.ChunkedArray,
    batch_size: int,
    dtype: Any,
    reduction: str = "last",
) -> np.ndarray:
    values, parent = _flatten_with_parent(array, batch_size)
    raw = _values_to_numpy(values, dtype=dtype)
    output = np.zeros(batch_size, dtype=dtype)
    counts = _row_counts(parent, batch_size)
    rows = np.nonzero(counts > 0)[0]
    if rows.size == 0:
        return output

    starts = np.empty_like(counts)
    starts[0] = 0
    if counts.size > 1:
        np.cumsum(counts[:-1], out=starts[1:])

    if reduction == "first":
        output[rows] = raw[starts[rows]]
    elif reduction == "last":
        output[rows] = raw[starts[rows] + counts[rows] - 1]
    elif reduction == "mean":
        sums = np.bincount(parent, weights=raw.astype(np.float64, copy=False), minlength=batch_size)
        output[rows] = (sums[rows] / counts[rows]).astype(dtype, copy=False)
    else:
        raise ValueError(f"Unsupported scalar reduction: {reduction}")
    return output


def column_to_sparse_scalar(
    array: pa.Array | pa.ChunkedArray,
    batch_size: int,
    bucket_size: int,
    reduction: str = "last",
) -> np.ndarray:
    raw = column_to_scalar(array, batch_size=batch_size, dtype=np.int64, reduction=reduction)
    return bucket_array(raw, bucket_size)


def column_to_dense_scalar(
    array: pa.Array | pa.ChunkedArray,
    batch_size: int,
    already_logged: bool,
    reduction: str = "last",
) -> np.ndarray:
    raw = column_to_scalar(array, batch_size=batch_size, dtype=np.float32, reduction=reduction)
    return raw.astype(np.float32, copy=False) if already_logged else signed_log1p_array(raw)


def column_to_fixed_matrix(
    array: pa.Array | pa.ChunkedArray,
    batch_size: int,
    width: int,
    truncation: str,
    dtype: Any,
    bucket_size: int | None = None,
    already_logged: bool = False,
    sparse_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if width <= 0:
        raise ValueError("width must be positive")
    if truncation not in {"head", "tail"}:
        raise ValueError("truncation must be 'head' or 'tail'")

    values, parent = _flatten_with_parent(array, batch_size)
    raw = _values_to_numpy(values, dtype=np.float32 if np.dtype(dtype).kind == "f" else np.int64)
    matrix = np.zeros((batch_size, width), dtype=dtype)
    mask = np.zeros((batch_size, width), dtype=bool)
    if parent.size == 0:
        return matrix, mask

    counts = _row_counts(parent, batch_size)
    pos = _row_positions(parent, counts)

    if truncation == "head":
        keep = pos < width
        out_pos = pos
    else:
        row_start = np.maximum(counts[parent] - width, 0)
        keep = pos >= row_start
        out_pos = pos - row_start
    keep &= out_pos < width

    if not np.any(keep):
        return matrix, mask

    kept_parent = parent[keep]
    kept_pos = out_pos[keep]
    kept_values = raw[keep]
    if bucket_size is not None:
        encoded = bucket_array(kept_values.astype(np.int64, copy=False), bucket_size)
        matrix[kept_parent, kept_pos] = encoded
        mask[kept_parent, kept_pos] = encoded != 0 if sparse_mask else True
    else:
        dense = kept_values.astype(np.float32, copy=False)
        if not already_logged:
            dense = signed_log1p_array(dense)
        matrix[kept_parent, kept_pos] = dense.astype(dtype, copy=False)
        mask[kept_parent, kept_pos] = True
    return matrix, mask
