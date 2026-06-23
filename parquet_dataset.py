"""Parquet-backed IterableDataset for RankMixer training.

Streams parquet files (local or HDFS) with PyArrow's pre_buffer prefetch.
Feature column selection driven by FeatureConfig.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import numpy as np
import pyarrow.parquet as pq
import torch
from pyarrow import fs
from torch.utils.data import IterableDataset

from .feature_config import FeatureConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File discovery (local / HDFS)
# ---------------------------------------------------------------------------

def get_files_local(data_path: List[List[str]]) -> List[List[str]]:
    """List parquet files under local directories.

    Args:
        data_path: list of data-source lists, e.g. [["/data/hr=00", "/data/hr=01"]]
    Returns:
        List[List[str]]: parquet file paths grouped by data source.
    """
    data_files = []
    for data_source in data_path:
        source_files = []
        for dp in data_source:
            if not os.path.isdir(dp):
                logger.warning(f"Skipping non-existent local dir: {dp}")
                continue
            for file in os.listdir(dp):
                if file.endswith(".parquet"):
                    source_files.append(os.path.join(dp, file))
        data_files.append(sorted(source_files))
    return data_files


def get_files_hdfs(data_path: List[List[str]]) -> Tuple[List[List[str]], str]:
    """List parquet files under HDFS directories.

    Returns:
        (List[List[str]], str): (file paths grouped by source, hdfs domain)
    """
    all_data_files = []
    domain = None
    for data_source in data_path:
        source_files = []
        for hdfs_path in data_source:
            split_result = urlsplit(hdfs_path)
            domain = f"{split_result.scheme}://{split_result.netloc}"
            directory = split_result.path
            client = fs.HadoopFileSystem.from_uri(domain)
            selector = fs.FileSelector(directory, recursive=True)
            files_info = client.get_file_info(selector)
            for fi in files_info:
                if fi.type == fs.FileType.File and "_SUCCESS" not in fi.path and "done" not in fi.path:
                    source_files.append(domain + fi.path)
        all_data_files.append(sorted(source_files))
    return all_data_files, domain


# ---------------------------------------------------------------------------
# IterableDataset
# ---------------------------------------------------------------------------

class ParquetIterableDataset(IterableDataset):
    """Streams samples from parquet files, driven by FeatureConfig.

    Yields one pandas DataFrame batch at a time (each batch has
    ``batch_size`` rows). The consumer (rankmixer_collate_fn) converts
    batches into RankMixer input tensors.
    """

    def __init__(
        self,
        data_path: List[List[str]],
        feature_config: FeatureConfig,
        from_hdfs: bool = False,
        batch_size: int = 4096,
        sample_rate: float = 0,
        split: Optional[str] = None,
        train_fraction: float = 0.857,
        seed: int = 2024,
    ):
        """
        Args:
            data_path: list of data-source lists, e.g. [["/data/dir1", "/data/dir2"]]
            feature_config: parsed feature.yaml config
            from_hdfs: True for HDFS, False for local filesystem
            batch_size: rows per parquet batch (iter_batches)
            sample_rate: fraction of files to use (0 = all, 0.5 = half)
            split: "train", "valid", or None (use all files)
            train_fraction: fraction of files for training (default 0.857)
            seed: random seed for file shuffling
        """
        super().__init__()
        self.feature_config = feature_config
        self.from_hdfs = from_hdfs
        self.batch_size = batch_size
        self.seed = seed
        self.domain = None
        self.hdfs_client = None

        # Discover files
        if from_hdfs:
            all_files, self.domain = get_files_hdfs(data_path)
        else:
            all_files = get_files_local(data_path)

        # Merge all data sources into a single file list
        merged = []
        for source_files in all_files:
            merged.extend(source_files)
        merged.sort()

        # Apply sample rate
        if sample_rate > 0 and sample_rate < 1.0:
            keep = max(int(sample_rate * len(merged)), 1)
            merged = merged[:keep]
            logger.info(f"[parquet] sample_rate={sample_rate}, keeping {keep}/{len(merged)} files")

        # Train/valid split at file level
        if split is not None:
            if len(merged) <= 1:
                # Single file: use it for both train and valid
                self.files = merged
            else:
                train_end = int(round(len(merged) * train_fraction))
                # Ensure at least 1 file for validation
                train_end = min(train_end, len(merged) - 1)
                train_end = max(train_end, 1)
                if split == "train":
                    self.files = merged[:train_end]
                elif split == "valid":
                    self.files = merged[train_end:]
                else:
                    raise ValueError(f"split must be 'train', 'valid', or None, got '{split}'")
        else:
            self.files = merged

        logger.info(
            f"[parquet] split={split} files={len(self.files)} "
            f"from_hdfs={from_hdfs} batch_size={batch_size}"
        )

        # Validate that all required columns exist in the parquet files
        if self.files:
            self._validate_columns()

    def _validate_columns(self):
        """Read schema from the first parquet file and check all required columns exist."""
        first_file = self.files[0]
        try:
            if self.from_hdfs:
                if self.hdfs_client is None:
                    import pyarrow.fs as pafs
                    self.hdfs_client = pafs.HadoopFileSystem.from_uri(self.domain)
                native_file = self.hdfs_client.open_input_file(first_file)
                schema = pq.ParquetFile(native_file).schema_arrow
                native_file.close()
            else:
                schema = pq.ParquetFile(first_file).schema_arrow

            parquet_columns = set(schema.names)
            required_columns = set(self.feature_config.all_columns)
            missing = required_columns - parquet_columns
            if missing:
                raise KeyError(
                    f"Feature config requires columns that are missing from parquet data: "
                    f"{sorted(missing)}. "
                    f"Available columns ({len(parquet_columns)}): {sorted(parquet_columns)[:20]}..."
                )
            logger.info(
                f"[parquet] column validation passed: {len(required_columns)} required columns "
                f"all found in {first_file}"
            )
        except Exception as e:
            if isinstance(e, KeyError):
                raise
            logger.warning(f"[parquet] could not validate columns from {first_file}: {e}")

    def _open_file(self, file_path: str):
        """Open a single parquet file, returning (ParquetFile, native_file|None)."""
        if self.from_hdfs:
            import time
            for attempt in range(5):
                try:
                    native_file = self.hdfs_client.open_input_file(file_path)
                    pq_file = pq.ParquetFile(native_file, pre_buffer=True)
                    return pq_file, native_file
                except Exception as exc:
                    logger.warning(f"HDFS open failed (attempt {attempt+1}/5): {file_path}: {exc}")
                    time.sleep(0.1)
            raise RuntimeError(f"Failed to open HDFS file after 5 retries: {file_path}")
        else:
            return pq.ParquetFile(file_path), None

    def __iter__(self):
        # Shuffle train files each epoch
        file_list = list(self.files)
        if len(file_list) > 1:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(file_list)
            self.seed += 1  # different shuffle next epoch

        if self.from_hdfs:
            self.hdfs_client = fs.HadoopFileSystem.from_uri(self.domain)

        columns = self.feature_config.all_columns

        try:
            for file_path in file_list:
                pq_file, native_file = self._open_file(file_path)
                try:
                    for record_batch in pq_file.iter_batches(
                        batch_size=self.batch_size, columns=columns
                    ):
                        yield record_batch.to_pandas()
                finally:
                    if native_file is not None:
                        try:
                            native_file.close()
                        except Exception:
                            pass
        finally:
            if self.hdfs_client is not None:
                try:
                    self.hdfs_client.close()
                except Exception:
                    pass
                self.hdfs_client = None


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def _extract_scalar(val, default=0):
    """Extract a scalar int64 from a possibly-nested parquet cell.

    Parquet cells can be:
      - None -> return default
      - list<int64> of length 1 -> return element [0]
      - list<list<int64>> with outer_len=1 -> return [0][0]
      - numpy array equivalent of the above (including dtype=object)
      - Elements inside arrays may be None -> return default
    """
    if val is None:
        return default
    try:
        arr = np.asarray(val)
        if arr.ndim == 0:
            return int(arr)
        if arr.size == 0:
            return default
        # dtype=object means nested arrays (list<list<int64>>)
        # arr[0] is itself an ndarray, need to recurse one level
        if arr.dtype == object:
            first = arr.flat[0]
            if first is None:
                return default
            inner = np.asarray(first)
            if inner.size == 0:
                return default
            return int(inner.flat[0])
        # Non-object dtype: arr is a plain 1-D (or 2-D) numeric array
        if arr.ndim >= 2:
            return int(arr[0, 0]) if arr.shape[1] > 0 else default
        return int(arr[0])
    except (ValueError, TypeError, IndexError):
        return default


def _extract_sequence(val, seq_len, default=0):
    """Extract a fixed-length int64 sequence from a possibly-nested parquet cell.

    Parquet cells can be:
      - None -> zero-filled length seq_len
      - list<int64> -> truncate/pad to seq_len
      - list<list<int64>> with outer_len=1 -> inner sequence, truncate/pad to seq_len
      - numpy array equivalent of the above (including dtype=object)
    Returns numpy array of shape (seq_len,) dtype=int64.
    """
    result = np.full(seq_len, default, dtype=np.int64)
    if val is None:
        return result
    try:
        arr = np.asarray(val)
        if arr.ndim == 0:
            result[0] = int(arr)
            return result
        if arr.size == 0:
            return result
        # dtype=object means nested arrays (list<list<int64>>)
        if arr.dtype == object:
            inner = np.asarray(arr.flat[0])
            if inner.size == 0:
                return result
            arr = inner
        # Flatten and truncate/pad
        if arr.ndim > 1:
            arr = arr.flatten()
        arr = arr.astype(np.int64)
        length = min(len(arr), seq_len)
        result[:length] = arr[:length]
    except (ValueError, TypeError, IndexError):
        pass
    return result


IS_PANDAS_DATAFRAME = None  # lazy import flag


def _extract_column_scalars(series, default=0):
    """Extract scalar int64 values from a pandas Series of nested arrays.

    Handles parquet list<int64> and list<list<int64>> column types,
    including None values and dtype=object arrays.

    Returns numpy array of shape (len(series),) dtype=int64.
    """
    batch_size = len(series)
    result = np.full(batch_size, default, dtype=np.int64)

    # Get underlying numpy array for faster access
    values = series.values

    for i in range(batch_size):
        val = values[i]
        if val is None:
            continue
        try:
            arr = np.asarray(val)
            if arr.ndim == 0:
                result[i] = int(arr)
                continue
            if arr.size == 0:
                continue
            # dtype=object: nested arrays (list<list<int64>>)
            if arr.dtype == object:
                first = arr.flat[0]
                if first is None:
                    continue
                inner = np.asarray(first)
                if inner.size > 0:
                    result[i] = int(inner.flat[0])
                continue
            # Non-object numeric array
            if arr.ndim >= 2:
                if arr.shape[1] > 0:
                    result[i] = int(arr[0, 0])
            else:
                result[i] = int(arr[0])
        except (ValueError, TypeError, IndexError):
            continue

    return result


def _extract_column_sequence(series, seq_len, default=0):
    """Extract fixed-length int64 sequences from a pandas Series of nested arrays.

    Handles parquet list<int64> and list<list<int64>> column types,
    including None values.

    Returns numpy array of shape (len(series), seq_len) dtype=int64.
    """
    batch_size = len(series)
    result = np.full((batch_size, seq_len), default, dtype=np.int64)
    values = series.values

    for i in range(batch_size):
        val = values[i]
        if val is None:
            continue
        try:
            arr = np.asarray(val)
            if arr.ndim == 0:
                result[i, 0] = int(arr)
                continue
            if arr.size == 0:
                continue
            # dtype=object: nested arrays (list<list<int64>>)
            if arr.dtype == object:
                inner = np.asarray(arr.flat[0])
                if inner.size == 0:
                    continue
                arr = inner
            if arr.ndim > 1:
                arr = arr.flatten()
            arr = arr.astype(np.int64)
            length = min(len(arr), seq_len)
            result[i, :length] = arr[:length]
        except (ValueError, TypeError, IndexError):
            continue

    return result


def rankmixer_collate_fn(
    batch,
    feature_config: FeatureConfig,
    vocab_caps: Optional[List[int]] = None,
):
    """Convert parquet DataFrame(s) into RankMixer input tensors.

    Handles the nested array format produced by PyArrow when reading parquet
    files with list-type columns. Each row in the parquet represents a group
    of examples (though group_size is typically 1 in this dataset).

    For sparse features: extracts scalar int64 IDs (handles both list<int64>
    and list<list<int64>> parquet types).

    For dense features: extracts scalar int64 hash values and converts to
    float32 (handles both list<list<int64>> single-hash and list<int64>
    multi-hash formats; multi-hash features are reduced to their first
    element).

    For sequential features: extracts variable-length int64 ID sequences
    (handles both list<int64> and list<list<int64>> types).

    For labels: extracts scalar from list<list<int64>> format.

    Args:
        batch: a single pandas DataFrame (from IterableDataset) or list of
            DataFrames (from map-style DataLoader)
        feature_config: feature schema
        vocab_caps: optional per-sparse-field cap on vocab size (modulo clamp)

    Returns:
        If feature_config.has_sequential:
            ((sparse, dense, sequential_list), labels)
        Else:
            ((sparse, dense), labels)
    """
    global IS_PANDAS_DATAFRAME
    if IS_PANDAS_DATAFRAME is None:
        import pandas as pd
        IS_PANDAS_DATAFRAME = pd.DataFrame

    if isinstance(batch, IS_PANDAS_DATAFRAME):
        merged = batch
    else:
        import pandas as pd
        merged = pd.concat(batch, ignore_index=True)

    batch_size = len(merged)

    # Sparse features -> int64 tensor (B, num_sparse)
    sparse_cols = feature_config.sparse_columns
    if sparse_cols:
        sparse_data = np.zeros((batch_size, len(sparse_cols)), dtype=np.int64)
        for j, col in enumerate(sparse_cols):
            if col in merged.columns:
                sparse_data[:, j] = _extract_column_scalars(merged[col], default=0)
        # Clamp negatives and apply vocab cap
        if vocab_caps is not None:
            np.maximum(sparse_data, 0, out=sparse_data)
            for j, cap in enumerate(vocab_caps):
                sparse_data[:, j] %= int(cap)
        sparse_tensor = torch.from_numpy(sparse_data)
    else:
        sparse_tensor = torch.zeros(batch_size, 0, dtype=torch.int64)

    # Dense features -> float32 tensor (B, num_dense)
    # In the parquet dataset, dense features are int64 hashes; we convert to float.
    # Single-hash features (list<list>): extract scalar and cast.
    # Multi-hash features (list<int>): extract first element and cast.
    dense_cols = feature_config.dense_columns
    if dense_cols:
        dense_data = np.zeros((batch_size, len(dense_cols)), dtype=np.float32)
        for j, col in enumerate(dense_cols):
            if col in merged.columns:
                dense_data[:, j] = _extract_column_scalars(merged[col], default=0).astype(np.float32)
        # Log1p preprocessing to match CriteoDataset (after clamping negatives)
        np.maximum(dense_data, 0.0, out=dense_data)
        np.log1p(dense_data, out=dense_data)
        dense_tensor = torch.from_numpy(dense_data)
    else:
        dense_tensor = torch.zeros(batch_size, 0, dtype=np.float32)

    # Labels -> float32 tensor
    label_col = feature_config.label_columns[0]
    labels = _extract_column_scalars(merged[label_col], default=0).astype(np.float32) if label_col in merged.columns else np.zeros(batch_size, dtype=np.float32)
    labels = torch.from_numpy(labels)

    # Sequential features -> list of (B, seq_len) int64 tensors
    if feature_config.has_sequential:
        sequential_inputs = []
        for sf in feature_config.sequential_features:
            seq_len = sf.seq_len
            if sf.name in merged.columns:
                seq_tensor = _extract_column_sequence(merged[sf.name], seq_len, default=0)
            else:
                seq_tensor = np.zeros((batch_size, seq_len), dtype=np.int64)
            # Apply vocab cap if set (modulo on shared embedding index)
            if vocab_caps is not None and sf.shared_emb_idx is not None:
                cap = vocab_caps[sf.shared_emb_idx]
                seq_tensor %= int(cap)
            sequential_inputs.append(torch.from_numpy(seq_tensor))
        model_inputs = (sparse_tensor, dense_tensor, sequential_inputs)
    else:
        model_inputs = (sparse_tensor, dense_tensor)

    return model_inputs, labels

