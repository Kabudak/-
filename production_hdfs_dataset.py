"""Production HDFSDataset - stream parquet from HDFS with production preprocessing.

Reads parquet files (local or HDFS) and preprocesses each batch using the same
logic as ``utils/production_data.build_tensors()``, yielding the 8-tuple tensor
format that ``TAACHyFormerClassifier.forward()`` expects.

Feature selection is driven by ``selectedfeaturefinal.txt`` (not ``feature.yaml``),
and features are classified into 5 categories:
  - non_seq_sparse       (scalar ID -> bucketized)
  - non_seq_sparse_bag   (multi-value ID array -> bucketized + mask)
  - non_seq_dense        (numeric -> signed_log1p or raw)
  - seq_sparse           (sequence of IDs -> bucketized + mask)
  - seq_dense            (sequence of numerics -> signed_log1p + mask)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from pyarrow import fs
from torch.utils.data import IterableDataset, get_worker_info

from utils.production_data import (
    ALREADY_LOGGED_FIELDS,
    LABEL_COLUMN,
    NS_SPARSE_BAG_FIELDS,
    ProductionFeatureSchema,
    bucket_scalar_array,
    bucket_size_for_field,
    extract_dense_matrix,
    extract_dense_spec_array,
    extract_sparse_matrix,
    first_scalar,
    load_feature_schema,
    safe_int,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File discovery (local / HDFS) - copied from parquet_dataset.py to avoid
# its broken ``from .feature_config import FeatureConfig`` import.
# ---------------------------------------------------------------------------

def get_files_local(data_path: List[List[str]]) -> List[List[str]]:
    """List parquet files under local directories."""
    data_files: List[List[str]] = []
    for data_source in data_path:
        source_files: List[str] = []
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
    all_data_files: List[List[str]] = []
    domain: str | None = None
    for data_source in data_path:
        source_files: List[str] = []
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
    return all_data_files, domain or ""


def _open_hdfs_file(
    hdfs_client: fs.HadoopFileSystem,
    file_path: str,
    max_retries: int = 5,
) -> Tuple[pq.ParquetFile, Any]:
    """Open a single HDFS parquet file with retries and pre_buffer prefetch."""
    for attempt in range(max_retries):
        try:
            native_file = hdfs_client.open_input_file(file_path)
            pq_file = pq.ParquetFile(native_file, pre_buffer=True)
            return pq_file, native_file
        except Exception as exc:
            logger.warning(f"HDFS open failed (attempt {attempt + 1}/{max_retries}): {file_path}: {exc}")
            time.sleep(0.1)
    raise RuntimeError(f"Failed to open HDFS file after {max_retries} retries: {file_path}")


# ---------------------------------------------------------------------------
# IterableDataset
# ---------------------------------------------------------------------------

class ProductionHDFSDataset(IterableDataset):
    """Streams production parquet files and yields preprocessed 8-tuple tensors.

    Each yielded item is a tuple of 8 tensors matching the input format of
    ``TAACHyFormerClassifier.forward()``:

        (non_seq_sparse, non_seq_sparse_bag, non_seq_sparse_bag_mask,
         non_seq_dense, seq_sparse, seq_dense, seq_mask, labels)
    """

    def __init__(
        self,
        data_path: List[List[str]],
        feature_file: Path,
        from_hdfs: bool = True,
        seq_len: int = 100,
        sequence_truncation: str = "tail",
        non_seq_bag_len: int = 64,
        sequence_lens: Dict[str, int] | None = None,
        non_seq_array_reduction: str = "last",
        batch_size: int = 4096,
        sample_rate: float = 0,
        split: Optional[str] = None,
        train_fraction: float = 0.857,
        seed: int = 2024,
    ):
        """
        Args:
            data_path: list of data-source lists, e.g.
                [["hdfs://nn:8020/data/hr=00", "hdfs://nn:8020/data/hr=01"]]
            feature_file: path to selectedfeaturefinal.txt
            from_hdfs: True for HDFS, False for local filesystem
            seq_len: max sequence length (overridden per-branch by sequence_lens)
            sequence_truncation: "head" or "tail" when sequences exceed max length
            non_seq_bag_len: max length per non-sequence sparse bag feature
            sequence_lens: optional per-branch lengths, e.g. {"click_seq": 100}
            non_seq_array_reduction: how dense array features are reduced ("last" / "mean")
            batch_size: rows per parquet batch (iter_batches)
            sample_rate: fraction of files to use (0 = all, 0.5 = half)
            split: "train", "valid", or None (use all files)
            train_fraction: fraction of files for training
            seed: random seed for file shuffling
        """
        super().__init__()
        self.from_hdfs = from_hdfs
        self.batch_size = batch_size
        self.seed = seed
        self.non_seq_array_reduction = non_seq_array_reduction
        self.domain: str | None = None
        self.hdfs_client: fs.HadoopFileSystem | None = None

        # ---- 1. Load feature schema from selectedfeaturefinal.txt ----
        self.schema = load_feature_schema(
            feature_file=feature_file,
            seq_len=seq_len,
            sequence_truncation=sequence_truncation,
            non_seq_bag_len=non_seq_bag_len,
            sequence_lens=sequence_lens,
        )

        # ---- 2. Compute required parquet columns ----
        self.all_columns = self._compute_required_columns()

        # ---- 3. Discover files ----
        if from_hdfs:
            all_files, self.domain = get_files_hdfs(data_path)
        else:
            all_files = get_files_local(data_path)

        # Merge all data sources into a single file list
        merged: List[str] = []
        for source_files in all_files:
            merged.extend(source_files)
        merged.sort()

        # Apply sample rate
        if 0 < sample_rate < 1.0:
            keep = max(int(sample_rate * len(merged)), 1)
            merged = merged[:keep]
            logger.info(f"[production_hdfs] sample_rate={sample_rate}, keeping {keep}/{len(merged)} files")

        # Train/valid split at file level
        if split is not None:
            if len(merged) <= 1:
                self.files = merged
            else:
                train_end = int(round(len(merged) * train_fraction))
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
            f"[production_hdfs] split={split} files={len(self.files)} "
            f"from_hdfs={from_hdfs} batch_size={batch_size} "
            f"seq_len={self.schema.seq_len} "
            f"non_seq_sparse={len(self.schema.non_seq_sparse_fields)} "
            f"non_seq_sparse_bag={len(self.schema.non_seq_sparse_bag_fields)} "
            f"non_seq_dense={len(self.schema.non_seq_dense_specs)} "
            f"seq_sparse={len(self.schema.seq_sparse_fields)} "
            f"seq_dense={len(self.schema.seq_dense_fields)}"
        )

        # ---- 4. Pre-compute index maps for fast lookup ----
        self._seq_sparse_index: Dict[str, int] = {
            field: idx for idx, field in enumerate(self.schema.seq_sparse_fields)
        }
        self._seq_dense_index: Dict[str, int] = {
            field: idx for idx, field in enumerate(self.schema.seq_dense_fields)
        }
        self._seq_sparse_set = set(self.schema.seq_sparse_fields)

        # ---- 5. Validate columns ----
        if self.files:
            self._validate_columns()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_required_columns(self) -> List[str]:
        """Gather all column names referenced by the schema."""
        cols: set[str] = {LABEL_COLUMN}
        cols.update(self.schema.non_seq_sparse_fields)
        cols.update(self.schema.non_seq_sparse_bag_fields)
        for spec in self.schema.non_seq_dense_specs:
            cols.add(spec.source)
        cols.update(self.schema.seq_sparse_fields)
        cols.update(self.schema.seq_dense_fields)
        return sorted(cols)

    def _validate_columns(self) -> None:
        """Check that the first parquet file contains all required columns."""
        first_file = self.files[0]
        try:
            if self.from_hdfs:
                if self.hdfs_client is None:
                    self.hdfs_client = fs.HadoopFileSystem.from_uri(self.domain)
                native_file = self.hdfs_client.open_input_file(first_file)
                parquet_schema = pq.ParquetFile(native_file).schema_arrow
                native_file.close()
            else:
                parquet_schema = pq.ParquetFile(first_file).schema_arrow

            parquet_columns = set(parquet_schema.names)
            required = set(self.all_columns)
            missing = required - parquet_columns
            if missing:
                raise KeyError(
                    f"Feature schema requires columns missing from parquet: "
                    f"{sorted(missing)}. "
                    f"Available ({len(parquet_columns)}): {sorted(parquet_columns)[:20]}..."
                )
            logger.info(
                f"[production_hdfs] column validation passed: {len(required)} columns "
                f"all found in {first_file}"
            )
        except Exception as e:
            if isinstance(e, KeyError):
                raise
            logger.warning(f"[production_hdfs] could not validate columns from {first_file}: {e}")

    def _open_file(self, file_path: str) -> Tuple[pq.ParquetFile, Any]:
        """Open a single parquet file, returning (ParquetFile, native_file|None)."""
        if self.from_hdfs:
            return _open_hdfs_file(self.hdfs_client, file_path)
        else:
            return pq.ParquetFile(file_path), None

    def _extract_labels(self, columns: Dict[str, List[Any]], batch_size: int) -> np.ndarray:
        """Vectorized label extraction from label_click column."""
        if LABEL_COLUMN not in columns:
            raise KeyError(f"Missing required label column: {LABEL_COLUMN}")
        raw_list = columns[LABEL_COLUMN]
        return np.array([safe_int(first_scalar(v)) for v in raw_list], dtype=np.int64)

    # ------------------------------------------------------------------
    # Core preprocessing - mirrors build_tensors() per-batch
    # ------------------------------------------------------------------

    def _preprocess_batch(self, batch_df: pd.DataFrame) -> Tuple[torch.Tensor, ...]:
        """Convert a pandas DataFrame batch into the 8-tuple tensor format.

        This follows the same logic as ``production_data.build_tensors()``
        but operates on a single batch instead of the entire file.
        """
        B = len(batch_df)
        schema = self.schema

        # Convert DataFrame columns to Python lists (production_data functions expect this)
        columns: Dict[str, List[Any]] = {col: batch_df[col].tolist() for col in batch_df.columns}

        # ---- Labels ----
        labels_np = self._extract_labels(columns, B)

        # ---- 1. Non-seq sparse scalars ----
        non_seq_sparse_np = np.zeros((B, len(schema.non_seq_sparse_fields)), dtype=np.int32)
        for idx, field in enumerate(schema.non_seq_sparse_fields):
            non_seq_sparse_np[:, idx] = bucket_scalar_array(
                columns[field], bucket_size_for_field(field), reduction="last"
            )

        # ---- 2. Non-seq sparse bags ----
        non_seq_sparse_bag_np = np.zeros(
            (B, len(schema.non_seq_sparse_bag_fields), schema.non_seq_bag_len),
            dtype=np.int32,
        )
        non_seq_sparse_bag_mask_np = np.zeros_like(non_seq_sparse_bag_np, dtype=bool)
        for idx, field in enumerate(schema.non_seq_sparse_bag_fields):
            matrix, mask = extract_sparse_matrix(
                columns[field],
                width=schema.non_seq_bag_len,
                truncation=schema.sequence_truncation,
                bucket_size=bucket_size_for_field(field),
            )
            non_seq_sparse_bag_np[:, idx, :] = matrix
            non_seq_sparse_bag_mask_np[:, idx, :] = mask

        # ---- 3. Non-seq dense ----
        non_seq_dense_np = np.zeros((B, len(schema.non_seq_dense_specs)), dtype=np.float32)
        for idx, spec in enumerate(schema.non_seq_dense_specs):
            non_seq_dense_np[:, idx] = extract_dense_spec_array(
                columns[spec.source],
                spec,
                array_reduction=self.non_seq_array_reduction,
            )

        # ---- 4. Sequence tensors ----
        max_seq_len = schema.seq_len
        num_sequences = len(schema.sequence_names)

        seq_sparse_np = np.zeros(
            (B, num_sequences, max_seq_len, len(schema.seq_sparse_fields)),
            dtype=np.int32,
        )
        seq_dense_np = np.zeros(
            (B, num_sequences, max_seq_len, len(schema.seq_dense_fields)),
            dtype=np.float32,
        )
        seq_mask_np = np.zeros((B, num_sequences, max_seq_len), dtype=bool)

        for branch_idx, branch_name in enumerate(schema.sequence_names):
            branch_len = schema.sequence_lens[branch_name]
            backbone_field = schema.sequence_backbone_fields[branch_name]
            for field in schema.sequence_fields[branch_name]:
                if field in self._seq_sparse_set:
                    field_idx = self._seq_sparse_index[field]
                    matrix, mask = extract_sparse_matrix(
                        columns[field],
                        width=branch_len,
                        truncation=schema.sequence_truncation,
                        bucket_size=bucket_size_for_field(field),
                    )
                    seq_sparse_np[:, branch_idx, :branch_len, field_idx] = matrix
                    if field == backbone_field:
                        seq_mask_np[:, branch_idx, :branch_len] = mask
                else:
                    field_idx = self._seq_dense_index[field]
                    matrix, mask = extract_dense_matrix(
                        columns[field],
                        width=branch_len,
                        truncation=schema.sequence_truncation,
                        already_logged=field in ALREADY_LOGGED_FIELDS,
                    )
                    seq_dense_np[:, branch_idx, :branch_len, field_idx] = matrix
                    if field == backbone_field:
                        seq_mask_np[:, branch_idx, :branch_len] = mask

        return (
            torch.from_numpy(non_seq_sparse_np),
            torch.from_numpy(non_seq_sparse_bag_np),
            torch.from_numpy(non_seq_sparse_bag_mask_np),
            torch.from_numpy(non_seq_dense_np),
            torch.from_numpy(seq_sparse_np),
            torch.from_numpy(seq_dense_np),
            torch.from_numpy(seq_mask_np),
            torch.from_numpy(labels_np),
        )

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self):
        # Shuffle train files each epoch
        file_list = list(self.files)
        if len(file_list) > 1:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(file_list)
            self.seed += 1  # different shuffle next epoch

        worker_info = get_worker_info()
        if worker_info is not None:
            file_list = file_list[worker_info.id::worker_info.num_workers]

        if self.from_hdfs:
            self.hdfs_client = fs.HadoopFileSystem.from_uri(self.domain)

        try:
            for file_path in file_list:
                pq_file, native_file = self._open_file(file_path)
                try:
                    for record_batch in pq_file.iter_batches(
                        batch_size=self.batch_size, columns=self.all_columns
                    ):
                        batch_df = record_batch.to_pandas()
                        yield self._preprocess_batch(batch_df)
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

    # ------------------------------------------------------------------
    # Metadata for model construction
    # ------------------------------------------------------------------

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata dict for constructing TAACHyFormerClassifier.

        This matches the 9th return value of ``production_data.build_tensors()``,
        minus per-file statistics (num_samples, pos_rate, etc.) which are not
        available in a streaming context.
        """
        schema = self.schema
        return {
            "dataset": "production_parquet",
            "feature_version": "production_field_aware_hyformer_v2",
            "label_column": LABEL_COLUMN,
            "label_mapping": {"0": 0, "1": 1},
            "seq_len": schema.seq_len,
            "sequence_lens": schema.sequence_lens,
            "sequence_truncation": schema.sequence_truncation,
            "non_seq_bag_len": schema.non_seq_bag_len,
            "num_sequences": len(schema.sequence_names),
            "sequence_names": schema.sequence_names,
            "sequence_fields": schema.sequence_fields,
            "sequence_backbone_fields": schema.sequence_backbone_fields,
            "non_seq_sparse_fields": schema.non_seq_sparse_fields,
            "non_seq_sparse_bag_fields": schema.non_seq_sparse_bag_fields,
            "non_seq_dense_fields": [spec.name for spec in schema.non_seq_dense_specs],
            "non_seq_dense_sources": [
                {"name": spec.name, "source": spec.source, "stat": spec.stat}
                for spec in schema.non_seq_dense_specs
            ],
            "seq_sparse_fields": schema.seq_sparse_fields,
            "seq_dense_fields": schema.seq_dense_fields,
            "sparse_field_cardinalities": schema.sparse_field_cardinalities,
            "token_groups": schema.token_groups,
            "already_logged_fields": sorted(ALREADY_LOGGED_FIELDS),
            "non_seq_array_policies": {
                "sparse_bag": sorted(NS_SPARSE_BAG_FIELDS),
                "dense_array_reduction": self.non_seq_array_reduction,
                "sparse_scalar_array_reduction": "last",
            },
            "notes": {
                "label": f"Binary label from {LABEL_COLUMN} column.",
                "feature_schema": "Manually curated feature groups from selectedfeaturefinal.txt.",
                "already_logged": "Fields in already_logged_fields skip signed_log1p to avoid double-log.",
                "non_seq_sparse_bag": "Sparse array features are bucketized and mean-pooled inside the model.",
                "sequence_mask": "Each sequence mask is built only from that branch's first selected backbone field.",
            },
        }
