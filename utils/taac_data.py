from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import Dataset, DownloadConfig, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files


ARRAY_STATS = ("mean", "std", "min", "max", "last", "length")


@dataclass
class FlatSchema:
    scalar_cols: list[str] = field(default_factory=list)
    array_cols: list[str] = field(default_factory=list)
    seq_cols: list[str] = field(default_factory=list)


@dataclass
class RawSchema:
    scalar_names: list[str] = field(default_factory=list)
    array_names: list[str] = field(default_factory=list)
    seq_names: list[str] = field(default_factory=list)


def is_flat_schema(row: dict[str, Any]) -> bool:
    return "label_type" in row and any(key.startswith("domain_") for key in row)


def safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
            return int(digest[:12], 16) / float(16**12)
    return 0.0


def squash_numeric(value: float) -> float:
    if value == 0.0:
        return 0.0
    return math.copysign(math.log1p(abs(value)), value)


def sanitize_sequence(values: Any) -> list[float]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, dict)):
        return []
    return [squash_numeric(safe_float(v)) for v in values]


def summarize_array(values: Any) -> list[float]:
    arr = sanitize_sequence(values)
    if not arr:
        return [0.0] * len(ARRAY_STATS)

    mean = sum(arr) / len(arr)
    variance = sum((value - mean) ** 2 for value in arr) / len(arr)
    return [
        mean,
        math.sqrt(variance),
        min(arr),
        max(arr),
        arr[-1],
        math.log1p(len(arr)),
    ]


def scalar_feature(value: Any) -> float:
    return squash_numeric(safe_float(value))


def detect_flat_schema(rows: list[dict[str, Any]]) -> FlatSchema:
    keys = sorted(rows[0].keys())
    schema = FlatSchema()
    for key in keys:
        if key in {"label_type", "label_time"}:
            continue

        sample_value = None
        for row in rows:
            if row.get(key) is not None:
                sample_value = row[key]
                break

        if key.startswith("domain_"):
            schema.seq_cols.append(key)
        elif isinstance(sample_value, list):
            schema.array_cols.append(key)
        else:
            schema.scalar_cols.append(key)
    return schema


def extract_raw_feature_maps(row: dict[str, Any]) -> tuple[dict[str, float], dict[str, list[float]], dict[str, list[float]]]:
    scalar_map: dict[str, float] = {
        "user_id": scalar_feature(row.get("user_id")),
        "item_id": scalar_feature(row.get("item_id")),
        "timestamp": scalar_feature(row.get("timestamp")),
    }
    array_map: dict[str, list[float]] = {}
    seq_map: dict[str, list[float]] = {}

    for prefix, feature_key in (("user", "user_feature"), ("item", "item_feature")):
        for feature in row.get(feature_key) or []:
            feature_id = feature.get("feature_id")
            if feature_id is None:
                continue

            feature_stub = f"{prefix}_{feature_id}"
            if feature.get("int_value") is not None:
                scalar_map[f"{feature_stub}_int_value"] = scalar_feature(feature["int_value"])
            if feature.get("float_value") is not None:
                scalar_map[f"{feature_stub}_float_value"] = scalar_feature(feature["float_value"])

            int_array = feature.get("int_array")
            if int_array:
                array_map[f"{feature_stub}_int_array"] = sanitize_sequence(int_array)

            float_array = feature.get("float_array")
            if float_array:
                array_map[f"{feature_stub}_float_array"] = sanitize_sequence(float_array)

    for group_name, feature_group in (row.get("seq_feature") or {}).items():
        for feature in feature_group or []:
            feature_id = feature.get("feature_id")
            if feature_id is None:
                continue

            values = feature.get("int_array") or feature.get("float_array") or []
            seq_map[f"{group_name}_{feature_id}"] = sanitize_sequence(values)

    return scalar_map, array_map, seq_map


def detect_raw_schema(rows: list[dict[str, Any]]) -> RawSchema:
    scalar_names = {"user_id", "item_id", "timestamp"}
    array_names: set[str] = set()
    seq_names: set[str] = set()

    for row in rows:
        scalar_map, array_map, seq_map = extract_raw_feature_maps(row)
        scalar_names.update(scalar_map.keys())
        array_names.update(array_map.keys())
        seq_names.update(seq_map.keys())

    return RawSchema(
        scalar_names=sorted(scalar_names),
        array_names=sorted(array_names),
        seq_names=sorted(seq_names),
    )


def flat_label(row: dict[str, Any]) -> int:
    return int(row["label_type"])


def raw_label(row: dict[str, Any]) -> int:
    label_entries = row.get("label") or []
    if not label_entries:
        raise ValueError("Raw row does not contain any label entry.")
    return int(label_entries[0]["action_type"])


def build_sequence_groups(names: list[str], num_sequences: int) -> list[list[str]]:
    if not names:
        return []
    actual_num_sequences = max(1, min(num_sequences, len(names)))
    groups = [[] for _ in range(actual_num_sequences)]
    for idx, name in enumerate(names):
        groups[idx % actual_num_sequences].append(name)
    return groups


def build_grouped_sequence_matrices(seq_map: dict[str, list[float]], groups: list[list[str]], seq_len: int) -> list[list[list[float]]]:
    grouped_matrices: list[list[list[float]]] = []
    for group in groups:
        group_matrix = [[0.0 for _ in group] for _ in range(seq_len)]
        for feature_idx, feature_name in enumerate(group):
            for step_idx, value in enumerate(seq_map.get(feature_name, [])[:seq_len]):
                group_matrix[step_idx][feature_idx] = value
        grouped_matrices.append(group_matrix)
    return grouped_matrices


def vectorize_flat_row(
    row: dict[str, Any],
    schema: FlatSchema,
    seq_len: int,
    groups: list[list[str]],
) -> tuple[list[float], list[list[list[float]]]]:
    non_seq = [scalar_feature(row.get(col)) for col in schema.scalar_cols]
    for col in schema.array_cols:
        non_seq.extend(summarize_array(row.get(col)))
    seq_map = {col: sanitize_sequence(row.get(col)) for col in schema.seq_cols}
    return non_seq, build_grouped_sequence_matrices(seq_map, groups, seq_len)


def vectorize_raw_row(
    row: dict[str, Any],
    schema: RawSchema,
    seq_len: int,
    groups: list[list[str]],
) -> tuple[list[float], list[list[list[float]]]]:
    scalar_map, array_map, seq_map = extract_raw_feature_maps(row)
    non_seq = [scalar_map.get(name, 0.0) for name in schema.scalar_names]
    for name in schema.array_names:
        non_seq.extend(summarize_array(array_map.get(name, [])))
    return non_seq, build_grouped_sequence_matrices(seq_map, groups, seq_len)


def pad_grouped_sequences(
    grouped_matrices: list[list[list[list[float]]]],
    seq_len: int,
    num_sequences: int,
    max_seq_feature_dim: int,
) -> torch.Tensor:
    tensor = torch.zeros(len(grouped_matrices), num_sequences, seq_len, max_seq_feature_dim, dtype=torch.float32)
    for row_idx, row_groups in enumerate(grouped_matrices):
        for seq_idx, matrix in enumerate(row_groups):
            for step_idx, step in enumerate(matrix[:seq_len]):
                tensor[row_idx, seq_idx, step_idx, : len(step)] = torch.tensor(step, dtype=torch.float32)
    return tensor


def make_label_mapping(rows: list[dict[str, Any]], label_fn: Any) -> dict[int, int]:
    labels = sorted({int(label_fn(row)) for row in rows})
    return {label: idx for idx, label in enumerate(labels)}


def build_tensors(
    rows: list[dict[str, Any]],
    seq_len: int,
    num_sequences: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not rows:
        raise ValueError("Dataset is empty.")

    if is_flat_schema(rows[0]):
        schema = detect_flat_schema(rows)
        label_fn = flat_label
        seq_names = schema.seq_cols
        schema_name = "flat"
        groups = build_sequence_groups(seq_names, num_sequences)
        vectorize_fn = vectorize_flat_row
        schema_payload: dict[str, Any] = {
            "schema": "flat",
            "scalar_cols": schema.scalar_cols,
            "array_cols": schema.array_cols,
            "seq_cols": schema.seq_cols,
        }
    else:
        schema = detect_raw_schema(rows)
        label_fn = raw_label
        seq_names = schema.seq_names
        schema_name = "raw"
        groups = build_sequence_groups(seq_names, num_sequences)
        vectorize_fn = vectorize_raw_row
        schema_payload = {
            "schema": "raw",
            "scalar_names": schema.scalar_names,
            "array_names": schema.array_names,
            "seq_names": schema.seq_names,
        }

    if not groups:
        raise ValueError(f"No sequence features were detected under the {schema_name} schema.")

    label_mapping = make_label_mapping(rows, label_fn)
    non_seq_vectors: list[list[float]] = []
    grouped_sequences: list[list[list[list[float]]]] = []
    labels: list[int] = []

    for row in rows:
        non_seq_vec, grouped_seq = vectorize_fn(row, schema, seq_len, groups)
        non_seq_vectors.append(non_seq_vec)
        grouped_sequences.append(grouped_seq)
        labels.append(label_mapping[int(label_fn(row))])

    non_seq_dim = len(non_seq_vectors[0])
    sequence_group_dims = [len(group) for group in groups]
    max_seq_feature_dim = max(sequence_group_dims)
    non_seq_tensor = torch.tensor(non_seq_vectors, dtype=torch.float32)
    seq_tensor = pad_grouped_sequences(
        grouped_sequences,
        seq_len=seq_len,
        num_sequences=len(groups),
        max_seq_feature_dim=max_seq_feature_dim,
    )
    label_tensor = torch.tensor(labels, dtype=torch.long)
    metadata = {
        **schema_payload,
        "label_mapping": {str(label): idx for label, idx in label_mapping.items()},
        "non_seq_dim": non_seq_dim,
        "num_sequences": len(groups),
        "sequence_groups": groups,
        "sequence_group_dims": sequence_group_dims,
        "max_seq_feature_dim": max_seq_feature_dim,
        "seq_len": seq_len,
    }
    return non_seq_tensor, seq_tensor, label_tensor, metadata


def fallback_download(dataset_id: str, cache_dir: Path, retries: int = 3) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_files = list_repo_files(dataset_id, repo_type="dataset")
    parquet_files = [name for name in repo_files if name.endswith(".parquet")]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet file found under dataset repo {dataset_id}.")

    last_error: Exception | None = None
    for filename in parquet_files:
        for attempt in range(1, retries + 1):
            try:
                path = hf_hub_download(
                    repo_id=dataset_id,
                    repo_type="dataset",
                    filename=filename,
                    cache_dir=cache_dir,
                    etag_timeout=60,
                    resume_download=True,
                )
                return Path(path)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"[download] {filename} attempt {attempt}/{retries} failed: {exc}")
                time.sleep(attempt)

    raise RuntimeError(f"Failed to download parquet from {dataset_id}") from last_error


def load_train_split(dataset_id: str, cache_dir: Path, local_parquet: Path | None = None) -> Dataset:
    if local_parquet is not None:
        print(f"[data] reading local parquet: {local_parquet}")
        return Dataset.from_parquet(str(local_parquet))

    cache_dir.mkdir(parents=True, exist_ok=True)
    download_config = DownloadConfig(
        cache_dir=str(cache_dir),
        max_retries=5,
        resume_download=True,
    )

    try:
        print(f"[data] loading dataset via datasets.load_dataset: {dataset_id}")
        return load_dataset(dataset_id, split="train", download_config=download_config)
    except Exception as exc:  # noqa: BLE001
        print(f"[data] load_dataset failed, switching to hf_hub_download fallback: {exc}")

    parquet_path = fallback_download(dataset_id, cache_dir)
    print(f"[data] reading downloaded parquet: {parquet_path}")
    return Dataset.from_parquet(str(parquet_path))
