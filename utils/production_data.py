from __future__ import annotations

import math
import sys
from collections import Counter
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch


LABEL_COLUMNS = {"rel_level", "rel_score_bkt"}
LABEL_MODES = {
    "rel_score_present",
    "rel_score_positive",
    "rel_level_present",
    "rel_level_positive",
}
SEQUENCE_BRANCH_ORDER = [
    "search_seq",
    "click_seq",
    "view_seq",
    "cart_seq",
    "buy_seq",
    "impression_seq",
    "item_hit_seq",
]
SUMMARY_STATS = ("length_log", "mean", "std", "min", "max", "last")


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    value_type: str
    non_null_rows: int
    non_empty_rows: int
    max_flat_len: int
    mean_flat_len: float
    scalar_like: bool


@dataclass(frozen=True)
class DenseFeatureSpec:
    name: str
    source: str
    stat: str


@dataclass
class ProductionFeatureSchema:
    label_mode: str
    seq_len: int
    sequence_truncation: str
    non_seq_sparse_fields: list[str]
    non_seq_dense_specs: list[DenseFeatureSpec]
    seq_sparse_fields: list[str]
    seq_dense_fields: list[str]
    sequence_names: list[str]
    sequence_fields: dict[str, list[str]]
    token_groups: dict[str, list[str]]
    sparse_field_cardinalities: dict[str, int]
    column_infos: list[ColumnInfo]


def import_pyarrow_parquet() -> Any:
    try:
        import pyarrow.parquet as pq
        return pq
    except ImportError:
        pass

    candidates = [
        Path(sys.base_prefix) / "Lib" / "site-packages" / "pyarrow",
        Path(sys.base_exec_prefix) / "Lib" / "site-packages" / "pyarrow",
        Path("C:/Program Files/Python39/lib/site-packages/pyarrow"),
    ]
    for package_dir in candidates:
        init_file = package_dir / "__init__.py"
        if not init_file.exists():
            continue
        spec = importlib.util.spec_from_file_location(
            "pyarrow",
            init_file,
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules["pyarrow"] = module
        spec.loader.exec_module(module)
        import pyarrow.parquet as pq
        return pq

    raise ImportError(
        "Reading production parquet requires pyarrow. Install it into the active "
        "environment with: pip install pyarrow"
    )


def load_parquet_columns(path: Path, max_rows: int | None = None) -> tuple[dict[str, list[Any]], dict[str, str]]:
    pq = import_pyarrow_parquet()

    table = pq.read_table(path)
    if max_rows is not None:
        table = table.slice(0, max_rows)

    columns = {name: table[name].to_pylist() for name in table.column_names}
    arrow_types = {field.name: str(field.type) for field in table.schema}
    return columns, arrow_types


def flatten_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(flatten_values(item))
        return flattened
    return [value]


def sequence_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        return flatten_values(value[0])
    return flatten_values(value)


def first_scalar(value: Any) -> Any:
    if value is None:
        return None
    while isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    return value


def safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return 0.0
        try:
            numeric = float(value)
            if math.isnan(numeric) or math.isinf(numeric):
                return 0.0
            return numeric
        except ValueError:
            return 0.0
    return 0.0


def safe_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return 0
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return 0
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def signed_log1p(value: Any) -> float:
    numeric = safe_float(value)
    if numeric == 0.0:
        return 0.0
    return math.copysign(math.log1p(abs(numeric)), numeric)


def bucket_id(value: Any, bucket_size: int) -> int:
    numeric = safe_int(value)
    if numeric == 0:
        return 0
    return abs(numeric) % bucket_size + 1


def dense_summary(values: list[Any], stat: str) -> float:
    numeric_values = [signed_log1p(value) for value in values]
    if stat == "length_log":
        return math.log1p(len(numeric_values))
    if not numeric_values:
        return 0.0

    if stat == "mean":
        return float(sum(numeric_values) / len(numeric_values))
    if stat == "std":
        mean = sum(numeric_values) / len(numeric_values)
        variance = sum((value - mean) ** 2 for value in numeric_values) / len(numeric_values)
        return float(math.sqrt(variance))
    if stat == "min":
        return float(min(numeric_values))
    if stat == "max":
        return float(max(numeric_values))
    if stat == "last":
        return float(numeric_values[-1])
    raise ValueError(f"Unsupported summary stat: {stat}")


def infer_column_infos(columns: dict[str, list[Any]], arrow_types: dict[str, str]) -> list[ColumnInfo]:
    infos: list[ColumnInfo] = []
    for name, values in columns.items():
        flat_lens = [len(flatten_values(value)) for value in values]
        non_null_rows = sum(value is not None for value in values)
        non_empty_rows = sum(length > 0 for length in flat_lens)
        max_flat_len = max(flat_lens, default=0)
        mean_flat_len = sum(flat_lens) / max(len(flat_lens), 1)
        scalar_like = max_flat_len <= 1
        infos.append(
            ColumnInfo(
                name=name,
                value_type=arrow_types.get(name, "unknown"),
                non_null_rows=non_null_rows,
                non_empty_rows=non_empty_rows,
                max_flat_len=max_flat_len,
                mean_flat_len=mean_flat_len,
                scalar_like=scalar_like,
            )
        )
    return infos


def is_dense_name(name: str) -> bool:
    low = name.lower()
    dense_parts = (
        "cnt",
        "count",
        "num",
        "rate",
        "score",
        "ctr",
        "cvr",
        "sales",
        "timediff",
        "diff",
        "gap",
        "rank",
        "idx",
        "index",
        "level",
        "bkt",
        "len",
        "length",
        "size",
        "sim",
        "dis",
        "dist",
        "val",
        "ratio",
    )
    sparse_parts = ("id", "ids", "hash")
    if any(part in low for part in sparse_parts):
        return False
    if low in {"cat1_id", "cat2_id", "cat3_id", "cat4_id", "mall_id", "goods_id", "user_id"}:
        return False
    return any(part in low for part in dense_parts)


def bucket_size_for_field(name: str) -> int:
    low = name.lower()
    if low == "user_id":
        return 1_048_576
    if "goods" in low or "mall" in low or "hash" in low:
        return 1_048_576
    if "cat" in low:
        return 262_144
    if low in {"site_id", "scene_id", "page_sn", "page_elsn", "bg_id", "opt_id"}:
        return 65_536
    if low in {"language", "currency", "region", "timezone", "plat", "search_method"}:
        return 4_096
    return 262_144


def sequence_branch_for_column(name: str, scalar_like: bool) -> str | None:
    if scalar_like:
        return None

    low = name.lower()
    if low in {"query_hash", "query_terms_hash", "query_len", "query_cat", "query_cat3", "query_cat4"}:
        return None
    if low.startswith("last_query") or low.startswith("sess_q2q") or low.startswith("log_query"):
        return "search_seq"
    if low.startswith("ups_cart") or "cart" in low:
        return "cart_seq"
    if low.startswith("ups_buy") or "buy" in low:
        return "buy_seq"
    if low.startswith("list_clk") or low.startswith("ups_clk") or low.startswith("ups_clkv2") or "_clk" in low:
        return "click_seq"
    if low.startswith("last_view") or low.startswith("ups_view") or "_view" in low:
        return "view_seq"
    if "impr" in low or low.startswith("pagesn") or low.startswith("cur_pagesn"):
        return "impression_seq"
    if low.startswith("i2i") or "_hit_" in low or low.endswith("_hit_val"):
        return "item_hit_seq"
    return None


def token_group_for_feature(name: str) -> str:
    source_name = name.split("__", 1)[0]
    low = source_name.lower()
    if low == "user_id":
        return "user_token"
    if low in {
        "site_id",
        "scene_id",
        "language",
        "currency",
        "region",
        "timezone",
        "plat",
        "page_sn",
        "page_elsn",
        "search_method",
        "bg_id",
        "opt_id",
    }:
        return "context_token"
    if low.startswith("query") or low.startswith("origin_query") or low.startswith("reduction_query"):
        return "query_token"
    if low.startswith("q2") or low.startswith("q_") or low.startswith("sess_q2q"):
        return "query_token"
    if "goods" in low or "mall" in low or low.startswith("cat") or low.startswith("req_") or low.startswith("main_"):
        return "candidate_token"
    if low.startswith("u2") or low.startswith("i2") or low.startswith("site_") or low.startswith("target_"):
        return "cross_token"
    if low.startswith(("ups_", "list_", "last_", "log_all_", "pagesn", "cur_pagesn")):
        return "history_summary_token"
    return "misc_token"


def make_token_groups(non_seq_sparse_fields: list[str], non_seq_dense_specs: list[DenseFeatureSpec]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for name in non_seq_sparse_fields:
        grouped.setdefault(token_group_for_feature(name), []).append(name)
    for spec in non_seq_dense_specs:
        grouped.setdefault(token_group_for_feature(spec.name), []).append(spec.name)

    preferred_order = [
        "user_token",
        "context_token",
        "query_token",
        "candidate_token",
        "cross_token",
        "history_summary_token",
        "misc_token",
    ]
    return {name: grouped[name] for name in preferred_order if grouped.get(name)}


def select_sequence_fields(
    sequence_fields: dict[str, list[str]],
    column_info_by_name: dict[str, ColumnInfo],
    max_seq_fields_per_branch: int | None,
) -> dict[str, list[str]]:
    if max_seq_fields_per_branch is None:
        return sequence_fields

    selected: dict[str, list[str]] = {}
    for branch_name, fields in sequence_fields.items():
        ranked = sorted(
            fields,
            key=lambda field: (
                column_info_by_name[field].non_empty_rows,
                column_info_by_name[field].mean_flat_len,
                field,
            ),
            reverse=True,
        )
        selected[branch_name] = sorted(ranked[:max_seq_fields_per_branch])
    return selected


def infer_feature_schema(
    columns: dict[str, list[Any]],
    arrow_types: dict[str, str],
    seq_len: int,
    label_mode: str,
    sequence_truncation: str,
    max_seq_fields_per_branch: int | None = None,
) -> ProductionFeatureSchema:
    if label_mode not in LABEL_MODES:
        raise ValueError(f"Unsupported label_mode={label_mode}. Valid modes: {sorted(LABEL_MODES)}")
    if sequence_truncation not in {"head", "tail"}:
        raise ValueError("sequence_truncation must be either 'head' or 'tail'")

    column_infos = infer_column_infos(columns, arrow_types)
    info_by_name = {info.name: info for info in column_infos}
    sequence_fields: dict[str, list[str]] = {name: [] for name in SEQUENCE_BRANCH_ORDER}
    non_seq_sparse_fields: list[str] = []
    non_seq_dense_specs: list[DenseFeatureSpec] = []

    for info in column_infos:
        if info.name in LABEL_COLUMNS or info.non_empty_rows == 0:
            continue

        branch_name = sequence_branch_for_column(info.name, info.scalar_like)
        if branch_name is not None:
            sequence_fields[branch_name].append(info.name)
            for stat in SUMMARY_STATS:
                non_seq_dense_specs.append(DenseFeatureSpec(name=f"{info.name}__{stat}", source=info.name, stat=stat))
            continue

        if info.scalar_like:
            if is_dense_name(info.name):
                non_seq_dense_specs.append(DenseFeatureSpec(name=info.name, source=info.name, stat="scalar"))
            else:
                non_seq_sparse_fields.append(info.name)
            continue

        for stat in SUMMARY_STATS:
            non_seq_dense_specs.append(DenseFeatureSpec(name=f"{info.name}__{stat}", source=info.name, stat=stat))

    sequence_fields = {
        branch_name: fields
        for branch_name, fields in sequence_fields.items()
        if fields
    }
    sequence_fields = select_sequence_fields(sequence_fields, info_by_name, max_seq_fields_per_branch)
    sequence_fields = {name: fields for name, fields in sequence_fields.items() if fields}
    if not sequence_fields:
        raise ValueError("No behavior sequence fields were inferred from the parquet sample.")

    seq_sparse_fields: list[str] = []
    seq_dense_fields: list[str] = []
    for fields in sequence_fields.values():
        for field in fields:
            target = seq_dense_fields if is_dense_name(field) else seq_sparse_fields
            if field not in target:
                target.append(field)

    non_seq_sparse_fields = sorted(non_seq_sparse_fields)
    non_seq_dense_specs = sorted(non_seq_dense_specs, key=lambda spec: spec.name)
    seq_sparse_fields = sorted(seq_sparse_fields)
    seq_dense_fields = sorted(seq_dense_fields)
    sequence_fields = {
        branch_name: sorted(sequence_fields[branch_name])
        for branch_name in SEQUENCE_BRANCH_ORDER
        if branch_name in sequence_fields
    }
    token_groups = make_token_groups(
        non_seq_sparse_fields=non_seq_sparse_fields,
        non_seq_dense_specs=non_seq_dense_specs,
    )
    sparse_field_cardinalities = {
        field: bucket_size_for_field(field) + 1
        for field in sorted(set(non_seq_sparse_fields) | set(seq_sparse_fields))
    }

    return ProductionFeatureSchema(
        label_mode=label_mode,
        seq_len=seq_len,
        sequence_truncation=sequence_truncation,
        non_seq_sparse_fields=non_seq_sparse_fields,
        non_seq_dense_specs=non_seq_dense_specs,
        seq_sparse_fields=seq_sparse_fields,
        seq_dense_fields=seq_dense_fields,
        sequence_names=list(sequence_fields.keys()),
        sequence_fields=sequence_fields,
        token_groups=token_groups,
        sparse_field_cardinalities=sparse_field_cardinalities,
        column_infos=column_infos,
    )


def label_from_columns(columns: dict[str, list[Any]], row_idx: int, label_mode: str) -> int:
    rel_score = first_scalar(columns.get("rel_score_bkt", [None])[row_idx])
    rel_level = first_scalar(columns.get("rel_level", [None])[row_idx])

    if label_mode == "rel_score_present":
        return int(rel_score is not None and safe_int(rel_score) >= 0)
    if label_mode == "rel_score_positive":
        return int(rel_score is not None and safe_int(rel_score) > 0)
    if label_mode == "rel_level_present":
        return int(rel_level is not None)
    if label_mode == "rel_level_positive":
        return int(rel_level is not None and safe_int(rel_level) > 0)
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def trim_values(values: list[Any], seq_len: int, sequence_truncation: str) -> list[Any]:
    if len(values) <= seq_len:
        return values
    if sequence_truncation == "head":
        return values[:seq_len]
    return values[-seq_len:]


def build_tensors(
    columns: dict[str, list[Any]],
    arrow_types: dict[str, str],
    seq_len: int,
    label_mode: str = "rel_score_present",
    sequence_truncation: str = "tail",
    max_seq_fields_per_branch: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not columns:
        raise ValueError("No columns were loaded from the production parquet.")
    row_count = len(next(iter(columns.values())))
    if row_count == 0:
        raise ValueError("The production parquet sample is empty.")

    schema = infer_feature_schema(
        columns=columns,
        arrow_types=arrow_types,
        seq_len=seq_len,
        label_mode=label_mode,
        sequence_truncation=sequence_truncation,
        max_seq_fields_per_branch=max_seq_fields_per_branch,
    )

    non_seq_sparse_np = np.zeros((row_count, len(schema.non_seq_sparse_fields)), dtype=np.int32)
    non_seq_dense_np = np.zeros((row_count, len(schema.non_seq_dense_specs)), dtype=np.float32)
    seq_sparse_np = np.zeros(
        (row_count, len(schema.sequence_names), seq_len, len(schema.seq_sparse_fields)),
        dtype=np.int32,
    )
    seq_dense_np = np.zeros(
        (row_count, len(schema.sequence_names), seq_len, len(schema.seq_dense_fields)),
        dtype=np.float32,
    )
    seq_mask_np = np.zeros((row_count, len(schema.sequence_names), seq_len), dtype=bool)
    labels_np = np.zeros(row_count, dtype=np.int64)

    non_seq_sparse_index = {field: idx for idx, field in enumerate(schema.non_seq_sparse_fields)}
    seq_sparse_index = {field: idx for idx, field in enumerate(schema.seq_sparse_fields)}
    seq_dense_index = {field: idx for idx, field in enumerate(schema.seq_dense_fields)}

    for row_idx in range(row_count):
        labels_np[row_idx] = label_from_columns(columns, row_idx, schema.label_mode)

        for field in schema.non_seq_sparse_fields:
            raw_value = first_scalar(columns[field][row_idx])
            non_seq_sparse_np[row_idx, non_seq_sparse_index[field]] = bucket_id(
                raw_value,
                bucket_size_for_field(field),
            )

        for dense_idx, spec in enumerate(schema.non_seq_dense_specs):
            raw_value = columns[spec.source][row_idx]
            if spec.stat == "scalar":
                non_seq_dense_np[row_idx, dense_idx] = signed_log1p(first_scalar(raw_value))
            else:
                non_seq_dense_np[row_idx, dense_idx] = dense_summary(flatten_values(raw_value), spec.stat)

        for branch_idx, branch_name in enumerate(schema.sequence_names):
            for field in schema.sequence_fields[branch_name]:
                values = trim_values(sequence_values(columns[field][row_idx]), seq_len, schema.sequence_truncation)
                if not values:
                    continue
                seq_mask_np[row_idx, branch_idx, : len(values)] = True
                if field in seq_sparse_index:
                    field_idx = seq_sparse_index[field]
                    bucket_size = bucket_size_for_field(field)
                    for step_idx, value in enumerate(values):
                        seq_sparse_np[row_idx, branch_idx, step_idx, field_idx] = bucket_id(value, bucket_size)
                else:
                    field_idx = seq_dense_index[field]
                    for step_idx, value in enumerate(values):
                        seq_dense_np[row_idx, branch_idx, step_idx, field_idx] = signed_log1p(value)

    label_counts = Counter(labels_np.tolist())
    sequence_non_empty = {
        branch_name: int(seq_mask_np[:, branch_idx, :].any(axis=1).sum())
        for branch_idx, branch_name in enumerate(schema.sequence_names)
    }
    metadata = {
        "dataset": "production_parquet",
        "feature_version": "production_field_aware_hyformer_v1",
        "label_mode": schema.label_mode,
        "label_mapping": {"0": 0, "1": 1},
        "num_samples": row_count,
        "positive_samples": int(label_counts.get(1, 0)),
        "negative_samples": int(label_counts.get(0, 0)),
        "pos_rate": round(int(label_counts.get(1, 0)) / max(row_count, 1), 6),
        "seq_len": schema.seq_len,
        "sequence_truncation": schema.sequence_truncation,
        "num_sequences": len(schema.sequence_names),
        "sequence_names": schema.sequence_names,
        "sequence_fields": schema.sequence_fields,
        "sequence_non_empty_samples": sequence_non_empty,
        "non_seq_sparse_fields": schema.non_seq_sparse_fields,
        "non_seq_dense_fields": [spec.name for spec in schema.non_seq_dense_specs],
        "non_seq_dense_sources": [
            {"name": spec.name, "source": spec.source, "stat": spec.stat}
            for spec in schema.non_seq_dense_specs
        ],
        "seq_sparse_fields": schema.seq_sparse_fields,
        "seq_dense_fields": schema.seq_dense_fields,
        "sparse_field_cardinalities": schema.sparse_field_cardinalities,
        "token_groups": schema.token_groups,
        "column_overview": [
            {
                "name": info.name,
                "type": info.value_type,
                "non_null_rows": info.non_null_rows,
                "non_empty_rows": info.non_empty_rows,
                "max_flat_len": info.max_flat_len,
                "mean_flat_len": round(info.mean_flat_len, 4),
                "scalar_like": info.scalar_like,
            }
            for info in schema.column_infos
        ],
        "notes": {
            "label": "Default label treats rel_score_bkt >= 0 as positive and -1/missing as negative.",
            "sequence_grouping": "Behavior columns are grouped by production-style name prefixes instead of reconstructing histories.",
            "query_generation": "Non-sequence semantic tokens and pooled sequence summaries are used by the model to generate branch queries.",
        },
    }

    return (
        torch.from_numpy(non_seq_sparse_np),
        torch.from_numpy(non_seq_dense_np),
        torch.from_numpy(seq_sparse_np),
        torch.from_numpy(seq_dense_np),
        torch.from_numpy(seq_mask_np),
        torch.from_numpy(labels_np),
        metadata,
    )
