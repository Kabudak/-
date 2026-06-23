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


LABEL_COLUMN = "label_click"

# Mapping from group names in selectedfeaturefinal.txt to sequence branch names.
# A None value means the group is a non-sequence token group.
GROUP_TO_BRANCH = {
    "label": None,
    "context": None,
    "item": None,
    "impr": "impression_seq",
    "click": "click_seq",
    "buy": "buy_seq",
}

GROUP_TO_TOKEN = {
    "context": "context_token",
    "item": "item_token",
}

SEQUENCE_BRANCH_ORDER = [
    "click_seq",
    "impression_seq",
    "buy_seq",
]

# Features that are already log-transformed and should NOT have signed_log1p applied
ALREADY_LOGGED_FIELDS = {"log_all_impr_tg_1d"}
SUMMARY_STATS = ("length_log", "mean", "std", "min", "max", "last")

NS_SPARSE_BAG_FIELDS = {
    "flip_cat1_ids",
    "flip_goods_ids",
    "list_clk_cat_ids_l20_x",
    "list_clk_goods_ids",
    "list_clk_mall_ids",
    "goods_name_bigram_hash",
    "i2cat2_hit_ups_clk_tg",
    "i2i_hit_clk_ids",
    "i2i_hit_clk_ids_1d",
    "i2i_hit_clk_ids_3d",
    "i2i_hit_view_ids",
    "i2i_list_swingv3gmv",
}

NS_DENSE_FIXED_WIDTHS = {
    "goods_cos_clk_sim_dis_cut3": 3,
    "goods_cos_view_sim_dis_cut3": 3,
    "u_clk_cnt_mix_d_kpos": 3,
}

NS_DENSE_SUMMARY_FIELDS = {
    "i2cat2_hit_clk_timediff",
    "i2i_hit_clk_timediff_3d",
    "i2i_hit_clk_timediff_l10",
    "ups_clk_hit_coclk_i2i_rank",
    "ups_clk_hit_i2i_rank",
    "ups_clk_hit_i2i_rank_1d",
}


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
    non_seq_bag_len: int
    non_seq_sparse_fields: list[str]
    non_seq_sparse_bag_fields: list[str]
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


def trim_flat_values(value: Any, max_len: int, truncation: str) -> list[Any]:
    values = sequence_values(value)
    if len(values) <= max_len:
        return values
    if truncation == "head":
        return values[:max_len]
    return values[-max_len:]


def extract_scalar_array(values: list[Any], dtype: Any = np.int64) -> np.ndarray:
    output = np.zeros(len(values), dtype=dtype)
    for row_idx, value in enumerate(values):
        if dtype == np.float32:
            output[row_idx] = safe_float(first_scalar(value))
        else:
            output[row_idx] = safe_int(first_scalar(value))
    return output


def bucket_scalar_array(values: list[Any], bucket_size: int) -> np.ndarray:
    raw = extract_scalar_array(values, dtype=np.int64)
    return np.where(raw == 0, 0, np.abs(raw) % bucket_size + 1).astype(np.int32)


def extract_sparse_matrix(
    values: list[Any],
    width: int,
    truncation: str,
    bucket_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(values), width), dtype=np.int32)
    mask = np.zeros((len(values), width), dtype=bool)
    for row_idx, value in enumerate(values):
        row_values = trim_flat_values(value, width, truncation)
        if not row_values:
            continue
        length = len(row_values)
        raw = np.asarray([safe_int(item) for item in row_values], dtype=np.int64)
        bucketed = np.where(raw == 0, 0, np.abs(raw) % bucket_size + 1).astype(np.int32)
        matrix[row_idx, :length] = bucketed
        mask[row_idx, :length] = bucketed != 0
    return matrix, mask


def extract_dense_matrix(
    values: list[Any],
    width: int,
    truncation: str,
    already_logged: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(values), width), dtype=np.float32)
    mask = np.zeros((len(values), width), dtype=bool)
    for row_idx, value in enumerate(values):
        row_values = trim_flat_values(value, width, truncation)
        if not row_values:
            continue
        length = len(row_values)
        if already_logged:
            matrix[row_idx, :length] = np.asarray([safe_float(item) for item in row_values], dtype=np.float32)
        else:
            matrix[row_idx, :length] = np.asarray([signed_log1p(item) for item in row_values], dtype=np.float32)
        mask[row_idx, :length] = True
    return matrix, mask


def extract_dense_spec_array(values: list[Any], spec: DenseFeatureSpec) -> np.ndarray:
    output = np.zeros(len(values), dtype=np.float32)
    already_logged = spec.source in ALREADY_LOGGED_FIELDS
    if spec.stat == "scalar":
        for row_idx, value in enumerate(values):
            scalar = first_scalar(value)
            output[row_idx] = safe_float(scalar) if already_logged else signed_log1p(scalar)
        return output
    if spec.stat.startswith("pos"):
        pos_idx = int(spec.stat[3:])
        for row_idx, value in enumerate(values):
            row_values = flatten_values(value)
            if pos_idx < len(row_values):
                output[row_idx] = safe_float(row_values[pos_idx]) if already_logged else signed_log1p(row_values[pos_idx])
        return output
    for row_idx, value in enumerate(values):
        raw_values = flatten_values(value)
        output[row_idx] = dense_summary(raw_values, spec.stat)
    return output


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


def make_manual_token_groups(non_seq_group_fields: dict[str, list[str]]) -> dict[str, list[str]]:
    token_groups: dict[str, list[str]] = {}
    for group_name, fields in non_seq_group_fields.items():
        token_name = GROUP_TO_TOKEN.get(group_name)
        if token_name is not None and fields:
            token_groups[token_name] = fields
    return token_groups


def load_feature_schema(
    feature_file: Path,
    seq_len: int,
    sequence_truncation: str,
    non_seq_bag_len: int,
) -> ProductionFeatureSchema:
    """Load feature schema from a manually curated feature grouping file.

    The file format (selectedfeaturefinal.txt) uses section headers like
    ``context:``, ``item:``, ``impr:``, ``click:``, ``buy:`` followed by one
    feature name per line.  Blank lines and lines without a colon are ignored.
    """
    if sequence_truncation not in {"head", "tail"}:
        raise ValueError("sequence_truncation must be either 'head' or 'tail'")

    text = feature_file.read_text(encoding="utf-8")
    groups: dict[str, list[str]] = {}
    current_group: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(":"):
            current_group = line[:-1].strip().lower()
            groups.setdefault(current_group, [])
            continue
        if current_group is not None:
            groups[current_group].append(line)

    # Partition features into non-sequence vs. sequence branches
    if non_seq_bag_len <= 0:
        raise ValueError("non_seq_bag_len must be positive")

    non_seq_sparse_fields: list[str] = []
    non_seq_sparse_bag_fields: list[str] = []
    non_seq_dense_specs: list[DenseFeatureSpec] = []
    non_seq_group_fields: dict[str, list[str]] = {}
    sequence_fields: dict[str, list[str]] = {}

    for group_name, features in groups.items():
        if group_name not in GROUP_TO_BRANCH:
            raise ValueError(
                f"Unknown feature group '{group_name}' in {feature_file}. "
                f"Expected groups: {sorted(GROUP_TO_BRANCH.keys())}"
            )

        branch_name = GROUP_TO_BRANCH[group_name]
        if branch_name is None:
            # Non-sequence group (for example "context" or "item").
            group_fields: list[str] = []
            for feat in features:
                if feat == LABEL_COLUMN:
                    continue
                if feat in NS_SPARSE_BAG_FIELDS:
                    non_seq_sparse_bag_fields.append(feat)
                    group_fields.append(feat)
                elif feat in NS_DENSE_FIXED_WIDTHS:
                    for pos_idx in range(NS_DENSE_FIXED_WIDTHS[feat]):
                        spec_name = f"{feat}__pos{pos_idx}"
                        non_seq_dense_specs.append(
                            DenseFeatureSpec(name=spec_name, source=feat, stat=f"pos{pos_idx}")
                        )
                        group_fields.append(spec_name)
                elif feat in NS_DENSE_SUMMARY_FIELDS:
                    for stat in SUMMARY_STATS:
                        spec_name = f"{feat}__{stat}"
                        non_seq_dense_specs.append(
                            DenseFeatureSpec(name=spec_name, source=feat, stat=stat)
                        )
                        group_fields.append(spec_name)
                elif is_dense_name(feat):
                    non_seq_dense_specs.append(
                        DenseFeatureSpec(name=feat, source=feat, stat="scalar")
                    )
                    group_fields.append(feat)
                else:
                    non_seq_sparse_fields.append(feat)
                    group_fields.append(feat)
            if group_fields:
                non_seq_group_fields[group_name] = group_fields
        else:
            # Sequence branch
            branch_feats: list[str] = []
            for feat in features:
                if feat == LABEL_COLUMN:
                    continue
                branch_feats.append(feat)
                # Scalar dense fields in a sequence branch still go into the
                # sequence dense tensor (not a summary stat).
            if branch_feats:
                sequence_fields[branch_name] = branch_feats

    # Ensure branches follow canonical order
    sequence_fields = {
        name: sequence_fields[name]
        for name in SEQUENCE_BRANCH_ORDER
        if name in sequence_fields
    }
    if not sequence_fields:
        raise ValueError(
            f"No sequence features found in {feature_file}. "
            f"Expected groups: {sorted(GROUP_TO_BRANCH.keys())}"
        )

    # Classify sequence fields into sparse vs. dense
    seq_sparse_fields: list[str] = []
    seq_dense_fields: list[str] = []
    for fields in sequence_fields.values():
        for field in fields:
            target = seq_dense_fields if is_dense_name(field) else seq_sparse_fields
            if field not in target:
                target.append(field)

    non_seq_sparse_fields = sorted(non_seq_sparse_fields)
    non_seq_sparse_bag_fields = sorted(non_seq_sparse_bag_fields)
    non_seq_dense_specs = sorted(non_seq_dense_specs, key=lambda spec: spec.name)
    seq_sparse_fields = sorted(seq_sparse_fields)
    seq_dense_fields = sorted(seq_dense_fields)

    token_groups = make_manual_token_groups(non_seq_group_fields)
    if not token_groups:
        token_groups = make_token_groups(
            non_seq_sparse_fields=non_seq_sparse_fields,
            non_seq_dense_specs=non_seq_dense_specs,
        )
    sparse_field_cardinalities = {
        field: bucket_size_for_field(field) + 1
        for field in sorted(set(non_seq_sparse_fields) | set(non_seq_sparse_bag_fields) | set(seq_sparse_fields))
    }

    return ProductionFeatureSchema(
        label_mode="label_click",
        seq_len=seq_len,
        sequence_truncation=sequence_truncation,
        non_seq_bag_len=non_seq_bag_len,
        non_seq_sparse_fields=non_seq_sparse_fields,
        non_seq_sparse_bag_fields=non_seq_sparse_bag_fields,
        non_seq_dense_specs=non_seq_dense_specs,
        seq_sparse_fields=seq_sparse_fields,
        seq_dense_fields=seq_dense_fields,
        sequence_names=list(sequence_fields.keys()),
        sequence_fields=sequence_fields,
        token_groups=token_groups,
        sparse_field_cardinalities=sparse_field_cardinalities,
        column_infos=[],
    )


def label_from_columns(columns: dict[str, list[Any]], row_idx: int) -> int:
    """Extract binary label from label_click column."""
    raw = first_scalar(columns.get(LABEL_COLUMN, [None])[row_idx])
    return safe_int(raw)


def trim_values(values: list[Any], seq_len: int, sequence_truncation: str) -> list[Any]:
    if len(values) <= seq_len:
        return values
    if sequence_truncation == "head":
        return values[:seq_len]
    return values[-seq_len:]


def build_tensors(
    columns: dict[str, list[Any]],
    feature_file: Path,
    seq_len: int,
    sequence_truncation: str = "tail",
    non_seq_bag_len: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if not columns:
        raise ValueError("No columns were loaded from the production parquet.")
    row_count = len(next(iter(columns.values())))
    if row_count == 0:
        raise ValueError("The production parquet sample is empty.")

    schema = load_feature_schema(
        feature_file=feature_file,
        seq_len=seq_len,
        sequence_truncation=sequence_truncation,
        non_seq_bag_len=non_seq_bag_len,
    )

    non_seq_sparse_np = np.zeros((row_count, len(schema.non_seq_sparse_fields)), dtype=np.int32)
    non_seq_sparse_bag_np = np.zeros(
        (row_count, len(schema.non_seq_sparse_bag_fields), schema.non_seq_bag_len),
        dtype=np.int32,
    )
    non_seq_sparse_bag_mask_np = np.zeros(
        (row_count, len(schema.non_seq_sparse_bag_fields), schema.non_seq_bag_len),
        dtype=bool,
    )
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
    non_seq_sparse_bag_index = {field: idx for idx, field in enumerate(schema.non_seq_sparse_bag_fields)}
    seq_sparse_index = {field: idx for idx, field in enumerate(schema.seq_sparse_fields)}
    seq_dense_index = {field: idx for idx, field in enumerate(schema.seq_dense_fields)}

    labels_np = np.asarray([label_from_columns(columns, row_idx) for row_idx in range(row_count)], dtype=np.int64)

    for field, field_idx in non_seq_sparse_index.items():
        non_seq_sparse_np[:, field_idx] = bucket_scalar_array(columns[field], bucket_size_for_field(field))

    for field, field_idx in non_seq_sparse_bag_index.items():
        matrix, mask = extract_sparse_matrix(
            columns[field],
            width=schema.non_seq_bag_len,
            truncation=schema.sequence_truncation,
            bucket_size=bucket_size_for_field(field),
        )
        non_seq_sparse_bag_np[:, field_idx, :] = matrix
        non_seq_sparse_bag_mask_np[:, field_idx, :] = mask

    for dense_idx, spec in enumerate(schema.non_seq_dense_specs):
        non_seq_dense_np[:, dense_idx] = extract_dense_spec_array(columns[spec.source], spec)

    for branch_idx, branch_name in enumerate(schema.sequence_names):
        for field in schema.sequence_fields[branch_name]:
            if field in seq_sparse_index:
                matrix, mask = extract_sparse_matrix(
                    columns[field],
                    width=seq_len,
                    truncation=schema.sequence_truncation,
                    bucket_size=bucket_size_for_field(field),
                )
                seq_sparse_np[:, branch_idx, :, seq_sparse_index[field]] = matrix
                seq_mask_np[:, branch_idx, :] |= mask
            else:
                matrix, mask = extract_dense_matrix(
                    columns[field],
                    width=seq_len,
                    truncation=schema.sequence_truncation,
                    already_logged=field in ALREADY_LOGGED_FIELDS,
                )
                seq_dense_np[:, branch_idx, :, seq_dense_index[field]] = matrix
                seq_mask_np[:, branch_idx, :] |= mask

    label_counts = Counter(labels_np.tolist())
    sequence_non_empty = {
        branch_name: int(seq_mask_np[:, branch_idx, :].any(axis=1).sum())
        for branch_idx, branch_name in enumerate(schema.sequence_names)
    }
    metadata = {
        "dataset": "production_parquet",
        "feature_version": "production_field_aware_hyformer_v2",
        "label_column": LABEL_COLUMN,
        "label_mapping": {"0": 0, "1": 1},
        "num_samples": row_count,
        "positive_samples": int(label_counts.get(1, 0)),
        "negative_samples": int(label_counts.get(0, 0)),
        "pos_rate": round(int(label_counts.get(1, 0)) / max(row_count, 1), 6),
        "seq_len": schema.seq_len,
        "sequence_truncation": schema.sequence_truncation,
        "non_seq_bag_len": schema.non_seq_bag_len,
        "num_sequences": len(schema.sequence_names),
        "sequence_names": schema.sequence_names,
        "sequence_fields": schema.sequence_fields,
        "sequence_non_empty_samples": sequence_non_empty,
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
            "dense_fixed": dict(sorted(NS_DENSE_FIXED_WIDTHS.items())),
            "dense_summary": sorted(NS_DENSE_SUMMARY_FIELDS),
            "summary_stats": list(SUMMARY_STATS),
        },
        "notes": {
            "label": f"Binary label from {LABEL_COLUMN} column.",
            "feature_schema": f"Manually curated feature groups from {feature_file.name}.",
            "already_logged": f"Fields in already_logged_fields skip signed_log1p to avoid double-log.",
            "non_seq_sparse_bag": "Sparse array features are bucketized and mean-pooled inside the model.",
        },
    }

    return (
        torch.from_numpy(non_seq_sparse_np),
        torch.from_numpy(non_seq_sparse_bag_np),
        torch.from_numpy(non_seq_sparse_bag_mask_np),
        torch.from_numpy(non_seq_dense_np),
        torch.from_numpy(seq_sparse_np),
        torch.from_numpy(seq_dense_np),
        torch.from_numpy(seq_mask_np),
        torch.from_numpy(labels_np),
        metadata,
    )
