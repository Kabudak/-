

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUPPORTED_NUM_SEQUENCES = 2
DEFAULT_DEDUP_WINDOW_SEC = 5
TZ_OFFSET_SEC = 8 * 3600

NON_SEQ_SPARSE_FIELDS = [
    "user",
    "adgroup_id",
    "cate_id",
    "campaign_id",
    "customer",
    "brand",
    "pid_type",
    "pid_id",
    "final_gender_code",
    "age_level",
    "pvalue_level",
    "shopping_level",
    "occupation",
    "new_user_class_level",
]

NON_SEQ_DENSE_FIELDS = [
    "price_log",
    "hour_of_day",
    "day_of_week",
    "click_hist_len_log",
    "exposure_hist_len_log",
    "click_last_gap_log",
    "exposure_last_gap_log",
    "click_same_ad_count_log",
    "exposure_same_ad_count_log",
    "click_same_cate_count_log",
    "exposure_same_cate_count_log",
    "click_same_brand_count_log",
    "exposure_same_brand_count_log",
    "click_same_campaign_count_log",
    "exposure_same_campaign_count_log",
    "click_same_customer_count_log",
    "exposure_same_customer_count_log",
    "event_dup_count_log",
    "event_cluster_span_log",
]

SEQ_SPARSE_FIELDS = [
    "adgroup_id",
    "cate_id",
    "campaign_id",
    "customer",
    "brand",
    "pid_type",
    "pid_id",
]

SEQ_DENSE_FIELDS = [
    "price_log",
    "price_delta_log",
    "price_ratio_log",
    "time_gap_log",
    "same_ad_as_target",
    "same_cate_as_target",
    "same_brand_as_target",
    "same_campaign_as_target",
    "same_customer_as_target",
    "recency_rank_log",
    "relative_position",
]

SPARSE_FIELD_BUCKETS = {
    "user": 524_288,
    "adgroup_id": 524_288,
    "cate_id": 131_072,
    "campaign_id": 262_144,
    "customer": 262_144,
    "brand": 262_144,
    "pid_type": 4_096,
    "pid_id": 65_536,
    "final_gender_code": 64,
    "age_level": 64,
    "pvalue_level": 64,
    "shopping_level": 64,
    "occupation": 64,
    "new_user_class_level": 64,
}

TOKEN_GROUPS = {
    "user_profile_token": [
        "final_gender_code",
        "age_level",
        "pvalue_level",
        "shopping_level",
        "occupation",
        "new_user_class_level",
    ],
    "user_identity_token": ["user"],
    "target_ad_identity_token": ["adgroup_id"],
    "target_ad_attribute_token": ["cate_id", "campaign_id", "customer", "brand"],
    "target_price_token": ["price_log"],
    "context_token": ["pid_type", "pid_id", "hour_of_day", "day_of_week"],
    "history_summary_token_click": [
        "click_hist_len_log",
        "click_last_gap_log",
        "click_same_ad_count_log",
        "click_same_cate_count_log",
        "click_same_brand_count_log",
        "click_same_campaign_count_log",
        "click_same_customer_count_log",
    ],
    "history_summary_token_exposure": [
        "exposure_hist_len_log",
        "exposure_last_gap_log",
        "exposure_same_ad_count_log",
        "exposure_same_cate_count_log",
        "exposure_same_brand_count_log",
        "exposure_same_campaign_count_log",
        "exposure_same_customer_count_log",
    ],
    "current_event_token": ["event_dup_count_log", "event_cluster_span_log"],
}


@dataclass
class UserHistory:
    click_ts: list[int]
    click_events: list[tuple[int, int, int, int, int, float, int, int]]
    expose_ts: list[int]
    expose_events: list[tuple[int, int, int, int, int, float, int, int]]


def safe_int(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
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


def safe_float(value: object) -> float:
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


def log1p_nonneg(value: object) -> float:
    numeric = safe_float(value)
    return math.log1p(max(numeric, 0.0))


def signed_log1p(value: object) -> float:
    numeric = safe_float(value)
    if numeric == 0.0:
        return 0.0
    return math.copysign(math.log1p(abs(numeric)), numeric)


def stable_bucket_id(value: object, bucket_size: int) -> int:
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    numeric = safe_int(value)
    if numeric == 0:
        return 0
    return abs(numeric) % bucket_size + 1


def deduplicate_raw_sample(raw_sample: pd.DataFrame, dedup_window_sec: int) -> tuple[pd.DataFrame, dict[str, float]]:
    print(f"{dedup_window_sec} ")

    df = raw_sample.sort_values(["user", "adgroup_id", "pid_type", "pid_id", "time_stamp"]).reset_index(drop=True)
    prev_keys = df[["user", "adgroup_id", "pid_type", "pid_id"]].shift()
    same_key = (
        (df["user"] == prev_keys["user"])
        & (df["adgroup_id"] == prev_keys["adgroup_id"])
        & (df["pid_type"] == prev_keys["pid_type"])
        & (df["pid_id"] == prev_keys["pid_id"])
    )
    prev_ts = df["time_stamp"].shift()
    time_gap = (df["time_stamp"] - prev_ts).fillna(dedup_window_sec + 1)
    same_cluster = same_key & (time_gap <= dedup_window_sec)
    cluster_id = (~same_cluster).cumsum()

    deduped = (
        df.groupby(cluster_id, sort=False)
        .agg(
            user=("user", "first"),
            time_stamp=("time_stamp", "min"),
            adgroup_id=("adgroup_id", "first"),
            pid_type=("pid_type", "first"),
            pid_id=("pid_id", "first"),
            label=("label", "max"),
            dup_count=("label", "size"),
            cluster_span_sec=("time_stamp", lambda col: int(col.max() - col.min())),
        )
        .reset_index(drop=True)
    )

    before = len(df)
    after = len(deduped)
    avg_cluster_size = float(deduped["dup_count"].mean()) if after else 0.0
    removed_ratio = 0.0 if before == 0 else 1.0 - after / before

    print(f"  {before:,}")
    print(f"  {after:,}")
    print(f"  {avg_cluster_size:.4f}")
    print(f"   {removed_ratio * 100:.2f}%")

    stats = {
        "dedup_window_sec": dedup_window_sec,
        "rows_before_dedup": before,
        "rows_after_dedup": after,
        "removed_ratio": round(removed_ratio, 6),
        "avg_cluster_size": round(avg_cluster_size, 6),
        "max_cluster_size": int(deduped["dup_count"].max()) if after else 0,
    }
    return deduped, stats


def load_raw_data(data_dir: Path, dedup_window_sec: int, max_raw_rows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    raw_sample = pd.read_csv(data_dir / "raw_sample.csv", nrows=max_raw_rows)
    ad_feature = pd.read_csv(data_dir / "ad_feature.csv")
    user_profile = pd.read_csv(data_dir / "user_profile.csv")
    if max_raw_rows is not None:
        print(f"  raw_sample smoke rows: {len(raw_sample):,}")

    raw_sample["time_stamp"] = raw_sample["time_stamp"].astype(np.int64)
    raw_sample["user"] = raw_sample["user"].astype(np.int64)
    raw_sample["adgroup_id"] = raw_sample["adgroup_id"].astype(np.int64)
    pid_split = raw_sample["pid"].str.split("_", expand=True)
    raw_sample["pid_type"] = pid_split[0].astype(np.int64)
    raw_sample["pid_id"] = pid_split[1].astype(np.int64)
    raw_sample.drop(columns=["pid", "nonclk"], inplace=True)
    raw_sample.rename(columns={"clk": "label"}, inplace=True)

    if max_raw_rows is not None and max_raw_rows < len(raw_sample):
        raw_sample = raw_sample.sample(n=max_raw_rows, random_state=42).reset_index(drop=True)
        print(f"  raw_sample: {max_raw_rows:,} ")

    deduped_raw_sample, dedup_stats = deduplicate_raw_sample(raw_sample, dedup_window_sec)

    ad_feature["adgroup_id"] = ad_feature["adgroup_id"].astype(np.int64)
    ad_feature["cate_id"] = ad_feature["cate_id"].astype(np.int64)
    ad_feature["campaign_id"] = ad_feature["campaign_id"].astype(np.int64)
    ad_feature["customer"] = ad_feature["customer"].astype(np.int64)
    ad_feature["brand"] = ad_feature["brand"].replace("NULL", np.nan)
    ad_feature["brand"] = ad_feature["brand"].fillna(0).astype(np.int64)
    ad_feature["price"] = pd.to_numeric(ad_feature["price"], errors="coerce").fillna(0.0)

    user_profile.columns = user_profile.columns.str.strip()
    user_profile["userid"] = user_profile["userid"].astype(np.int64)
    for col in [
        "cms_segid",
        "cms_group_id",
        "final_gender_code",
        "age_level",
        "pvalue_level",
        "shopping_level",
        "occupation",
        "new_user_class_level",
    ]:
        user_profile[col] = pd.to_numeric(user_profile[col], errors="coerce").fillna(0).astype(np.int64)

    print(f"  raw_sample: {len(deduped_raw_sample):,} ")
    print(f"  ad_feature: {len(ad_feature):,} ")
    print(f"  user_profile: {len(user_profile):,} ")
    return deduped_raw_sample, ad_feature, user_profile, dedup_stats


def join_tables(raw_sample: pd.DataFrame, ad_feature: pd.DataFrame, user_profile: pd.DataFrame) -> pd.DataFrame:

    df = raw_sample.merge(ad_feature, on="adgroup_id", how="left")
    df = df.merge(user_profile, left_on="user", right_on="userid", how="left")

    if "userid" in df.columns:
        df.drop(columns=["userid"], inplace=True)

    for col in ["cate_id", "campaign_id", "customer", "brand"]:
        df[col] = df[col].fillna(0).astype(np.int64)
    df["price"] = df["price"].fillna(0.0)
    for col in [
        "final_gender_code",
        "age_level",
        "pvalue_level",
        "shopping_level",
        "occupation",
        "new_user_class_level",
    ]:
        df[col] = df[col].fillna(0).astype(np.int64)

    print(f"   {len(df):,}")
    print(f"   {df.shape[1]}")
    return df


def build_user_behavior_sequences(df: pd.DataFrame) -> dict[int, UserHistory]:

    histories: dict[int, UserHistory] = {}
    event_cols = ["adgroup_id", "cate_id", "campaign_id", "customer", "brand", "price", "pid_type", "pid_id"]

    click_df = df[df["label"] == 1][["user", "time_stamp"] + event_cols].copy()
    click_df.sort_values(["user", "time_stamp"], inplace=True)
    for uid, group in click_df.groupby("user", sort=False):
        histories[int(uid)] = UserHistory(
            click_ts=group["time_stamp"].astype(np.int64).tolist(),
            click_events=list(group[event_cols].itertuples(index=False, name=None)),
            expose_ts=[],
            expose_events=[],
        )

    expose_df = df[df["label"] == 0][["user", "time_stamp"] + event_cols].copy()
    expose_df.sort_values(["user", "time_stamp"], inplace=True)
    for uid, group in expose_df.groupby("user", sort=False):
        uid = int(uid)
        history = histories.get(uid)
        if history is None:
            histories[uid] = UserHistory(
                click_ts=[],
                click_events=[],
                expose_ts=group["time_stamp"].astype(np.int64).tolist(),
                expose_events=list(group[event_cols].itertuples(index=False, name=None)),
            )
        else:
            history.expose_ts = group["time_stamp"].astype(np.int64).tolist()
            history.expose_events = list(group[event_cols].itertuples(index=False, name=None))

    users_with_click = sum(1 for hist in histories.values() if hist.click_ts)
    users_with_expose = sum(1 for hist in histories.values() if hist.expose_ts)
    print(f" {len(histories):,}")
    print(f"  click : {users_with_click:,}")
    print(f"  exposure : {users_with_expose:,}")
    return histories


def encode_sparse_values(values: dict[str, object], field_names: list[str]) -> list[int]:
    return [stable_bucket_id(values[field], SPARSE_FIELD_BUCKETS[field]) for field in field_names]


def encode_seq_sparse_event(event: tuple[int, int, int, int, int, float, int, int]) -> list[int]:
    (
        adgroup_id,
        cate_id,
        campaign_id,
        customer,
        brand,
        _price,
        pid_type,
        pid_id,
    ) = event
    values = {
        "adgroup_id": adgroup_id,
        "cate_id": cate_id,
        "campaign_id": campaign_id,
        "customer": customer,
        "brand": brand,
        "pid_type": pid_type,
        "pid_id": pid_id,
    }
    return [stable_bucket_id(values[field], SPARSE_FIELD_BUCKETS[field]) for field in SEQ_SPARSE_FIELDS]


def summarize_history(
    history_events: list[tuple[int, int, int, int, int, float, int, int]],
    history_ts: list[int],
    history_count: int,
    current_ts: int,
    target_adgroup: int,
    target_cate: int,
    target_campaign: int,
    target_customer: int,
    target_brand: int,
) -> tuple[float, float, float, float, float, float, float]:
    hist_len_log = log1p_nonneg(history_count)
    if history_ts:
        last_gap_log = log1p_nonneg(current_ts - history_ts[-1])
    else:
        last_gap_log = 0.0

    same_ad_count = 0
    same_cate_count = 0
    same_brand_count = 0
    same_campaign_count = 0
    same_customer_count = 0
    for event in history_events:
        adgroup_id, cate_id, campaign_id, customer, brand, _, _, _ = event
        if adgroup_id == target_adgroup:
            same_ad_count += 1
        if cate_id == target_cate:
            same_cate_count += 1
        if brand == target_brand:
            same_brand_count += 1
        if campaign_id == target_campaign:
            same_campaign_count += 1
        if customer == target_customer:
            same_customer_count += 1

    return (
        hist_len_log,
        last_gap_log,
        log1p_nonneg(same_ad_count),
        log1p_nonneg(same_cate_count),
        log1p_nonneg(same_brand_count),
        log1p_nonneg(same_campaign_count),
        log1p_nonneg(same_customer_count),
    )


def select_evenly_spaced_rows(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or max_rows >= len(df):
        return df
    indices = np.linspace(0, len(df) - 1, num=max_rows, dtype=np.int64)
    indices = np.unique(indices)
    return df.iloc[indices].reset_index(drop=True)


def fill_sequence_branch(
    seq_sparse_tensor: np.ndarray,
    seq_dense_tensor: np.ndarray,
    seq_mask_tensor: np.ndarray,
    row_idx: int,
    branch_idx: int,
    selected_events: list[tuple[int, int, int, int, int, float, int, int]],
    selected_ts: list[int],
    current_ts: int,
    target_adgroup: int,
    target_cate: int,
    target_campaign: int,
    target_customer: int,
    target_brand: int,
    target_price: float,
) -> None:
    selected_len = len(selected_events)
    for step_idx, (event, hist_ts) in enumerate(zip(selected_events, selected_ts)):
        (
            hist_adgroup,
            hist_cate,
            hist_campaign,
            hist_customer,
            hist_brand,
            hist_price,
            _hist_pid_type,
            _hist_pid_id,
        ) = event
        hist_price = safe_float(hist_price)
        target_price = safe_float(target_price)
        price_ratio_log = math.log1p(hist_price / target_price) if target_price > 0.0 else 0.0
        relative_position = 0.0 if selected_len <= 1 else step_idx / (selected_len - 1)
        seq_sparse_tensor[row_idx, branch_idx, step_idx, :] = np.asarray(encode_seq_sparse_event(event), dtype=np.int64)
        seq_dense_tensor[row_idx, branch_idx, step_idx, :] = np.asarray(
            [
                log1p_nonneg(hist_price),
                signed_log1p(hist_price - target_price),
                price_ratio_log,
                log1p_nonneg(current_ts - hist_ts),
                float(hist_adgroup == target_adgroup),
                float(hist_cate == target_cate),
                float(hist_brand == target_brand),
                float(hist_campaign == target_campaign),
                float(hist_customer == target_customer),
                log1p_nonneg(selected_len - step_idx),
                float(relative_position),
            ],
            dtype=np.float32,
        )
        seq_mask_tensor[row_idx, branch_idx, step_idx] = True


def vectorize_dataset(
    df: pd.DataFrame,
    user_histories: dict[int, UserHistory],
    seq_len: int,
    num_sequences: int,
    max_rows: int | None,
    dedup_stats: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    if num_sequences != SUPPORTED_NUM_SEQUENCES:
        raise ValueError(
            f" {SUPPORTED_NUM_SEQUENCES} "
            f" num_sequences={num_sequences}。"
        )


    df = df.sort_values("time_stamp").reset_index(drop=True)
    df = select_evenly_spaced_rows(df, max_rows).copy()

    n = len(df)
    print(f"  {n:,}")

    non_seq_sparse_np = np.zeros((n, len(NON_SEQ_SPARSE_FIELDS)), dtype=np.int64)
    non_seq_dense_np = np.zeros((n, len(NON_SEQ_DENSE_FIELDS)), dtype=np.float32)
    seq_sparse_np = np.zeros((n, num_sequences, seq_len, len(SEQ_SPARSE_FIELDS)), dtype=np.int64)
    seq_dense_np = np.zeros((n, num_sequences, seq_len, len(SEQ_DENSE_FIELDS)), dtype=np.float32)
    seq_mask_np = np.zeros((n, num_sequences, seq_len), dtype=bool)
    labels_np = df["label"].astype(np.int64).to_numpy()
    timestamps_np = df["time_stamp"].astype(np.int64).to_numpy()

    users_np = df["user"].astype(np.int64).to_numpy()
    target_adgroup_np = df["adgroup_id"].astype(np.int64).to_numpy()
    target_cate_np = df["cate_id"].astype(np.int64).to_numpy()
    target_campaign_np = df["campaign_id"].astype(np.int64).to_numpy()
    target_customer_np = df["customer"].astype(np.int64).to_numpy()
    target_brand_np = df["brand"].astype(np.int64).to_numpy()
    profile_gender_np = df["final_gender_code"].astype(np.int64).to_numpy()
    profile_age_np = df["age_level"].astype(np.int64).to_numpy()
    profile_pvalue_np = df["pvalue_level"].astype(np.int64).to_numpy()
    profile_shopping_np = df["shopping_level"].astype(np.int64).to_numpy()
    profile_occupation_np = df["occupation"].astype(np.int64).to_numpy()
    profile_new_user_np = df["new_user_class_level"].astype(np.int64).to_numpy()
    pid_type_np = df["pid_type"].astype(np.int64).to_numpy()
    pid_id_np = df["pid_id"].astype(np.int64).to_numpy()
    price_np = df["price"].astype(np.float32).to_numpy()
    dup_count_np = df["dup_count"].astype(np.int64).to_numpy()
    cluster_span_np = df["cluster_span_sec"].astype(np.int64).to_numpy()
    dt_index = pd.to_datetime(timestamps_np + TZ_OFFSET_SEC, unit="s")
    hour_np = (dt_index.hour.to_numpy(dtype=np.float32) / 23.0).astype(np.float32)
    dow_np = (dt_index.dayofweek.to_numpy(dtype=np.float32) / 6.0).astype(np.float32)

    for row_idx in range(n):
        current_ts = int(timestamps_np[row_idx])
        current_user = int(users_np[row_idx])
        target_adgroup = int(target_adgroup_np[row_idx])
        target_cate = int(target_cate_np[row_idx])
        target_campaign = int(target_campaign_np[row_idx])
        target_customer = int(target_customer_np[row_idx])
        target_brand = int(target_brand_np[row_idx])

        history = user_histories.get(current_user)
        if history is None:
            click_cut = 0
            expose_cut = 0
            click_selected_events: list[tuple[int, int, int, int, int, float, int, int]] = []
            click_selected_ts: list[int] = []
            expose_selected_events = []
            expose_selected_ts = []
        else:
            click_cut = bisect_left(history.click_ts, current_ts)
            expose_cut = bisect_left(history.expose_ts, current_ts)

            click_start = max(0, click_cut - seq_len)
            expose_start = max(0, expose_cut - seq_len)

            click_selected_events = history.click_events[click_start:click_cut]
            click_selected_ts = history.click_ts[click_start:click_cut]
            expose_selected_events = history.expose_events[expose_start:expose_cut]
            expose_selected_ts = history.expose_ts[expose_start:expose_cut]

        click_summary = summarize_history(
            click_selected_events,
            click_selected_ts,
            click_cut,
            current_ts,
            target_adgroup,
            target_cate,
            target_campaign,
            target_customer,
            target_brand,
        )
        expose_summary = summarize_history(
            expose_selected_events,
            expose_selected_ts,
            expose_cut,
            current_ts,
            target_adgroup,
            target_cate,
            target_campaign,
            target_customer,
            target_brand,
        )

        non_seq_sparse_values = {
            "user": current_user,
            "adgroup_id": target_adgroup,
            "cate_id": target_cate,
            "campaign_id": target_campaign,
            "customer": target_customer,
            "brand": target_brand,
            "pid_type": int(pid_type_np[row_idx]),
            "pid_id": int(pid_id_np[row_idx]),
            "final_gender_code": int(profile_gender_np[row_idx]),
            "age_level": int(profile_age_np[row_idx]),
            "pvalue_level": int(profile_pvalue_np[row_idx]),
            "shopping_level": int(profile_shopping_np[row_idx]),
            "occupation": int(profile_occupation_np[row_idx]),
            "new_user_class_level": int(profile_new_user_np[row_idx]),
        }
        non_seq_sparse_np[row_idx, :] = np.asarray(
            encode_sparse_values(non_seq_sparse_values, NON_SEQ_SPARSE_FIELDS),
            dtype=np.int64,
        )
        non_seq_dense_np[row_idx, :] = np.asarray(
            [
                log1p_nonneg(price_np[row_idx]),
                float(hour_np[row_idx]),
                float(dow_np[row_idx]),
                click_summary[0],
                expose_summary[0],
                click_summary[1],
                expose_summary[1],
                click_summary[2],
                expose_summary[2],
                click_summary[3],
                expose_summary[3],
                click_summary[4],
                expose_summary[4],
                click_summary[5],
                expose_summary[5],
                click_summary[6],
                expose_summary[6],
                log1p_nonneg(dup_count_np[row_idx]),
                log1p_nonneg(cluster_span_np[row_idx]),
            ],
            dtype=np.float32,
        )

        fill_sequence_branch(
            seq_sparse_np,
            seq_dense_np,
            seq_mask_np,
            row_idx=row_idx,
            branch_idx=0,
            selected_events=click_selected_events,
            selected_ts=click_selected_ts,
            current_ts=current_ts,
            target_adgroup=target_adgroup,
            target_cate=target_cate,
            target_campaign=target_campaign,
            target_customer=target_customer,
            target_brand=target_brand,
            target_price=float(price_np[row_idx]),
        )
        fill_sequence_branch(
            seq_sparse_np,
            seq_dense_np,
            seq_mask_np,
            row_idx=row_idx,
            branch_idx=1,
            selected_events=expose_selected_events,
            selected_ts=expose_selected_ts,
            current_ts=current_ts,
            target_adgroup=target_adgroup,
            target_cate=target_cate,
            target_campaign=target_campaign,
            target_customer=target_customer,
            target_brand=target_brand,
            target_price=float(price_np[row_idx]),
        )

        if row_idx > 0 and row_idx % 500000 == 0:
            print(f"    {row_idx:,}/{n:,} ...")

    labels = torch.from_numpy(labels_np).long()
    timestamps = torch.from_numpy(timestamps_np).long()
    non_seq_sparse = torch.from_numpy(non_seq_sparse_np).long()
    non_seq_dense = torch.from_numpy(non_seq_dense_np).float()
    seq_sparse = torch.from_numpy(seq_sparse_np).long()
    seq_dense = torch.from_numpy(seq_dense_np).float()
    seq_mask = torch.from_numpy(seq_mask_np).bool()

    pos = int(labels.sum().item())
    neg = int(len(labels) - pos)
    click_non_empty = int(seq_mask[:, 0].any(dim=1).sum().item())
    expose_non_empty = int(seq_mask[:, 1].any(dim=1).sum().item())

    metadata = {
        "dataset": "ad_ctr_public_three_table",
        "feature_version": "field_aware_hyformer_v2",
        "history_source": {
            "source_table": "raw_sample",
            "available_tables": ["raw_sample", "ad_feature", "user_profile"],
            "sequence_names": ["click_seq", "exposure_seq"],
            "note": "Histories are reconstructed from prior ad display/click rows, not from raw_behavior_log.",
        },
        "num_samples": n,
        "num_sequences": num_sequences,
        "sequence_names": ["click_seq", "exposure_seq"],
        "seq_len": seq_len,
        "label_mapping": {"0": 0, "1": 1},
        "positive_samples": pos,
        "negative_samples": neg,
        "pos_rate": round(pos / max(n, 1), 6),
        "samples_with_click_seq": click_non_empty,
        "samples_with_exposure_seq": expose_non_empty,
        "non_seq_sparse_fields": NON_SEQ_SPARSE_FIELDS,
        "non_seq_dense_fields": NON_SEQ_DENSE_FIELDS,
        "seq_sparse_fields": SEQ_SPARSE_FIELDS,
        "seq_dense_fields": SEQ_DENSE_FIELDS,
        "sparse_field_cardinalities": {field: bucket_size + 1 for field, bucket_size in SPARSE_FIELD_BUCKETS.items()},
        "token_groups": TOKEN_GROUPS,
        "dedup": dedup_stats,
        "time_semantics": {
            "timezone": "Asia",
            "timestamp_unit": "seconds",
        },
        "time_range": {
            "min_timestamp": int(timestamps.min().item()) if n else 0,
            "max_timestamp": int(timestamps.max().item()) if n else 0,
        },
    }

    return non_seq_sparse, non_seq_dense, seq_sparse, seq_dense, seq_mask, labels, timestamps, metadata


def save_outputs(
    output_dir: Path,
    non_seq_sparse: torch.Tensor,
    non_seq_dense: torch.Tensor,
    seq_sparse: torch.Tensor,
    seq_dense: torch.Tensor,
    seq_mask: torch.Tensor,
    labels: torch.Tensor,
    timestamps: torch.Tensor,
    metadata: dict,
) -> None:
    print("[5/5]")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(non_seq_sparse, output_dir / "non_seq_sparse.pt")
    torch.save(non_seq_dense, output_dir / "non_seq_dense.pt")
    torch.save(seq_sparse, output_dir / "seq_sparse.pt")
    torch.save(seq_dense, output_dir / "seq_dense.pt")
    torch.save(seq_mask, output_dir / "seq_mask.pt")
    torch.save(labels, output_dir / "labels.pt")
    torch.save(timestamps, output_dir / "timestamps.pt")

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"  {output_dir}")
    print(f"  non_seq_sparse: {tuple(non_seq_sparse.shape)}")
    print(f"  non_seq_dense:  {tuple(non_seq_dense.shape)}")
    print(f"  seq_sparse:     {tuple(seq_sparse.shape)}")
    print(f"  seq_dense:      {tuple(seq_dense.shape)}")
    print(f"  seq_mask:       {tuple(seq_mask.shape)}")
    print(f"  labels:         {tuple(labels.shape)}")
    print(f"  timestamps:     {tuple(timestamps.shape)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTR ")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "archive",
        help="CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="data/processed）",
    )
    parser.add_argument(
        "--max-raw-rows",
        type=int,
        default=None,
        help="",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=100,
        help="",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=2,
        help="",
    )
    parser.add_argument(
        "--dedup-window-sec",
        type=int,
        default=DEFAULT_DEDUP_WINDOW_SEC,
        help="",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_sample, ad_feature, user_profile, dedup_stats = load_raw_data(args.data_dir, args.dedup_window_sec, args.max_raw_rows)
    df = join_tables(raw_sample, ad_feature, user_profile)
    user_histories = build_user_behavior_sequences(df)

    non_seq_sparse, non_seq_dense, seq_sparse, seq_dense, seq_mask, labels, timestamps, metadata = vectorize_dataset(
        df=df,
        user_histories=user_histories,
        seq_len=args.seq_len,
        num_sequences=args.num_sequences,
        max_rows=args.max_rows,
        dedup_stats=dedup_stats,
    )

    save_outputs(
        output_dir=args.output_dir,
        non_seq_sparse=non_seq_sparse,
        non_seq_dense=non_seq_dense,
        seq_sparse=seq_sparse,
        seq_dense=seq_dense,
        seq_mask=seq_mask,
        labels=labels,
        timestamps=timestamps,
        metadata=metadata,
    )

    print("\n finished")
    print(f"  python scripts/run_baotao.py --data-dir {args.output_dir}")


if __name__ == "__main__":
    main()
