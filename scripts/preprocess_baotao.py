

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
TZ_OFFSET_SEC = 8 * 3600

NON_SEQ_SPARSE_FIELDS = [
    "user",
    "adgroup_id",
    "cate_id",
    "campaign_id",
    "customer",
    "brand",
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
]

SEQ_SPARSE_FIELDS = [
    "adgroup_id",
    "cate_id",
    "campaign_id",
    "customer",
    "brand",
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
    "pid_id": 4,
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
    "context_token": ["pid_id", "hour_of_day", "day_of_week"],
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
}


@dataclass
class UserHistory:
    click_ts: list[int]
    click_events: list[tuple[int, int, int, int, int, float, int]]
    expose_ts: list[int]
    expose_events: list[tuple[int, int, int, int, int, float, int]]


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


def load_raw_data(data_dir: Path, max_raw_rows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_sample = pd.read_csv(data_dir / "raw_sample.csv", nrows=max_raw_rows)
    ad_feature = pd.read_csv(data_dir / "ad_feature.csv")
    user_profile = pd.read_csv(data_dir / "user_profile.csv")

    raw_sample["time_stamp"] = raw_sample["time_stamp"].astype(np.int64)
    raw_sample["user"] = raw_sample["user"].astype(np.int64)
    raw_sample["adgroup_id"] = raw_sample["adgroup_id"].astype(np.int64)
    pid_split = raw_sample["pid"].str.split("_", expand=True)
    raw_sample["pid_id"] = pid_split[1].astype(np.int64)
    raw_sample.drop(columns=["pid", "nonclk"], inplace=True)
    raw_sample.rename(columns={"clk": "label"}, inplace=True)

    if max_raw_rows is not None:
        raw_sample = raw_sample.sample(n=max_raw_rows, random_state=42).reset_index(drop=True)
        print(f"  raw_sample_sample: {max_raw_rows:,} ")

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

    print(f"  raw_sample: {len(raw_sample):,} ")
    print(f"  ad_feature: {len(ad_feature):,} ")
    print(f"  user_profile: {len(user_profile):,} ")
    return raw_sample, ad_feature, user_profile


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
    event_cols = ["adgroup_id", "cate_id", "campaign_id", "customer", "brand", "price", "pid_id"]

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
    print(f"  sum : {len(histories):,}")
    print(f"  click : {users_with_click:,}")
    print(f"  exposure : {users_with_expose:,}")
    return histories


def encode_sparse_values(values: dict[str, object], field_names: list[str]) -> list[int]:
    return [stable_bucket_id(values[field], SPARSE_FIELD_BUCKETS[field]) for field in field_names]


def encode_seq_sparse_event(event: tuple[int, int, int, int, int, float, int]) -> list[int]:
    (
        adgroup_id,
        cate_id,
        campaign_id,
        customer,
        brand,
        _price,
        pid_id,
    ) = event
    values = {
        "adgroup_id": adgroup_id,
        "cate_id": cate_id,
        "campaign_id": campaign_id,
        "customer": customer,
        "brand": brand,
        "pid_id": pid_id,
    }
    return [stable_bucket_id(values[field], SPARSE_FIELD_BUCKETS[field]) for field in SEQ_SPARSE_FIELDS]


def summarize_history(
    history_events: list[tuple[int, int, int, int, int, float, int]],
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
        adgroup_id, cate_id, campaign_id, customer, brand, _, _ = event
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
    selected_events: list[tuple[int, int, int, int, int, float, int]],
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
            _hist_pid_id,
        ) = event
        hist_price = safe_float(hist_price)
        target_price = safe_float(target_price)
        price_ratio_log = math.log1p(hist_price / target_price) if target_price > 0.0 else 0.0
        relative_position = 0.0 if selected_len <= 1 else step_idx / (selected_len - 1)
        seq_sparse_tensor[row_idx, branch_idx, step_idx, :] = np.asarray(encode_seq_sparse_event(event), dtype=np.int32)
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
            dtype=np.float16,
        )
        seq_mask_tensor[row_idx, branch_idx, step_idx] = True


def vectorize_dataset(
    df: pd.DataFrame,
    user_histories: dict[int, UserHistory],
    seq_len: int,
    num_sequences: int,
    max_rows: int | None,
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

    non_seq_sparse_np = np.zeros((n, len(NON_SEQ_SPARSE_FIELDS)), dtype=np.int32)
    non_seq_dense_np = np.zeros((n, len(NON_SEQ_DENSE_FIELDS)), dtype=np.float32)
    seq_sparse_np = np.zeros((n, num_sequences, seq_len, len(SEQ_SPARSE_FIELDS)), dtype=np.int32)
    seq_dense_np = np.zeros((n, num_sequences, seq_len, len(SEQ_DENSE_FIELDS)), dtype=np.float16)
    seq_mask_np = np.zeros((n, num_sequences, seq_len), dtype=bool)
    labels_np = df["label"].astype(np.int32).to_numpy().copy()
    timestamps_np = df["time_stamp"].astype(np.int64).to_numpy().copy()

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
    pid_id_np = df["pid_id"].astype(np.int64).to_numpy()
    price_np = df["price"].astype(np.float32).to_numpy()
    dt_index = pd.to_datetime(timestamps_np + TZ_OFFSET_SEC, unit="s")
    hour_np = (dt_index.hour.to_numpy(dtype=np.float32) / 23.0).astype(np.float32)
    dow_np = (dt_index.dayofweek.to_numpy(dtype=np.float32) / 6.0).astype(np.float32)

    # ── Vectorized non_seq_sparse ────────────────────────────────────────
    # Bucket-encode all non-seq sparse fields at once via numpy broadcasting.
    for col_idx, field in enumerate(NON_SEQ_SPARSE_FIELDS):
        bucket = SPARSE_FIELD_BUCKETS[field]
        col_np = df[field].astype(np.int64).to_numpy()
        is_zero = col_np == 0
        non_seq_sparse_np[:, col_idx] = np.where(
            is_zero, 0, np.abs(col_np) % bucket + 1
        ).astype(np.int32)

    # ── Pre-extract user history arrays for fast lookup ──────────────────
    # Convert the dict-of-lists into arrays for vectorized bisect + indexing.
    _user_keys = np.array(list(user_histories.keys()), dtype=np.int64)
    _user_to_idx = {int(k): i for i, k in enumerate(_user_keys)}
    _click_ts_arrays = [np.array(user_histories[int(k)].click_ts, dtype=np.int64)
                        for k in _user_keys]
    _click_event_arrays = [user_histories[int(k)].click_events for k in _user_keys]
    _expose_ts_arrays = [np.array(user_histories[int(k)].expose_ts, dtype=np.int64)
                         for k in _user_keys]
    _expose_event_arrays = [user_histories[int(k)].expose_events for k in _user_keys]

    # Map each row's user to its history index; -1 for unknown users
    user_history_idx = np.full(n, -1, dtype=np.int64)
    for row_idx in range(n):
        uid = int(users_np[row_idx])
        if uid in _user_to_idx:
            user_history_idx[row_idx] = _user_to_idx[uid]

    has_history = user_history_idx >= 0
    rows_with_hist = np.where(has_history)[0]
    hist_indices = user_history_idx[rows_with_hist]

    # ── Vectorized bisect_left for click and expose timestamps ───────────
    click_cuts = np.zeros(n, dtype=np.int64)
    expose_cuts = np.zeros(n, dtype=np.int64)
    click_cuts[rows_with_hist] = np.array([
        int(np.searchsorted(_click_ts_arrays[hi], timestamps_np[ri], side="left"))
        for ri, hi in zip(rows_with_hist, hist_indices)
    ], dtype=np.int64)
    expose_cuts[rows_with_hist] = np.array([
        int(np.searchsorted(_expose_ts_arrays[hi], timestamps_np[ri], side="left"))
        for ri, hi in zip(rows_with_hist, hist_indices)
    ], dtype=np.int64)

    # ── Vectorized history summaries for non_seq_dense ───────────────────
    click_hist_len_log = np.zeros(n, dtype=np.float32)
    expose_hist_len_log = np.zeros(n, dtype=np.float32)
    click_last_gap_log = np.zeros(n, dtype=np.float32)
    expose_last_gap_log = np.zeros(n, dtype=np.float32)
    click_same_ad_log = np.zeros(n, dtype=np.float32)
    expose_same_ad_log = np.zeros(n, dtype=np.float32)
    click_same_cate_log = np.zeros(n, dtype=np.float32)
    expose_same_cate_log = np.zeros(n, dtype=np.float32)
    click_same_brand_log = np.zeros(n, dtype=np.float32)
    expose_same_brand_log = np.zeros(n, dtype=np.float32)
    click_same_campaign_log = np.zeros(n, dtype=np.float32)
    expose_same_campaign_log = np.zeros(n, dtype=np.float32)
    click_same_customer_log = np.zeros(n, dtype=np.float32)
    expose_same_customer_log = np.zeros(n, dtype=np.float32)

    # Compute last-gap (only non-zero for users with history)
    for arr_idx, ri in enumerate(rows_with_hist):
        hi = int(hist_indices[arr_idx])
        cc = int(click_cuts[ri])
        ec = int(expose_cuts[ri])
        click_hist_len_log[ri] = math.log1p(cc)
        expose_hist_len_log[ri] = math.log1p(ec)
        if cc > 0:
            click_last_gap_log[ri] = math.log1p(int(timestamps_np[ri]) - int(_click_ts_arrays[hi][cc - 1]))
        if ec > 0:
            expose_last_gap_log[ri] = math.log1p(int(timestamps_np[ri]) - int(_expose_ts_arrays[hi][ec - 1]))

    # Compute same_* counts by scanning the selected history slice for each row
    for arr_idx, ri in enumerate(rows_with_hist):
        hi = int(hist_indices[arr_idx])
        cc = int(click_cuts[ri])
        ec = int(expose_cuts[ri])
        c_start = max(0, cc - seq_len)
        e_start = max(0, ec - seq_len)
        t_ad = int(target_adgroup_np[ri])
        t_ca = int(target_cate_np[ri])
        t_br = int(target_brand_np[ri])
        t_cp = int(target_campaign_np[ri])
        t_cu = int(target_customer_np[ri])

        # Click history
        if cc > 0:
            c_ad = c_ca = c_br = c_cp = c_cu = 0
            for ev in _click_event_arrays[hi][c_start:cc]:
                if ev[0] == t_ad: c_ad += 1
                if ev[1] == t_ca: c_ca += 1
                if ev[4] == t_br: c_br += 1
                if ev[2] == t_cp: c_cp += 1
                if ev[3] == t_cu: c_cu += 1
            click_same_ad_log[ri] = math.log1p(c_ad)
            click_same_cate_log[ri] = math.log1p(c_ca)
            click_same_brand_log[ri] = math.log1p(c_br)
            click_same_campaign_log[ri] = math.log1p(c_cp)
            click_same_customer_log[ri] = math.log1p(c_cu)

        # Expose history
        if ec > 0:
            e_ad = e_ca = e_br = e_cp = e_cu = 0
            for ev in _expose_event_arrays[hi][e_start:ec]:
                if ev[0] == t_ad: e_ad += 1
                if ev[1] == t_ca: e_ca += 1
                if ev[4] == t_br: e_br += 1
                if ev[2] == t_cp: e_cp += 1
                if ev[3] == t_cu: e_cu += 1
            expose_same_ad_log[ri] = math.log1p(e_ad)
            expose_same_cate_log[ri] = math.log1p(e_ca)
            expose_same_brand_log[ri] = math.log1p(e_br)
            expose_same_campaign_log[ri] = math.log1p(e_cp)
            expose_same_customer_log[ri] = math.log1p(e_cu)

    # Fill non_seq_dense
    non_seq_dense_np[:, 0] = np.log1p(np.maximum(price_np, 0)).astype(np.float32)
    non_seq_dense_np[:, 1] = hour_np
    non_seq_dense_np[:, 2] = dow_np
    non_seq_dense_np[:, 3] = click_hist_len_log
    non_seq_dense_np[:, 4] = expose_hist_len_log
    non_seq_dense_np[:, 5] = click_last_gap_log
    non_seq_dense_np[:, 6] = expose_last_gap_log
    non_seq_dense_np[:, 7] = click_same_ad_log
    non_seq_dense_np[:, 8] = expose_same_ad_log
    non_seq_dense_np[:, 9] = click_same_cate_log
    non_seq_dense_np[:, 10] = expose_same_cate_log
    non_seq_dense_np[:, 11] = click_same_brand_log
    non_seq_dense_np[:, 12] = expose_same_brand_log
    non_seq_dense_np[:, 13] = click_same_campaign_log
    non_seq_dense_np[:, 14] = expose_same_campaign_log
    non_seq_dense_np[:, 15] = click_same_customer_log
    non_seq_dense_np[:, 16] = expose_same_customer_log

    # ── Vectorized sequence branches ─────────────────────────────────────
    for arr_idx, ri in enumerate(rows_with_hist):
        hi = int(hist_indices[arr_idx])
        cc = int(click_cuts[ri])
        ec = int(expose_cuts[ri])
        c_start = max(0, cc - seq_len)
        e_start = max(0, ec - seq_len)
        current_ts = int(timestamps_np[ri])
        t_ad = int(target_adgroup_np[ri])
        t_ca = int(target_cate_np[ri])
        t_cp = int(target_campaign_np[ri])
        t_cu = int(target_customer_np[ri])
        t_br = int(target_brand_np[ri])
        t_price = float(price_np[ri])

        # Fill click branch (branch_idx=0)
        click_evts = _click_event_arrays[hi][c_start:cc]
        click_tss = _click_ts_arrays[hi][c_start:cc]
        sel_len = len(click_evts)
        for step_idx, (event, hist_ts) in enumerate(zip(click_evts, click_tss)):
            hist_ad, hist_ca, hist_cp, hist_cu, hist_br, hist_price, hist_pid = event
            hist_price_f = safe_float(hist_price)
            price_ratio_log = math.log1p(hist_price_f / t_price) if t_price > 0.0 else 0.0
            relative_position = 0.0 if sel_len <= 1 else step_idx / (sel_len - 1)
            seq_sparse_np[ri, 0, step_idx, :] = np.asarray(
                [stable_bucket_id(hist_ad, SPARSE_FIELD_BUCKETS["adgroup_id"]),
                 stable_bucket_id(hist_ca, SPARSE_FIELD_BUCKETS["cate_id"]),
                 stable_bucket_id(hist_cp, SPARSE_FIELD_BUCKETS["campaign_id"]),
                 stable_bucket_id(hist_cu, SPARSE_FIELD_BUCKETS["customer"]),
                 stable_bucket_id(hist_br, SPARSE_FIELD_BUCKETS["brand"]),
                 stable_bucket_id(hist_pid, SPARSE_FIELD_BUCKETS["pid_id"])],
                dtype=np.int32,
            )
            seq_dense_np[ri, 0, step_idx, :] = np.asarray(
                [log1p_nonneg(hist_price_f),
                 signed_log1p(hist_price_f - t_price),
                 price_ratio_log,
                 log1p_nonneg(current_ts - int(hist_ts)),
                 float(hist_ad == t_ad),
                 float(hist_ca == t_ca),
                 float(hist_br == t_br),
                 float(hist_cp == t_cp),
                 float(hist_cu == t_cu),
                 log1p_nonneg(sel_len - step_idx),
                 float(relative_position)],
                dtype=np.float16,
            )
            seq_mask_np[ri, 0, step_idx] = True

        # Fill expose branch (branch_idx=1)
        expose_evts = _expose_event_arrays[hi][e_start:ec]
        expose_tss = _expose_ts_arrays[hi][e_start:ec]
        sel_len = len(expose_evts)
        for step_idx, (event, hist_ts) in enumerate(zip(expose_evts, expose_tss)):
            hist_ad, hist_ca, hist_cp, hist_cu, hist_br, hist_price, hist_pid = event
            hist_price_f = safe_float(hist_price)
            price_ratio_log = math.log1p(hist_price_f / t_price) if t_price > 0.0 else 0.0
            relative_position = 0.0 if sel_len <= 1 else step_idx / (sel_len - 1)
            seq_sparse_np[ri, 1, step_idx, :] = np.asarray(
                [stable_bucket_id(hist_ad, SPARSE_FIELD_BUCKETS["adgroup_id"]),
                 stable_bucket_id(hist_ca, SPARSE_FIELD_BUCKETS["cate_id"]),
                 stable_bucket_id(hist_cp, SPARSE_FIELD_BUCKETS["campaign_id"]),
                 stable_bucket_id(hist_cu, SPARSE_FIELD_BUCKETS["customer"]),
                 stable_bucket_id(hist_br, SPARSE_FIELD_BUCKETS["brand"]),
                 stable_bucket_id(hist_pid, SPARSE_FIELD_BUCKETS["pid_id"])],
                dtype=np.int32,
            )
            seq_dense_np[ri, 1, step_idx, :] = np.asarray(
                [log1p_nonneg(hist_price_f),
                 signed_log1p(hist_price_f - t_price),
                 price_ratio_log,
                 log1p_nonneg(current_ts - int(hist_ts)),
                 float(hist_ad == t_ad),
                 float(hist_ca == t_ca),
                 float(hist_br == t_br),
                 float(hist_cp == t_cp),
                 float(hist_cu == t_cu),
                 log1p_nonneg(sel_len - step_idx),
                 float(relative_position)],
                dtype=np.float16,
            )
            seq_mask_np[ri, 1, step_idx] = True

        if ri > 0 and ri % 500000 == 0:
            print(f"    {ri:,}/{n:,} ...")

    # Keep sparse tensors as int32 for storage; run_baotao.py will .long() at load time.
    labels = torch.from_numpy(labels_np)
    del labels_np
    timestamps = torch.from_numpy(timestamps_np)
    del timestamps_np
    non_seq_sparse = torch.from_numpy(non_seq_sparse_np)
    del non_seq_sparse_np
    non_seq_dense = torch.from_numpy(non_seq_dense_np)
    del non_seq_dense_np
    seq_sparse = torch.from_numpy(seq_sparse_np)
    del seq_sparse_np
    # Keep seq_dense as float16 for storage; run_baotao.py will .float() at load time.
    seq_dense = torch.from_numpy(seq_dense_np)
    del seq_dense_np
    seq_mask = torch.from_numpy(seq_mask_np)
    del seq_mask_np

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
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # [1/7] Load raw CSV
    raw_sample, ad_feature, user_profile = load_raw_data(args.data_dir, args.max_raw_rows)

    # [2/7] Join tables
    df = join_tables(raw_sample, ad_feature, user_profile)
    del raw_sample, ad_feature, user_profile

    # [3/7] Per-day split — split all data into per-day DataFrames.
    #       Progressive validation will evaluate each day before training on it.
    TZ_OFFSET_SEC = 8 * 3600
    timestamps_all = df["time_stamp"].astype(np.int64).values
    day_ids = (timestamps_all + TZ_OFFSET_SEC) // 86400
    unique_days = np.unique(day_ids)
    num_days = len(unique_days)

    day_dfs = []
    for day in unique_days:
        mask = day_ids == day
        day_dfs.append(df[mask].copy())

    day_sizes = [len(d) for d in day_dfs]
    print(f"[3/7] Per-day split: {num_days} days  sizes={day_sizes}")

    # [4/7] Build user histories from ALL data.
    #       The bisect_left() cutoff in vectorize_dataset() ensures each sample
    #       only sees history from before its own timestamp — no per-sample leakage.
    #       Using all data gives later-day samples richer historical context.
    print("[4/7] Building user histories from ALL data (bisect_left prevents leakage) ...")
    user_histories = build_user_behavior_sequences(df)

    # [5/7] Use full data (no negative sampling)
    print("[5/7] Using full data (no negative sampling)")

    # [6/7] Sort each day's data by timestamp
    for day_idx in range(num_days):
        day_dfs[day_idx] = day_dfs[day_idx].sort_values("time_stamp").reset_index(drop=True)

    # [7/7] Vectorize per-day and save to per-day subdirectories
    day_metas = []
    total_pos = 0
    total_neg = 0
    total_click_nonempty = 0
    total_expose_nonempty = 0
    min_ts_all = float("inf")
    max_ts_all = 0

    for day_idx in range(num_days):
        day_label = f"day_{day_idx + 1}"
        day_dir = args.output_dir / day_label
        day_dir.mkdir(parents=True, exist_ok=True)

        print(f"[7/7] Vectorizing {day_label} ({len(day_dfs[day_idx]):,} rows) ...")
        non_seq_sparse, non_seq_dense, seq_sparse, seq_dense, seq_mask, labels, timestamps, day_meta = vectorize_dataset(
            df=day_dfs[day_idx],
            user_histories=user_histories,
            seq_len=args.seq_len,
            num_sequences=args.num_sequences,
            max_rows=None,  # per-day, no subsampling
        )

        # Save per-day tensors
        torch.save(non_seq_sparse, day_dir / "non_seq_sparse.pt")
        torch.save(non_seq_dense, day_dir / "non_seq_dense.pt")
        torch.save(seq_sparse, day_dir / "seq_sparse.pt")
        torch.save(seq_dense, day_dir / "seq_dense.pt")
        torch.save(seq_mask, day_dir / "seq_mask.pt")
        torch.save(labels, day_dir / "labels.pt")
        torch.save(timestamps, day_dir / "timestamps.pt")

        n = len(labels)
        pos = int(labels.sum().item())
        neg = n - pos
        click_nonempty = int(seq_mask[:, 0].any(dim=1).sum().item())
        expose_nonempty = int(seq_mask[:, 1].any(dim=1).sum().item())
        ts_min = int(timestamps.min().item()) if n else 0
        ts_max = int(timestamps.max().item()) if n else 0

        total_pos += pos
        total_neg += neg
        total_click_nonempty += click_nonempty
        total_expose_nonempty += expose_nonempty
        min_ts_all = min(min_ts_all, ts_min)
        max_ts_all = max(max_ts_all, ts_max)

        day_meta["day_label"] = day_label
        day_meta["day_index"] = day_idx
        day_meta["day_id"] = int(unique_days[day_idx])
        day_meta["num_samples"] = n
        day_meta["timestamp_range"] = [ts_min, ts_max]
        day_metas.append(day_meta)

        print(f"  {day_label} saved to {day_dir}")
        print(f"    non_seq_sparse: {tuple(non_seq_sparse.shape)}")
        print(f"    non_seq_dense:  {tuple(non_seq_dense.shape)}")
        print(f"    seq_sparse:     {tuple(seq_sparse.shape)}")
        print(f"    seq_dense:      {tuple(seq_dense.shape)}")
        print(f"    seq_mask:       {tuple(seq_mask.shape)}")
        print(f"    labels:         {tuple(labels.shape)}")
        print(f"    timestamps:     {tuple(timestamps.shape)}")

    del day_dfs, user_histories

    # Write global metadata
    metadata = {
        "dataset": "ad_ctr_public_three_table",
        "feature_version": "field_aware_hyformer_v2",
        "history_source": {
            "source_table": "raw_sample",
            "available_tables": ["raw_sample", "ad_feature", "user_profile"],
            "sequence_names": ["click_seq", "exposure_seq"],
            "note": "Histories are reconstructed from prior ad display/click rows, not from raw_behavior_log.",
        },
        "num_samples": total_pos + total_neg,
        "num_sequences": args.num_sequences,
        "sequence_names": ["click_seq", "exposure_seq"],
        "seq_len": args.seq_len,
        "label_mapping": {"0": 0, "1": 1},
        "positive_samples": total_pos,
        "negative_samples": total_neg,
        "pos_rate": round(total_pos / max(total_pos + total_neg, 1), 6),
        "samples_with_click_seq": total_click_nonempty,
        "samples_with_exposure_seq": total_expose_nonempty,
        "non_seq_sparse_fields": NON_SEQ_SPARSE_FIELDS,
        "non_seq_dense_fields": NON_SEQ_DENSE_FIELDS,
        "seq_sparse_fields": SEQ_SPARSE_FIELDS,
        "seq_dense_fields": SEQ_DENSE_FIELDS,
        "sparse_field_cardinalities": {field: bucket_size + 1 for field, bucket_size in SPARSE_FIELD_BUCKETS.items()},
        "token_groups": TOKEN_GROUPS,
        "time_semantics": {
            "timezone": "Asia",
            "timestamp_unit": "seconds",
        },
        "time_range": {
            "min_timestamp": int(min_ts_all) if min_ts_all != float("inf") else 0,
            "max_timestamp": int(max_ts_all),
        },
        "progressive_split": {
            "split_mode": "progressive_time",
            "num_days": num_days,
            "timezone": "Asia",
            "days": day_metas,
            "note": "Each day's data is stored in a separate subdirectory (day_1/, day_2/, ...). "
                    "Progressive validation: evaluate on day D BEFORE training on it. "
                    "Last day is test-only (no training).",
        },
    }

    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n finished")
    print(f"  python scripts/run_baotao.py --data-dir {args.output_dir}")


if __name__ == "__main__":
    main()
