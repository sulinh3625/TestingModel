from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .kcore import iterative_k_core


@dataclass
class InteractionData:
    df: pd.DataFrame
    n_users: int
    n_items: int
    user2idx: dict
    item2idx: dict


def aggregate_unique_user_item(events: pd.DataFrame) -> pd.DataFrame:
    """Gộp transaction lặp trước khi k-core.

    Giữ cả số lần tương tác, tổng value và mốc thời gian đầu/cuối.
    last_source_order là tie-breaker deterministic khi nhiều item cùng timestamp.
    """
    required = {"user_raw", "item_raw", "timestamp", "value_raw", "source_order"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Thiếu cột chuẩn hóa: {sorted(missing)}")

    ordered = events.sort_values(["timestamp", "source_order"], kind="mergesort")
    agg = (
        ordered.groupby(["user_raw", "item_raw"], sort=False)
        .agg(
            interaction_count=("item_raw", "size"),
            value_sum=("value_raw", "sum"),
            first_timestamp=("timestamp", "first"),
            last_timestamp=("timestamp", "last"),
            last_source_order=("source_order", "last"),
        )
        .reset_index()
    )
    return agg


def reindex_interactions(df: pd.DataFrame) -> InteractionData:
    users = sorted(df["user_raw"].unique().tolist(), key=lambda x: str(x))
    items = sorted(df["item_raw"].unique().tolist(), key=lambda x: str(x))
    user2idx = {u: idx for idx, u in enumerate(users)}
    item2idx = {i: idx for idx, i in enumerate(items)}

    out = df.copy()
    out["user"] = out["user_raw"].map(user2idx).astype(np.int64)
    out["item"] = out["item_raw"].map(item2idx).astype(np.int64)
    out = out.sort_values(
        ["user", "last_timestamp", "last_source_order", "item"], kind="mergesort"
    ).reset_index(drop=True)
    return InteractionData(out, len(users), len(items), user2idx, item2idx)


def build_interactions(events: pd.DataFrame, k_core: int) -> InteractionData:
    aggregated = aggregate_unique_user_item(events)
    filtered = iterative_k_core(aggregated, k_core)
    if filtered.empty:
        raise ValueError(f"Không còn dữ liệu sau k-core={k_core}.")
    return reindex_interactions(filtered)


def apply_feedback_weights(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mode: str = "binary",
    confidence_alpha: float = 1.0,
):
    """Tạo sample_weight; mọi thống kê biến đổi chỉ fit trên TRAIN.

    binary: sample_weight = 1.
    weighted_confidence: 1 + alpha * normalized log1p(value_sum).
    Đây là confidence weight cho BCE, không biến Sales/price thành nhãn.
    """
    train, val, test = train_df.copy(), val_df.copy(), test_df.copy()
    if mode == "binary":
        for frame in (train, val, test):
            frame["sample_weight"] = 1.0
        return train, val, test, {"mode": mode}

    if mode != "weighted_confidence":
        raise ValueError("feedback.mode phải là 'binary' hoặc 'weighted_confidence'.")

    train_log = np.log1p(train["value_sum"].clip(lower=0).astype(float))
    scale = float(train_log.max()) if len(train_log) else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    for frame in (train, val, test):
        normalized = np.log1p(frame["value_sum"].clip(lower=0).astype(float)) / scale
        normalized = normalized.clip(0.0, 1.0)
        frame["sample_weight"] = 1.0 + confidence_alpha * normalized

    return train, val, test, {
        "mode": mode,
        "confidence_alpha": confidence_alpha,
        "train_log_scale": scale,
    }
