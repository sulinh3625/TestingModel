from __future__ import annotations

import pandas as pd


def temporal_leave_one_out(df: pd.DataFrame, min_interactions: int = 3):
    """Temporal LOO trên unique user-item interactions.

    test = item cuối, validation = item áp chót, train = phần trước đó.
    Với timestamp bằng nhau, last_source_order và item index là tie-breaker ổn định.
    """
    ordered = df.sort_values(
        ["user", "last_timestamp", "last_source_order", "item"], kind="mergesort"
    )
    train_parts, val_parts, test_parts = [], [], []

    for _, g in ordered.groupby("user", sort=False):
        if len(g) < min_interactions:
            train_parts.append(g)
            continue
        train_parts.append(g.iloc[:-2])
        val_parts.append(g.iloc[-2:-1])
        test_parts.append(g.iloc[-1:])

    train = pd.concat(train_parts, ignore_index=True) if train_parts else ordered.iloc[0:0].copy()
    val = pd.concat(val_parts, ignore_index=True) if val_parts else ordered.iloc[0:0].copy()
    test = pd.concat(test_parts, ignore_index=True) if test_parts else ordered.iloc[0:0].copy()
    return train, val, test


def user_item_pairs(df: pd.DataFrame) -> set[tuple[int, int]]:
    return set(zip(df["user"].astype(int), df["item"].astype(int)))


def assert_disjoint_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    t, v, s = user_item_pairs(train), user_item_pairs(val), user_item_pairs(test)
    tv = t & v
    ts = t & s
    vs = v & s
    if tv or ts or vs:
        raise AssertionError(
            f"User-item leakage giữa splits: train∩val={len(tv)}, "
            f"train∩test={len(ts)}, val∩test={len(vs)}"
        )
