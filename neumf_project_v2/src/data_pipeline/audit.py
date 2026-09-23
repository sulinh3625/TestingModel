from __future__ import annotations

import math
import pandas as pd

from .kcore import iterative_k_core
from .preprocessing import aggregate_unique_user_item
from .splitting import user_item_pairs


def _summary(df: pd.DataFrame) -> dict:
    users = int(df["user_raw"].nunique()) if len(df) else 0
    items = int(df["item_raw"].nunique()) if len(df) else 0
    interactions = int(len(df))
    density = interactions / (users * items) if users and items else 0.0
    return {"users": users, "items": items, "interactions": interactions, "density": density}


def audit_events(events: pd.DataFrame, k_values: list[int]) -> tuple[dict, pd.DataFrame]:
    aggregated = aggregate_unique_user_item(events)
    repeat_rows = int(len(events) - len(aggregated))
    report = {
        "raw_rows": int(len(events)),
        "raw_users": int(events["user_raw"].nunique()),
        "raw_items": int(events["item_raw"].nunique()),
        "unique_user_item_pairs": int(len(aggregated)),
        "repeat_transaction_rows": repeat_rows,
        "repeat_row_rate": repeat_rows / len(events) if len(events) else 0.0,
        "kcore": {},
    }
    rows = []
    for k in k_values:
        filtered = iterative_k_core(aggregated, int(k))
        stats = _summary(filtered)
        report["kcore"][str(k)] = stats
        rows.append({"k": int(k), **stats})
    return report, pd.DataFrame(rows)


def audit_splits(train, val, test) -> dict:
    tp, vp, sp = user_item_pairs(train), user_item_pairs(val), user_item_pairs(test)
    return {
        "train_interactions": int(len(train)),
        "validation_interactions": int(len(val)),
        "test_interactions": int(len(test)),
        "train_val_overlap": len(tp & vp),
        "train_test_overlap": len(tp & sp),
        "val_test_overlap": len(vp & sp),
    }
