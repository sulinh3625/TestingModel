from __future__ import annotations

import math


def define_head_items(train_df, n_items: int, head_fraction: float = 0.10) -> set[int]:
    n_head = max(1, int(math.ceil(n_items * head_fraction)))
    counts = train_df["item"].value_counts().sort_values(ascending=False)
    return set(int(i) for i in counts.index[:n_head])


def split_records_head_tail(records, head_items: set[int]):
    head, tail = [], []
    for record in records:
        (head if record.positive_item in head_items else tail).append(record)
    return head, tail
