from __future__ import annotations

import pandas as pd


def iterative_k_core(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Bipartite k-core trên bảng unique user-item.

    Mỗi dòng phải đại diện đúng một cạnh user-item. Lọc user và item lặp
    đến fixed point để mọi node còn lại có degree >= k.
    """
    if k <= 0:
        return df.copy().reset_index(drop=True)

    cur = df.copy()
    while True:
        before = len(cur)
        user_degree = cur.groupby("user_raw", sort=False)["item_raw"].nunique()
        item_degree = cur.groupby("item_raw", sort=False)["user_raw"].nunique()
        valid_users = user_degree[user_degree >= k].index
        valid_items = item_degree[item_degree >= k].index
        cur = cur[cur["user_raw"].isin(valid_users) & cur["item_raw"].isin(valid_items)].copy()
        if len(cur) == before:
            break
    return cur.reset_index(drop=True)
