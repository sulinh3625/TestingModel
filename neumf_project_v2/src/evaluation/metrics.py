from __future__ import annotations

import math


def hr_at_k(rank: int, k: int) -> float:
    return 1.0 if rank <= k else 0.0


def ndcg_at_k(rank: int, k: int) -> float:
    return 1.0 / math.log2(rank + 1) if rank <= k else 0.0


def precision_at_k(rank: int, k: int) -> float:
    return 1.0 / k if rank <= k else 0.0


def recall_at_k(rank: int, k: int) -> float:
    return hr_at_k(rank, k)
