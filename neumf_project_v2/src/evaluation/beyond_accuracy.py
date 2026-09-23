from __future__ import annotations

import numpy as np


def catalog_coverage(recommendations: dict[int, list[int]], n_items: int) -> float:
    """Tỷ lệ % catalog item xuất hiện trong ít nhất 1 recommendation list."""
    if n_items <= 0 or not recommendations:
        return 0.0
    unique = {item for recs in recommendations.values() for item in recs}
    return len(unique) / n_items


def average_recommendation_popularity(recommendations: dict[int, list[int]], item_counts) -> float:
    """Trung bình số lần xuất hiện trong train của các item được recommend.

    item_counts: dict {item_id -> count_in_train}
    Giá trị cao → model thiên về popular items.
    """
    values = []
    for recs in recommendations.values():
        values.extend(float(item_counts.get(item, 0.0)) for item in recs)
    return float(np.mean(values)) if values else 0.0


def head_recommendation_rate(recommendations: dict[int, list[int]], head_items: set[int]) -> float:
    """Tỷ lệ trung bình các item được recommend thuộc nhóm head (popular) items.

    head_items: tập item thuộc top-fraction% phổ biến nhất trong train.
    Giá trị cao → model bias về head items, novelty thấp.
    """
    if not recommendations:
        return 0.0
    user_rates = []
    for recs in recommendations.values():
        if not recs:
            continue
        rate = sum(1 for item in recs if item in head_items) / len(recs)
        user_rates.append(rate)
    return float(np.mean(user_rates)) if user_rates else 0.0


def novelty_score(recommendations: dict[int, list[int]], item_counts, n_train: int) -> float:
    """Novelty dựa trên self-information: -log2(P(item)).

    Tham khảo: Vargas & Castells (2011).
    item_counts: dict {item_id -> count_in_train}
    n_train: tổng số training interactions (để tính P(item)).
    """
    if not recommendations or n_train <= 0:
        return 0.0
    values = []
    for recs in recommendations.values():
        for item in recs:
            count = item_counts.get(item, 0)
            p = (count + 1e-9) / n_train  # smoothing
            values.append(-np.log2(p))
    return float(np.mean(values)) if values else 0.0
