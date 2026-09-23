from __future__ import annotations

import numpy as np


def deterministic_tie_key(user: int, item_ids: np.ndarray, seed: int) -> np.ndarray:
    # 64-bit integer mixing; stable between Python runs.
    x = item_ids.astype(np.uint64)
    x ^= np.uint64((int(user) + 1) * 0x9E3779B1)
    x ^= np.uint64((int(seed) + 1) * 0x85EBCA77)
    x *= np.uint64(0xC2B2AE3D)
    x ^= x >> np.uint64(16)
    return x


def rank_positive(scores: np.ndarray, candidates: np.ndarray, positive_item: int, user: int, tie_seed: int) -> int:
    scores = np.asarray(scores, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.int64)
    tie = deterministic_tie_key(user, candidates, tie_seed)
    # primary: score descending; secondary: deterministic pseudo-random key ascending
    order = np.lexsort((tie, -scores))
    pos_idx = np.flatnonzero(candidates[order] == int(positive_item))
    if len(pos_idx) != 1:
        raise ValueError("positive_item phải xuất hiện đúng 1 lần trong candidate set")
    return int(pos_idx[0]) + 1
