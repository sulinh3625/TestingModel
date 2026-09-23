from __future__ import annotations

import numpy as np

from src.data_pipeline.negative_sampling import sample_eval_negatives
from .full_ranking import EvalRecord


def build_sampled_ranking_records(eval_df, n_items: int, all_positive_sets, n_negatives: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    records = []
    for u, pos in zip(eval_df["user"].values, eval_df["item"].values):
        u, pos = int(u), int(pos)
        # Negatives must exclude every known positive, including the target itself.
        exclude = set(all_positive_sets[u])
        negs = sample_eval_negatives(exclude, n_items, n_negatives, rng)
        candidates = np.concatenate(([pos], negs.astype(np.int64)))
        records.append(EvalRecord(u, pos, candidates))
    return records
