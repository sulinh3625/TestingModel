from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.evaluation.full_ranking import build_full_ranking_records, evaluate_torch_model, EvalRecord
from src.evaluation.ranking_utils import rank_positive
from src.evaluation.long_tail import define_head_items


def test_full_ranking_excludes_seen_and_keeps_positive():
    eval_df = pd.DataFrame({"user": [0], "item": [4]})
    seen = [{0, 1, 2, 3}]
    rec = build_full_ranking_records(eval_df, 6, seen)[0]
    assert 4 in rec.candidates
    assert not ({0, 1, 2, 3} & set(rec.candidates))
    assert set(rec.candidates) == {4, 5}


def test_tie_ranking_is_deterministic():
    candidates = np.array([1, 2, 3, 4])
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    r1 = rank_positive(scores, candidates, 3, user=7, tie_seed=2026)
    r2 = rank_positive(scores, candidates, 3, user=7, tie_seed=2026)
    assert r1 == r2
    assert 1 <= r1 <= 4


def test_perfect_model_hr_ndcg_one():
    class Perfect(torch.nn.Module):
        def forward(self, u, i):
            return (i == 5).float() * 10.0

    records = [EvalRecord(0, 5, np.array([1, 2, 5, 6], dtype=np.int64))]
    result = evaluate_torch_model(Perfect(), records, [1, 3], tie_seed=1)
    assert result["HR@1"] == 1.0
    assert result["NDCG@1"] == 1.0


def test_head_fraction_uses_train_only_counts():
    train = pd.DataFrame({"item": [0] * 10 + [1] * 5 + [2] * 2 + [3]})
    head = define_head_items(train, n_items=4, head_fraction=0.25)
    assert head == {0}


def test_sampled_candidates_do_not_duplicate_positive():
    from src.evaluation.sampled_ranking import build_sampled_ranking_records
    eval_df = pd.DataFrame({"user": [0], "item": [2]})
    all_pos = [{0, 1, 2}]
    rec = build_sampled_ranking_records(eval_df, n_items=6, all_positive_sets=all_pos, n_negatives=3, seed=1)[0]
    assert list(rec.candidates).count(2) == 1
    assert not ({0, 1} & set(rec.candidates))


# ─────────────────────────────────────────────────────────
# Beyond-accuracy metrics tests
# ─────────────────────────────────────────────────────────
from src.evaluation.beyond_accuracy import (
    catalog_coverage, average_recommendation_popularity,
    head_recommendation_rate, novelty_score,
)


def test_catalog_coverage_full():
    recs = {0: [0, 1], 1: [1, 2], 2: [2, 3]}
    assert catalog_coverage(recs, n_items=4) == 1.0


def test_catalog_coverage_partial():
    recs = {0: [0], 1: [0]}  # chỉ item 0
    assert catalog_coverage(recs, n_items=4) == 0.25


def test_catalog_coverage_empty():
    assert catalog_coverage({}, n_items=10) == 0.0


def test_arp_most_popular_higher():
    # item 0 xuất hiện 100 lần, item 1 xuất hiện 1 lần
    pop_recs = {0: [0, 0]}        # recommend popular
    tail_recs = {0: [1, 1]}       # recommend tail
    counts = {0: 100, 1: 1}
    assert average_recommendation_popularity(pop_recs, counts) > \
           average_recommendation_popularity(tail_recs, counts)


def test_head_rec_rate_all_head():
    head = {0, 1}
    recs = {0: [0, 1], 1: [0, 1]}
    assert head_recommendation_rate(recs, head) == 1.0


def test_head_rec_rate_no_head():
    head = {0, 1}
    recs = {0: [2, 3], 1: [4, 5]}
    assert head_recommendation_rate(recs, head) == 0.0


def test_novelty_popular_lower():
    # item 0 rất phổ biến → novelty thấp; item 99 hiếm → novelty cao
    counts = {0: 1000, 99: 1}
    n_train = 1001
    pop_recs = {0: [0]}
    rare_recs = {0: [99]}
    assert novelty_score(pop_recs, counts, n_train) < novelty_score(rare_recs, counts, n_train)

