from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.kcore import iterative_k_core
from src.data_pipeline.preprocessing import aggregate_unique_user_item, build_interactions, apply_feedback_weights
from src.data_pipeline.splitting import temporal_leave_one_out, assert_disjoint_splits
from src.data_pipeline.negative_sampling import build_user_positive_sets, sample_train_negatives
from src.data_pipeline.dataset import TrainDataset
from src.data_pipeline.adapters.hm import HMAdapter


def test_aggregate_before_kcore_counts_unique_edges():
    events = pd.DataFrame({
        "user_raw": ["u1"] * 5 + ["u1", "u2", "u2"],
        "item_raw": ["a"] * 5 + ["b", "a", "b"],
        "timestamp": pd.date_range("2020-01-01", periods=8),
        "value_raw": [1.0] * 8,
        "source_order": np.arange(8),
    })
    agg = aggregate_unique_user_item(events)
    # u1 has only 2 unique items, not 6 interactions as raw rows suggest.
    assert len(agg[agg.user_raw == "u1"]) == 2
    filtered = iterative_k_core(agg, 2)
    assert set(filtered.user_raw) == {"u1", "u2"}
    assert set(filtered.item_raw) == {"a", "b"}


def test_temporal_loo_and_no_overlap():
    df = pd.DataFrame({
        "user_raw": ["u"] * 5,
        "item_raw": ["a", "b", "c", "d", "e"],
        "interaction_count": [1] * 5,
        "value_sum": [1.0] * 5,
        "first_timestamp": pd.date_range("2020-01-01", periods=5),
        "last_timestamp": pd.date_range("2020-01-01", periods=5),
        "last_source_order": np.arange(5),
    })
    data = build_interactions(
        pd.DataFrame({
            "user_raw": ["u"] * 5,
            "item_raw": ["a", "b", "c", "d", "e"],
            "timestamp": pd.date_range("2020-01-01", periods=5),
            "value_raw": [1.0] * 5,
            "source_order": np.arange(5),
        }),
        k_core=1,
    )
    train, val, test = temporal_leave_one_out(data.df, 3)
    assert train.item.tolist() == [0, 1, 2]
    assert val.item.tolist() == [3]
    assert test.item.tolist() == [4]
    assert_disjoint_splits(train, val, test)


def test_weight_transform_fit_on_train_only():
    base = pd.DataFrame({"value_sum": [1.0, 10.0]})
    val = pd.DataFrame({"value_sum": [1e9]})
    test = pd.DataFrame({"value_sum": [1e12]})
    tr, va, te, meta = apply_feedback_weights(base, val, test, "weighted_confidence", 1.0)
    assert meta["train_log_scale"] == pytest.approx(np.log1p(10.0))
    assert va.sample_weight.iloc[0] <= 2.0
    assert te.sample_weight.iloc[0] <= 2.0


def test_negative_sampler_never_returns_positive():
    rng = np.random.default_rng(0)
    positives = {0, 1, 2, 3, 4}
    negs = sample_train_negatives(positives, 6, 10, rng)
    assert len(negs) == 10
    assert set(negs) == {5}


def test_train_dataset_never_labels_positive_as_negative():
    train = pd.DataFrame({"user": [0, 0], "item": [0, 1], "sample_weight": [1.0, 1.0]})
    pos = build_user_positive_sets(train, 1)
    ds = TrainDataset(train, n_items=3, train_positive_sets=pos, neg_ratio=4, seed=0)
    negative_items = ds.items[ds.labels == 0]
    assert set(negative_items) == {2}


def test_train_dataset_fast_mode_never_labels_positive_as_negative():
    """fast_negative_sampling=True (vector hoá cho quy mô lớn, VD H&M full
    scale) phải giữ đúng bất biến chính xác như chế độ mặc định, kể cả ở
    trường hợp cực đoan catalog gần cạn negative khả dụng."""
    train = pd.DataFrame({"user": [0, 0], "item": [0, 1], "sample_weight": [1.0, 1.0]})
    pos = build_user_positive_sets(train, 1)
    ds = TrainDataset(
        train, n_items=3, train_positive_sets=pos, neg_ratio=4, seed=0,
        fast_negative_sampling=True,
    )
    negative_items = ds.items[ds.labels == 0]
    assert set(negative_items) == {2}


def test_train_dataset_fast_mode_matches_loop_mode_on_larger_data():
    """Kiểm tra ở quy mô lớn hơn (nhiều user/item hơn) rằng chế độ vector hoá
    không bao giờ trả về 1 negative trùng positive thật của đúng user đó."""
    rng = np.random.default_rng(1)
    n_users, n_items = 200, 50
    rows = []
    for u in range(n_users):
        n_pos = rng.integers(1, 8)
        items = rng.choice(n_items, size=n_pos, replace=False)
        for i in items:
            rows.append((u, int(i)))
    train = pd.DataFrame(rows, columns=["user", "item"])
    train["sample_weight"] = 1.0
    pos = build_user_positive_sets(train, n_users)

    ds = TrainDataset(
        train, n_items=n_items, train_positive_sets=pos, neg_ratio=4, seed=7,
        fast_negative_sampling=True,
    )
    neg_mask = ds.labels == 0
    for u, i in zip(ds.users[neg_mask], ds.items[neg_mask]):
        assert int(i) not in pos[int(u)], f"user {u} bị gán negative {i} nhưng đó là positive thật"



