from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from .metrics import hr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from .ranking_utils import rank_positive, deterministic_tie_key


@dataclass
class EvalRecord:
    user: int
    positive_item: int
    candidates: np.ndarray


def build_full_ranking_records(eval_df, n_items: int, seen_positive_sets) -> list[EvalRecord]:
    all_items = np.arange(n_items, dtype=np.int64)
    records = []
    for u, pos in zip(eval_df["user"].values, eval_df["item"].values):
        u, pos = int(u), int(pos)
        seen = set(seen_positive_sets[u])
        seen.discard(pos)
        if seen:
            mask = np.ones(n_items, dtype=bool)
            mask[np.fromiter(seen, dtype=np.int64)] = False
            candidates = all_items[mask]
        else:
            candidates = all_items.copy()
        if pos not in candidates:
            candidates = np.append(candidates, pos)
        records.append(EvalRecord(u, pos, candidates.astype(np.int64, copy=False)))
    return records


def _empty_metric_lists(k_values, include_redundant):
    out = {f"HR@{k}": [] for k in k_values}
    out.update({f"NDCG@{k}": [] for k in k_values})
    if include_redundant:
        out.update({f"Precision@{k}": [] for k in k_values})
        out.update({f"Recall@{k}": [] for k in k_values})
    return out


def _append(metrics, rank, k_values, include_redundant):
    for k in k_values:
        metrics[f"HR@{k}"].append(hr_at_k(rank, k))
        metrics[f"NDCG@{k}"].append(ndcg_at_k(rank, k))
        if include_redundant:
            metrics[f"Precision@{k}"].append(precision_at_k(rank, k))
            metrics[f"Recall@{k}"].append(recall_at_k(rank, k))


@torch.no_grad()
def evaluate_torch_model(model, records, k_values, device="cpu", batch_size=16384, tie_seed=2026, include_redundant=False, return_topk=False):
    """Evaluate PyTorch model with flattened batched scoring.

    Tránh gọi model một lần cho từng user; toàn bộ candidate pairs được flatten
    rồi score theo batch lớn, sau đó cắt lại theo offsets để tính rank từng user.
    """
    model.eval()
    metrics = _empty_metric_lists(k_values, include_redundant)
    max_k = max(k_values)
    recommendations = {} if return_topk else None
    if not records:
        summary = {name: float("nan") for name in metrics}
        return (summary, recommendations) if return_topk else summary

    lengths = np.fromiter((len(r.candidates) for r in records), dtype=np.int64, count=len(records))
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    total = int(offsets[-1])
    users_flat = np.empty(total, dtype=np.int64)
    items_flat = np.empty(total, dtype=np.int64)
    for idx, record in enumerate(records):
        a, b = int(offsets[idx]), int(offsets[idx + 1])
        users_flat[a:b] = record.user
        items_flat[a:b] = record.candidates

    score_flat = np.empty(total, dtype=np.float32)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        users = torch.from_numpy(users_flat[start:end]).long().to(device)
        items = torch.from_numpy(items_flat[start:end]).long().to(device)
        score_flat[start:end] = model(users, items).detach().cpu().numpy()

    for idx, record in enumerate(records):
        a, b = int(offsets[idx]), int(offsets[idx + 1])
        cand = items_flat[a:b]
        scores = score_flat[a:b]
        rank = rank_positive(scores, cand, record.positive_item, record.user, tie_seed)
        _append(metrics, rank, k_values, include_redundant)
        if return_topk:
            tie = deterministic_tie_key(record.user, cand, tie_seed)
            order = np.lexsort((tie, -scores))[:max_k]
            recommendations[record.user] = cand[order].tolist()

    summary = {name: float(np.mean(vals)) if vals else float("nan") for name, vals in metrics.items()}
    return (summary, recommendations) if return_topk else summary


def evaluate_score_function(score_fn, records, k_values, tie_seed=2026, include_redundant=False, return_topk=False):
    metrics = _empty_metric_lists(k_values, include_redundant)
    max_k = max(k_values)
    recommendations = {} if return_topk else None
    for record in records:
        cand = record.candidates
        scores = np.fromiter((score_fn(record.user, int(i)) for i in cand), dtype=np.float64, count=len(cand))
        rank = rank_positive(scores, cand, record.positive_item, record.user, tie_seed)
        _append(metrics, rank, k_values, include_redundant)
        if return_topk:
            tie = deterministic_tie_key(record.user, cand, tie_seed)
            order = np.lexsort((tie, -scores))[:max_k]
            recommendations[record.user] = cand[order].tolist()
    summary = {name: float(np.mean(vals)) if vals else float("nan") for name, vals in metrics.items()}
    return (summary, recommendations) if return_topk else summary
