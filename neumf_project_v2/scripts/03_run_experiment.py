from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import build_adapter
from src.baselines import RandomBaseline, MostPopularBaseline, ItemKNNBaseline, BPRMFBaseline, CategoryPopularityBaseline
from src.data_pipeline.dataset import TrainDataset
from src.data_pipeline.negative_sampling import build_user_positive_sets
from src.data_pipeline.preprocessing import build_interactions, apply_feedback_weights
from src.data_pipeline.splitting import temporal_leave_one_out, assert_disjoint_splits
from src.evaluation.beyond_accuracy import (
    catalog_coverage, average_recommendation_popularity,
    head_recommendation_rate, novelty_score,
)
from src.evaluation.full_ranking import build_full_ranking_records, evaluate_torch_model, evaluate_score_function
from src.evaluation.sampled_ranking import build_sampled_ranking_records
from src.evaluation.long_tail import define_head_items, split_records_head_tail
from src.evaluation.cold_start import define_cold_users, split_records_by_coldness
from src.models.neumf import GMF, MLP, NeuMF
from src.models.early_fusion import EarlyFusionModel
from src.models.lightgcn import LightGCN, build_norm_adj
from src.models.sasrec import SASRec, build_user_sequences
from src.training.trainer import train_one_model, make_optimizer, get_device
from src.utils.seed import seed_everything
from src.utils.io import ensure_dir, write_json

METHOD_ORDER = [
    "Random", "MostPopular", "ItemKNN", "BPR-MF", "CategoryPopularity",
    "GMF", "MLP", "EarlyFusion", "LightGCN", "SASRec", "NeuMF-Scratch", "NeuMF-Pretrained",
]


def build_eval_records(cfg, val_df, test_df, train_df, full_df, n_users, n_items):
    train_pos = build_user_positive_sets(train_df, n_users)
    train_val = pd.concat([train_df, val_df], ignore_index=True)
    train_val_pos = build_user_positive_sets(train_val, n_users)
    all_pos = build_user_positive_sets(full_df, n_users)

    if cfg.evaluation.primary == "full_ranking":
        val_records = build_full_ranking_records(val_df, n_items, train_pos)
        test_records = build_full_ranking_records(test_df, n_items, train_val_pos)
    elif cfg.evaluation.primary == "sampled":
        val_records = build_sampled_ranking_records(
            val_df, n_items, all_pos, cfg.evaluation.sampled_negatives, seed=cfg.training.seed + 1
        )
        test_records = build_sampled_ranking_records(
            test_df, n_items, all_pos, cfg.evaluation.sampled_negatives, seed=cfg.training.seed + 2
        )
    else:
        raise ValueError(f"evaluation.primary không hợp lệ: {cfg.evaluation.primary}")

    sampled_test = build_sampled_ranking_records(
        test_df, n_items, all_pos, cfg.evaluation.sampled_negatives, seed=cfg.training.seed + 2
    )
    return train_pos, val_records, test_records, sampled_test


def run(config_path: str, run_tag: str | None = None):
    experiment_start = time.perf_counter()
    cfg, adapter = build_adapter(config_path)
    seed_everything(cfg.training.seed)

    print("\n" + "═"*60)
    print("  🧪 NeuMF EXPERIMENT")
    print("═"*60)

    # Auto-detect device
    print("\n📱 Thiết bị:")
    device = get_device(cfg.training.device)

    run_tag = run_tag or f"{cfg.dataset.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ensure_dir(cfg.output_root / "experiments" / run_tag)
    ckpt_dir = ensure_dir(cfg.output_root / "checkpoints" / run_tag)
    print(f"\n📂 Output: {run_dir}")

    # 1) Data pipeline
    print("\n" + "─"*60)
    print("  📊 PHASE 1: Data Pipeline")
    print("─"*60)
    t0 = time.perf_counter()
    print("  Loading raw events...")
    events = adapter.load_events()
    print(f"  ✓ {len(events):,} raw events loaded")
    print("  Building interactions (aggregate → k-core)...")
    data = build_interactions(events, cfg.dataset.k_core)
    print(f"  ✓ {data.n_users:,} users | {data.n_items:,} items | {len(data.df):,} interactions")
    print("  Splitting (temporal Leave-One-Out)...")
    train_df, val_df, test_df = temporal_leave_one_out(data.df, cfg.dataset.min_interactions_for_loo)
    assert_disjoint_splits(train_df, val_df, test_df)
    train_df, val_df, test_df, feedback_meta = apply_feedback_weights(
        train_df, val_df, test_df, cfg.feedback.mode, cfg.feedback.confidence_alpha
    )
    print(f"  ✓ Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print("  Building evaluation records...")
    train_pos, val_records, test_records, sampled_test = build_eval_records(
        cfg, val_df, test_df, train_df, data.df, data.n_users, data.n_items
    )
    train_dataset = TrainDataset(
        train_df, data.n_items, train_pos, cfg.training.negative_ratio, seed=cfg.training.seed,
        fast_negative_sampling=cfg.training.fast_negative_sampling,
    )
    print(f"  ✓ Data pipeline hoàn thành ({time.perf_counter() - t0:.1f}s)")

    eval_kwargs = dict(
        k_values=cfg.evaluation.k_values,
        device=device,
        tie_seed=cfg.evaluation.tie_break_seed,
        include_redundant=cfg.evaluation.include_redundant_metrics,
    )

    def eval_torch(model, records):
        return evaluate_torch_model(model, records, **eval_kwargs)

    results_primary = {}
    results_sampled = {}
    train_meta = {}
    models = {}
    score_fns = {}

    print("\n" + "─"*60)
    print("  🧠 PHASE 2: Neural Models Training")
    print("─"*60)

    # 2) Pretrain GMF
    gmf = GMF(data.n_users, data.n_items, cfg.model.embedding_dim).to(device)
    opt = make_optimizer(cfg.training.pretrain_optimizer, gmf.parameters(), cfg.training.pretrain_lr, cfg.training.weight_decay)
    gmf, hist, meta = train_one_model(
        gmf, train_dataset, val_records, eval_torch, opt, device,
        cfg.training.max_epochs_pretrain, cfg.training.patience, cfg.training.batch_size,
        monitor=cfg.training.monitor, seed=cfg.training.seed + 11, model_name="GMF"
    )
    models["GMF"] = gmf
    train_meta["GMF"] = meta
    pd.DataFrame(hist).to_csv(run_dir / "history_gmf.csv", index=False)
    torch.save(gmf.state_dict(), ckpt_dir / "gmf.pt")

    # 3) Pretrain MLP
    mlp = MLP(data.n_users, data.n_items, cfg.model.embedding_dim, cfg.model.mlp_layers, cfg.model.dropout).to(device)
    opt = make_optimizer(cfg.training.pretrain_optimizer, mlp.parameters(), cfg.training.pretrain_lr, cfg.training.weight_decay)
    mlp, hist, meta = train_one_model(
        mlp, train_dataset, val_records, eval_torch, opt, device,
        cfg.training.max_epochs_pretrain, cfg.training.patience, cfg.training.batch_size,
        monitor=cfg.training.monitor, seed=cfg.training.seed + 22, model_name="MLP"
    )
    models["MLP"] = mlp
    train_meta["MLP"] = meta
    pd.DataFrame(hist).to_csv(run_dir / "history_mlp.csv", index=False)
    torch.save(mlp.state_dict(), ckpt_dir / "mlp.pt")

    # 3b) Early Fusion baseline — đối chứng trực tiếp với NeuMF (Late Fusion).
    # Dùng cùng optimizer/LR/budget với NeuMF-Scratch (finetune_*) để so sánh
    # công bằng: chỉ khác nhau ở chỗ kết hợp sớm (concat rồi 1 mạng chung)
    # hay muộn (2 nhánh riêng rồi mới nối ở cuối).
    if cfg.training.train_early_fusion:
        early_fusion = EarlyFusionModel(
            data.n_users, data.n_items, cfg.model.embedding_dim, cfg.model.mlp_layers, cfg.model.dropout
        ).to(device)
        opt = make_optimizer(
            cfg.training.finetune_optimizer, early_fusion.parameters(),
            cfg.training.finetune_lr, cfg.training.weight_decay
        )
        early_fusion, hist, meta = train_one_model(
            early_fusion, train_dataset, val_records, eval_torch, opt, device,
            cfg.training.max_epochs_finetune, cfg.training.patience, cfg.training.batch_size,
            monitor=cfg.training.monitor, seed=cfg.training.seed + 55, model_name="EarlyFusion"
        )
        models["EarlyFusion"] = early_fusion
        train_meta["EarlyFusion"] = meta
        pd.DataFrame(hist).to_csv(run_dir / "history_early_fusion.csv", index=False)
        torch.save(early_fusion.state_dict(), ckpt_dir / "early_fusion.pt")

    # 3c) LightGCN — mã hoá tương tác bậc cao qua lan truyền đồ thị User--Item,
    # đối lập với GMF/MLP/NeuMF vốn chỉ dùng Embedding ID thuần (không có cấu
    # trúc đồ thị). Đồ thị chuẩn hoá chỉ xây từ TRAIN (build_norm_adj nhận
    # train_df), khớp nguyên tắc chống rò rỉ dữ liệu chung của dự án. Huấn
    # luyện bằng optimizer/budget của nhóm pretrain (giống GMF/MLP) vì đây là
    # kiến trúc độc lập, không có bước pretrain->finetune riêng.
    if cfg.training.train_lightgcn:
        norm_adj = build_norm_adj(train_df, data.n_users, data.n_items).to(device)
        lightgcn = LightGCN(
            data.n_users, data.n_items, cfg.model.embedding_dim, norm_adj, cfg.model.lightgcn_layers
        ).to(device)
        opt = make_optimizer(cfg.training.pretrain_optimizer, lightgcn.parameters(), cfg.training.pretrain_lr, cfg.training.weight_decay)
        lightgcn, hist, meta = train_one_model(
            lightgcn, train_dataset, val_records, eval_torch, opt, device,
            cfg.training.max_epochs_pretrain, cfg.training.patience, cfg.training.batch_size,
            monitor=cfg.training.monitor, seed=cfg.training.seed + 66, model_name="LightGCN"
        )
        models["LightGCN"] = lightgcn
        train_meta["LightGCN"] = meta
        pd.DataFrame(hist).to_csv(run_dir / "history_lightgcn.csv", index=False)
        torch.save(lightgcn.state_dict(), ckpt_dir / "lightgcn.pt")

    # 3d) SASRec — mã hoá THỨ TỰ THỜI GIAN của lịch sử mua hàng bằng
    # Self-Attention causal, đối lập với mọi model ở trên (coi lịch sử là
    # tập hợp không thứ tự). Chuỗi mỗi user chỉ xây từ TRAIN (build_user_sequences
    # nhận train_df), khớp nguyên tắc chống rò rỉ dữ liệu chung của dự án.
    if cfg.training.train_sasrec:
        user_seq = build_user_sequences(train_df, data.n_users, data.n_items, cfg.model.sasrec_seq_len).to(device)
        sasrec = SASRec(
            data.n_users, data.n_items, cfg.model.embedding_dim, user_seq,
            cfg.model.sasrec_seq_len, cfg.model.sasrec_heads, cfg.model.sasrec_blocks, cfg.model.dropout,
        ).to(device)
        opt = make_optimizer(cfg.training.pretrain_optimizer, sasrec.parameters(), cfg.training.pretrain_lr, cfg.training.weight_decay)
        sasrec, hist, meta = train_one_model(
            sasrec, train_dataset, val_records, eval_torch, opt, device,
            cfg.training.max_epochs_pretrain, cfg.training.patience, cfg.training.batch_size,
            monitor=cfg.training.monitor, seed=cfg.training.seed + 77, model_name="SASRec"
        )
        models["SASRec"] = sasrec
        train_meta["SASRec"] = meta
        pd.DataFrame(hist).to_csv(run_dir / "history_sasrec.csv", index=False)
        torch.save(sasrec.state_dict(), ckpt_dir / "sasrec.pt")

    # 4) Controlled pretraining ablation: same optimizer/LR/budget for scratch & pretrained.
    scratch = NeuMF(data.n_users, data.n_items, cfg.model.embedding_dim, cfg.model.mlp_layers, cfg.model.dropout).to(device)
    opt = make_optimizer(cfg.training.finetune_optimizer, scratch.parameters(), cfg.training.finetune_lr, cfg.training.weight_decay)
    scratch, hist, meta = train_one_model(
        scratch, train_dataset, val_records, eval_torch, opt, device,
        cfg.training.max_epochs_finetune, cfg.training.patience, cfg.training.batch_size,
        monitor=cfg.training.monitor, seed=cfg.training.seed + 33, model_name="NeuMF-Scratch"
    )
    models["NeuMF-Scratch"] = scratch
    train_meta["NeuMF-Scratch"] = meta
    pd.DataFrame(hist).to_csv(run_dir / "history_neumf_scratch.csv", index=False)
    torch.save(scratch.state_dict(), ckpt_dir / "neumf_scratch.pt")

    pretrained = NeuMF(data.n_users, data.n_items, cfg.model.embedding_dim, cfg.model.mlp_layers, cfg.model.dropout).to(device)
    pretrained.load_pretrained(gmf, mlp, alpha=cfg.model.pretrain_alpha)
    opt = make_optimizer(cfg.training.finetune_optimizer, pretrained.parameters(), cfg.training.finetune_lr, cfg.training.weight_decay)
    pretrained, hist, meta = train_one_model(
        pretrained, train_dataset, val_records, eval_torch, opt, device,
        cfg.training.max_epochs_finetune, cfg.training.patience, cfg.training.batch_size,
        monitor=cfg.training.monitor, seed=cfg.training.seed + 44, model_name="NeuMF-Pretrained"
    )
    models["NeuMF-Pretrained"] = pretrained
    train_meta["NeuMF-Pretrained"] = meta
    pd.DataFrame(hist).to_csv(run_dir / "history_neumf_pretrained.csv", index=False)
    torch.save(pretrained.state_dict(), ckpt_dir / "neumf_pretrained.pt")

    # 5) Classical baselines
    print("\n" + "─"*60)
    print("  📐 PHASE 3: Classical Baselines")
    print("─"*60)
    enabled = {x.lower() for x in cfg.baselines.enabled}
    if "random" in enabled:
        print("  → Random baseline...")
        random_bl = RandomBaseline(cfg.training.seed)
        score_fns["Random"] = random_bl.score
        print("    ✓ Random baseline ready")
    if "popularity" in enabled:
        print("  → MostPopular baseline...")
        pop_bl = MostPopularBaseline(train_df, data.n_items)
        score_fns["MostPopular"] = pop_bl.score
        print("    ✓ MostPopular baseline ready")
    if "itemknn" in enabled:
        print("  → ItemKNN baseline (computing similarity matrix)...")
        t0 = time.perf_counter()
        itemknn = ItemKNNBaseline(train_df, data.n_users, data.n_items)
        knn_time = time.perf_counter() - t0
        train_meta["ItemKNN"] = {"train_time_s": knn_time, "n_parameters": 0}
        score_fns["ItemKNN"] = itemknn.score
        print(f"    ✓ ItemKNN ready ({knn_time:.1f}s)")
    if "bpr" in enabled:
        print(f"  → BPR-MF baseline ({cfg.baselines.bpr.epochs} epochs)...")
        bpr = BPRMFBaseline(
            data.n_users, data.n_items, cfg.baselines.bpr.embedding_dim, seed=cfg.training.seed
        )
        t0 = time.perf_counter()
        bpr.fit(
            train_df,
            epochs=cfg.baselines.bpr.epochs,
            lr=cfg.baselines.bpr.lr,
            reg=cfg.baselines.bpr.reg,
            seed=cfg.training.seed,
        )
        bpr_time = time.perf_counter() - t0
        train_meta["BPR-MF"] = {
            "train_time_s": bpr_time,
            "n_parameters": int(bpr.P.size + bpr.Q.size),
        }
        score_fns["BPR-MF"] = bpr.score
        print(f"    ✓ BPR-MF ready ({bpr_time:.1f}s)")

    # 5b) CategoryPopularity — content-based, giảm nhẹ Cold-start (mục 5.4 đề
    # cương chi tiết). Chỉ khả dụng cho DataCo (có Category Name trong CSV
    # gốc); H&M chưa có adapter tương ứng nên bỏ qua nếu bật nhầm config.
    if "categorypopularity" in enabled:
        if cfg.dataset.name != "dataco" or not hasattr(adapter, "load_item_categories"):
            print("  ⚠️  Bỏ qua CategoryPopularity: chỉ hỗ trợ DataCo (cần Category Name).")
        else:
            print("  → CategoryPopularity baseline (content-based, cold-start)...")
            raw_categories = adapter.load_item_categories()
            item_category = {
                idx: raw_categories.get(item_raw, "UNKNOWN")
                for item_raw, idx in data.item2idx.items()
            }
            cat_pop = CategoryPopularityBaseline(train_df, item_category, data.n_items)
            score_fns["CategoryPopularity"] = cat_pop.score
            train_meta["CategoryPopularity"] = {"train_time_s": 0.0, "n_parameters": 0}
            print(f"    ✓ CategoryPopularity ready ({len(set(raw_categories.values()))} danh mục)")

    # 6) Evaluate primary + sampled reproduction protocol
    print("\n" + "─"*60)
    print("  📈 PHASE 4: Evaluation")
    print("─"*60)

    # Tính item popularity từ TRAIN để dùng cho ARP (Average Recommendation Popularity)
    item_popularity = train_df["item"].value_counts().to_dict()

    results_topk = {}  # Actual Top-K lists cho beyond-accuracy
    max_k = max(cfg.evaluation.k_values)
    eval_topk_kwargs = dict(**eval_kwargs, return_topk=True)

    all_eval_items = list(models.items()) + [(n, None) for n in score_fns]
    eval_bar = tqdm(all_eval_items, desc="  Evaluating", unit="model", ncols=80)
    for name, _ in eval_bar:
        eval_bar.set_postfix_str(name)
        if name in models:
            metrics, topk = evaluate_torch_model(models[name], test_records, **eval_topk_kwargs)
            results_primary[name] = metrics
            results_topk[name] = topk
            results_sampled[name] = evaluate_torch_model(models[name], sampled_test, **eval_kwargs)
        else:
            fn = score_fns[name]
            metrics, topk = evaluate_score_function(
                fn, test_records, cfg.evaluation.k_values,
                tie_seed=cfg.evaluation.tie_break_seed,
                include_redundant=cfg.evaluation.include_redundant_metrics,
                return_topk=True,
            )
            results_primary[name] = metrics
            results_topk[name] = topk
            results_sampled[name] = evaluate_score_function(
                fn, sampled_test, cfg.evaluation.k_values,
                tie_seed=cfg.evaluation.tie_break_seed,
                include_redundant=cfg.evaluation.include_redundant_metrics,
            )
    eval_bar.close()

    # 7) Long-tail segmentation from TRAIN only (trước beyond-accuracy để có head_items)
    print("\n  📊 Long-tail evaluation...")
    head_items = define_head_items(train_df, data.n_items, cfg.evaluation.head_fraction)
    head_records, tail_records = split_records_head_tail(test_records, head_items)
    print(f"    Head items: {len(head_items)} | Head test users: {len(head_records)} | Tail test users: {len(tail_records)}")

    # Tính beyond-accuracy metrics từ actual recommendations (sau khi có head_items)
    print("  📊 Computing beyond-accuracy metrics (coverage, ARP, HRR, novelty)...")
    n_train_interactions = int(len(train_df))
    results_beyond = {}
    for name, topk in results_topk.items():
        if topk:
            cov  = catalog_coverage(topk, data.n_items)
            arp  = average_recommendation_popularity(topk, item_popularity)
            hrr  = head_recommendation_rate(topk, head_items)
            nov  = novelty_score(topk, item_popularity, n_train_interactions)
            results_beyond[name] = {
                "catalog_coverage": round(cov, 6),
                "avg_rec_popularity": round(arp, 4),
                "head_rec_rate": round(hrr, 6),
                "novelty": round(nov, 6),
            }
        else:
            results_beyond[name] = {
                "catalog_coverage": 0.0, "avg_rec_popularity": 0.0,
                "head_rec_rate": 0.0, "novelty": 0.0,
            }
    results_tail = {}
    for name, model in models.items():
        results_tail[name] = eval_torch(model, tail_records) if tail_records else {}
    for name, fn in score_fns.items():
        results_tail[name] = evaluate_score_function(
            fn, tail_records, cfg.evaluation.k_values,
            tie_seed=cfg.evaluation.tie_break_seed,
            include_redundant=cfg.evaluation.include_redundant_metrics,
        ) if tail_records else {}

    # 8) Cold-start: user thuộc nhóm cold_fraction% ÍT tương tác nhất trong TRAIN
    # (mục 5.4 đề cương chi tiết). Item cold-start đã được bao phủ bởi Long-tail
    # ở trên (xem docstring src/evaluation/cold_start.py) — không lặp lại.
    print("\n  📊 Cold-start evaluation...")
    cold_users = define_cold_users(train_df, data.n_users, cfg.evaluation.cold_fraction)
    cold_records, warm_records = split_records_by_coldness(test_records, cold_users)
    print(f"    Cold users: {len(cold_users)} | Cold test records: {len(cold_records)} | Warm test records: {len(warm_records)}")

    results_cold = {}
    for name, model in models.items():
        results_cold[name] = eval_torch(model, cold_records) if cold_records else {}
    for name, fn in score_fns.items():
        results_cold[name] = evaluate_score_function(
            fn, cold_records, cfg.evaluation.k_values,
            tie_seed=cfg.evaluation.tie_break_seed,
            include_redundant=cfg.evaluation.include_redundant_metrics,
        ) if cold_records else {}

    # Save consistent tables
    order = [m for m in METHOD_ORDER if m in results_primary]
    pd.DataFrame.from_dict(results_primary, orient="index").reindex(order).to_csv(run_dir / "results_primary.csv")
    pd.DataFrame.from_dict(results_sampled, orient="index").reindex(order).to_csv(run_dir / "results_sampled_99.csv")
    pd.DataFrame.from_dict(results_tail, orient="index").reindex(order).to_csv(run_dir / "results_long_tail.csv")
    pd.DataFrame.from_dict(results_cold, orient="index").reindex(order).to_csv(run_dir / "results_cold_start.csv")

    metadata = {
        "run_tag": run_tag,
        "dataset": cfg.dataset.name,
        "seed": cfg.training.seed,
        "n_users": data.n_users,
        "n_items": data.n_items,
        "n_interactions": len(data.df),
        "train": len(train_df), "validation": len(val_df), "test": len(test_df),
        "k_core": cfg.dataset.k_core,
        "feedback": feedback_meta,
        "evaluation_primary": cfg.evaluation.primary,
        "head_fraction": cfg.evaluation.head_fraction,
        "n_head_items": len(head_items),
        "n_head_test_users": len(head_records),
        "n_long_tail_test_users": len(tail_records),
        "cold_fraction": cfg.evaluation.cold_fraction,
        "n_cold_users": len(cold_users),
        "n_cold_test_users": len(cold_records),
        "n_warm_test_users": len(warm_records),
        "training": train_meta,
    }
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "results.json", {
        "primary": results_primary,
        "sampled_99": results_sampled,
        "long_tail": results_tail,
        "cold_start": results_cold,
        "beyond_accuracy": results_beyond,
        "item_popularity": {str(k): int(v) for k, v in item_popularity.items()},
        "metadata": metadata,
    })

    total_time = time.perf_counter() - experiment_start
    print("\n" + "═"*60)
    print("  ✅ EXPERIMENT HOÀN THÀNH")
    print("═"*60)
    print(f"  ⏱ Tổng thời gian: {total_time:.1f}s ({total_time/60:.1f} phút)")
    print(f"  📂 Kết quả: {run_dir}")
    print(f"  💾 Checkpoints: {ckpt_dir}")
    _primary_label = "Full-Ranking" if cfg.evaluation.primary == "full_ranking" else "Sampled (giao thức chính)"
    print(f"\n📋 KẾT QUẢ PRIMARY ({_primary_label}):")
    print(pd.DataFrame.from_dict(results_primary, orient="index").reindex(order).to_string())
    print("\n📋 KẾT QUẢ SAMPLED (99 negatives):")
    print(pd.DataFrame.from_dict(results_sampled, orient="index").reindex(order).to_string())
    print(f"\n📋 KẾT QUẢ COLD-START (cold_fraction={cfg.evaluation.cold_fraction:.0%} user ít tương tác nhất trong train):")
    print(pd.DataFrame.from_dict(results_cold, orient="index").reindex(order).to_string())
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataco.yaml")
    ap.add_argument("--run-tag", default=None)
    args = ap.parse_args()
    run(args.config, args.run_tag)


if __name__ == "__main__":
    main()