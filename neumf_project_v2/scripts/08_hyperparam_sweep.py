"""scripts/08_hyperparam_sweep.py — Sweep có hệ thống 3 siêu tham số kiến
trúc/huấn luyện (embedding_dim, số lớp MLP, negative_ratio) cho NeuMF, đúng
mục 4.5/5.3 đề cương chi tiết ("Tinh chỉnh siêu tham số" / "Ablation study:
ảnh hưởng của số chiều embedding, số lớp MLP, tỉ lệ negative sampling").

Thiết kế:
- Coordinate sweep: mỗi siêu tham số được quét ĐỘC LẬP quanh cấu hình mặc
  định (baseline lấy từ configs/*.yaml), giữ nguyên 2 trục còn lại — rẻ hơn
  grid đầy đủ (3^3=27 trial) và cho ra đúng 3 đường "sensitivity" dễ đọc,
  đúng tinh thần mục 5.3 báo cáo (phân tích TỪNG yếu tố).
- Mỗi trial chỉ huấn luyện NeuMF-Scratch (bỏ qua pretrain GMF/MLP riêng,
  EarlyFusion, baseline cổ điển) để giữ chi phí sweep hợp lý — pipeline dữ
  liệu (audit/split/negative-sampling) chỉ chạy 1 lần, dùng lại cho mọi trial.
- Chọn cấu hình thắng cuộc theo NDCG@10 trên VALIDATION, KHÔNG dùng test để
  chọn siêu tham số (đúng lưu ý chống rò rỉ dữ liệu ở "Đề cương sơ bộ" mục 3:
  "Optuna/Grid Search kết hợp tập validation riêng biệt, không dùng tập test
  để tránh rò rỉ dữ liệu"). Cột test_NDCG@10/test_HR@10 trong bảng kết quả
  chỉ để MINH HOẠ xu hướng ablation, không phải số liệu chọn model.
- Cấu hình thắng cuộc cần được chạy lại bằng 03_run_experiment.py (đầy đủ
  pretrain GMF/MLP + mọi baseline) để lấy số liệu CHÍNH THỨC báo cáo ở
  Chương 4/5 — script này chỉ phục vụ khâu tinh chỉnh, không thay thế
  03_run_experiment.py.

Chạy:
    python scripts/08_hyperparam_sweep.py --config configs/dataco.yaml
    python scripts/08_hyperparam_sweep.py --config configs/hm_subset.yaml --seed 42

Kết quả:
    outputs/tables/sweep_<dataset>/sweep_embedding_dim.csv
    outputs/tables/sweep_<dataset>/sweep_mlp_layers.csv
    outputs/tables/sweep_<dataset>/sweep_negative_ratio.csv
    outputs/tables/sweep_<dataset>/sweep_all.csv
    outputs/tables/sweep_<dataset>/best_config.json
    outputs/figures/sweep_<dataset>/hyperparam_sensitivity.png
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import build_adapter
from src.data_pipeline.dataset import TrainDataset
from src.data_pipeline.negative_sampling import build_user_positive_sets
from src.data_pipeline.preprocessing import build_interactions, apply_feedback_weights
from src.data_pipeline.splitting import temporal_leave_one_out, assert_disjoint_splits
from src.evaluation.full_ranking import build_full_ranking_records, evaluate_torch_model
from src.models.neumf import NeuMF
from src.training.trainer import train_one_model, make_optimizer, get_device
from src.utils.seed import seed_everything
from src.utils.io import ensure_dir, write_json

EMBEDDING_DIM_GRID = [8, 16, 32, 64, 128]
MLP_DEPTH_GRID = [2, 3, 4, 5, 6]
NEGATIVE_RATIO_GRID = [1, 2, 4, 8, 16]


def make_tower(input_size: int, depth: int) -> list[int]:
    """Tower giảm dần kiểu halving bắt đầu từ input_size = 2*embedding_dim
    (bắt buộc theo NeuMF.__init__), vd. make_tower(64, 4) = [64, 32, 16, 8]."""
    layers = [input_size]
    for _ in range(depth - 1):
        layers.append(max(2, layers[-1] // 2))
    return layers


def build_pipeline(cfg, adapter):
    """Chạy 1 lần: audit -> k-core -> split -> feedback weights -> eval records.
    Không phụ thuộc siêu tham số kiến trúc nên dùng lại được cho mọi trial."""
    events = adapter.load_events()
    data = build_interactions(events, cfg.dataset.k_core)
    train_df, val_df, test_df = temporal_leave_one_out(data.df, cfg.dataset.min_interactions_for_loo)
    assert_disjoint_splits(train_df, val_df, test_df)
    train_df, val_df, test_df, _ = apply_feedback_weights(
        train_df, val_df, test_df, cfg.feedback.mode, cfg.feedback.confidence_alpha
    )
    train_pos = build_user_positive_sets(train_df, data.n_users)
    val_records = build_full_ranking_records(val_df, data.n_items, train_pos)
    train_val = pd.concat([train_df, val_df], ignore_index=True)
    train_val_pos = build_user_positive_sets(train_val, data.n_users)
    test_records = build_full_ranking_records(test_df, data.n_items, train_val_pos)
    return data, train_df, train_pos, val_records, test_records


def run_trial(cfg, pipeline, embedding_dim, mlp_layers, negative_ratio, device, seed):
    data, train_df, train_pos, val_records, test_records = pipeline

    seed_everything(seed)
    train_dataset = TrainDataset(train_df, data.n_items, train_pos, negative_ratio, seed=seed)
    model = NeuMF(data.n_users, data.n_items, embedding_dim, mlp_layers, cfg.model.dropout).to(device)
    opt = make_optimizer(
        cfg.training.finetune_optimizer, model.parameters(),
        cfg.training.finetune_lr, cfg.training.weight_decay,
    )

    def eval_fn(m, records):
        return evaluate_torch_model(
            m, records, k_values=cfg.evaluation.k_values, device=device,
            tie_seed=cfg.evaluation.tie_break_seed,
        )

    t0 = time.perf_counter()
    model, _, meta = train_one_model(
        model, train_dataset, val_records, eval_fn, opt, device,
        cfg.training.max_epochs_finetune, cfg.training.patience, cfg.training.batch_size,
        monitor=cfg.training.monitor, seed=seed,
        model_name=f"sweep_d{embedding_dim}_L{len(mlp_layers)}_neg{negative_ratio}",
    )
    train_time = time.perf_counter() - t0
    test_metrics = eval_fn(model, test_records)

    return {
        "embedding_dim": embedding_dim,
        "mlp_layers": "-".join(map(str, mlp_layers)),
        "n_mlp_layers": len(mlp_layers),
        "negative_ratio": negative_ratio,
        "val_NDCG@10_best": meta["best_metric"],
        "best_epoch": meta["best_epoch"],
        "test_NDCG@10": test_metrics.get("NDCG@10"),
        "test_HR@10": test_metrics.get("HR@10"),
        "n_parameters": meta["n_parameters"],
        "train_time_s": round(train_time, 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/dataco.yaml")
    ap.add_argument("--seed", type=int, default=42, help="Seed cố định cho mọi trial (cô lập ảnh hưởng của siêu tham số, không phải của random seed)")
    ap.add_argument(
        "--tag", default=None,
        help="Tên thư mục output (mặc định: cfg.dataset.name). BẮT BUỘC truyền riêng khi sweep "
             "'configs/hm.yaml' và 'configs/hm_subset.yaml' trong cùng dự án -- cả hai đều có "
             "dataset.name='hm' nên sẽ ghi đè lẫn nhau nếu không phân biệt bằng --tag (vd. --tag hm_subset).",
    )
    args = ap.parse_args()

    cfg, adapter = build_adapter(args.config)
    tag = args.tag or cfg.dataset.name
    device = get_device(cfg.training.device)

    base_dim = cfg.model.embedding_dim
    base_layers = list(cfg.model.mlp_layers)
    base_depth = len(base_layers)
    base_neg = cfg.training.negative_ratio

    print(f"\n{'='*70}\n  🔬 HYPERPARAMETER SWEEP — {tag}\n{'='*70}")
    print(f"  Baseline: embedding_dim={base_dim}, mlp_layers={base_layers}, negative_ratio={base_neg}")
    print("  Xây dựng pipeline dữ liệu (chạy 1 lần, dùng lại cho mọi trial)...")
    pipeline = build_pipeline(cfg, adapter)
    print(f"  ✓ {pipeline[0].n_users:,} users | {pipeline[0].n_items:,} items\n")

    sweeps = {
        "embedding_dim": [
            (d, make_tower(2 * d, base_depth), base_neg) for d in EMBEDDING_DIM_GRID
        ],
        "mlp_layers": [
            (base_dim, make_tower(2 * base_dim, depth), base_neg) for depth in MLP_DEPTH_GRID
        ],
        "negative_ratio": [
            (base_dim, base_layers, r) for r in NEGATIVE_RATIO_GRID
        ],
    }

    out_dir = ensure_dir(PROJECT_ROOT / "outputs" / "tables" / f"sweep_{tag}")
    fig_dir = ensure_dir(PROJECT_ROOT / "outputs" / "figures" / f"sweep_{tag}")

    all_rows = []
    for axis, trials in sweeps.items():
        print(f"--- Sweep: {axis} ({len(trials)} trials) ---")
        rows = []
        for dim, layers, neg in trials:
            row = run_trial(cfg, pipeline, dim, layers, neg, device, args.seed)
            row["sweep_axis"] = axis
            rows.append(row)
            all_rows.append(row)
            print(
                f"  dim={dim:>4} layers={layers!s:<22} neg={neg:>2} -> "
                f"val_NDCG@10={row['val_NDCG@10_best']:.4f}  test_NDCG@10={row['test_NDCG@10']:.4f} "
                f"({row['train_time_s']:.0f}s)"
            )
        pd.DataFrame(rows).to_csv(out_dir / f"sweep_{axis}.csv", index=False)
        print()

    full_df = pd.DataFrame(all_rows)
    full_df.to_csv(out_dir / "sweep_all.csv", index=False)

    best = full_df.loc[full_df["val_NDCG@10_best"].idxmax()]
    best_layers = [int(x) for x in str(best["mlp_layers"]).split("-")]
    print(f"{'='*70}\n  🏆 CẤU HÌNH TỐT NHẤT (theo VALIDATION NDCG@10 — chưa nhìn test)\n{'='*70}")
    print(f"  embedding_dim={int(best['embedding_dim'])}, mlp_layers={best_layers}, "
          f"negative_ratio={int(best['negative_ratio'])}")
    print(f"  val_NDCG@10={best['val_NDCG@10_best']:.4f}")
    print(
        f"\n  ➜ Cập nhật configs/{tag}.yaml (model.embedding_dim, model.mlp_layers, "
        f"training.negative_ratio) theo cấu hình trên rồi chạy lại:\n"
        f"      python run.py all --dataset {tag}\n"
        f"    để lấy số liệu test CHÍNH THỨC (đầy đủ pretrain + baseline) cho báo cáo."
    )

    write_json(out_dir / "best_config.json", {
        "dataset": cfg.dataset.name,
        "tag": tag,
        "embedding_dim": int(best["embedding_dim"]),
        "mlp_layers": best_layers,
        "negative_ratio": int(best["negative_ratio"]),
        "val_NDCG@10": float(best["val_NDCG@10_best"]),
        "selected_by": "validation NDCG@10 (test không dùng để chọn hyperparameter)",
        "seed": args.seed,
    })

    _plot_sensitivity(full_df, tag, fig_dir)
    print(f"\n  → Bảng:   {out_dir}\n  → Biểu đồ: {fig_dir}")


def _plot_sensitivity(full_df: pd.DataFrame, dataset_name: str, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes_spec = [
        ("embedding_dim", "embedding_dim", "Embedding dim"),
        ("mlp_layers", "n_mlp_layers", "Số lớp MLP"),
        ("negative_ratio", "negative_ratio", "Negative ratio"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (axis, xcol, xlabel) in zip(axes, axes_spec):
        sub = full_df[full_df["sweep_axis"] == axis].sort_values(xcol)
        if sub.empty:
            continue
        ax.plot(sub[xcol], sub["val_NDCG@10_best"], "o-", label="Validation (dùng để chọn)", color="#2980b9")
        ax.plot(sub[xcol], sub["test_NDCG@10"], "s--", label="Test (chỉ minh hoạ)", color="#e74c3c", alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("NDCG@10")
        ax.set_title(f"Sensitivity: {axis}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Hyperparameter Sensitivity — {dataset_name} (NeuMF-Scratch)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "hyperparam_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
