"""07_compare_datasets.py — So sánh trực quan hiệu năng mô hình giữa DataCo và H&M.

Do quy mô catalog và mật độ tương tác của hai bộ dữ liệu chênh lệch rất lớn
(DataCo: 100 sản phẩm, mật độ 8,16%; H&M: 1.080 sản phẩm, mật độ 0,62%), việc
vẽ chung một biểu đồ cột với một trục giá trị duy nhất sẽ khiến các cột của
H&M gần như biến mất cạnh DataCo và gây hiểu lầm về mức độ chênh lệch tuyệt đối.

Giải pháp: dùng bố cục small-multiples 2x2 (hàng = HR@10 / NDCG@10, cột =
DataCo / H&M), mỗi ô một trục giá trị riêng phù hợp thang đo của chính bộ dữ
liệu đó -- không phải trục kép (dual-axis) trên cùng một ô. Màu sắc tái sử dụng
đúng bảng MODEL_COLORS đã dùng xuyên suốt các biểu đồ khác trong dự án
(scripts/05_evaluate.py) để giữ tính nhất quán trực quan của toàn báo cáo.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METHOD_ORDER = [
    "Random", "MostPopular", "ItemKNN", "BPR-MF",
    "GMF", "MLP", "EarlyFusion", "NeuMF-Scratch", "NeuMF-Pretrained",
]
MODEL_COLORS = {
    "Random":           "#95a5a6",
    "MostPopular":      "#e67e22",
    "ItemKNN":          "#27ae60",
    "BPR-MF":           "#2980b9",
    "GMF":              "#8e44ad",
    "MLP":              "#c0392b",
    "EarlyFusion":      "#1abc9c",
    "NeuMF-Scratch":    "#16a085",
    "NeuMF-Pretrained": "#d4ac0d",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FCFCFB",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "grid.color": "#E1E0D9",
})


def _panel(ax, df, metric, dataset_label, exclude_random=True):
    models = [m for m in METHOD_ORDER if m in df.index]
    if exclude_random:
        models = [m for m in models if m != "Random"]
    vals = df.loc[models, metric].values
    colors = [MODEL_COLORS.get(m, "#555") for m in models]
    best_idx = vals.argmax()

    bars = ax.barh(models, vals, color=colors, edgecolor="white", height=0.62, alpha=0.92)
    bars[best_idx].set_edgecolor("#0b0b0b")
    bars[best_idx].set_linewidth(2.2)
    for bar, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{v:.4f}", va="center", fontsize=9, color="#0b0b0b")
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_title(f"{metric} — {dataset_label}")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=9.5)


def main():
    dataco_dir = PROJECT_ROOT / "outputs" / "tables" / "dataco_20260918_verify"
    hm_dir = PROJECT_ROOT / "outputs" / "tables" / "hm_20260918_verify"
    out_dir = PROJECT_ROOT / "outputs" / "figures" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataco = pd.read_csv(dataco_dir / "primary_results.csv", index_col=0)
    hm = pd.read_csv(hm_dir / "primary_results.csv", index_col=0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _panel(axes[0, 0], dataco, "HR@10", "DataCo (100 sản phẩm)")
    _panel(axes[0, 1], hm, "HR@10", "H&M (1.080 sản phẩm)")
    _panel(axes[1, 0], dataco, "NDCG@10", "DataCo (100 sản phẩm)")
    _panel(axes[1, 1], hm, "NDCG@10", "H&M (1.080 sản phẩm)")

    fig.suptitle(
        "So sánh hiệu năng mô hình: DataCo và H&M (Full Ranking, mỗi biểu đồ có thang trục riêng)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out_path = out_dir / "dataco_vs_hm_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
