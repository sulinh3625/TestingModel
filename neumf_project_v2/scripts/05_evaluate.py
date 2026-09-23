"""05_evaluate.py — Đánh giá mô hình toàn diện V2.

Sinh ra:
  outputs/figures/<run_tag>/
    01_training_loss_curves.png
    02_validation_ndcg_curves.png
    03_ndcg_comparison.png
    04_hr_comparison.png
    05_k_sensitivity_ndcg.png
    06_k_sensitivity_hr.png
    07_long_tail_analysis.png
    08_beyond_accuracy.png
    09_radar_chart.png
    10_training_summary.png
    11_heatmap_all_metrics.png
    12_cold_start_analysis.png
  outputs/tables/<run_tag>/
    primary_results.csv
    sampled_results.csv
    long_tail_results.csv
    cold_start_results.csv
    training_summary.csv
    beyond_accuracy.csv
    evaluation_report.txt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ──────────────────────────────────────────────────────────
# Global style
# ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#F8F9FA",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

MODEL_COLORS = {
    "Random":           "#95a5a6",
    "MostPopular":      "#e67e22",
    "ItemKNN":          "#27ae60",
    "BPR-MF":           "#2980b9",
    "CategoryPopularity": "#f39c12",
    "GMF":              "#8e44ad",
    "MLP":              "#c0392b",
    "EarlyFusion":      "#1abc9c",
    "LightGCN":         "#34495e",
    "SASRec":           "#7f8c8d",
    "NeuMF-Scratch":    "#16a085",
    "NeuMF-Pretrained": "#d4ac0d",
}
MODEL_MARKERS = {
    "Random": "x", "MostPopular": "s", "ItemKNN": "D",
    "BPR-MF": "^", "GMF": "o", "MLP": "v",
    "EarlyFusion": "h", "LightGCN": "8", "CategoryPopularity": "p", "SASRec": "X",
    "NeuMF-Scratch": "*", "NeuMF-Pretrained": "P",
}
METHOD_ORDER = [
    "Random", "MostPopular", "ItemKNN", "BPR-MF", "CategoryPopularity",
    "GMF", "MLP", "EarlyFusion", "LightGCN", "SASRec", "NeuMF-Scratch", "NeuMF-Pretrained",
]
NEURAL_MODELS = ["GMF", "MLP", "EarlyFusion", "LightGCN", "SASRec", "NeuMF-Scratch", "NeuMF-Pretrained"]
K_SENSITIVITY = [1, 3, 5, 10, 20]


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────
def find_latest_run(experiments_dir: Path) -> str:
    runs = [d for d in experiments_dir.iterdir()
            if d.is_dir() and d.name not in ("_temp_configs",)]
    if not runs:
        raise FileNotFoundError(f"Không tìm thấy run nào trong {experiments_dir}")
    return sorted(runs, key=lambda d: d.stat().st_mtime)[-1].name


def load_run(run_tag: str):
    run_dir = PROJECT_ROOT / "outputs" / "experiments" / run_tag
    if not run_dir.exists():
        raise FileNotFoundError(f"Run '{run_tag}' không tồn tại: {run_dir}")
    with open(run_dir / "results.json", encoding="utf-8") as f:
        results = json.load(f)
    with open(run_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    histories = {}
    for name in NEURAL_MODELS:
        fname = f"history_{name.lower().replace('-', '_').replace(' ', '_')}.csv"
        p = run_dir / fname
        if p.exists():
            histories[name] = pd.read_csv(p)
    return results, metadata, histories, run_dir


def _order(df: pd.DataFrame) -> pd.DataFrame:
    order = [m for m in METHOD_ORDER if m in df.index]
    return df.reindex(order)


def _savefig(fig, path: Path, name: str):
    fig.savefig(path / name, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✓ {name}")


# ────────────────────────────────────────────────────────
# Beyond-accuracy metrics — lấy từ results['beyond_accuracy'] đã lưu
# (03_run_experiment.py lưu catalog_coverage và avg_rec_popularity thực)
# ────────────────────────────────────────────────────────
def compute_beyond_accuracy(results: dict, metadata: dict) -> pd.DataFrame:
    """Lấy beyond-accuracy metrics đã được tính thực từ actual Top-K recommendations.

    Nếu run cũ chưa có 'beyond_accuracy' trong results.json (trước khi sửa
    03_run_experiment.py), hàm này trả về DataFrame rỗng và in cảnh báo.
    """
    beyond_data = results.get("beyond_accuracy", {})
    n_items = metadata.get("n_items", 1)

    if not beyond_data:
        print(
            "  ⚠️  'beyond_accuracy' không có trong results.json.\n"
            "      Chạy lại 03_run_experiment.py để tính giá trị thực."
        )
        return pd.DataFrame()

    rows = []
    for name in METHOD_ORDER:
        if name not in beyond_data:
            continue
        ba = beyond_data[name]
        cov = ba.get("catalog_coverage", 0.0)
        arp = ba.get("avg_rec_popularity", 0.0)
        hrr = ba.get("head_rec_rate", None)
        nov = ba.get("novelty", None)
        # Normalize ARP theo số training interaction (để có thể so sánh giữa dataset)
        n_train = metadata.get("train", 1)
        arp_norm = arp / max(n_train, 1)  # giá trị tương đối, càng cao càng popular
        row = {
            "Model": name,
            "Coverage@K": round(cov, 4),
            "Avg Rec Popularity": round(arp, 2),
            "ARP (norm ×100)": round(arp_norm * 100, 4),
        }
        if hrr is not None:
            row["Head Rec Rate"] = round(hrr, 4)
        if nov is not None:
            row["Novelty"] = round(nov, 4)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Model")



# ────────────────────────────────────────────────────────
# K-sensitivity: chỉ dùng K values đã được evaluate thực, không ước tính
# ────────────────────────────────────────────────────────
def build_k_sensitivity(results: dict) -> dict[str, dict[str, list]]:
    """Xây dựng dữ liệu K-sensitivity từ các k_values đã tính thực trong results.

    Chỉ dùng giá trị đã evaluate thực — không extrapolate hay estimate.
    K_SENSITIVITY sẽ là tập các K thực sự có trong kết quả.
    """
    primary = results.get("primary", {})
    if not primary:
        return {}

    # Lấy danh sách k_values thực từ cột HR@K và NDCG@K
    first_method = next(iter(primary.values()), {})
    real_k_values = sorted(set(
        int(key.split("@")[1])
        for key in first_method
        if "HR@" in key or "NDCG@" in key
    ))

    sens = {}
    for name in METHOD_ORDER:
        if name not in primary:
            continue
        r = primary[name]
        hr_vals = [r.get(f"HR@{k}", 0.0) for k in real_k_values]
        ndcg_vals = [r.get(f"NDCG@{k}", 0.0) for k in real_k_values]
        sens[name] = {"HR": hr_vals, "NDCG": ndcg_vals, "K": real_k_values}
    return sens


# ──────────────────────────────────────────────────────────
# Tables
# ──────────────────────────────────────────────────────────
def generate_tables(results, metadata, beyond_df, table_dir: Path):
    table_dir.mkdir(parents=True, exist_ok=True)

    primary_df = _order(pd.DataFrame.from_dict(results["primary"], orient="index"))
    primary_df.index.name = "Model"
    primary_df.to_csv(table_dir / "primary_results.csv")

    sampled_df = _order(pd.DataFrame.from_dict(results["sampled_99"], orient="index"))
    sampled_df.index.name = "Model"
    sampled_df.to_csv(table_dir / "sampled_results.csv")

    tail_df = _order(pd.DataFrame.from_dict(results["long_tail"], orient="index"))
    tail_df.index.name = "Model"
    tail_df.to_csv(table_dir / "long_tail_results.csv")

    cold_df = _order(pd.DataFrame.from_dict(results.get("cold_start", {}), orient="index"))
    cold_df.index.name = "Model"
    cold_df.to_csv(table_dir / "cold_start_results.csv")

    beyond_df.to_csv(table_dir / "beyond_accuracy.csv")

    # Training summary
    train_info = metadata.get("training", {})
    rows = []
    for name in [m for m in METHOD_ORDER if m in primary_df.index]:
        info = train_info.get(name, {})
        rows.append({
            "Model": name,
            "Parameters": info.get("n_parameters", "N/A"),
            "Best Epoch": info.get("best_epoch", "N/A"),
            "Best Val NDCG@10": round(info.get("best_metric", 0), 4) if info.get("best_metric") else "N/A",
            "Train Time (s)": round(info.get("train_time_s", 0), 1),
        })
    train_df = pd.DataFrame(rows).set_index("Model")
    train_df.to_csv(table_dir / "training_summary.csv")

    # Text report
    report = _build_report(primary_df, sampled_df, tail_df, cold_df, beyond_df, train_df, metadata)
    (table_dir / "evaluation_report.txt").write_text(report, encoding="utf-8")
    print(report)

    return primary_df, sampled_df, tail_df, cold_df, train_df


def _build_report(primary_df, sampled_df, tail_df, cold_df, beyond_df, train_df, metadata):
    L = []
    sep = "=" * 70
    sub = "─" * 50
    L += [sep, "  BÁO CÁO ĐÁNH GIÁ MÔ HÌNH — NeuMF Recommendation V2", sep, ""]

    L += ["📊 THÔNG TIN DATASET", sub]
    for k, v in [
        ("Dataset", metadata.get("dataset")),
        ("Users", f"{metadata.get('n_users',0):,}"),
        ("Items", f"{metadata.get('n_items',0):,}"),
        ("Interactions", f"{metadata.get('n_interactions',0):,}"),
        ("Density", f"{metadata['n_interactions']/(metadata['n_users']*metadata['n_items']):.4%}"),
        ("K-core", metadata.get("k_core")),
        ("Feedback", metadata.get("feedback", {}).get("mode")),
        ("Train/Val/Test", f"{metadata.get('train',0):,} / {metadata.get('validation',0):,} / {metadata.get('test',0):,}"),
        ("Head items", metadata.get("n_head_items")),
        ("Long-tail test users", metadata.get("n_long_tail_test_users")),
    ]:
        L.append(f"  {k:<22} {v}")
    L.append("")

    L += ["📋 FULL-RANKING (Primary)", sub]
    L.append(primary_df.to_string(float_format="%.4f")); L.append("")
    best = primary_df["NDCG@10"].idxmax()
    L.append(f"  🏆 Best: {best} — NDCG@10 = {primary_df.loc[best,'NDCG@10']:.4f}"); L.append("")

    L += ["📋 SAMPLED-99 (Secondary)", sub]
    L.append(sampled_df.to_string(float_format="%.4f")); L.append("")

    L += ["📋 LONG-TAIL", sub]
    L.append(f"  Head items: {metadata.get('n_head_items')} | "
             f"Head test users: {metadata.get('n_head_test_users')} | "
             f"Tail test users: {metadata.get('n_long_tail_test_users')}")
    L.append(tail_df.to_string(float_format="%.4f")); L.append("")

    L += ["📋 COLD-START", sub]
    cold_frac = metadata.get("cold_fraction")
    L.append(f"  Cold users: {metadata.get('n_cold_users')} ({cold_frac:.0%} ít tương tác nhất trong train) | "
             f"Cold test users: {metadata.get('n_cold_test_users')} | Warm test users: {metadata.get('n_warm_test_users')}"
             if cold_frac is not None else "  (run cũ, chưa có dữ liệu cold-start — chạy lại 03_run_experiment.py)")
    if not cold_df.empty:
        L.append(cold_df.to_string(float_format="%.4f"))
    L.append("")

    L += ["📋 BEYOND-ACCURACY", sub]
    L.append(beyond_df.to_string(float_format="%.4f")); L.append("")

    L += ["⏱ TRAINING SUMMARY", sub]
    L.append(train_df.to_string()); L.append("")

    L += ["📝 PHÂN TÍCH", sub]
    neural = [m for m in NEURAL_MODELS if m in primary_df.index]
    classical = [m for m in ["MostPopular","ItemKNN","BPR-MF"] if m in primary_df.index]
    if neural and classical:
        bn = primary_df.loc[neural, "NDCG@10"].idxmax()
        bc = primary_df.loc[classical, "NDCG@10"].idxmax()
        imp = (primary_df.loc[bn,"NDCG@10"] - primary_df.loc[bc,"NDCG@10"]) / primary_df.loc[bc,"NDCG@10"] * 100
        L.append(f"  Neural vs Classical: {bn} vs {bc} → {imp:+.2f}%")
    if "NeuMF-Scratch" in primary_df.index and "NeuMF-Pretrained" in primary_df.index:
        s = primary_df.loc["NeuMF-Scratch","NDCG@10"]
        p = primary_df.loc["NeuMF-Pretrained","NDCG@10"]
        L.append(f"  Pretrained vs Scratch: {(p-s)/s*100:+.2f}% → {'Pretrained' if p>s else 'Scratch'} wins")
    if "GMF" in primary_df.index and "MLP" in primary_df.index:
        g = primary_df.loc["GMF","NDCG@10"]; m = primary_df.loc["MLP","NDCG@10"]
        L.append(f"  GMF vs MLP: {g:.4f} vs {m:.4f} → {'GMF' if g>m else 'MLP'} wins")
    # Early Fusion vs Late Fusion (NeuMF)
    if "EarlyFusion" in primary_df.index and "NeuMF-Scratch" in primary_df.index:
        ef = primary_df.loc["EarlyFusion","NDCG@10"]
        ns = primary_df.loc["NeuMF-Scratch","NDCG@10"]
        winner = "Late Fusion (NeuMF-Scratch)" if ns >= ef else "Early Fusion"
        diff = abs(ns - ef) / max(ef, 1e-9) * 100
        L.append(f"  Early vs Late Fusion: EarlyFusion={ef:.4f} vs NeuMF-Scratch={ns:.4f} → {winner} wins (+{diff:.2f}%)")
    L += ["", sep]
    return "\n".join(L)


# ──────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────
def plot_training_loss(histories, fig_dir):
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, hist in histories.items():
        c = MODEL_COLORS.get(name, "#333")
        ax.plot(hist["epoch"], hist["loss"], marker=MODEL_MARKERS.get(name,"o"),
                markersize=5, label=name, color=c, linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
    ax.set_title("Training Loss Curves — Neural Models")
    ax.legend(loc="upper right"); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    fig.tight_layout(); _savefig(fig, fig_dir, "01_training_loss_curves.png")


def plot_validation_ndcg(histories, metadata, fig_dir):
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, hist in histories.items():
        if "NDCG@10" not in hist.columns: continue
        c = MODEL_COLORS.get(name, "#333")
        ax.plot(hist["epoch"], hist["NDCG@10"], marker=MODEL_MARKERS.get(name,"o"),
                markersize=5, label=name, color=c, linewidth=2)
        info = metadata.get("training", {}).get(name, {})
        ep, val = info.get("best_epoch"), info.get("best_metric")
        if ep and val:
            ax.scatter([ep], [val], color=c, s=120, zorder=6, edgecolors="black", linewidths=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("NDCG@10")
    ax.set_title("Validation NDCG@10 — Neural Models (★ = best epoch)")
    ax.legend(loc="lower right"); ax.set_xlim(left=0)
    fig.tight_layout(); _savefig(fig, fig_dir, "02_validation_ndcg_curves.png")


def plot_ndcg_bar(primary_df, fig_dir):
    models = [m for m in METHOD_ORDER if m in primary_df.index and m != "Random"]
    vals = primary_df.loc[models, "NDCG@10"].values
    colors = [MODEL_COLORS.get(m,"#555") for m in models]
    best_idx = vals.argmax()

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(models, vals, color=colors, edgecolor="white", height=0.65, alpha=0.90)
    bars[best_idx].set_edgecolor("black"); bars[best_idx].set_linewidth(2.5)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontweight="bold", fontsize=10)
    ax.set_xlabel("NDCG@10"); ax.set_title("NDCG@10 — So sánh tổng thể (Full-Ranking)")
    ax.set_xlim(0, max(vals)*1.12); ax.invert_yaxis()
    fig.tight_layout(); _savefig(fig, fig_dir, "03_ndcg_comparison.png")


def plot_hr_bar(primary_df, fig_dir):
    models = [m for m in METHOD_ORDER if m in primary_df.index and m != "Random"]
    vals5  = primary_df.loc[models, "HR@5"].values
    vals10 = primary_df.loc[models, "HR@10"].values
    x = np.arange(len(models)); w = 0.36

    fig, ax = plt.subplots(figsize=(12, 6))
    b5  = ax.bar(x - w/2, vals5,  w, label="HR@5",  color="#3498db", alpha=0.85)
    b10 = ax.bar(x + w/2, vals10, w, label="HR@10", color="#e74c3c", alpha=0.85)
    for bar, v in [(b5, vals5), (b10, vals10)]:
        for b, val in zip(bar, v):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Hit Rate"); ax.set_title("HR@5 và HR@10 — So sánh các Model")
    ax.legend(); ax.set_ylim(0, max(vals10)*1.12)
    fig.tight_layout(); _savefig(fig, fig_dir, "04_hr_comparison.png")


def plot_k_sensitivity(sens, fig_dir):
    models_show = [m for m in METHOD_ORDER if m in sens and m != "Random"]
    if not models_show:
        print("  ⚠️  Không có dữ liệu K-sensitivity để vẽ.")
        return

    # Lấy K values thực từ data (không hardcode)
    k_vals = sens[models_show[0]].get("K", [5, 10])

    # NDCG
    fig, ax = plt.subplots(figsize=(11, 6))
    for name in models_show:
        c = MODEL_COLORS.get(name, "#555")
        mk = MODEL_MARKERS.get(name, "o")
        ax.plot(k_vals, sens[name]["NDCG"], marker=mk, markersize=6,
                label=name, color=c, linewidth=2)
    ax.set_xlabel("K"); ax.set_ylabel("NDCG@K")
    ax.set_title("K-Sensitivity — NDCG@K (giá trị thực)")
    ax.set_xticks(k_vals); ax.legend(loc="lower right")
    fig.tight_layout(); _savefig(fig, fig_dir, "05_k_sensitivity_ndcg.png")

    # HR
    fig, ax = plt.subplots(figsize=(11, 6))
    for name in models_show:
        c = MODEL_COLORS.get(name, "#555")
        mk = MODEL_MARKERS.get(name, "o")
        ax.plot(k_vals, sens[name]["HR"], marker=mk, markersize=6,
                label=name, color=c, linewidth=2)
    ax.set_xlabel("K"); ax.set_ylabel("HR@K")
    ax.set_title("K-Sensitivity — HR@K (giá trị thực)")
    ax.set_xticks(k_vals); ax.legend(loc="lower right")
    fig.tight_layout(); _savefig(fig, fig_dir, "06_k_sensitivity_hr.png")



def plot_long_tail(primary_df, tail_df, metadata, fig_dir):
    models = [m for m in METHOD_ORDER if m in primary_df.index and m in tail_df.index and m != "Random"]
    overall = primary_df.loc[models, "NDCG@10"].values
    tail    = tail_df.loc[models, "NDCG@10"].values
    x = np.arange(len(models)); w = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Left: grouped bars
    ax = axes[0]
    ax.bar(x-w/2, overall, w, label="Overall", color="#3498db", alpha=0.88)
    ax.bar(x+w/2, tail,    w, label="Long-tail", color="#e74c3c", alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("NDCG@10"); ax.set_title("Overall vs Long-tail NDCG@10")
    ax.legend()

    # Right: drop% (performance drop from overall to tail)
    drop = [(o - t) / max(o, 1e-9) * 100 for o, t in zip(overall, tail)]
    colors_drop = [MODEL_COLORS.get(m,"#555") for m in models]
    ax2 = axes[1]
    bars = ax2.barh(models, drop, color=colors_drop, edgecolor="white", height=0.6, alpha=0.88)
    for bar, d in zip(bars, drop):
        ax2.text(d + 0.5, bar.get_y()+bar.get_height()/2,
                 f"{d:.1f}%", va="center", fontsize=10)
    ax2.set_xlabel("Performance Drop (%)")
    ax2.set_title("Long-tail Performance Drop\n(Overall → Tail, NDCG@10)")
    ax2.invert_yaxis()

    n_tail = metadata.get("n_long_tail_test_users", "?")
    n_head = metadata.get("n_head_items", "?")
    fig.suptitle(f"Long-tail Analysis (Head={n_head} items, Tail={n_tail} test users)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); _savefig(fig, fig_dir, "07_long_tail_analysis.png")


def plot_cold_start(primary_df, cold_df, metadata, fig_dir):
    if cold_df.empty:
        print("  ⚠️  Không có cold-start data để vẽ (cần chạy lại 03_run_experiment.py).")
        return
    models = [m for m in METHOD_ORDER if m in primary_df.index and m in cold_df.index and m != "Random"]
    if not models:
        return
    overall = primary_df.loc[models, "NDCG@10"].values
    cold = cold_df.loc[models, "NDCG@10"].values
    x = np.arange(len(models)); w = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.bar(x - w/2, overall, w, label="Overall", color="#3498db", alpha=0.88)
    ax.bar(x + w/2, cold, w, label="Cold-start users", color="#e74c3c", alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("NDCG@10"); ax.set_title("Overall vs Cold-start NDCG@10")
    ax.legend()

    drop = [(o - c) / max(o, 1e-9) * 100 for o, c in zip(overall, cold)]
    colors_drop = [MODEL_COLORS.get(m, "#555") for m in models]
    ax2 = axes[1]
    bars = ax2.barh(models, drop, color=colors_drop, edgecolor="white", height=0.6, alpha=0.88)
    for bar, d in zip(bars, drop):
        ax2.text(d + 0.5, bar.get_y() + bar.get_height()/2, f"{d:.1f}%", va="center", fontsize=10)
    ax2.set_xlabel("Performance Drop (%)")
    ax2.set_title("Cold-start Performance Drop\n(Overall → Cold, NDCG@10)")
    ax2.invert_yaxis()

    n_cold = metadata.get("n_cold_test_users", "?")
    frac = metadata.get("cold_fraction")
    frac_str = f"{frac:.0%}" if frac is not None else "?"
    fig.suptitle(f"Cold-start Analysis (cold_fraction={frac_str}, Cold test users={n_cold})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); _savefig(fig, fig_dir, "12_cold_start_analysis.png")


def plot_beyond_accuracy(beyond_df, fig_dir):
    if beyond_df.empty:
        print("  ⚠️  Không có beyond-accuracy data để vẽ (cần chạy lại 03_run_experiment.py).")
        return

    models = [m for m in METHOD_ORDER if m in beyond_df.index]
    if not models:
        return

    # Chỉ lấy các metric có trong DataFrame thực (dynamic)
    available_metrics = [c for c in beyond_df.columns if beyond_df.loc[models, c].notna().any()]
    df = beyond_df.loc[models, available_metrics]

    colors_m = ["#3498db", "#e74c3c", "#f39c12", "#2ecc71", "#8e44ad", "#1abc9c"]
    x = np.arange(len(models))
    w = max(0.12, 0.8 / max(len(available_metrics), 1))

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (metric, col) in enumerate(zip(available_metrics, colors_m)):
        bars = ax.bar(x + i * w, df[metric], w, label=metric, color=col, alpha=0.85)
        for bar, v in zip(bars, df[metric]):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + w * (len(available_metrics) - 1) / 2)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Beyond-Accuracy Metrics (giá trị thực từ actual Top-K recommendations)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(df.max().max() * 1.2, 1.0))
    fig.tight_layout()
    _savefig(fig, fig_dir, "08_beyond_accuracy.png")



def plot_radar(primary_df, beyond_df, fig_dir):
    """Radar chart so sánh GMF, NeuMF-Scratch, NeuMF-Pretrained, MostPopular."""
    highlight = [m for m in ["MostPopular","GMF","NeuMF-Scratch","NeuMF-Pretrained"]
                 if m in primary_df.index]
    if len(highlight) < 2:
        return

    radar_metrics = ["NDCG@10","HR@10","NDCG@5","HR@5"]
    data = {}
    for m in highlight:
        vals = [primary_df.loc[m, met] for met in radar_metrics]
        data[m] = vals

    # Normalize to [0,1] per metric
    arr = np.array([data[m] for m in highlight])
    mn, mx = arr.min(axis=0), arr.max(axis=0)
    rng = np.where(mx - mn < 1e-9, 1.0, mx - mn)
    norm = {m: list((np.array(data[m]) - mn) / rng) for m in highlight}

    N = len(radar_metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for m in highlight:
        vals = norm[m] + norm[m][:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=MODEL_COLORS.get(m,"#555"),
                label=m, markersize=6)
        ax.fill(angles, vals, alpha=0.08, color=MODEL_COLORS.get(m,"#555"))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics, fontsize=12)
    ax.set_yticklabels([]); ax.set_ylim(0, 1)
    ax.set_title("Radar — Accuracy Metrics (normalized)", pad=20, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout(); _savefig(fig, fig_dir, "09_radar_chart.png")


def plot_training_summary(metadata, fig_dir):
    train_info = metadata.get("training", {})
    models_t = [m for m in METHOD_ORDER if m in train_info and train_info[m].get("train_time_s", 0) > 0]
    if not models_t:
        return

    times  = [train_info[m].get("train_time_s", 0) for m in models_t]
    params = [train_info[m].get("n_parameters", 0) for m in models_t]
    colors = [MODEL_COLORS.get(m,"#555") for m in models_t]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Training time
    ax = axes[0]
    bars = ax.barh(models_t, times, color=colors, edgecolor="white", height=0.6, alpha=0.9)
    for bar, t in zip(bars, times):
        lbl = f"{t:.0f}s" if t < 60 else f"{t/60:.1f}m"
        ax.text(t + max(times)*0.015, bar.get_y()+bar.get_height()/2,
                lbl, va="center", fontweight="bold", fontsize=10)
    ax.set_xlabel("Seconds"); ax.set_title("Training Time")
    ax.invert_yaxis()

    # Parameters
    ax2 = axes[1]
    bars2 = ax2.barh(models_t, params, color=colors, edgecolor="white", height=0.6, alpha=0.9)
    for bar, p in zip(bars2, params):
        if p > 0:
            ax2.text(p + max(params)*0.015, bar.get_y()+bar.get_height()/2,
                     f"{p:,}", va="center", fontweight="bold", fontsize=10)
    ax2.set_xlabel("Parameters")
    ax2.set_title("Model Parameters")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    ax2.invert_yaxis()

    fig.suptitle("Training Summary", fontsize=14, fontweight="bold")
    fig.tight_layout(); _savefig(fig, fig_dir, "10_training_summary.png")


def plot_heatmap(primary_df, tail_df, beyond_df, fig_dir):
    models = [m for m in METHOD_ORDER if m in primary_df.index and m != "Random"]
    cols_primary = ["HR@5","HR@10","NDCG@5","NDCG@10"]
    cols_tail    = ["NDCG@10"]
    cols_beyond  = ["Coverage@K","Novelty"]

    parts = [primary_df.loc[models, cols_primary]]
    if not tail_df.empty and all(c in tail_df.columns for c in cols_tail):
        t = tail_df.loc[[m for m in models if m in tail_df.index], cols_tail].copy()
        t.columns = ["Tail NDCG@10"]
        parts.append(t)
    if not beyond_df.empty and all(c in beyond_df.columns for c in cols_beyond):
        b = beyond_df.loc[[m for m in models if m in beyond_df.index], cols_beyond]
        parts.append(b)

    combined = pd.concat(parts, axis=1).loc[models]
    # Normalize each column to [0,1]
    norm = combined.copy()
    for c in norm.columns:
        mn, mx = norm[c].min(), norm[c].max()
        norm[c] = (norm[c] - mn) / max(mx - mn, 1e-9)

    fig, ax = plt.subplots(figsize=(max(10, len(norm.columns)*1.3), max(6, len(models)*0.7)))
    im = ax.imshow(norm.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(norm.columns)))
    ax.set_xticklabels(norm.columns, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)

    for i in range(len(models)):
        for j in range(len(norm.columns)):
            raw_val = combined.iloc[i, j]
            ax.text(j, i, f"{raw_val:.3f}", ha="center", va="center",
                    fontsize=9, color="black" if 0.25 < norm.iloc[i,j] < 0.85 else "white")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Normalized Score")
    ax.set_title("Heatmap — Tất cả Metrics (normalized)", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout(); _savefig(fig, fig_dir, "11_heatmap_all_metrics.png")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def run_evaluation(run_tag: str | None = None):
    experiments_dir = PROJECT_ROOT / "outputs" / "experiments"
    tag = run_tag or find_latest_run(experiments_dir)

    print("═" * 65)
    print("  📊 ĐÁNH GIÁ MÔ HÌNH — NeuMF Recommendation V2")
    print("═" * 65)
    print(f"\n  Run tag : {tag}")

    results, metadata, histories, run_dir = load_run(tag)

    # Beyond-accuracy
    beyond_df = compute_beyond_accuracy(results, metadata)

    # K-sensitivity
    sens = build_k_sensitivity(results)

    # Tables
    print("\n📋 Sinh bảng kết quả...")
    table_dir = PROJECT_ROOT / "outputs" / "tables" / tag
    primary_df, sampled_df, tail_df, cold_df, train_df = generate_tables(
        results, metadata, beyond_df, table_dir
    )
    print(f"\n  → Saved: {table_dir}")

    # Figures
    print("\n📈 Sinh biểu đồ...")
    fig_dir = PROJECT_ROOT / "outputs" / "figures" / tag
    fig_dir.mkdir(parents=True, exist_ok=True)

    if histories:
        plot_training_loss(histories, fig_dir)
        plot_validation_ndcg(histories, metadata, fig_dir)

    plot_ndcg_bar(primary_df, fig_dir)
    plot_hr_bar(primary_df, fig_dir)
    plot_k_sensitivity(sens, fig_dir)
    plot_long_tail(primary_df, tail_df, metadata, fig_dir)
    plot_cold_start(primary_df, cold_df, metadata, fig_dir)
    plot_beyond_accuracy(beyond_df, fig_dir)
    plot_radar(primary_df, beyond_df, fig_dir)
    plot_training_summary(metadata, fig_dir)
    plot_heatmap(primary_df, tail_df, beyond_df, fig_dir)

    print(f"\n  → Saved: {fig_dir}")
    print("\n" + "═" * 65)
    print("  ✅ ĐÁNH GIÁ HOÀN THÀNH")
    print("═" * 65)
    print(f"  📋 Bảng:    {table_dir}")
    print(f"  📈 Biểu đồ: {fig_dir}\n")
    return table_dir, fig_dir


def main():
    ap = argparse.ArgumentParser(description="Đánh giá mô hình NeuMF V2")
    ap.add_argument("--run-tag", default=None, help="Tên run (mặc định: mới nhất)")
    args = ap.parse_args()
    run_evaluation(args.run_tag)


if __name__ == "__main__":
    main()

