from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.statistics import summarize, paired_wilcoxon
from src.utils.io import ensure_dir, write_json

# Các cặp so sánh chính theo RQ đã chốt trong METHODOLOGY_V2.md:
# RQ1 (NeuMF vs MF/CF), RQ2 (vai trò GMF/MLP + fusion), RQ3 (pretraining).
DEFAULT_COMPARISONS = [
    ("NeuMF-Pretrained", "BPR-MF"),
    ("NeuMF-Pretrained", "MLP"),
    ("NeuMF-Pretrained", "GMF"),
    ("NeuMF-Pretrained", "NeuMF-Scratch"),
    ("NeuMF-Pretrained", "EarlyFusion"),
]


def load_run_results(experiments_dir: Path, run_tags: list[str], split: str = "primary") -> dict[str, dict]:
    """Đọc results.json[split] cho từng run-tag.

    split: "primary" (full-ranking, mặc định) hoặc "sampled_99"/"long_tail"
    nếu muốn tổng hợp multi-seed trên các protocol khác.
    """
    per_seed = {}
    for tag in run_tags:
        path = experiments_dir / tag / "results.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {path}. Chạy scripts/04_multi_seed.py trước, "
                f"hoặc kiểm tra lại run-tag."
            )
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if split not in payload:
            raise KeyError(f"'{split}' không có trong {path} (có: {sorted(payload)})")
        per_seed[tag] = payload[split]
    return per_seed


def build_summary_table(per_seed: dict[str, dict]) -> pd.DataFrame:
    """mean±std cho mọi (method, metric) qua các seed."""
    tags = list(per_seed.keys())
    methods = list(per_seed[tags[0]].keys())
    metrics = [m for m in per_seed[tags[0]][methods[0]].keys()]

    rows = []
    for method in methods:
        row = {"method": method, "n_seeds": len(tags)}
        for metric in metrics:
            values = [per_seed[tag][method][metric] for tag in tags]
            s = summarize(values)
            row[f"{metric}_mean"] = s["mean"]
            row[f"{metric}_std"] = s["std"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def build_significance_table(
    per_seed: dict[str, dict], comparisons: list[tuple[str, str]], primary_metric: str = "NDCG@10"
) -> pd.DataFrame:
    """Wilcoxon signed-rank giữa các cặp method, ghép theo seed, trên primary_metric."""
    tags = list(per_seed.keys())
    rows = []
    for a_name, b_name in comparisons:
        missing = [name for name in (a_name, b_name) if name not in per_seed[tags[0]]]
        if missing:
            rows.append({
                "comparison": f"{a_name} vs {b_name}", "metric": primary_metric,
                "statistic": None, "p_value": None, "note": f"thiếu method: {missing}",
            })
            continue
        a = [per_seed[tag][a_name][primary_metric] for tag in tags]
        b = [per_seed[tag][b_name][primary_metric] for tag in tags]
        result = paired_wilcoxon(a, b)
        mean_diff = sum(a) / len(a) - sum(b) / len(b)
        rows.append({
            "comparison": f"{a_name} vs {b_name}",
            "metric": primary_metric,
            "mean_diff": mean_diff,
            "relative_improvement_pct": (mean_diff / (sum(b) / len(b))) * 100 if sum(b) else None,
            "statistic": result["statistic"],
            "p_value": result["p_value"],
            "significant_at_0.05": result["p_value"] < 0.05,
            "note": "",
        })
    return pd.DataFrame(rows)


def run(config_name: str, seeds: list[int], experiments_root: Path, output_dir: Path, split: str, primary_metric: str):
    run_tags = [f"{config_name}_seed_{seed}" for seed in seeds]
    per_seed = load_run_results(experiments_root, run_tags, split=split)

    summary_df = build_summary_table(per_seed)
    sig_df = build_significance_table(per_seed, DEFAULT_COMPARISONS, primary_metric=primary_metric)

    # Thư mục con riêng theo config_name -- tránh dataco/hm ghi đè lẫn nhau
    # (trước đây output_dir dùng chung 1 đường dẫn cho mọi dataset).
    out_dir = ensure_dir(Path(output_dir) / config_name)
    summary_path = out_dir / f"multi_seed_summary_{split}.csv"
    sig_path = out_dir / f"multi_seed_significance_{split}.csv"
    summary_df.to_csv(summary_path)
    sig_df.to_csv(sig_path, index=False)

    write_json(out_dir / f"multi_seed_meta_{split}.json", {
        "config_name": config_name,
        "seeds": seeds,
        "run_tags": run_tags,
        "split": split,
        "primary_metric": primary_metric,
    })

    print(f"\n✅ Tổng hợp {len(seeds)} seed cho '{config_name}' (split={split})")
    print(f"\n📋 Mean ± Std ({summary_path}):")
    print(summary_df.to_string())
    print(f"\n📋 Wilcoxon signed-rank trên {primary_metric} ({sig_path}):")
    print(sig_df.to_string(index=False))
    return summary_df, sig_df


def main():
    ap = argparse.ArgumentParser(description="Tổng hợp kết quả multi-seed: mean±std + Wilcoxon signed-rank.")
    ap.add_argument("--config-name", required=True, help="Tên dataset trong config, VD 'dataco' — phải khớp run-tag của 04_multi_seed.py")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 2025, 2026, 3407])
    ap.add_argument("--experiments-root", default=str(PROJECT_ROOT / "outputs" / "experiments"))
    ap.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "tables"))
    ap.add_argument("--split", default="primary", choices=["primary", "sampled_99", "long_tail"])
    ap.add_argument("--primary-metric", default="NDCG@10")
    args = ap.parse_args()

    run(
        args.config_name, args.seeds, Path(args.experiments_root), Path(args.output_dir),
        args.split, args.primary_metric,
    )


if __name__ == "__main__":
    main()