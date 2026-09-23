"""scripts/09_full_report.py — Pipeline "chạy 1 lần lấy đủ số liệu chính thức
cho báo cáo". Nguyên tắc: KHÔNG cherry-pick số liệu rời rạc từ nhiều lần
chạy khác nhau — toàn bộ các bước dùng CHUNG đúng 1 cấu hình kiến trúc (được
chọn qua sweep trên validation), nối tiếp nhau tự động theo đúng 1 trình tự
khoa học duy nhất.

Trình tự (mỗi bước phụ thuộc đầu ra bước trước, không nhảy bước, không đổi
cấu hình giữa chừng):
  1. 01_data_audit.py         — audit dữ liệu gốc + k-core sensitivity
  2. 08_hyperparam_sweep.py   — sweep ĐẦY ĐỦ (embedding_dim, số lớp MLP,
                                 negative_ratio), CHỈ chọn theo validation
  3. Sinh config "final"      = config gốc + best_config.json (bước 2)
                                 + include_redundant_metrics=true
  4. 03_run_experiment.py     --config final --run-tag <tag>_final
                                 (mọi model + baseline, Precision/Recall,
                                 long-tail, cold-start, beyond-accuracy —
                                 TRONG CÙNG 1 lần chạy, không train lại)
  5. 04_multi_seed.py         --config final --seeds ... (mặc định 5 seed)
                                 xác nhận ý nghĩa thống kê CHO ĐÚNG cấu hình
                                 đã chọn ở bước 2-4, không phải cấu hình mặc
                                 định gốc trong configs/*.yaml
  6. 06_aggregate_seeds.py    — mean±std + Wilcoxon signed-rank
  7. Sinh config "weighted"   = config final + feedback.mode=weighted_confidence
  8. 03_run_experiment.py     --config weighted --run-tag <tag>_weighted
  9. So sánh trực tiếp results.json "final" (binary) vs "weighted"
 10. 05_evaluate.py           --run-tag <tag>_final -> bảng + biểu đồ chính thức

Số liệu CHÍNH THỨC dùng cho báo cáo PHẢI lấy từ run-tag "<tag>_final" (bước
4/10) và "<config_name>_seed_*" (bước 5/6) — KHÔNG dùng số liệu ở bước 2
(sweep chỉ huấn luyện NeuMF-Scratch, không pretrain, không baseline cổ điển,
chỉ phục vụ mục đích CHỌN kiến trúc).

Lưu ý hm/hm_subset: cả 2 config đều có dataset.name="hm" bên trong YAML.
Script này dùng --dataset CLI (dataco/hm/hm_subset) làm tag phân biệt cho
sweep và cho run-tag final/weighted (không đụng nhau). Multi-seed/aggregate
(bước 5-6) vẫn theo đúng quy ước sẵn có của 04_multi_seed.py/06_aggregate_seeds.py
(dựa trên cfg['dataset']['name'] bên trong YAML, không sửa trong lần này) —
nghĩa là KHÔNG được chạy multi-seed cho cả hm VÀ hm_subset trong cùng một
outputs/ vì tag sẽ trùng "hm_seed_*". Hiện báo cáo chỉ multi-seed trên
hm_subset (hm full-scale chưa khả thi, xem README mục 4) nên không phát sinh.

Chạy (khuyến nghị trên Colab GPU — xem README mục "Chạy trên Colab"):
    python scripts/09_full_report.py --dataset dataco
    python scripts/09_full_report.py --dataset hm_subset --seeds 42 2024 2025 2026 3407

Resume sau khi mất kết nối giữa chừng (Colab hay ngắt phiên):
    python scripts/09_full_report.py --dataset dataco --skip-audit --skip-sweep
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir  # noqa: E402

DATASET_CONFIGS = {
    "dataco": "configs/dataco.yaml",
    "hm": "configs/hm.yaml",
    "hm_subset": "configs/hm_subset.yaml",
}
DEFAULT_SEEDS = [42, 2024, 2025, 2026, 3407]


def _run(*args: str) -> None:
    cmd = [sys.executable, *args]
    print("\n+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def build_final_config(base_config_path: Path, best_config: dict) -> dict:
    cfg = copy.deepcopy(load_yaml(base_config_path))
    cfg["model"]["embedding_dim"] = best_config["embedding_dim"]
    cfg["model"]["mlp_layers"] = best_config["mlp_layers"]
    cfg["training"]["negative_ratio"] = best_config["negative_ratio"]
    cfg["evaluation"]["include_redundant_metrics"] = True
    return cfg


def compare_feedback_variants(final_tag: str, weighted_tag: str, out_name: str) -> pd.DataFrame:
    def load_primary(tag: str) -> dict:
        path = PROJECT_ROOT / "outputs" / "experiments" / tag / "results.json"
        return json.loads(path.read_text(encoding="utf-8"))["primary"]

    binary = load_primary(final_tag)
    weighted = load_primary(weighted_tag)

    rows = []
    for model, metrics in binary.items():
        if model not in weighted:
            continue
        for metric, b in metrics.items():
            w = weighted[model].get(metric)
            if w is None:
                continue
            rows.append({
                "model": model,
                "metric": metric,
                "binary": b,
                "weighted_confidence": w,
                "delta": w - b,
                "relative_pct": (w - b) / b * 100 if b else None,
            })
    df = pd.DataFrame(rows)
    out_path = ensure_dir(PROJECT_ROOT / "outputs" / "tables") / f"feedback_comparison_{out_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n📋 So sánh Binary vs Weighted-confidence feedback ({out_path}):")
    print(df.to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASET_CONFIGS), required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--sweep-seed", type=int, default=42)
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--skip-sweep", action="store_true", help="Bỏ qua sweep, dùng lại best_config.json đã có sẵn")
    ap.add_argument("--skip-multiseed", action="store_true", help="Bỏ qua bước 5-6 (multi-seed + aggregate)")
    ap.add_argument("--skip-weighted", action="store_true", help="Bỏ qua bước 7-9 (biến thể weighted-feedback)")
    args = ap.parse_args()

    tag = args.dataset
    base_config_path = PROJECT_ROOT / DATASET_CONFIGS[tag]
    temp_dir = ensure_dir(PROJECT_ROOT / "outputs" / "experiments" / "_temp_configs")

    print(f"\n{'#'*70}\n  📦 FULL REPORT PIPELINE — {tag}\n{'#'*70}")

    # BƯỚC 1: Audit
    if not args.skip_audit:
        _run(str(PROJECT_ROOT / "scripts" / "01_data_audit.py"), "--config", str(base_config_path))

    # BƯỚC 2: Sweep đầy đủ (không thu gọn grid — theo đúng yêu cầu số liệu
    # chính xác, không cherry-pick)
    best_config_path = PROJECT_ROOT / "outputs" / "tables" / f"sweep_{tag}" / "best_config.json"
    if not args.skip_sweep or not best_config_path.exists():
        _run(
            str(PROJECT_ROOT / "scripts" / "08_hyperparam_sweep.py"),
            "--config", str(base_config_path), "--tag", tag, "--seed", str(args.sweep_seed),
        )
    best_config = json.loads(best_config_path.read_text(encoding="utf-8"))
    print(f"\n✓ Cấu hình được chọn theo validation NDCG@10: {best_config}")

    # BƯỚC 3: Sinh config "final"
    final_cfg = build_final_config(base_config_path, best_config)
    final_cfg_path = temp_dir / f"{tag}_final.yaml"
    dump_yaml(final_cfg, final_cfg_path)
    config_name = final_cfg["dataset"]["name"]

    # BƯỚC 4: Train + eval đầy đủ với cấu hình final (số liệu CHÍNH THỨC)
    final_tag = f"{tag}_final"
    _run(
        str(PROJECT_ROOT / "scripts" / "03_run_experiment.py"),
        "--config", str(final_cfg_path), "--run-tag", final_tag,
    )

    # BƯỚC 5-6: Multi-seed + Wilcoxon trên ĐÚNG cấu hình final
    if not args.skip_multiseed:
        _run(
            str(PROJECT_ROOT / "scripts" / "04_multi_seed.py"),
            "--config", str(final_cfg_path), "--seeds", *map(str, args.seeds),
        )
        _run(
            str(PROJECT_ROOT / "scripts" / "06_aggregate_seeds.py"),
            "--config-name", config_name, "--seeds", *map(str, args.seeds),
        )

    # BƯỚC 7-9: Biến thể weighted-feedback, so sánh trực tiếp với "final" (binary)
    if not args.skip_weighted:
        weighted_cfg = copy.deepcopy(final_cfg)
        weighted_cfg["feedback"]["mode"] = "weighted_confidence"
        weighted_cfg_path = temp_dir / f"{tag}_weighted.yaml"
        dump_yaml(weighted_cfg, weighted_cfg_path)

        weighted_tag = f"{tag}_weighted"
        _run(
            str(PROJECT_ROOT / "scripts" / "03_run_experiment.py"),
            "--config", str(weighted_cfg_path), "--run-tag", weighted_tag,
        )
        compare_feedback_variants(final_tag, weighted_tag, tag)

    # BƯỚC 10: Bảng + biểu đồ chính thức (bao gồm cold-start, Precision/Recall)
    _run(str(PROJECT_ROOT / "scripts" / "05_evaluate.py"), "--run-tag", final_tag)

    print(f"\n{'#'*70}\n  ✅ HOÀN THÀNH FULL REPORT PIPELINE — {tag}\n{'#'*70}")
    print(f"  Số liệu CHÍNH THỨC:         outputs/tables/{final_tag}/, outputs/figures/{final_tag}/")
    if not args.skip_multiseed:
        print(f"  Multi-seed + Wilcoxon:      outputs/tables/{config_name}/multi_seed_summary_primary.csv, multi_seed_significance_primary.csv")
    if not args.skip_weighted:
        print(f"  So sánh binary vs weighted: outputs/tables/feedback_comparison_{tag}.csv")
    print(f"  Sweep siêu tham số:         outputs/tables/sweep_{tag}/")


if __name__ == "__main__":
    main()
