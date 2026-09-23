"""Entry point tổng hợp cho pipeline NeuMF — gom các lệnh trong scripts/ lại một chỗ.

Ví dụ:
    python run.py all --dataset dataco                  # audit + train + evaluate, trọn gói
    python run.py all --dataset hm_subset --run-tag hm_v1
    python run.py audit --dataset hm
    python run.py train --dataset dataco --run-tag my_tag
    python run.py multi-seed --dataset dataco --seeds 42 2024 2025
    python run.py aggregate --dataset dataco
    python run.py compare
    python run.py sweep --dataset dataco             # sweep embedding_dim / mlp_layers / negative_ratio
    python run.py prepare-hm-cache
    python run.py demo

Mỗi lệnh chỉ gọi thẳng script tương ứng trong scripts/ (xem README mục 3-8
để biết chi tiết từng bước) — file này không chứa logic huấn luyện/đánh giá.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Console Windows mặc định dùng codepage (vd. cp1258) không encode được tiếng
# Việt -> crash khi in --help hoặc thông báo lỗi. Ép UTF-8 cho tiến trình này
# và các script con được gọi bên dưới.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

ROOT = Path(__file__).resolve().parent

DATASET_CONFIGS = {
    "dataco": "configs/dataco.yaml",
    "hm": "configs/hm.yaml",
    "hm_subset": "configs/hm_subset.yaml",
}
# dataset.name bên trong config (dùng cho run-tag/aggregate) — hm và hm_subset dùng chung "hm".
DATASET_CONFIG_NAME = {"dataco": "dataco", "hm": "hm", "hm_subset": "hm"}


def _run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=_ENV, check=True)


def cmd_prepare_hm_cache(args):
    _run("00_prepare_hm_cache.py")


def cmd_audit(args):
    _run("01_data_audit.py", "--config", DATASET_CONFIGS[args.dataset])


def cmd_preprocess(args):
    _run("02_preprocess.py", "--config", DATASET_CONFIGS[args.dataset])


def cmd_train(args):
    extra = ["--run-tag", args.run_tag] if args.run_tag else []
    _run("03_run_experiment.py", "--config", DATASET_CONFIGS[args.dataset], *extra)


def cmd_evaluate(args):
    extra = ["--run-tag", args.run_tag] if args.run_tag else []
    _run("05_evaluate.py", *extra)


def cmd_multi_seed(args):
    _run("04_multi_seed.py", "--config", DATASET_CONFIGS[args.dataset], "--seeds", *map(str, args.seeds))


def cmd_aggregate(args):
    _run("06_aggregate_seeds.py", "--config-name", DATASET_CONFIG_NAME[args.dataset], "--seeds", *map(str, args.seeds))


def cmd_compare(args):
    _run("07_compare_datasets.py")


def cmd_sweep(args):
    extra = ["--seed", str(args.seed)] if args.seed is not None else []
    # --tag = args.dataset (không phải cfg.dataset.name) để hm/hm_subset không
    # ghi đè kết quả sweep của nhau (cả hai đều có dataset.name="hm").
    _run("08_hyperparam_sweep.py", "--config", DATASET_CONFIGS[args.dataset], "--tag", args.dataset, *extra)


def cmd_report_full(args):
    extra = ["--seeds", *map(str, args.seeds)] if args.seeds else []
    for flag, val in [
        ("--skip-audit", args.skip_audit), ("--skip-sweep", args.skip_sweep),
        ("--skip-multiseed", args.skip_multiseed), ("--skip-weighted", args.skip_weighted),
    ]:
        if val:
            extra.append(flag)
    _run("09_full_report.py", "--dataset", args.dataset, *extra)


def cmd_demo(args):
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "demo.backend.main:app", "--port", "8000", "--host", "127.0.0.1"],
        cwd=ROOT, env=_ENV, check=True,
    )


def cmd_all(args):
    """Audit + train + evaluate trọn gói cho một dataset (tương đương README mục 3/4)."""
    cmd_audit(args)
    extra = ["--run-tag", args.run_tag] if args.run_tag else []
    _run("run_all.py", "--config", DATASET_CONFIGS[args.dataset], *extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def with_dataset(p):
        p.add_argument("--dataset", choices=list(DATASET_CONFIGS), default="dataco", help="Bộ dữ liệu (mặc định: dataco)")
        return p

    with_dataset(sub.add_parser("all", help="Audit + train + evaluate trọn gói cho 1 dataset")).add_argument("--run-tag", default=None)
    with_dataset(sub.add_parser("audit", help="Audit dữ liệu — bắt buộc trước khi train"))
    with_dataset(sub.add_parser("preprocess", help="Sinh splits độc lập"))
    with_dataset(sub.add_parser("train", help="Chỉ huấn luyện, không vẽ biểu đồ")).add_argument("--run-tag", default=None)
    sub.add_parser("evaluate", help="Đánh giá + xuất biểu đồ cho 1 run-tag").add_argument("--run-tag", default=None, help="Mặc định: run mới nhất")
    with_dataset(sub.add_parser("multi-seed", help="Lặp lại train trên nhiều seed")).add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 2025, 2026, 3407])
    with_dataset(sub.add_parser("aggregate", help="Tổng hợp mean/std + Wilcoxon từ multi-seed")).add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 2025, 2026, 3407])
    sub.add_parser("compare", help="Biểu đồ so sánh DataCo vs H&M")
    with_dataset(sub.add_parser("sweep", help="Sweep siêu tham số: embedding_dim, số lớp MLP, negative_ratio")).add_argument("--seed", type=int, default=None)
    p_report = with_dataset(sub.add_parser("report-full", help="Pipeline đầy đủ 1 lần: audit+sweep+train+multi-seed+weighted-feedback+evaluate"))
    p_report.add_argument("--seeds", nargs="+", type=int, default=None)
    p_report.add_argument("--skip-audit", action="store_true")
    p_report.add_argument("--skip-sweep", action="store_true")
    p_report.add_argument("--skip-multiseed", action="store_true")
    p_report.add_argument("--skip-weighted", action="store_true")
    sub.add_parser("prepare-hm-cache", help="Nén CSV H&M gốc thành cache Parquet (chạy 1 lần)")
    sub.add_parser("demo", help="Chạy giao diện demo (uvicorn, port 8000)")

    args = parser.parse_args()
    {
        "all": cmd_all,
        "audit": cmd_audit,
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "multi-seed": cmd_multi_seed,
        "aggregate": cmd_aggregate,
        "compare": cmd_compare,
        "sweep": cmd_sweep,
        "report-full": cmd_report_full,
        "prepare-hm-cache": cmd_prepare_hm_cache,
        "demo": cmd_demo,
    }[args.command](args)


if __name__ == "__main__":
    main()
