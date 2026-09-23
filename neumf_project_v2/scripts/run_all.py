from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Dynamic imports vì tên file script có tiền tố số (03_..., 05_...)
run_experiment = importlib.import_module("scripts.03_run_experiment").run
run_evaluation = importlib.import_module("scripts.05_evaluate").run_evaluation


def run_pipeline(config_path: str, run_tag: str | None = None, skip_eval: bool = False):
    print("\n" + "🚀" * 35)
    print("  BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE (END-TO-END)")
    print(f"  Config: {config_path}")
    print("🚀" * 35 + "\n")

    # Bước 1: Huấn luyện toàn bộ mô hình
    run_dir = run_experiment(config_path, run_tag=run_tag)
    tag = run_dir.name

    # Bước 2: Tự động đánh giá và xuất biểu đồ PNG
    if not skip_eval:
        print("\n" + "🎨" * 35)
        print(f"  TỰ ĐỘNG ĐÁNH GIÁ & XUẤT BIỂU ĐỒ CHO RUN: {tag}")
        print("🎨" * 35 + "\n")
        run_evaluation(run_tag=tag)

    print("\n" + "✨" * 35)
    print("  🎉 HOÀN TẤT TOÀN BỘ PIPELINE THÀNH CÔNG!")
    print(f"  📂 Thư mục thí nghiệm: outputs/experiments/{tag}")
    print(f"  📋 Thư mục bảng số liệu: outputs/tables/{tag}")
    print(f"  📈 Thư mục ảnh biểu đồ:  outputs/figures/{tag}")
    print("✨" * 35 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Chạy trọn gói Huấn luyện + Đánh giá + Xuất biểu đồ")
    parser.add_argument("--config", default="configs/dataco.yaml", help="Đường dẫn file config YAML")
    parser.add_argument("--run-tag", default=None, help="Tên đặt cho lần chạy này")
    parser.add_argument("--skip-eval", action="store_true", help="Bỏ qua bước vẽ biểu đồ")
    args = parser.parse_args()

    run_pipeline(args.config, args.run_tag, args.skip_eval)


if __name__ == "__main__":
    main()
