from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataco.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 2025, 2026, 3407])
    args = ap.parse_args()

    cfg_path = PROJECT_ROOT / args.config
    base = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    temp_dir = PROJECT_ROOT / "outputs" / "experiments" / "_temp_configs"
    temp_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        cfg = deepcopy(base)
        cfg.setdefault("training", {})["seed"] = seed
        temp_cfg = temp_dir / f"seed_{seed}.yaml"
        temp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        tag = f"{cfg['dataset']['name']}_seed_{seed}"
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "03_run_experiment.py"),
             "--config", str(temp_cfg), "--run-tag", tag],
            cwd=PROJECT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
