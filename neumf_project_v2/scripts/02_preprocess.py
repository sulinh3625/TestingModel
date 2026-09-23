from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import build_adapter
from src.data_pipeline.preprocessing import build_interactions, apply_feedback_weights
from src.data_pipeline.splitting import temporal_leave_one_out, assert_disjoint_splits
from src.utils.io import ensure_dir, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataco.yaml")
    args = ap.parse_args()

    cfg, adapter = build_adapter(args.config)
    events = adapter.load_events()
    data = build_interactions(events, cfg.dataset.k_core)
    train, val, test = temporal_leave_one_out(data.df, cfg.dataset.min_interactions_for_loo)
    assert_disjoint_splits(train, val, test)
    train, val, test, feedback_meta = apply_feedback_weights(
        train, val, test, cfg.feedback.mode, cfg.feedback.confidence_alpha
    )

    out = ensure_dir(PROJECT_ROOT / "data" / "splits" / cfg.dataset.name)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "validation.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    write_json(out / "meta.json", {
        "n_users": data.n_users,
        "n_items": data.n_items,
        "k_core": cfg.dataset.k_core,
        "feedback": feedback_meta,
    })
    print(f"Saved splits -> {out}")


if __name__ == "__main__":
    main()
