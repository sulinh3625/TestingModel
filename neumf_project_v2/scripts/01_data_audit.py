from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import build_adapter
from src.data_pipeline.audit import audit_events, audit_splits
from src.data_pipeline.preprocessing import build_interactions, apply_feedback_weights
from src.data_pipeline.splitting import temporal_leave_one_out, assert_disjoint_splits
from src.utils.io import ensure_dir, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataco.yaml")
    args = ap.parse_args()

    cfg, adapter = build_adapter(args.config)
    events = adapter.load_events()
    report, table = audit_events(events, cfg.dataset.audit_k_values)

    data = build_interactions(events, cfg.dataset.k_core)
    train, val, test = temporal_leave_one_out(data.df, cfg.dataset.min_interactions_for_loo)
    assert_disjoint_splits(train, val, test)
    train, val, test, feedback_meta = apply_feedback_weights(
        train, val, test, cfg.feedback.mode, cfg.feedback.confidence_alpha
    )
    report["selected_k"] = cfg.dataset.k_core
    report["selected_dataset"] = {
        "users": data.n_users,
        "items": data.n_items,
        "interactions": len(data.df),
        "density": len(data.df) / (data.n_users * data.n_items),
    }
    report["splits"] = audit_splits(train, val, test)
    report["feedback"] = feedback_meta

    out_dir = ensure_dir(cfg.output_root / "data_audit" / cfg.dataset.name)
    table.to_csv(out_dir / "kcore_sensitivity.csv", index=False)
    write_json(out_dir / "data_audit.json", report)

    print(table.to_string(index=False))
    print("\nSelected k:", cfg.dataset.k_core)
    print("Users / Items / Interactions:", data.n_users, data.n_items, len(data.df))
    print("Splits:", report["splits"])
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
