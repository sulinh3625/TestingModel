from __future__ import annotations

from pathlib import Path

from src.config import load_config, resolve_project_path
from src.data_pipeline.adapters import DataCoAdapter, HMAdapter


def build_adapter(config_path: str):
    cfg = load_config(config_path)
    path = resolve_project_path(cfg.dataset.raw_path)
    name = cfg.dataset.name.lower()
    if name == "dataco":
        adapter = DataCoAdapter(path, encoding=cfg.dataset.encoding, nrows=cfg.dataset.nrows)
    elif name == "hm":
        adapter = HMAdapter(
            path,
            encoding=cfg.dataset.encoding,
            nrows=cfg.dataset.nrows,
            start_date=cfg.dataset.start_date,
            end_date=cfg.dataset.end_date,
        )
    else:
        raise ValueError(f"Dataset chưa hỗ trợ: {cfg.dataset.name}")
    return cfg, adapter
