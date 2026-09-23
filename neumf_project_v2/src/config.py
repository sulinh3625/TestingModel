from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DatasetConfig:
    name: str
    raw_path: str
    encoding: str = "utf-8"
    k_core: int = 5
    audit_k_values: list[int] = field(default_factory=lambda: [3, 5, 10])
    min_interactions_for_loo: int = 3
    nrows: int | None = None
    start_date: str | None = None
    end_date: str | None = None


@dataclass
class FeedbackConfig:
    mode: str = "binary"  # binary | weighted_confidence
    confidence_alpha: float = 1.0


@dataclass
class ModelConfig:
    embedding_dim: int = 32
    mlp_layers: list[int] = field(default_factory=lambda: [64, 32, 16, 8])
    dropout: float = 0.2
    pretrain_alpha: float = 0.5
    lightgcn_layers: int = 3
    sasrec_seq_len: int = 20
    sasrec_heads: int = 2
    sasrec_blocks: int = 2


@dataclass
class TrainingConfig:
    batch_size: int = 256
    negative_ratio: int = 4
    pretrain_optimizer: str = "adam"
    pretrain_lr: float = 1e-3
    finetune_optimizer: str = "sgd"
    finetune_lr: float = 1e-2
    weight_decay: float = 1e-6
    max_epochs_pretrain: int = 50
    max_epochs_finetune: int = 50
    patience: int = 10
    monitor: str = "NDCG@10"
    device: str = "cpu"
    seed: int = 42
    # EarlyFusionModel dùng cùng optimizer/LR/budget với NeuMF-Scratch
    # (finetune_*) để ablation Early vs Late fusion công bằng.
    train_early_fusion: bool = True
    train_lightgcn: bool = True
    train_sasrec: bool = True
    fast_negative_sampling: bool = False


@dataclass
class EvaluationConfig:
    primary: str = "full_ranking"  # full_ranking | sampled
    k_values: list[int] = field(default_factory=lambda: [5, 10])
    sampled_negatives: int = 99
    tie_break_seed: int = 2026
    include_redundant_metrics: bool = False
    head_fraction: float = 0.10
    cold_fraction: float = 0.20


@dataclass
class BPRConfig:
    embedding_dim: int = 32
    epochs: int = 30
    lr: float = 0.03
    reg: float = 0.005


@dataclass
class BaselineConfig:
    enabled: list[str] = field(default_factory=lambda: ["random", "popularity", "itemknn", "bpr"])
    bpr: BPRConfig = field(default_factory=BPRConfig)


@dataclass
class PathsConfig:
    outputs_dir: str = "outputs"


@dataclass
class ProjectConfig:
    dataset: DatasetConfig
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @property
    def output_root(self) -> Path:
        p = Path(self.paths.outputs_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p


def _merge_dataclass(cls, data: dict[str, Any] | None):
    data = data or {}
    if cls is BaselineConfig:
        bpr_data = data.get("bpr", {})
        payload = {k: v for k, v in data.items() if k != "bpr"}
        return BaselineConfig(**payload, bpr=BPRConfig(**bpr_data))
    return cls(**data)


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    dataset = DatasetConfig(**raw["dataset"])
    cfg = ProjectConfig(
        dataset=dataset,
        feedback=_merge_dataclass(FeedbackConfig, raw.get("feedback")),
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        training=_merge_dataclass(TrainingConfig, raw.get("training")),
        evaluation=_merge_dataclass(EvaluationConfig, raw.get("evaluation")),
        baselines=_merge_dataclass(BaselineConfig, raw.get("baselines")),
        paths=_merge_dataclass(PathsConfig, raw.get("paths")),
    )
    return cfg


def resolve_project_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p