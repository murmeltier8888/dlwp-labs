from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    path: str = "data/era5_5p6.zarr"
    stats_path: str = "data/stats.zarr"
    variables: list[str] = field(default_factory=lambda: ["T2M", "U10M", "V10M", "TP6h", "Z500", "T850", "Q700", "U250", "V250"])
    sequence_length: int = 2
    time_slice: dict[str, Any] | None = None
    lat_slice: dict[str, Any] | None = None
    lon_slice: dict[str, Any] | None = None
    batch_size: int = 16
    num_workers: int = 0


@dataclass
class NetworkConfig:
    name: str = "vit"
    dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dim_heads: int = 32
    patch_size: list[int] = field(default_factory=lambda: [4, 4])
    expansion_factor: int = 2
    separable_embed: bool = False
    metadata_embed: bool = False
    dim_metadata: int = 8


@dataclass
class ObjectiveConfig:
    name: str = "mse"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainerConfig:
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    max_steps: int = 500
    rollout_steps: int = 4
    val_time_slice: dict[str, Any] | None = None
    predict_steps: int = 20
    predict_stride: int = 5
    train_rollout_steps: int = 1
    pre_steps: int = 0


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Config:
        return cls(
            dataset=DatasetConfig(**cfg.get("dataset", {})),
            network=NetworkConfig(**cfg.get("network", {})),
            objective=ObjectiveConfig(**cfg.get("objective", {})),
            trainer=TrainerConfig(**cfg.get("trainer", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls.from_dict(cfg)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
