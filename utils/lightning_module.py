from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import lightning as L
from lightning.pytorch.callbacks import Callback

from utils.components import Persistence, ViT
from utils.config import Config
from utils.dataset import WeatherDataset
from utils.loss_fn import MSE, WeightedMSE


class ForecastModule(L.LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.save_hyperparameters(config.to_dict())
        self.config = config

        assert config.dataset.sequence_length >= config.trainer.train_rollout_steps + 1, (
            f"sequence_length ({config.dataset.sequence_length}) must be >= "
            f"train_rollout_steps + 1 ({config.trainer.train_rollout_steps + 1})"
        )

        self.train_dataset = WeatherDataset(config.dataset)

        val_config = dataclasses.replace(
            config.dataset,
            time_slice=config.trainer.val_time_slice,
            sequence_length=config.trainer.rollout_steps + 1,
        )
        self.val_dataset = WeatherDataset(val_config)

        num_variables = len(config.dataset.variables)
        H, W = self.train_dataset.tensor.shape[2], self.train_dataset.tensor.shape[3]

        if config.network.name == "vit":
            self.model = ViT(config.network, num_variables, (H, W))
        else:
            self.model = Persistence()

        self._data_step = 6.0  # hours between consecutive timesteps (six-hourly data)

        if config.objective.name == "weighted_mse":
            self.loss = WeightedMSE(
                latitude=self.train_dataset._lat,
                variable_weights=config.objective.kwargs.get("variable_weights"),
                latitude_weighting=config.objective.kwargs.get("latitude_weighting", True),
                variables=config.dataset.variables,
            )
        else:
            self.loss = MSE(**config.objective.kwargs)

        self.val_loss = MSE()
        self.automatic_optimization = isinstance(self.model, ViT)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        states, time = batch
        steps = 1 if self.global_step < self.config.trainer.pre_steps else self.config.trainer.train_rollout_steps
        x = states[:, :, 0]
        t = time
        total_loss = torch.tensor(0.0, device=x.device)
        for k in range(steps):
            pred = self.model(x, time=t)
            target = states[:, :, k + 1]
            total_loss = total_loss + self.loss(pred, target)
            x = pred
            if t is not None:
                t = t + self._data_step
        loss = total_loss / steps
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        states, time = batch
        x = states[:, :, 0]
        t = time
        for k in range(1, self.config.trainer.rollout_steps + 1):
            x = self.model(x, time=t)
            target = states[:, :, k]
            loss = self.val_loss(x, target)
            self.log(f"val/loss_step{k}", loss, prog_bar=True)
            if t is not None:
                t = t + self._data_step

    def forecast(self, x: torch.Tensor, steps: int, time: torch.Tensor | None = None) -> torch.Tensor:
        preds = []
        h = x
        t = time
        for _ in range(steps):
            h = self.model(h, time=t)
            preds.append(h)
            if t is not None:
                t = t + self._data_step
        return torch.stack(preds, dim=2)

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        states, time = batch
        return self.forecast(states[:, :, 0], self.config.trainer.predict_steps, time=time).cpu()

    def predict_dataloader(self) -> DataLoader:
        cfg = self.config.trainer
        predict_seq_length = cfg.predict_steps + 1
        pred_dataset_cfg = dataclasses.replace(
            self.config.dataset,
            time_slice=cfg.val_time_slice,
            sequence_length=predict_seq_length,
        )
        pred_dataset = WeatherDataset(pred_dataset_cfg)
        indices = list(range(0, len(pred_dataset), cfg.predict_stride))
        subset = Subset(pred_dataset, indices)
        return DataLoader(
            subset,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            shuffle=False,
        )

    def configure_optimizers(self):
        if not self.automatic_optimization:
            return []
        cfg = self.config.trainer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=tuple(cfg.betas),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def train_dataloader(self) -> DataLoader:
        cfg = self.config.dataset
        return DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        cfg = self.config.dataset
        return DataLoader(
            self.val_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
        )


def plot_metrics(log_dir: str, ax=None, label=None):
    """Plot train/loss and val/loss_step1 from a run's metrics.csv."""
    import matplotlib.pyplot as plt

    metrics_path = Path(log_dir) / "metrics.csv"
    if not metrics_path.exists():
        return ax
    df = pd.read_csv(metrics_path)

    if ax is None:
        _, ax = plt.subplots()

    if "train/loss" in df.columns:
        train = df.dropna(subset=["train/loss"])
        if not train.empty:
            rolling = train["train/loss"].rolling(10, min_periods=1).mean()
            ax.plot(train["step"], rolling, label=label or "train/loss", alpha=0.8)
            ax.scatter(train["step"], train["train/loss"], s=1, alpha=0.3)

    if "val/loss_step1" in df.columns:
        val = df.dropna(subset=["val/loss_step1"])
        if not val.empty:
            ax.scatter(val["step"], val["val/loss_step1"], marker="x", s=20,
                       label=(label or "") + " val/loss_step1")

    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    if label:
        ax.legend()
    return ax


class LossCurve(Callback):
    def on_train_end(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        if trainer.logger is not None:
            trainer.logger.save()
            plot_metrics(trainer.logger.log_dir)
