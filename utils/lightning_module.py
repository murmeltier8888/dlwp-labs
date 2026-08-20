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
    """Lightning wrapper for the weather forecasting model.

    Handles training (single-step or autoregressive rollout), validation (multi-step rollout),
    prediction (long roll-out for evaluation year), and the data loaders.

    All batch tensors carry the axes (batch, variable, time, latitude, longitude) unless noted.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.save_hyperparameters(config.to_dict())    # stores config as hyperparameters for checkpointing
        self.config = config

        # The training window must be long enough for the autoregressive rollout
        assert config.dataset.sequence_length >= config.trainer.train_rollout_steps + 1, (
            f"sequence_length ({config.dataset.sequence_length}) must be >= "
            f"train_rollout_steps + 1 ({config.trainer.train_rollout_steps + 1})"
        )

        # --- Datasets ---
        # Training dataset: full years, windows of `sequence_length` states
        self.train_dataset = WeatherDataset(config.dataset)

        # Validation dataset: evaluation year, windows of `rollout_steps + 1` states
        # (one initial state + rollout_steps targets for multi-step validation)
        val_config = dataclasses.replace(
            config.dataset,
            time_slice=config.trainer.val_time_slice,
            sequence_length=config.trainer.rollout_steps + 1,
        )
        self.val_dataset = WeatherDataset(val_config)

        # --- Model ---
        num_variables = len(config.dataset.variables)
        H, W = self.train_dataset.tensor.shape[2], self.train_dataset.tensor.shape[3]

        if config.network.name == "vit":
            self.model = ViT(config.network, num_variables, (H, W))
        else:
            self.model = Persistence()

        # Time step between consecutive states in hours (six-hourly ERA5 data)
        self._data_step = 6.0

        # --- Loss ---
        # Training loss: MSE or WeightedMSE depending on config
        if config.objective.name == "weighted_mse":
            self.loss = WeightedMSE(
                latitude=self.train_dataset._lat,       # (H,) latitude in degrees
                variable_weights=config.objective.kwargs.get("variable_weights"),
                latitude_weighting=config.objective.kwargs.get("latitude_weighting", True),
                variables=config.dataset.variables,
            )
        else:
            self.loss = MSE(**config.objective.kwargs)

        # Validation loss is always plain MSE for fair comparison across runs
        self.val_loss = MSE()
        self.automatic_optimization = isinstance(self.model, ViT)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Single-step or autoregressive training.

        With train_rollout_steps=1 (default): standard one-step prediction.
        With train_rollout_steps>1: applies the model repeatedly on its own output,
        the loss is the mean over the rollout steps, gradients flow through the whole chain.

        Args:
            batch: tuple of (states, time)
                states: (b, v, seq, H, W) — a window of seq consecutive states
                time:   (b,) — hours since Jan 1 for the first state
        Returns:
            scalar loss
        """
        states, time = batch
        # During pre_steps phase (global_step < pre_steps), use single-step training;
        # after that, switch to the full rollout
        steps = 1 if self.global_step < self.config.trainer.pre_steps else self.config.trainer.train_rollout_steps

        x = states[:, :, 0]                            # (b, v, H, W) — initial state
        t = time                                        # (b,) — time of the initial state
        total_loss = torch.tensor(0.0, device=x.device)
        for k in range(steps):
            pred = self.model(x, time=t)               # (b, v, H, W) — one-step prediction
            target = states[:, :, k + 1]                # (b, v, H, W) — ground truth at step k+1
            total_loss = total_loss + self.loss(pred, target)
            x = pred                                    # feed prediction back as input (autoregressive)
            if t is not None:
                t = t + self._data_step                 # advance time by one data step (6 hours)
        loss = total_loss / steps                       # mean loss over the rollout
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Multi-step autoregressive validation.

        Rolls out the model for `rollout_steps` steps from the initial state,
        logging the loss at each step separately (val/loss_step1, val/loss_step2, ...).

        Args:
            batch: tuple of (states, time)
                states: (b, v, rollout_steps+1, H, W) — initial state + rollout_steps targets
                time:   (b,) — hours since Jan 1 for the initial state
        """
        states, time = batch
        x = states[:, :, 0]                            # (b, v, H, W) — initial state
        t = time
        for k in range(1, self.config.trainer.rollout_steps + 1):
            x = self.model(x, time=t)                  # (b, v, H, W) — predicted state at step k
            target = states[:, :, k]                    # (b, v, H, W) — ground truth at step k
            loss = self.val_loss(x, target)
            self.log(f"val/loss_step{k}", loss, prog_bar=True)
            if t is not None:
                t = t + self._data_step

    def forecast(self, x: torch.Tensor, steps: int, time: torch.Tensor | None = None) -> torch.Tensor:
        """Autoregressive roll-out: apply the model `steps` times from an initial state.

        Args:
            x:     (b, v, H, W) — initial state
            steps: number of autoregressive steps
            time:  (b,) or None — time of the initial state
        Returns:
            (b, v, steps, H, W) — the predicted states at steps 1..steps
        """
        preds = []
        h = x                                          # current state, starts as initial
        t = time
        for _ in range(steps):
            h = self.model(h, time=t)                  # (b, v, H, W) — next prediction
            preds.append(h)
            if t is not None:
                t = t + self._data_step
        return torch.stack(preds, dim=2)               # (b, v, steps, H, W)

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Trainer.predict entry point: roll out from the initial state for predict_steps.

        Args:
            batch: tuple of (states, time) from predict_dataloader
                states: (b, v, predict_steps+1, H, W)
                time:   (b,)
        Returns:
            (b, v, predict_steps, H, W) on CPU
        """
        states, time = batch
        return self.forecast(states[:, :, 0], self.config.trainer.predict_steps, time=time).cpu()

    def predict_dataloader(self) -> DataLoader:
        """DataLoader for the evaluation year predictions.

        Creates a dataset with sequence_length = predict_steps + 1 (enough for one
        initial state + predict_steps targets), then takes every predict_stride-th
        window via a Subset. With predict_stride=5 on six-hourly data, the
        initialisations are 30 hours apart, cycling through 00, 06, 12, and 18 UTC.

        Returns:
            DataLoader of (states, time) tuples, unshuffled
        """
        cfg = self.config.trainer
        predict_seq_length = cfg.predict_steps + 1     # initial state + predict_steps targets
        pred_dataset_cfg = dataclasses.replace(
            self.config.dataset,
            time_slice=cfg.val_time_slice,
            sequence_length=predict_seq_length,
        )
        pred_dataset = WeatherDataset(pred_dataset_cfg)
        # Every predict_stride-th starting position: e.g. stride=5 -> indices 0, 5, 10, ...
        indices = list(range(0, len(pred_dataset), cfg.predict_stride))
        subset = Subset(pred_dataset, indices)
        return DataLoader(
            subset,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            shuffle=False,
        )

    def configure_optimizers(self):
        """AdamW optimizer with cosine annealing schedule."""
        if not self.automatic_optimization:
            return []                                   # persistence model has no trainable parameters
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            shuffle=False,
        )


def plot_metrics(log_dir: str, ax=None, label=None):
    """Plot train/loss and val/loss_step1 from a run's metrics.csv.

    The CSV is written by CSVLogger for every value passed to self.log.
    train/loss is plotted as a line (with 10-step rolling mean) and raw scatter;
    val/loss_step1 is plotted as x-markers.

    Args:
        log_dir: path to the run's log directory (contains metrics.csv)
        ax:      matplotlib axis to draw on (created if None)
        label:   label for the legend
    Returns:
        the matplotlib axis
    """
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
    """Callback that saves the logger and plots the loss curve at the end of training.

    Attach to every Trainer: callbacks=[LossCurve()]
    """
    def on_train_end(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        if trainer.logger is not None:
            trainer.logger.save()                       # flush metrics.csv to disk
            plot_metrics(trainer.logger.log_dir)        # plot train/loss and val/loss_step1
