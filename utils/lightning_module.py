from __future__ import annotations

import dataclasses

import lightning as L
import torch
from torch.utils.data import DataLoader

from utils.components import Persistence, ViT
from utils.config import Config
from utils.dataset import WeatherDataset
from utils.loss_fn import MSE


class ForecastModule(L.LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.save_hyperparameters(config.to_dict())
        self.config = config

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

        self.loss = MSE(**config.objective.kwargs)
        self.automatic_optimization = isinstance(self.model, ViT)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch[:, :, 0]
        y = batch[:, :, 1]
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        x = batch[:, :, 0]
        for k in range(1, self.config.trainer.rollout_steps + 1):
            x = self.model(x)
            target = batch[:, :, k]
            loss = self.loss(x, target)
            self.log(f"val/loss_step{k}", loss, prog_bar=True)

    def forecast(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        preds = []
        h = x
        for _ in range(steps):
            h = self.model(h)
            preds.append(h)
        return torch.stack(preds, dim=2)

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
