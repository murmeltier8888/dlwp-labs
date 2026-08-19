from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MSE(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        error = (prediction - target) ** 2
        weight = kwargs.get("weight")
        if weight is not None:
            error = error * weight
        return error.mean()


class WeightedMSE(nn.Module):
    def __init__(
        self,
        latitude: np.ndarray,
        variable_weights: dict[str, float] | None = None,
        latitude_weighting: bool = True,
        variables: list[str] | None = None,
    ) -> None:
        super().__init__()
        lat = torch.from_numpy(np.deg2rad(latitude)).float()
        if latitude_weighting:
            w_lat = torch.cos(lat)
            w_lat = w_lat / w_lat.mean()
        else:
            w_lat = torch.ones_like(lat)
        self.register_buffer("per_latitude", w_lat)

        n_vars = len(variables) if variables is not None else 1
        if variable_weights is not None and variables is not None:
            w_var = torch.tensor([variable_weights.get(v, 1.0) for v in variables], dtype=torch.float32)
        else:
            w_var = torch.ones(n_vars, dtype=torch.float32)
        self.register_buffer("per_variable", w_var)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """prediction, target: (b, v, h, w)"""
        se = (prediction - target) ** 2
        return torch.einsum("b v h w, v, h ->", se, self.per_variable, self.per_latitude) / se.numel()
