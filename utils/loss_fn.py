from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MSE(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        # prediction, target: (b, v, h, w) — batch, variable, latitude, longitude
        error = (prediction - target) ** 2          # (b, v, h, w) element-wise squared error
        weight = kwargs.get("weight")
        if weight is not None:
            error = error * weight                   # (b, v, h, w) optional broadcast weight
        return error.mean()                          # scalar


class WeightedMSE(nn.Module):
    """Squared error weighted per variable and per latitude in one einsum, then the mean.

    With uniform variable weights and no latitude weighting the value equals MSE.
    The weights are registered as buffers so they move with .to(device) but are not trained.
    """
    def __init__(
        self,
        latitude: np.ndarray,                        # (h,) latitude values in degrees
        variable_weights: dict[str, float] | None = None,  # e.g. {"T2M": 1.0, "U10M": 0.1, ...}
        latitude_weighting: bool = True,
        variables: list[str] | None = None,          # dataset variable order, e.g. ["T2M", "U10M", ...]
    ) -> None:
        super().__init__()
        # cos(latitude) normalised to mean one — the relative area of each latitude band
        lat = torch.from_numpy(np.deg2rad(latitude)).float()   # (h,)
        if latitude_weighting:
            w_lat = torch.cos(lat)                              # (h,)
            w_lat = w_lat / w_lat.mean()                       # (h,) mean == 1
        else:
            w_lat = torch.ones_like(lat)                        # (h,) uniform
        self.register_buffer("per_latitude", w_lat)             # (h,) — moves with .to(), not trained

        # Per-variable scalar weights; the dataset's variable order maps names to positions
        n_vars = len(variables) if variables is not None else 1
        if variable_weights is not None and variables is not None:
            w_var = torch.tensor([variable_weights.get(v, 1.0) for v in variables], dtype=torch.float32)  # (v,)
        else:
            w_var = torch.ones(n_vars, dtype=torch.float32)    # (v,) uniform
        self.register_buffer("per_variable", w_var)             # (v,)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """prediction, target: (b, v, h, w) — batch, variable, latitude, longitude.
        Returns: scalar weighted MSE."""
        se = (prediction - target) ** 2                        # (b, v, h, w) squared error
        # einsum: multiply by per_variable[v] and per_latitude[h], sum over b,v,h,w, then divide by total count
        return torch.einsum("b v h w, v, h ->", se, self.per_variable, self.per_latitude) / se.numel()
