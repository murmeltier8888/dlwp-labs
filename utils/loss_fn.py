import torch
import torch.nn as nn


class MSE(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        error = (prediction - target) ** 2
        weight = kwargs.get("weight")
        if weight is not None:
            error = error * weight
        return error.mean()
