from __future__ import annotations

from typing import Any

import numpy as np
import torch
import xarray as xr

from utils.config import DatasetConfig


def _slice_from_dict(d: dict[str, Any] | None) -> slice | None:
    if d is None:
        return None
    return slice(d.get("start"), d.get("stop"), d.get("step"))


class WeatherDataset(torch.utils.data.Dataset):
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__()
        self.config = config

        ds = xr.open_zarr(config.path)
        ts = _slice_from_dict(config.time_slice)
        ls = _slice_from_dict(config.lat_slice)
        lo = _slice_from_dict(config.lon_slice)

        if ts is not None:
            ds = ds.sel(time=ts)
        if ls is not None:
            ds = ds.sel(latitude=ls)
        if lo is not None:
            ds = ds.sel(longitude=lo)

        ds = ds[config.variables]
        arr = ds.to_array(dim="variable")
        assert list(arr.coords["variable"].values) == config.variables, (
            f"Variable order mismatch: {list(arr.coords['variable'].values)} != {config.variables}"
        )

        stats = xr.open_zarr(config.stats_path)
        self._means = {v: float(stats[v].sel(statistic="mean").values) for v in config.variables}
        self._stds = {v: float(stats[v].sel(statistic="std").values) for v in config.variables}
        self._ds = ds
        self._lat = ds["latitude"].values
        self._lon = ds["longitude"].values
        self.time = ds["time"].values

        self.tensor = self.to_tensor(ds)

        times = ds["time"].values
        ts_ns = times.astype("datetime64[ns]")
        hours = ts_ns.astype("datetime64[h]").astype(int) % 24
        dayofyears = ((ts_ns - ts_ns.astype("datetime64[Y]")) / np.timedelta64(1, "D")).astype(int) + 1
        self._time_hours = torch.tensor((dayofyears - 1) * 24 + hours, dtype=torch.float32)

    def to_tensor(self, ds: xr.Dataset) -> torch.Tensor:
        arrays = []
        for v in self.config.variables:
            arr = torch.from_numpy(ds[v].values).float()
            arr = (arr - self._means[v]) / self._stds[v]
            arr = torch.nan_to_num(arr, nan=0.0)
            arrays.append(arr)
        return torch.stack(arrays, dim=0)

    def to_xarray(self, x: torch.Tensor, **coords: Any) -> xr.Dataset:
        if x.dim() == 4:
            # (variable, n, latitude, longitude) — name the middle axis
            dim_name = next(iter(coords))
            coord_vals = coords[dim_name]
            arrays = {}
            for i, v in enumerate(self.config.variables):
                arr = x[i].detach().cpu().numpy() * self._stds[v] + self._means[v]
                arrays[v] = ((dim_name, "latitude", "longitude"), arr)
            da = xr.Dataset(arrays, coords={dim_name: coord_vals, "latitude": self._lat, "longitude": self._lon})
        else:
            # (variable, latitude, longitude)
            arrays = {}
            for i, v in enumerate(self.config.variables):
                arr = x[i].detach().cpu().numpy() * self._stds[v] + self._means[v]
                arrays[v] = (("latitude", "longitude"), arr)
            da = xr.Dataset(arrays, coords={"latitude": self._lat, "longitude": self._lon})
        return da

    def __len__(self) -> int:
        return self.tensor.shape[1] - self.config.sequence_length + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tensor[:, idx: idx + self.config.sequence_length], self._time_hours[idx]
