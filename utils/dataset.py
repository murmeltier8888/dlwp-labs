from __future__ import annotations

from typing import Any

import numpy as np
import torch
import xarray as xr

from utils.config import DatasetConfig


def _slice_from_dict(d: dict[str, Any] | None) -> slice | None:
    """Convert a dict like {"start": "2015", "stop": "2018"} to a slice object."""
    if d is None:
        return None
    return slice(d.get("start"), d.get("stop"), d.get("step"))


class WeatherDataset(torch.utils.data.Dataset):
    """Loads ERA5 fields, normalises them, and serves sliding windows.

    The full tensor has shape (v, T, H, W) where:
        v = number of variables (e.g. 9)
        T = number of timesteps
        H = latitude points (e.g. 32)
        W = longitude points (e.g. 64)

    Each sample is a window of `sequence_length` consecutive timesteps:
        __getitem__(idx) -> (states, time)
        states: (v, sequence_length, H, W)  — normalised fields
        time:   scalar float32              — (dayofyear - 1) * 24 + hour of the first state
    """
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__()
        self.config = config

        # --- Load and slice the ERA5 zarr store ---
        ds = xr.open_zarr(config.path)
        ts = _slice_from_dict(config.time_slice)       # e.g. slice("2015", "2018")
        ls = _slice_from_dict(config.lat_slice)
        lo = _slice_from_dict(config.lon_slice)

        if ts is not None:
            ds = ds.sel(time=ts)                       # slice along time axis
        if ls is not None:
            ds = ds.sel(latitude=ls)
        if lo is not None:
            ds = ds.sel(longitude=lo)

        # Select only the variables we need, in the order listed in config
        ds = ds[config.variables]
        arr = ds.to_array(dim="variable")
        assert list(arr.coords["variable"].values) == config.variables, (
            f"Variable order mismatch: {list(arr.coords['variable'].values)} != {config.variables}"
        )

        # --- Load normalisation statistics (mean, std) from the stats store ---
        stats = xr.open_zarr(config.stats_path)
        self._means = {v: float(stats[v].sel(statistic="mean").values) for v in config.variables}
        self._stds = {v: float(stats[v].sel(statistic="std").values) for v in config.variables}
        self._ds = ds
        self._lat = ds["latitude"].values               # (H,) — latitude coordinates
        self._lon = ds["longitude"].values               # (W,) — longitude coordinates
        self.time = ds["time"].values                    # (T,) — numpy datetime64 array

        # --- Convert to normalised tensor: (v, T, H, W) ---
        self.tensor = self.to_tensor(ds)

        # --- Pre-compute the time of each state as hours since Jan 1 ---
        # Used by __getitem__ to return the time of the first state in each window
        times = ds["time"].values                        # (T,) datetime64
        ts_ns = times.astype("datetime64[ns]")
        hours = ts_ns.astype("datetime64[h]").astype(int) % 24            # (T,) hour of day: 0, 6, 12, 18
        dayofyears = ((ts_ns - ts_ns.astype("datetime64[Y]")) / np.timedelta64(1, "D")).astype(int) + 1  # (T,) day of year: 1..365
        self._time_hours = torch.tensor((dayofyears - 1) * 24 + hours, dtype=torch.float32)  # (T,) float: 0.0, 6.0, ... 8754.0

    def to_tensor(self, ds: xr.Dataset) -> torch.Tensor:
        """Convert the xarray Dataset to a normalised torch.Tensor.

        For each variable: (arr - mean) / std, then stack along a new dim=0.
        Returns: (v, T, H, W) — variable, time, latitude, longitude
        """
        arrays = []
        for v in self.config.variables:
            arr = torch.from_numpy(ds[v].values).float()    # (T, H, W)
            arr = (arr - self._means[v]) / self._stds[v]    # z-score normalisation
            arr = torch.nan_to_num(arr, nan=0.0)            # replace NaN with 0
            arrays.append(arr)
        return torch.stack(arrays, dim=0)                    # (v, T, H, W)

    def to_xarray(self, x: torch.Tensor, **coords: Any) -> xr.Dataset:
        """Denormalise a tensor back to physical units and wrap as an xarray Dataset.

        x can be:
            (v, n, H, W) — n is an arbitrary middle axis (e.g. time or prediction_timedelta)
            (v, H, W)    — a single field
        """
        if x.dim() == 4:
            # (v, n, H, W) — name the middle axis from the keyword argument
            dim_name = next(iter(coords))
            coord_vals = coords[dim_name]
            arrays = {}
            for i, v in enumerate(self.config.variables):
                arr = x[i].detach().cpu().numpy() * self._stds[v] + self._means[v]  # denormalise
                arrays[v] = ((dim_name, "latitude", "longitude"), arr)
            da = xr.Dataset(arrays, coords={dim_name: coord_vals, "latitude": self._lat, "longitude": self._lon})
        else:
            # (v, H, W) — a single field, no extra named dimension
            arrays = {}
            for i, v in enumerate(self.config.variables):
                arr = x[i].detach().cpu().numpy() * self._stds[v] + self._means[v]
                arrays[v] = (("latitude", "longitude"), arr)
            da = xr.Dataset(arrays, coords={"latitude": self._lat, "longitude": self._lon})
        return da

    def __len__(self) -> int:
        """Number of valid window starting positions."""
        return self.tensor.shape[1] - self.config.sequence_length + 1  # T - seq_len + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a window of consecutive states and the time of its first state.

        Returns:
            states: (v, sequence_length, H, W) — the window of normalised fields
            time:   scalar float32              — (dayofyear - 1) * 24 + hour of the first state
        """
        return self.tensor[:, idx: idx + self.config.sequence_length], self._time_hours[idx]


if __name__ == "__main__":
    import sys
    from utils.config import Config

    path = sys.argv[1] if len(sys.argv) > 1 else "configs/vit_mse.yaml"
    config = Config.from_yaml(path).dataset
    print(f"Loading dataset from {path}")
    print(f"  variables:  {config.variables}")
    print(f"  time_slice: {config.time_slice}")
    print(f"  sequence_length: {config.sequence_length}")

    ds = WeatherDataset(config)
    print(f"  tensor shape: {ds.tensor.shape}  (v, T, H, W)")
    print(f"  num samples:  {len(ds)}")

    states, time = ds[0]
    print(f"  sample[0]:   states {states.shape}, time = {time.item():.1f}h")
    print(f"  sample[-1]:  time = {ds._time_hours[-1].item():.1f}h")
