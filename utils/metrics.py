from __future__ import annotations

import numpy as np
import xarray as xr


def latitude_weights(latitude) -> xr.DataArray | np.ndarray:
    """cos(latitude) normalised to mean one."""
    w = np.cos(np.deg2rad(latitude))
    if isinstance(latitude, xr.DataArray):
        return w / w.mean("latitude")
    return w / w.mean()


def truth_at(era5: xr.Dataset, forecast: xr.Dataset) -> xr.Dataset:
    """The stored fields at the valid times ``time + prediction_timedelta`` of the forecast."""
    valid_times = forecast["time"].values[:, None] + forecast["prediction_timedelta"].values[None, :]
    valid_times_da = xr.DataArray(valid_times, dims=("time", "prediction_timedelta"))
    truth = era5[forecast.data_vars].rename({"time": "_era5_time"})
    truth = truth.sel(_era5_time=valid_times_da)
    truth = truth.rename({"_era5_time": "time"})
    truth["prediction_timedelta"] = forecast["prediction_timedelta"]
    return truth


def climatology_at(clim: xr.Dataset, forecast: xr.Dataset) -> xr.Dataset:
    """The (hour, dayofyear) climatology at the valid times of the forecast."""
    valid_times = forecast["time"].values[:, None] + forecast["prediction_timedelta"].values[None, :]
    valid_times_da = xr.DataArray(valid_times, dims=("time", "prediction_timedelta"))
    hours = valid_times_da.dt.hour
    doys = valid_times_da.dt.dayofyear

    clim_vars = {}
    for v in forecast.data_vars:
        clim_v = clim[v]
        hour_idx = xr.DataArray(hours.values, dims=("time", "prediction_timedelta"))
        doy_idx = xr.DataArray(doys.values, dims=("time", "prediction_timedelta"))
        clim_vars[v] = clim_v.sel(hour=hour_idx, dayofyear=doy_idx)
    result = xr.Dataset(clim_vars, coords={
        "time": forecast["time"],
        "prediction_timedelta": forecast["prediction_timedelta"],
        "latitude": forecast["latitude"],
        "longitude": forecast["longitude"],
    })
    return result


def rmse_per_initialisation(forecast: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
    """Area-weighted RMSE over latitude and longitude, one value per initialisation, lead, and variable."""
    w = latitude_weights(forecast["latitude"])
    result = {}
    for v in forecast.data_vars:
        err = forecast[v] - truth[v]
        result[v] = np.sqrt((err ** 2 * w).mean(dim=("latitude", "longitude")))
    return xr.Dataset(result)


def rmse(forecast: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
    """RMSE averaged over the initialisations, one value per lead and variable."""
    per_init = rmse_per_initialisation(forecast, truth)
    return per_init.mean(dim="time")


def skill_score(rmse_forecast: xr.Dataset, rmse_reference: xr.Dataset) -> xr.Dataset:
    """1 - rmse_forecast / rmse_reference; one is a perfect forecast, zero the reference."""
    result = {}
    for v in rmse_forecast.data_vars:
        result[v] = 1 - rmse_forecast[v] / rmse_reference[v]
    return xr.Dataset(result)
