from __future__ import annotations

import numpy as np
import xarray as xr


def latitude_weights(latitude) -> xr.DataArray | np.ndarray:
    """cos(latitude) normalised to mean one — the relative area of each latitude band.

    Args:
        latitude: (h,) latitude values in degrees, either xr.DataArray or np.ndarray
    Returns:
        Same shape as input, weights summing to h (mean = 1)
    """
    w = np.cos(np.deg2rad(latitude))       # cos(angle) — area of a latitude band
    if isinstance(latitude, xr.DataArray):
        return w / w.mean("latitude")      # normalise so mean = 1
    return w / w.mean()


def truth_at(era5: xr.Dataset, forecast: xr.Dataset) -> xr.Dataset:
    """The stored fields at the valid times of the forecast (time + prediction_timedelta).

    Uses xarray's vectorised indexing: a 2D DataArray of valid times selects from
    the 1D time axis of era5, producing a 2D result.

    Args:
        era5: the full ERA5 store with dims (time, latitude, longitude) per variable
        forecast: the forecast Dataset with dims (time, prediction_timedelta, latitude, longitude)
    Returns:
        xr.Dataset with the same dims and variables as forecast, containing truth values
    """
    # valid_times: (n_inits, n_leads) — each (i, j) is init_time[i] + lead_time[j]
    valid_times = forecast["time"].values[:, None] + forecast["prediction_timedelta"].values[None, :]
    valid_times_da = xr.DataArray(valid_times, dims=("time", "prediction_timedelta"))
    # Rename era5's time axis so it doesn't clash with the forecast's time (initialisation)
    truth = era5[forecast.data_vars].rename({"time": "_era5_time"})
    # Vectorised selection: for each (init, lead) pair, pick the right era5 timestep
    truth = truth.sel(_era5_time=valid_times_da)
    truth = truth.rename({"_era5_time": "time"})
    truth["prediction_timedelta"] = forecast["prediction_timedelta"]
    return truth


def climatology_at(clim: xr.Dataset, forecast: xr.Dataset) -> xr.Dataset:
    """The (hour, dayofyear) climatology at the valid times of the forecast.

    The climatology has dims (hour, dayofyear, latitude, longitude).
    For each valid time, we look up the climatological mean at that hour and day of year.

    Args:
        clim: the climatology Dataset with dims (hour, dayofyear, latitude, longitude)
        forecast: the forecast Dataset with dims (time, prediction_timedelta, latitude, longitude)
    Returns:
        xr.Dataset with the same dims as forecast, containing climatological values
    """
    # valid_times: (n_inits, n_leads)
    valid_times = forecast["time"].values[:, None] + forecast["prediction_timedelta"].values[None, :]
    valid_times_da = xr.DataArray(valid_times, dims=("time", "prediction_timedelta"))
    hours = valid_times_da.dt.hour                     # (n_inits, n_leads) — hour of day at each valid time
    doys = valid_times_da.dt.dayofyear                 # (n_inits, n_leads) — day of year at each valid time

    # For each variable, select the climatology at the (hour, dayofyear) of each valid time
    clim_vars = {}
    for v in forecast.data_vars:
        clim_v = clim[v]                               # (hour, dayofyear, latitude, longitude)
        hour_idx = xr.DataArray(hours.values, dims=("time", "prediction_timedelta"))
        doy_idx = xr.DataArray(doys.values, dims=("time", "prediction_timedelta"))
        clim_vars[v] = clim_v.sel(hour=hour_idx, dayofyear=doy_idx)  # (n_inits, n_leads, lat, lon)
    result = xr.Dataset(clim_vars, coords={
        "time": forecast["time"],
        "prediction_timedelta": forecast["prediction_timedelta"],
        "latitude": forecast["latitude"],
        "longitude": forecast["longitude"],
    })
    return result


def rmse_per_initialisation(forecast: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
    """Area-weighted RMSE over latitude and longitude.

    One value per initialisation time, lead time, and variable.

    Args:
        forecast, truth: xr.Datasets with dims (time, prediction_timedelta, latitude, longitude)
    Returns:
        xr.Dataset with dims (time, prediction_timedelta) per variable — shape (n_inits, n_leads)
    """
    w = latitude_weights(forecast["latitude"])         # (latitude,) — cos(lat) normalised
    result = {}
    for v in forecast.data_vars:
        err = forecast[v] - truth[v]                   # (n_inits, n_leads, lat, lon)
        # Weight by latitude, mean over lat and lon, then sqrt for RMSE
        result[v] = np.sqrt((err ** 2 * w).mean(dim=("latitude", "longitude")))  # (n_inits, n_leads)
    return xr.Dataset(result)


def rmse(forecast: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
    """RMSE averaged over the initialisations — one value per lead and variable.

    Args:
        forecast, truth: xr.Datasets with dims (time, prediction_timedelta, latitude, longitude)
    Returns:
        xr.Dataset with dim (prediction_timedelta) per variable — shape (n_leads,)
    """
    per_init = rmse_per_initialisation(forecast, truth)  # (n_inits, n_leads) per variable
    return per_init.mean(dim="time")                     # (n_leads,) per variable


def skill_score(rmse_forecast: xr.Dataset, rmse_reference: xr.Dataset) -> xr.Dataset:
    """Skill score: 1 - rmse_forecast / rmse_reference.

    1 = perfect forecast, 0 = reference (e.g. climatology), negative = worse than reference.

    Args:
        rmse_forecast: xr.Dataset with dim (prediction_timedelta) per variable
        rmse_reference: same shape as rmse_forecast
    Returns:
        xr.Dataset with the same dims, values in (-inf, 1]
    """
    result = {}
    for v in rmse_forecast.data_vars:
        result[v] = 1 - rmse_forecast[v] / rmse_reference[v]
    return xr.Dataset(result)
