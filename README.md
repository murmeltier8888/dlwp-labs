# Deep Learning for Weather Prediction: labs

The lab notebooks of the course and the package the labs fill in.
The first notebook asks a line of questions and names the objects the answers produce; the later ones state the pieces to build and check each one.
The code you write accumulates in `utils/`, so that by the end the repository holds a working forecasting pipeline.

## Structure

```
01_data_and_verification.ipynb   ERA5 in xarray, climatologies, tendencies, the persistence forecast, and its verification
02_dl_pipeline.ipynb             the training pipeline: Dataset, loss, Vision Transformer, LightningModule, and configuration
03_inductive_biases.ipynb        checkpoints and forecasts as xarray, then loss weights, roll-out training, per-variable embeddings, and the time of day
04_ensembles.ipynb               the empirical score, a noise input, and the verification of an ensemble
utils/                           the package the labs fill in; empty modules for now
configs/                         the configuration files the notebooks run from
environment.yaml                 the conda environment every notebook runs in
```

Notebooks live in the top level and import from `utils`, so start Jupyter from the repository root.
The later notebooks appear here as the course reaches them.

## Environment

```
conda env create -f environment.yaml
conda activate dlwp
jupyter lab
```

The environment carries every lab from the start: xarray with zarr, dask, and gcsfs for the cloud stores, torch, lightning, and einops for the models, and matplotlib and cartopy for the figures.
Nothing is added per lab.
Solving it takes a few minutes; `mamba env create -f environment.yaml` is faster if you have mamba.

## Data

All data are read from public Google Cloud buckets with anonymous access, and every notebook lists the stores it uses:

- WeatherBench 2 (`gs://weatherbench2/`): ERA5 at 5.625 and 1.5 degrees, the 1990 to 2019 climatology, and the HRES forecasts.
- ARCO-ERA5 (`gs://gcp-public-data-arco-era5/`): ERA5 at its native 0.25 degrees, updated to about a week ago.

Opening a store is lazy and costs nothing; downloading follows the store's chunks.
Load a subset once, cache it with `to_netcdf` or `to_zarr`, and work from the cache.
