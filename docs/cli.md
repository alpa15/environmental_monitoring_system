# CLI (eo)

This page documents the **EMS** command-line interface (`eo`), designed to run offline jobs (downloads, Earth Engine queries, preprocessing, dataset previews) separately from the Dash dashboard.

The CLI is implemented with **Typer** and organized into command groups (subcommands):

- `eo ee ...` for Google Earth Engine pipelines (Sentinel-2).
- `eo torchgeo ...` for TorchGeo utilities (dataset inspection / sampling previews).

---

## Quick start

### How to run the CLI

Recommended (after `pip install -e .`):

```bash
venv\Scripts\activate
python -m eo.cli --help
```

List subcommands:

```bash
python -m eo.cli ee --help
python -m eo.cli torchgeo --help
```

> Tip: always use `--help` on the exact command you want to run—Typer will show up-to-date options, defaults, and descriptions.

---

## Conventions and outputs

### AOI as a bounding box

Earth Engine commands require a bounding box AOI:

- `--min-lon`, `--min-lat`
- `--max-lon`, `--max-lat`

These parameters define the geographic rectangle used to query/process Sentinel‑2 imagery.

### Output directories

Several commands accept an `--out-dir`. In `cli.py`, defaults point under `src/eo/data/...`, but in practice it is usually better to override them to a top-level `data/` folder (e.g., `data/ee/outputs`, `data/torchgeo/inspect`) and keep it out of version control via `.gitignore`.

---

## Earth Engine commands (`eo ee`)

### `single-year-analysis`

Runs a single-year Sentinel‑2 analysis (RGB/NIR/NDVI/NDWI) for the selected AOI.

Typical steps:

- Initialize Earth Engine.
- Load a least-cloudy Sentinel‑2 image (filtered by cloud cover threshold).
- Render and optionally save plots.
- Optionally export rasters to Google Drive.


#### Example (save plots)

```bash
python -m eo.cli ee single-year-analysis --min-lon 12.45 --min-lat 41.85 --max-lon 12.55 --max-lat 41.95 --year 2024
```


#### Example (export to Google Drive)

```bash
python -m eo.cli ee single-year-analysis --min-lon 12.45 --min-lat 41.85 --max-lon 12.55 --max-lat 41.95 --year 2024 --export-drive
```

Key options:

- `--cloud-cover`: maximum `CLOUDY_PIXEL_PERCENTAGE` filter (default: 30).
- `--save-plot`: whether to save plots to disk (default: `True`).
- `--export-drive`: export rasters to Google Drive (default: `False`).

---

### `two-year-comparison`

Generates a simple visual comparison plot for two years over the same AOI.

#### Example

```bash
python -m eo.cli ee two-year-comparison --min-lon 12.45 --min-lat 41.85 --max-lon 12.55 --max-lat 41.95 --year-a 2019 --year-b 2024
```

Notes:

- This is currently a visualization-based comparison; explicit delta rasters (e.g., `NDVI_year_b - NDVI_year_a`) can be added later as an artifact-producing step. 

---

## TorchGeo commands (`eo torchgeo`)

### `inspect`

Downloads (if needed), loads, samples, and saves previews of `n` samples from a TorchGeo dataset.

Supported dataset names (passed as `--dataset`): `ChaBuD`, `MMFlood`, `DigitalTyphoon`, `ADVANCE`.

#### Example

```bash
python -m eo.cli torchgeo inspect --dataset MMFlood
```

Key options:

- `--dataset`: dataset name (string).
- `--root`: optional dataset root directory.
- `--n`: number of samples to preview (default: 3).
- `--chip-size`: chip size for GeoDatasets in pixels (default: 256).
- `--seed`: seed for reproducible sampling (default: 0).
- `--download`: force download even if the dataset exists locally (default: `False`).
- `--xarray`: enable xarray/rioxarray-based samples (default: `True`).

---

## Troubleshooting

### `ModuleNotFoundError: eo`

If `python -m eo.cli ...` fails, ensure the project is installed in editable mode (`pip install -e .`).

Also ensure that the CLI module imports from the installed package namespace (e.g., `from eo...`) rather than from local paths (e.g., `from src.eo...`), to avoid import issues depending on how you execute the script.

### Earth Engine auth/init

If Earth Engine fails to initialize:

- run `earthengine authenticate`
- verify `PROJECT_ID` in `src/eo/config/const.py`

---

## Quick links

- Back to the main README: `../README.md`