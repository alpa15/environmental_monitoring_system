# Environmental Monitoring System (EMS)

A personal Earth Observation (EO) project to experiment with environmental monitoring workflows using:
- Sentinel‑2 analysis via Google Earth Engine (GEE)
- Disaster-focused benchmark datasets via TorchGeo (ChaBuD, MMFlood, DigitalTyphoon, ADVANCE)
- An offline **CLI-first** pipeline and a separate **Dash** dashboard

> Design goal: keep compute-heavy tasks (downloads, GEE queries, preprocessing, future training/inference) outside the dashboard and run them via CLI. Typer supports command groups (subcommands) for this structure.

---

## Table of contents
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Earth Engine authentication](#earth-engine-authentication)
- [CLI usage](#cli-usage)
- [Dashboard (Dash)](#dashboard-dash)
- [Data & outputs](#data--outputs)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## Project structure

```text
src/
  eo/
    config/
      __init__.py
      const.py
    ee/
      __init__.py
      data_loader.py
      pipeline.py
    torchgeo/
      __init__.py
      data_loader.py
      pipeline.py
    cli.py
dashboard/
  app.py
  pages/
    1_change_detection.py
    2_torchgeo_dl.py
```


---

## Requirements

- Python 3.10+ recommended.
- Core libraries:
    - `typer` (CLI)
    - `torch`, `torchgeo` (datasets and sampling)
    - `matplotlib` (plots/previews)
- Earth Engine stack:
    - `earthengine-api`
    - `geemap`

---

## Installation

1) Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows PowerShell
```

2) Install dependencies (example):
```bash
pip install typer torch torchgeo matplotlib
pip install earthengine-api geemap
```

3) Install the project in editable mode so `import eo` works reliably:
```bash
pip install -e .
```


---

## Configuration

Create/edit `src/eo/config/const.py` and set your Earth Engine project:

```python
PROJECT_ID = "your-gcp-project-id"
```


---

## Earth Engine authentication

Earth Engine requires authentication and initialization; once authenticated, `ee.Initialize(project=...)` can reuse stored credentials and will only need re-auth if credentials are expired/invalid.

Typical local setup:

```bash
earthengine authenticate
```

If you implement an “authenticate-if-needed” pattern inside `initialize_ee()`, you can attempt `ee.Initialize(...)` first and call `ee.Authenticate()` only when initialization fails (and optionally `force=True` to re-create credentials).

---

 ## CLI usage
 
 The CLI is implemented with Typer and exposes subcommands such as `ee` and `torchgeo` (command groups).

 For a more detailed CLI reference (commands, options, conventions), see [`docs/cli.md`](docs/cli.md).

## Notes:

- If `--root` is not provided, datasets default to `data/torchgeo/<dataset_name>`.
- For GeoDatasets, sampling is performed via TorchGeo samplers (e.g., `RandomGeoSampler`)
and `stack_samples`.

---

## Dashboard (Dash)

The dashboard is intentionally kept separate from the CLI and should be run via `dashboard/app.py`.

Example:

```bash
python dashboard/app.py
```


---

## Data \& outputs

By default, the project writes outputs under `data/` (recommended outside `src/`):

```text
data/
  ee/
    outputs/
      single_year_<year>.png
      comparison_<year_a>_<year_b>.png
  torchgeo/
    ChaBuD/...
    MMFlood/...
    DigitalTyphoon/...
    ADVANCE/...
    inspect/...
```

This directory can be excluded from version control (add `data/` to `.gitignore`).

---

## Troubleshooting

### `ModuleNotFoundError: eo`

- Ensure editable install (`pip install -e .`) succeeded.


### Earth Engine errors (auth/init)

- Authenticate with `earthengine authenticate` and verify `PROJECT_ID`.
- If credentials are stale, rerun authentication (optionally with `--force`) and retry initialization.


### TorchGeo sampling looks “random”

- For GeoDatasets, a sample is a randomly extracted chip; set `--seed` for reproducibility and keep `--chip-size` consistent.

---

## Roadmap

- Compute and export explicit change maps (delta NDVI/NDWI) as raster artifacts.
- Add `eo torchgeo train` and `eo torchgeo predict` commands and store outputs for Dash.
- Integrate model baselines and later foundation-model fine-tuning once baselines are stable.

---

## License

No license.
