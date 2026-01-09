<p align="center">
  <img src="assets/logo.png" alt="EMS logo" width="160">
</p>

<h1 align="center">Environmental Monitoring System (EMS)</h1>

<p align="center">
  CLI-first Earth Observation workflows (GEE + TorchGeo) with a separate Dash dashboard.
</p>

<p align="center">
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue"></a>
  <a href="#"><img alt="CLI" src="https://img.shields.io/badge/CLI-Typer-black"></a>
  <a href="#"><img alt="DL" src="https://img.shields.io/badge/Deep%20Learning-Torch%20%2B%20TorchGeo-orange"></a>
  <a href="#"><img alt="EO" src="https://img.shields.io/badge/Earth%20Observation-GEE%20%2B%20Sentinel--2-green"></a>
</p>


A personal Earth Observation (EO) project to experiment with environmental monitoring workflows using:
- Sentinel‑2 analysis via Google Earth Engine (GEE)
- Disaster-focused benchmark datasets via TorchGeo (ChaBuD, MMFlood, DigitalTyphoon, ADVANCE)
- An offline **CLI-first** pipeline and a separate **Dash** dashboard
- Transformer-based change detection baseline integrated from **BIT_CD** (Bitemporal Image Transformer) for ChaBuD inference (pretrained weights); training/fine-tuning on ChaBuD is currently not included.

> Design goal: keep compute-heavy tasks (downloads, GEE queries, preprocessing, future training/inference) outside the dashboard and run them via CLI. Typer supports command groups (subcommands) for this structure.


---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pip install -e .
eo torchgeo inspect --dataset ChaBuD --n 3 --chip-size 256
```


---

## Table of contents
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Earth Engine authentication](#earth-engine-authentication)
- [CLI usage](#cli-usage)
- [Models](#models)
- [Dashboard (Dash)](#dashboard-dash)
- [Data & outputs](#data--outputs)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [License](#license)


---

## Project structure

```text
├── dashboard/
│   ├── app.py
│   └── pages/
│       ├── 1_change_detection.py
│       └── 2_torchgeo_dl.py
├── docs/
│   └── cli.md
├── src/
│   └── eo/
│       ├── cli.py
│       ├── config/
│       ├── ee/
│       ├── models/
│       │   └── BIT_CD/
│       └── torchgeo/
├── .gitignore
├── README.md
└── requirements.txt
```


---

## Requirements

- Python 3.10+ recommended.
- Dependencies are pinned in `requirements.txt` (Earth Engine stack, TorchGeo/PyTorch, geospatial I/O, and dashboard libraries).


---

## Installation

1) Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows PowerShell
```

2) Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
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
and `stacksamples`.


---

## Models

### BIT_CD (ChaBuD change detection)

BIT_CD is the official PyTorch implementation of the **Bitemporal Image Transformer (BIT)** for remote sensing change detection.  
Given a pair of co-registered images (T1, T2), BIT models *cross-temporal interactions* with transformer attention and produces a pixel-wise change map (binary semantic segmentation).

In this project, the codebase is vendored under `src/eo/models/BIT_CD/` and currently used for inference; training/fine-tuning on ChaBuD is not part of the pipeline yet.

**Official resources**
- Paper (IEEE TGRS): https://ieeexplore.ieee.org/document/9491802  
- Code: https://github.com/justchenhao/BIT_CD


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

- Add dashboard pages.


---

## Citation

If you use the BIT model (or the vendored BIT_CD code) in academic work, please cite:

```bibtex
@Article{chen2021a,
    title={Remote Sensing Image Change Detection with Transformers},
    author={Hao Chen, Zipeng Qi and Zhenwei Shi},
    year={2021},
    journal={IEEE Transactions on Geoscience and Remote Sensing},
    volume={},
    number={},
    pages={1-14},
    doi={10.1109/TGRS.2021.3095166}
}
```


---

## License

No license.
