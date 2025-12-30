# Environmental Monitoring System

A comprehensive system for monitoring and analyzing environmental data using satellite imagery and geographic datasets. This project integrates Google Earth Engine and OSGeo data to provide advanced visualizations of environmental changes over time.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [EarthEngine Module](#earthengine-module)
  - [OSGeo Module](#osgeo-module)
- [Project Structure](#project-structure)
- [Available Datasets](#available-datasets)
- [Analysis and Visualizations](#analysis-and-visualizations)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The Environmental Monitoring System is a sophisticated tool for analyzing environmental and climate data. It enables:

- Load and process satellite imagery from Google Earth Engine (Sentinel-2)
- Analyze geographic and geospatial datasets
- Compare environmental changes over time
- Visualize satellite data interactively
- Export results to Google Drive

The system was designed to support environmental research, territory monitoring, and climate change analysis.

---

## Features

✨ **Key Functionalities:**

- 📡 **Google Earth Engine Integration** - Access to Sentinel-2 satellite datasets
- 🗺️ **OSGeo Support** - Load and analyze geospatial datasets
- 📊 **Advanced Visualizations** - Interactive charts and satellite maps
- 🔄 **Temporal Analysis** - Compare images from different periods
- 💾 **Data Export** - Automatic saving to Google Drive
- 📍 **Geographic Queries** - Select areas by lat/lon coordinates
- 🎨 **Multi-channel Rendering** - RGB visualization and advanced composites

---

## Architecture

```
environmental_monitoring_system/
├── backend/                        # Python Backend
│   ├── main.py                     # Main entry point
│   ├── data_load/                  # Data loading modules
│   │   ├── ee_data_loader.py       # Google Earth Engine loader
│   │   └── osgeo_data_loader.py    # OSGeo dataset loader
│   ├── analyses/                   # Analysis modules
│   │   ├── analyse_ee_data.py      # EE data analysis
│   │   └── analyse_osgeo_data.py   # OSGeo data analysis
│   ├── config/                     # Configuration
│   │   └── const.py                # Constants and parameters
│   └── data/                       # Datasets and data
│       ├── ChaBuD/                 # ChaBuD Dataset
│       └── MMFlood/                # MMFlood Dataset (activations)
└── frontend/                       # User Interface
```

---

## System Requirements

**Software:**
- Python 3.8+
- Node.js 14+ (for frontend)
- pip (Python package manager)

**Main Python Libraries:**
- `earthengine-api` - Access to Google Earth Engine
- `geopandas` - Geospatial data processing
- `rasterio` - Reading/writing raster files
- `matplotlib` - Visualizations
- `numpy` - Numerical computations
- `h5py` - Reading HDF5 files

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/alpa15/environmental_monitoring_system.git
cd environmental_monitoring_system
```

### 2. Configure Python Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Google Earth Engine Authentication

```bash
earthengine authenticate
```

This command will open a browser for authentication with your Google account.

### 5. Install Frontend (Optional)

```bash
cd frontend
npm install
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
EE_PROJECT_ID=your-ee-project-id
```

### Configuration Constants

The main constants are defined in [config/const.py](config/const.py):

```python
OSGEO_DB_NAME = "dataset_name"  # Name of the OSGeo dataset to load
```

---

## Usage

### Main Execution

```bash
python backend/main.py
```

This will run both modules (EarthEngine and OSGeo) with default configurations.

### EarthEngine Module

Loads Sentinel-2 satellite imagery for a specific geographic area.

**Parameters:**
- `start_lat`, `start_lon` - Start latitude/longitude coordinates
- `end_lat`, `end_lon` - End latitude/longitude coordinates
- `start_year`, `end_year` - Start and end year (default: current year)

**Example:**

```python
from data_load.ee_data_loader import initialize_ee, load_sentinel_data, visualize_single_day

initialize_ee()

# Load images for Rome
best_image, bounds, _ = load_sentinel_data(
    start_lat=41.85,
    start_lon=12.45,
    end_lat=41.95,
    end_lon=12.55
)

# Visualize results
visualize_single_day(
    start_lat=41.85,
    start_lon=12.45,
    end_lat=41.95,
    end_lon=12.55,
    image=best_image,
    bounds=bounds
)
```

### OSGeo Module

Loads and visualizes geospatial datasets.

**Example:**

```python
from data_load.osgeo_data_loader import load_dataset, visualize_sample
from config.const import OSGEO_DB_NAME

# Load dataset
dataset = load_dataset(dataset_name=OSGEO_DB_NAME)

# Visualize a sample
visualize_sample(dataset, sample_index=100, dataset_name=OSGEO_DB_NAME)
```

---

## Project Structure

### `backend/main.py`
Main entry point that coordinates the execution of data loading modules.

### `data_load/ee_data_loader.py`
Module for Google Earth Engine integration:
- `initialize_ee()` - Initialize the connection
- `load_sentinel_data()` - Load Sentinel-2 images
- `visualize_single_day()` - Visualize single image
- `visualize_comparison()` - Compare two images
- `export_to_drive()` - Export to Google Drive

### `data_load/osgeo_data_loader.py`
Module for loading geospatial datasets:
- `load_dataset()` - Load HDF5 dataset
- `visualize_sample()` - Visualize dataset samples

### `analyses/`
Modules dedicated to specialized analysis:
- `analyse_ee_data.py` - In-depth EarthEngine data analysis
- `analyse_osgeo_data.py` - OSGeo dataset analysis

### `config/const.py`
Contains global constants and configuration parameters.

---

## Available Datasets

### ChaBuD
- **Description:** Training/validation dataset
- **Format:** HDF5
- **Path:** `data/ChaBuD/train_eval.hdf5`

### MMFlood
- **Description:** MMFlood dataset for flood detection
- **Content:**
  - Neural network activations from 71 different events
  - EMSR (Emergency Management Service) identifiers
- **Path:** `data/MMFlood/activations/`
- **Format:** Compressed TAR archives + JSON metadata files

**Available Events:**
- EMSR107, EMSR117, EMSR118, EMSR120, EMSR122, EMSR141, EMSR147-150, EMSR151, EMSR154, EMSR156, EMSR162, EMSR165-167, EMSR184, EMSR187, EMSR192, EMSR199 and others

---

## Analysis and Visualizations

### Supported Visualization Types

1. **Single Images** - RGB visualization of Sentinel-2 images
2. **Temporal Comparisons** - Compare two images from different periods
3. **Multi-channel Composites** - NIR, SWIR and other combination visualizations
4. **Activation Maps** - Visualization of neural network activations
5. **Statistical Charts** - Distributions and time trends

### Analysis Parameters

Analyses use configurable parameters:
- Spectral indices (NDVI, NDWI)
- Cloud cover thresholds
- Time intervals
- Spatial resolution

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## Contact and Support

For questions, suggestions or issues:
- Open an [Issue](https://github.com/alpa15/environmental_monitoring_system/issues)
- Contact the development team

---

**Last updated:** December 2025
