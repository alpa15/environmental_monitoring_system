from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import ee
import geemap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eo.config.const import PROJECT_ID


def initialize_ee(
    project_id: str = PROJECT_ID,
    *,
    force_auth: bool = False,
    quiet_auth: bool = False,
    auth_mode: Optional[str] = None,
) -> None:
    """
    Initialize Google Earth Engine with a safe "authenticate-if-needed" behavior.

    Strategy:
        1) Try to initialize with existing credentials.
        2) If it fails (missing/expired credentials), authenticate and retry.

    Args:
        project_id: Google Cloud project ID with Earth Engine enabled.
        force_auth: If True, forces a new authentication flow (useful if token expired).
        quiet_auth: If True, reduces interactive output when possible.
        auth_mode: Optional auth mode passed to ee.Authenticate(auth_mode=...).
    """
    try:
        ee.Initialize(project=project_id)
        return
    except Exception:
        # Authenticate chooses the best mode for the environment and reuses
        # existing credentials if available, unless force=True. [web:295]
        ee.Authenticate(force=force_auth, quiet=quiet_auth, auth_mode=auth_mode)  # [web:312]
        ee.Initialize(project=project_id)


def calculate_all_indices(s2_image: ee.Image) -> ee.Image:
    """
    Compute RGB + NIR + NDVI + NDWI for Sentinel-2.

    Notes:
        Uses only 10m bands: B2, B3, B4, B8. Reflectance is scaled to [0, 1].

    Args:
        s2_image: Sentinel-2 SR image.

    Returns:
        ee.Image with bands:
        - B4, B3, B2 (scaled) for RGB display
        - B8 (scaled) as NIR
        - NDVI
        - NDWI
    """
    # Select 10m bands and scale to [0, 1] (S2 SR is typically 0..10000).
    image = s2_image.select(["B2", "B3", "B4", "B8"]).multiply(0.0001)

    nir = image.select("B8")
    red = image.select("B4")
    green = image.select("B3")

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")

    rgb = image.select(["B4", "B3", "B2"])
    return rgb.addBands([nir, ndvi, ndwi])


def load_sentinel_data(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    start_year: int,
    end_year: int,
    season: str = "summer",
    cloud_cover: int = 50,
) -> Tuple[Optional[ee.Image], Optional[ee.Geometry], int]:
    """
    Load a cloud-minimized Sentinel-2 SR image for a given AOI and year range.

    Strategy:
        Filters COPERNICUS/S2_SR_HARMONIZED by geometry, date range, and
        CLOUDY_PIXEL_PERCENTAGE, then picks the least cloudy image.

    Args:
        start_lat: AOI min latitude.
        start_lon: AOI min longitude.
        end_lat: AOI max latitude.
        end_lon: AOI max longitude.
        start_year: Start year (inclusive).
        end_year: End year (inclusive).
        cloud_cover: Maximum CLOUDY_PIXEL_PERCENTAGE threshold.

    Returns:
        (image_with_indices, bounds_geometry, n_images_in_collection)
    """
    bounds = ee.Geometry.Rectangle([start_lon, start_lat, end_lon, end_lat])

    # Date range for the season
    match season.lower():
        case "spring":
            start_date = f"{start_year}-03-01"
            end_date = f"{end_year}-05-31"
        case "summer":
            start_date = f"{start_year}-06-01"
            end_date = f"{end_year}-08-31"
        case "autumn":
            start_date = f"{start_year}-09-01"
            end_date = f"{end_year}-11-30"
        case "winter":
            start_date = f"{start_year}-01-01"
            end_date = f"{end_year}-02-28"
        case _:
            raise ValueError("Invalid season. Please choose from 'spring', 'summer', 'autumn', or 'winter'.")

    s2_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(bounds)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_cover))
    )

    count = int(s2_collection.size().getInfo())
    if count <= 0:
        return None, None, 0

    s2 = s2_collection.sort("CLOUDY_PIXEL_PERCENTAGE").first()
    image_final = calculate_all_indices(s2)
    return image_final, bounds, count


def _legend_text() -> str:
    """Return a compact legend string for NIR/NDVI/NDWI interpretation."""
    return (
        "NIR (B8) reflectance (0→1)\n"
        "0.00–0.15 water/shadows\n"
        "0.15–0.30 built-up/asphalt\n"
        "0.30–0.45 bare soil/rock\n"
        "0.45–0.60 sparse vegetation\n"
        "0.60–0.75 crops/meadows\n"
        "0.75–0.90 dense vegetation\n"
        ">0.90 very lush vegetation\n\n"
        "NDVI (-1→1)\n"
        "<0 water/bare soil\n"
        "0.2–0.4 shrubs/grass\n"
        ">0.4 dense vegetation\n\n"
        "NDWI (-1→1)\n"
        "<-0.1 dry/built-up\n"
        "-0.1–0 mixed\n"
        ">0 water bodies\n"
        "Resolution: 10m/pixel | Sentinel-2 SR"
    )


def visualize_single_day(
    image: ee.Image,
    bounds: ee.Geometry,
    show_plot: bool = False,
    save_output: bool = False,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot RGB, NIR, NDVI, NDWI for a single image.

    Args:
        image: ee.Image containing B4/B3/B2/B8 + NDVI + NDWI.
        bounds: ee.Geometry used for ee_to_numpy extraction.
        show_plot: If True, display the plot on screen.
        save_output: If True, save the figure to disk.
        output_path: Optional explicit output filename (PNG). If None and save_output is True,
            a default name is used in the current working directory.
    """
    spectral_params = {
        "RGB": {"bands": ["B4", "B3", "B2"]},
        "NIR": {"bands": ["B8"], "vmin": 0.0, "vmax": 1.0, "cmap": "Spectral"},
        "NDVI": {"bands": ["NDVI"], "vmin": -1.0, "vmax": 1.0, "cmap": "RdYlGn"},
        "NDWI": {"bands": ["NDWI"], "vmin": -1.0, "vmax": 1.0, "cmap": "Blues"},
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.ravel()

    for i, (name, params) in enumerate(spectral_params.items()):
        array = geemap.ee_to_numpy(image.select(params["bands"]), bounds, scale=10)
        if array is None:
            axes[i].set_title(f"{name} (n/a)")
            axes[i].axis("off")
            continue

        if array.ndim == 4:
            array = array[0]

        if name == "RGB":
            rgb = np.clip(array * 3.0, 0, 1)
            axes[i].imshow(rgb)
            axes[i].set_title("RGB (natural colors)")
        else:
            band = np.squeeze(array)
            im = axes[i].imshow(band, cmap=params["cmap"], vmin=params["vmin"], vmax=params["vmax"])
            axes[i].set_title(name)
            plt.colorbar(im, ax=axes[i], shrink=0.8, pad=0.02)

        axes[i].axis("off")

    fig.text(
        0.02,
        0.02,
        _legend_text(),
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.9),
    )

    plt.tight_layout()

    if save_output:
        if output_path is None:
            output_path = Path("single_day_analysis.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=250)

    if show_plot:
        plt.show()
    plt.close(fig)


def visualize_comparison(
    image_start: ee.Image,
    image_end: ee.Image,
    bounds: ee.Geometry,
    start_year: int,
    end_year: int,
    show_plot: bool = False,
    save_output: bool = False,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot a 2x4 comparison: (RGB, NIR, NDVI, NDWI) for two images/years.

    Args:
        image_start: ee.Image for the first year.
        image_end: ee.Image for the second year.
        bounds: ee.Geometry used for ee_to_numpy extraction.
        start_year: Label for the first year.
        end_year: Label for the second year.
        show_plot: If True, display the plot on screen.
        save_output: If True, save the figure to disk.
        output_path: Optional explicit output filename (PNG). If None and save_output is True,
            a default name is used in the current working directory.
    """
    spectral_params = {
        "RGB": {"bands": ["B4", "B3", "B2"]},
        "NIR": {"bands": ["B8"], "vmin": 0.0, "vmax": 1.0, "cmap": "Spectral"},
        "NDVI": {"bands": ["NDVI"], "vmin": -1.0, "vmax": 1.0, "cmap": "RdYlGn"},
        "NDWI": {"bands": ["NDWI"], "vmin": -1.0, "vmax": 1.0, "cmap": "Blues"},
    }

    images = [(start_year, image_start), (end_year, image_end)]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for row, (year, img) in enumerate(images):
        for col, (name, params) in enumerate(spectral_params.items()):
            ax = axes[row, col]
            array = geemap.ee_to_numpy(img.select(params["bands"]), bounds, scale=10)
            if array is None:
                ax.set_title(f"{name} {year} (n/a)")
                ax.axis("off")
                continue

            if array.ndim == 4:
                array = array[0]

            if name == "RGB":
                rgb = np.clip(array * 3.0, 0, 1)
                ax.imshow(rgb)
                ax.set_title(f"RGB - {year}")
            else:
                band = np.squeeze(array)
                im = ax.imshow(band, cmap=params["cmap"], vmin=params["vmin"], vmax=params["vmax"])
                ax.set_title(f"{name} - {year}")
                plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

            ax.axis("off")

    fig.text(
        0.02,
        0.02,
        _legend_text(),
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.9),
    )

    plt.tight_layout()

    if save_output:
        if output_path is None:
            output_path = Path("comparison_analysis.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=250)

    if show_plot:
        plt.show()
    plt.close(fig)


def export_to_drive(image: ee.Image, bounds: ee.Geometry, folder: str = "GEE_Analysis") -> None:
    """
    Export GeoTIFFs to Google Drive (Earth Engine batch export).

    Exports:
        - RGB (B4, B3, B2)
        - B8
        - NDVI
        - NDWI

    Args:
        image: ee.Image containing the bands above.
        bounds: Export region geometry.
        folder: Google Drive folder name.
    """
    exports = [
        ("Area_RGB", image.select(["B4", "B3", "B2"])),
        ("Area_B8", image.select(["B8"])),
        ("Area_NDVI", image.select(["NDVI"])),
        ("Area_NDWI", image.select(["NDWI"])),
    ]

    for desc, img in exports:
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=desc,
            folder=folder,
            scale=10,
            region=bounds,
            maxPixels=1e8,
            crs="EPSG:4326",
        )
        task.start()