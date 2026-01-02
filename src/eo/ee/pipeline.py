from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BBox:
    """
    Bounding box in WGS84 degrees.

    Attributes:
        min_lon: Minimum longitude.
        min_lat: Minimum latitude.
        max_lon: Maximum longitude.
        max_lat: Maximum latitude.
    """
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


def run_single_year_analysis(
    bbox: BBox,
    year: int,
    cloud_cover: int = 30,
    out_dir: Path = Path("eo/data/ee/outputs"),
    save_plot: bool = True,
    export_drive: bool = False,
) -> None:
    """
    Run a single-year analysis (RGB/NIR/NDVI/NDWI) for a given AOI.

    Args:
        bbox: Area of interest bounding box.
        year: Year to analyze.
        cloud_cover: CLOUDY_PIXEL_PERCENTAGE threshold.
        out_dir: Directory where plots are saved (when enabled).
        save_plot: If True, generate and save a plot to disk.
        export_drive: If True, export rasters to Google Drive.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    from eo.ee.data_loader import (
        initialize_ee,
        load_sentinel_data,
        visualize_single_day,
        export_to_drive,
    )

    initialize_ee()

    image, bounds, _ = load_sentinel_data(
        start_lat=bbox.min_lat,
        start_lon=bbox.min_lon,
        end_lat=bbox.max_lat,
        end_lon=bbox.max_lon,
        start_year=year,
        end_year=year,
        cloud_cover=cloud_cover,
    )
    if not image:
        return

    if save_plot:
        visualize_single_day(
            image=image,
            bounds=bounds,
            save_output=True,
            output_path=out_dir / f"single_year_{year}.png",
        )

    if export_drive:
        export_to_drive(image=image, bounds=bounds)


def run_two_year_comparison(
    bbox: BBox,
    year_a: int,
    year_b: int,
    cloud_cover: int = 30,
    out_dir: Path = Path("eo/data/ee/outputs"),
    save_plot: bool = True,
) -> None:
    """
    Compare two years for a given AOI using side-by-side plots.

    Notes:
        This is a visualization-based comparison. A next step could be computing
        explicit delta maps (e.g., NDVI_year_b - NDVI_year_a).

    Args:
        bbox: Area of interest bounding box.
        year_a: First year.
        year_b: Second year.
        cloud_cover: CLOUDY_PIXEL_PERCENTAGE threshold.
        out_dir: Directory where plots are saved (when enabled).
        save_plot: If True, generate and save a comparison plot to disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    from eo.ee.data_loader import initialize_ee, load_sentinel_data, visualize_comparison

    initialize_ee()

    image_a, bounds, _ = load_sentinel_data(
        start_lat=bbox.min_lat,
        start_lon=bbox.min_lon,
        end_lat=bbox.max_lat,
        end_lon=bbox.max_lon,
        start_year=year_a,
        end_year=year_a,
        cloud_cover=cloud_cover,
    )

    image_b, _, _ = load_sentinel_data(
        start_lat=bbox.min_lat,
        start_lon=bbox.min_lon,
        end_lat=bbox.max_lat,
        end_lon=bbox.max_lon,
        start_year=year_b,
        end_year=year_b,
        cloud_cover=cloud_cover,
    )

    if not (image_a and image_b):
        return

    if save_plot:
        visualize_comparison(
            image_start=image_a,
            image_end=image_b,
            bounds=bounds,
            start_year=year_a,
            end_year=year_b,
            save_output=True,
            output_path=out_dir / f"comparison_{year_a}_{year_b}.png",
        )