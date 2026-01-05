from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import pandas as pd

from eo.ee.data_analysis import compute_stats

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
    season: str = "summer",
    cloud_cover: int = 50,
    out_dir: Path = Path("eo") / "data" / "ee" / "outputs",
    show_plot: bool = False,
    save_plot: bool = True,
    export_drive: bool = False,
) -> None:
    """
    Run a single-year analysis (RGB/NIR/NDVI/NDWI) for a given AOI.

    Args:
        bbox: Area of interest bounding box.
        season: Season to analyze.
        year: Year to analyze.
        cloud_cover: CLOUDY_PIXEL_PERCENTAGE threshold.
        out_dir: Directory where plots are saved (when enabled).
        show_plot: If True, display the plot on screen.
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
        season=season,
        cloud_cover=cloud_cover,
    )
    if not image:
        return
    
    logging.info("Loaded Sentinel-2 image")

    if save_plot:
        visualize_single_day(
            image=image,
            bounds=bounds,
            show_plot=show_plot,
            save_output=True,
            output_path=out_dir / f"single_year_{year}.png",
        )
        logging.info("Saved single-year analysis plot")

    if export_drive:
        export_to_drive(image=image, bounds=bounds)

    if not (out_dir / "stats_df.csv").exists():
        stats_df = pd.DataFrame()
    else:
        stats_df = pd.read_csv(out_dir / "stats_df.csv", index_col=0)

    image_stats = compute_stats(
        image=image,
        bounds=bounds,
        year=year,
        season=season,
    )

    row_idx = len(stats_df)
    for key in image_stats.keys():
        stats_df.loc[row_idx, key] = image_stats[key]

    stats_df.to_csv(out_dir / f"stats.csv")

    logging.info("Saved statistics CSV")


def run_two_year_comparison(
    bbox: BBox,
    year_a: int,
    year_b: int,
    season: str = "summer",
    cloud_cover: int = 50,
    out_dir: Path = Path("eo") / "data" / "ee" / "outputs",
    show_plot: bool = False,
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
        season: Season to analyze.
        cloud_cover: CLOUDY_PIXEL_PERCENTAGE threshold.
        out_dir: Directory where plots are saved (when enabled).
        show_plot: If True, display the comparison plot on screen.
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
        season=season,
        cloud_cover=cloud_cover,
    )

    image_b, _, _ = load_sentinel_data(
        start_lat=bbox.min_lat,
        start_lon=bbox.min_lon,
        end_lat=bbox.max_lat,
        end_lon=bbox.max_lon,
        start_year=year_b,
        end_year=year_b,
        season=season,
        cloud_cover=cloud_cover,
    )

    if not (image_a and image_b):
        raise ValueError("Could not load images for the specified years and AOI. Try to change the year(s)"
        " or the season.")
    
    logging.info("Loaded Sentinel-2 images for both years")

    if save_plot:
        visualize_comparison(
            image_start=image_a,
            image_end=image_b,
            bounds=bounds,
            start_year=year_a,
            end_year=year_b,
            show_plot=show_plot,
            save_output=True,
            output_path=out_dir / f"comparison_{year_a}_{year_b}.png",
        )
        logging.info("Saved two-year comparison plot")

    from eo.ee.data_analysis import compute_delta_maps, visualize_delta

    delta_image = compute_delta_maps(image_a=image_a, image_b=image_b)

    if save_plot:
        visualize_delta(
            delta_image=delta_image,
            bounds=bounds,
            year_a=year_a,
            year_b=year_b,
            save_output=True,
            output_path=out_dir / f"delta_{year_a}_{year_b}.png",
        )
        logging.info("Saved delta comparison plot")

    image_a_stats = compute_stats(
        image=image_a,
        bounds=bounds,
        year=year_a,
        season=season,
    )

    image_b_stats = compute_stats(
        image=image_b,
        bounds=bounds,
        year=year_b,
        season=season,
    )

    if not (out_dir / "stats_df.csv").exists():
        stats_df = pd.DataFrame()
        row_a_idx = 0
        row_b_idx = 1
    else:
        stats_df = pd.read_csv(out_dir / "stats_df.csv", index_col=0)
        min_lon = image_a_stats["min_lon"]
        max_lon = image_a_stats["max_lon"]
        min_lat = image_a_stats["min_lat"]
        max_lat = image_a_stats["max_lat"]

        cols = ["min_lon", "max_lon", "min_lat", "max_lat", "year", "season"]
        values_a = [min_lon, max_lon, min_lat, max_lat, year_a, season]
        values_b = [min_lon, max_lon, min_lat, max_lat, year_b, season]

        # If the stats for year_a or year_b already exist, update them; otherwise, append new rows
        exists_a = (
            stats_df[cols]
            .eq(values_a)
            .all(axis=1)
            .any()
        )
        exists_b = (
            stats_df[cols]
            .eq(values_b)
            .all(axis=1)
            .any()
        )
        if exists_a:
            mask = stats_df[cols].eq(values_a).all(axis=1)
            row_a_idx = stats_df.index[mask][0]
            row_b_idx = len(stats_df)
        else:
            row_a_idx = len(stats_df)
            row_b_idx = row_a_idx + 1

        if exists_b:
            mask = stats_df[cols].eq(values_b).all(axis=1)
            row_b_idx = stats_df.index[mask][0]

    for key in image_a_stats.keys():
        stats_df.loc[row_a_idx, key] = image_a_stats[key]
        stats_df.loc[row_b_idx, key] = image_b_stats[key]

    stats_df.to_csv(out_dir / f"stats_df.csv")

    logging.info("Saved statistics CSV")