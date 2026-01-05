from __future__ import annotations

from pathlib import Path
from typing import Optional

import ee
import geemap
import numpy as np
from matplotlib import pyplot as plt

from eo.ee.data_loader import _legend_text


def compute_stats(image: ee.Image, bounds: ee.Geometry, year: int, season: str) -> dict[str, float]:
    """
    Compute basic statistics (mean, stddev, min, max) for NIR, NDVI, NDWI bands.

    Returns a dictionary with keys like: 'NIR_mean', 'NDVI_stddev', etc.
    """
    stats_dict: dict[str, float] = {}

    stats = image.select(["B8", "NDVI", "NDWI"]).reduceRegion(
        reducer=ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), "", True)
        .combine(ee.Reducer.min(), "", True)
        .combine(ee.Reducer.max(), "", True),
        geometry=bounds,
        scale=10,
        maxPixels=1e13,
    )

    band_names = {
        "B8": "NIR",
        "NDVI": "NDVI",
        "NDWI": "NDWI",
    }

    coords = bounds.coordinates().getInfo()  # 1 server round-trip
    # Rectangle -> GeoJSON-like polygon ring: [[min_lon, min_lat], ..., [max_lon, max_lat], ...]
    min_lon, min_lat = coords[0][0]
    max_lon, max_lat = coords[0][2]
    stats_dict["min_lon"] = float(min_lon)
    stats_dict["min_lat"] = float(min_lat)
    stats_dict["max_lon"] = float(max_lon)
    stats_dict["max_lat"] = float(max_lat)
    stats_dict["year"] = int(year)
    stats_dict["season"] = season

    for band_key, band_label in band_names.items():
        for stat in ["mean", "stdDev", "min", "max"]:
            key = f"{band_key}_{stat}"
            value = stats.get(key)
            if value is not None:
                stats_dict[f"{band_label}_{stat}"] = float(value.getInfo())

    return stats_dict


def compute_delta_maps(image_a: ee.Image, image_b: ee.Image) -> ee.Image:
    """
    Build a delta image for a two-year comparison.

    Assumptions:
    - image_a is the first year (baseline)
    - image_b is the second year
    - both images already contain: B4, B3, B2, B8, NDVI, NDWI

    Output bands:
    - RGB_A_B4, RGB_A_B3, RGB_A_B2  (RGB from image_a)
    - RGB_B_B4, RGB_B_B3, RGB_B_B2  (RGB from image_b)
    - Delta_NIR   = B8_b   - B8_a
    - Delta_NDVI  = NDVI_b - NDVI_a
    - Delta_NDWI  = NDWI_b - NDWI_a
    """
    rgb_a = image_a.select(["B4", "B3", "B2"]).rename(["RGB_A_B4", "RGB_A_B3", "RGB_A_B2"])
    rgb_b = image_b.select(["B4", "B3", "B2"]).rename(["RGB_B_B4", "RGB_B_B3", "RGB_B_B2"])

    delta_nir = image_b.select("B8").subtract(image_a.select("B8")).rename("Delta_NIR")
    delta_ndvi = image_b.select("NDVI").subtract(image_a.select("NDVI")).rename("Delta_NDVI")
    delta_ndwi = image_b.select("NDWI").subtract(image_a.select("NDWI")).rename("Delta_NDWI")

    return rgb_a.addBands(rgb_b).addBands([delta_nir, delta_ndvi, delta_ndwi])


def visualize_delta(
    delta_image: ee.Image,
    bounds: ee.Geometry,
    *,
    year_a: Optional[int] = None,
    year_b: Optional[int] = None,
    show_plot: bool = False,
    save_output: bool = False,
    output_path: Optional[Path] = None,
) -> None:
    """
    Layout (matching the sketch):
    - Left column: RGB A (top), RGB B (bottom)
    - Right side (3 tall panels): |ΔNIR|, |ΔNDVI|, |ΔNDWI|
      in grayscale from white (0) to black (max).
    """
    # --- Layout like the sketch: 3 rows x 4 cols, with tall delta panels on the right
    fig = plt.figure(figsize=(22, 8))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.55], width_ratios=[1.1, 1.6, 1.6, 1.6], wspace=0.25, hspace=0.15)

    ax_rgb_a = fig.add_subplot(gs[0, 0])
    ax_rgb_b = fig.add_subplot(gs[1, 0])
    ax_dnir  = fig.add_subplot(gs[0:2, 1])
    ax_dndvi = fig.add_subplot(gs[0:2, 2])
    ax_dndwi = fig.add_subplot(gs[0:2, 3])
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis("off")

    def _ee_to_array(bands: list[str]) -> Optional[np.ndarray]:
        arr = geemap.ee_to_numpy(delta_image.select(bands), bounds, scale=10)
        if arr is None:
            return None
        if arr.ndim == 4:  # (1, H, W, C)
            arr = arr[0]
        return arr

    # --- RGB A
    arr_a = _ee_to_array(["RGB_A_B4", "RGB_A_B3", "RGB_A_B2"])
    if arr_a is None:
        ax_rgb_a.set_title("RGB A (n/a)")
        ax_rgb_a.axis("off")
    else:
        rgb = np.clip(arr_a * 3.0, 0.0, 1.0)
        ax_rgb_a.imshow(rgb)
        ax_rgb_a.set_title(f"RGB - {year_a}" if year_a is not None else "RGB A")
        ax_rgb_a.axis("off")

    # --- RGB B
    arr_b = _ee_to_array(["RGB_B_B4", "RGB_B_B3", "RGB_B_B2"])
    if arr_b is None:
        ax_rgb_b.set_title("RGB B (n/a)")
        ax_rgb_b.axis("off")
    else:
        rgb = np.clip(arr_b * 3.0, 0.0, 1.0)
        ax_rgb_b.imshow(rgb)
        ax_rgb_b.set_title(f"RGB - {year_b}" if year_b is not None else "RGB B")
        ax_rgb_b.axis("off")

    # --- Delta panels (absolute value, white=0, black=max)
    delta_panels = [
        (ax_dnir, "NIR Δ", ["Delta_NIR"]),
        (ax_dndvi, "NDVI Δ", ["Delta_NDVI"]),
        (ax_dndwi, "NDWI Δ", ["Delta_NDWI"]),
    ]

    for ax, title, bands in delta_panels:
        arr = _ee_to_array(bands)
        if arr is None:
            ax.set_title(f"{title} (n/a)")
            ax.axis("off")
            continue

        band = np.squeeze(arr).astype(float)
        mag = np.abs(band)

        vmax = float(np.nanpercentile(mag, 98))
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1e-6

        im = ax.imshow(mag, cmap="gray_r", vmin=0.0, vmax=vmax)
        if (year_a is not None) and (year_b is not None):
            ax.set_title(f"{title} ({year_a} → {year_b})")
        else:
            ax.set_title(title)

        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        ax.axis("off")

    delta_note = ""
    if (year_a is not None) and (year_b is not None):
        delta_note = (
            f"Delta convention (signed): year_b - year_a = {year_b} - {year_a}\n"
            "Shown here as |delta| (magnitude).\n\n"
        )

    legend_str = delta_note + _legend_text()

    ax_legend.text(
        0.0,
        1.0,
        legend_str,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.9),
    )

    if save_output:
        if output_path is None:
            output_path = Path("delta_image.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=250)

    if show_plot:
        plt.show()
    plt.close(fig)
