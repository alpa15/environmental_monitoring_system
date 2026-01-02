from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from eo.ee.pipeline import BBox as EEBBox
from eo.ee import pipeline as ee_pipeline
from eo.torchgeo import pipeline as tg_pipeline

app = typer.Typer(
    name="eo",
    help="EO pipelines: Earth Engine + TorchGeo (offline jobs).",
    no_args_is_help=True,
)

ee_app = typer.Typer(help="Google Earth Engine pipelines.", no_args_is_help=True)
tg_app = typer.Typer(help="TorchGeo utilities (inspect/train/predict).", no_args_is_help=True)

app.add_typer(ee_app, name="ee")
app.add_typer(tg_app, name="torchgeo")


def _ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist and return the resolved path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


@ee_app.command("single-year-analysis")
def ee_year(
    min_lon: float = typer.Option(..., help="AOI min longitude."),
    min_lat: float = typer.Option(..., help="AOI min latitude."),
    max_lon: float = typer.Option(..., help="AOI max longitude."),
    max_lat: float = typer.Option(..., help="AOI max latitude."),
    year: int = typer.Option(..., help="Year to analyze."),
    cloud_cover: int = typer.Option(30, help="Max CLOUDY_PIXEL_PERCENTAGE filter."),
    out_dir: Path = typer.Option(Path("eo/data/ee/outputs"), help="Output directory."),
    save_plot: bool = typer.Option(True, help="Save the plot to disk."),
    export_drive: bool = typer.Option(False, help="Export rasters to Google Drive."),
) -> None:
    """
    Run single-year Sentinel-2 analysis (RGB/NIR/NDVI/NDWI) for an AOI.
    """
    out_dir = _ensure_dir(out_dir)

    bbox = EEBBox(min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat)
    ee_pipeline.run_single_year_analysis(
        bbox=bbox,
        year=year,
        cloud_cover=cloud_cover,
        out_dir=out_dir,
        save_plot=save_plot,
        export_drive=export_drive,
    )
    typer.echo(f"Done -> {out_dir}")


@ee_app.command("two-year-comparison")
def ee_two_year_comparison(
    min_lon: float = typer.Option(..., help="AOI min longitude."),
    min_lat: float = typer.Option(..., help="AOI min latitude."),
    max_lon: float = typer.Option(..., help="AOI max longitude."),
    max_lat: float = typer.Option(..., help="AOI max latitude."),
    year_a: int = typer.Option(..., help="Start year."),
    year_b: int = typer.Option(..., help="End year."),
    cloud_cover: int = typer.Option(30, help="Max CLOUDY_PIXEL_PERCENTAGE filter."),
    out_dir: Path = typer.Option(Path("eo/data/ee/outputs"), help="Output directory."),
    save_plot: bool = typer.Option(True, help="Save the comparison plot to disk."),
) -> None:
    """
    Run a simple 2-year comparison plot for an AOI.
    """
    out_dir = _ensure_dir(out_dir)

    bbox = EEBBox(min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat)
    ee_pipeline.run_two_year_comparison(
        bbox=bbox,
        year_a=year_a,
        year_b=year_b,
        cloud_cover=cloud_cover,
        out_dir=out_dir,
        save_plot=save_plot,
    )
    typer.echo(f"Done -> {out_dir}")


@tg_app.command("inspect")
def tg_inspect(
    dataset: str = typer.Option(..., help="Dataset name: ChaBuD, MMFlood, DigitalTyphoon, ADVANCE."),
    root: Optional[Path] = typer.Option(None, help="Dataset root directory."),
    n: int = typer.Option(3, help="Number of samples to preview."),
    chip_size: int = typer.Option(256, help="Chip size for GeoDatasets (pixels)."),
    seed: int = typer.Option(0, help="Seed for reproducible sampling."),
    out_dir: Path = typer.Option(Path("eo/data/torchgeo"), help="Output directory."),
    download: bool = typer.Option(False, help="Force download dataset (even if exists)."),
) -> None:
    """
    Inspect/preview samples from a TorchGeo dataset.

    The dataset will only be downloaded if it doesn't already exist locally.
    Use --download to force re-downloading even if the dataset folder exists.
    """
    out_dir = _ensure_dir(out_dir)
    tg_pipeline.inspect_dataset(
        dataset_name=dataset,
        root=root,
        n=n,
        chip_size=chip_size,
        seed=seed,
        out_dir=out_dir,
        download=download,
    )
    typer.echo(f"Done -> {out_dir}")


def main() -> None:
    """CLI entrypoint."""
    app()


if __name__ == "__main__":
    main()