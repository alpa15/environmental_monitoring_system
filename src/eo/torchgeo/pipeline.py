from __future__ import annotations

from pathlib import Path
from typing import Optional

from eo.torchgeo.data_loader import load_dataset, visualize_sample


def inspect_dataset(
    dataset_name: str,
    root: Optional[Path] = None,
    n: int = 3,
    chip_size: int = 256,
    seed: int = 0,
    out_dir: Path = Path("eo") / "data" / "torchgeo",
    download: bool = False,
    use_xarray: bool = False,
) -> None:
    """
    Load a dataset and visualize the first N samples (for debugging/learning).

    Notes:
    This function is intentionally simple: it focuses on quickly verifying
    shapes and sample structure before building a full training pipeline.

    The function now checks if the dataset directory already exists before
    attempting to download. Download is only performed if:
    - The dataset directory doesn't exist AND download=True, OR
    - download=True is explicitly requested regardless of directory existence

    Args:
    dataset_name: One of: ChaBuD, MMFlood, DigitalTyphoon, ADVANCE.
    root: Dataset root directory. If None, uses default in load_dataset.
    n: Number of samples to visualize.
    chip_size: Chip size (pixels) for GeoDatasets.
    seed: Base seed for deterministic sampling.
    out_dir: Reserved for future use (e.g., saving PNG previews + JSON metadata).
    download: If True, download dataset if missing.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine the actual root directory
    if root is None:
        root = Path("eo") / "data" / "torchgeo" / dataset_name
    else:
        root = Path(root)

    # Check if dataset already exists
    dataset_exists = root.exists() and any(root.iterdir())

    # Only download if explicitly requested OR if directory doesn't exist
    should_download = download or not dataset_exists

    if dataset_exists and not download:
        print(f"✓ Dataset directory found at {root}")
        print(f"  Skipping download (use download=True to force re-download)")

    ds = load_dataset(
        dataset_name=dataset_name,
        root=root,
        download=should_download,
    )

    for i in range(n):
        visualize_sample(
            ds,
            dataset_name=dataset_name,
            sample_index=i,
            chip_size=chip_size,
            seed=seed,
            use_xarray=use_xarray,
        )


def train(*args, **kwargs) -> None:
    """
    Placeholder for a future training pipeline.

    Keep the signature flexible while experimenting; later you can
    type it strongly once the training interface is stable.
    """
    raise NotImplementedError("Training pipeline not implemented yet.")


def predict(*args, **kwargs) -> None:
    """
    Placeholder for a future inference pipeline.

    This is where you will load a checkpoint and write predictions to disk
    for Dash to consume.
    """
    raise NotImplementedError("Inference pipeline not implemented yet.")