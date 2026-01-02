from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import ADVANCE, ChaBuD, DigitalTyphoon, MMFlood
from torchgeo.datasets.utils import stack_samples
from torchgeo.samplers import RandomGeoSampler

# Dataset registry (string -> TorchGeo dataset class)
_DATASETS = {
    "ChaBuD": ChaBuD,
    "MMFlood": MMFlood,
    "DigitalTyphoon": DigitalTyphoon,
    "ADVANCE": ADVANCE,
}


def load_dataset(
    dataset_name: str,
    root: Optional[Path] = None,
    download: bool = True,
    checksum: bool = True,
):
    """
    Load (and optionally download) a TorchGeo dataset.

    Args:
        dataset_name: One of: ChaBuD, MMFlood, DigitalTyphoon, ADVANCE.
        root: Dataset root directory. If None, uses "data/torchgeo/<dataset_name>".
        download: If True, download the dataset if missing.
        checksum: If True, verify checksums when downloading.

    Returns:
        Instantiated TorchGeo dataset object.
    """
    if dataset_name not in _DATASETS:
        raise ValueError(f"Unknown dataset_name={dataset_name}. Choose one of: {list(_DATASETS.keys())}")

    if root is None:
        root = Path("data") / "torchgeo" / dataset_name

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    ds_cls = _DATASETS[dataset_name]
    return ds_cls(
        root=str(root),
        download=download,
        checksum=checksum if download else False,
    )


def get_sample(
    dataset,
    sample_index: int = 0,
    chip_size: int = 256,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Return a single sample as a dict.

    Behavior:
        - If the dataset supports direct indexing (map-style), returns dataset[sample_index].
        - Otherwise, samples one geospatial chip with RandomGeoSampler and returns the batch item.

    Notes:
        For GeoDatasets the sample_index is used only to make sampling reproducible by
        setting the generator seed to (seed + sample_index).

    Args:
        dataset: TorchGeo dataset object.
        sample_index: Index for map-style datasets or deterministic sampling step for GeoDatasets.
        chip_size: Chip size in pixels for RandomGeoSampler.
        seed: Base seed for reproducible sampling.

    Returns:
        A dictionary with at least "image" and possibly "mask"/"crs"/etc. depending on dataset.
    """
    # 1) Map-style datasets (e.g., ChaBuD is usually directly indexable)
    try:
        sample = dataset[sample_index]
        if isinstance(sample, dict):
            return sample
    except Exception:
        pass

    # 2) GeoDatasets: sample a single chip
    g = torch.Generator().manual_seed(seed + int(sample_index))
    sampler = RandomGeoSampler(dataset, size=chip_size, length=1, generator=g)
    loader = DataLoader(dataset, sampler=sampler, batch_size=1, collate_fn=stack_samples)

    batch = next(iter(loader))
    sample_out: Dict[str, Any] = {}
    for k, v in batch.items():
        sample_out[k] = v[0] if torch.is_tensor(v) and v.ndim >= 1 else v
    return sample_out


def _normalize_to_hwc(img: torch.Tensor) -> torch.Tensor:
    """
    Convert (C,H,W) -> (H,W,C) and normalize to [0, 1] for plotting.

    Args:
        img: Image tensor.

    Returns:
        Float tensor in HWC format and [0,1] range.
    """
    img = img.detach().cpu()
    if img.ndim == 3:
        img = img.permute(1, 2, 0)

    img_min = float(img.min())
    img_max = float(img.max())
    denom = (img_max - img_min) if (img_max - img_min) > 1e-8 else 1.0
    return (img - img_min) / denom


def visualize_sample(
    dataset,
    dataset_name: str,
    sample_index: int = 0,
    chip_size: int = 256,
    seed: int = 0,
    figsize: Tuple[int, int] = (15, 5),
) -> None:
    """
    Visualize a sample from one of the supported datasets.

    Args:
        dataset: TorchGeo dataset object.
        dataset_name: Dataset name used for selecting a specific visualization path.
        sample_index: Sample index (map-style) or deterministic sampling step (GeoDataset).
        chip_size: Chip size used for GeoDatasets.
        seed: Base seed for deterministic sampling.
        figsize: Matplotlib figure size.
    """
    sample = get_sample(dataset, sample_index=sample_index, chip_size=chip_size, seed=seed)

    if dataset_name == "ChaBuD":
        img = sample.get("image")
        mask = sample.get("mask")

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        if img is not None and torch.is_tensor(img) and img.ndim >= 4:
            pre = _normalize_to_hwc(img[0, :3])
            post = _normalize_to_hwc(img[1, :3])
            axes[0].imshow(pre)
            axes[0].set_title("Pre-fire (RGB)")
            axes[1].imshow(post)
            axes[1].set_title("Post-fire (RGB)")
        else:
            axes[0].set_title("Pre-fire (n/a)")
            axes[1].set_title("Post-fire (n/a)")

        if mask is not None and torch.is_tensor(mask):
            axes[2].imshow(mask.squeeze().cpu(), cmap="Reds")
            axes[2].set_title("Mask")
        else:
            axes[2].set_title("Mask (n/a)")

        for ax in axes:
            ax.axis("off")
        plt.show()
        return

    img = sample.get("image")
    mask = sample.get("mask")

    fig, axes = plt.subplots(1, 2 if mask is not None else 1, figsize=figsize)
    if not isinstance(axes, (list, tuple)):
        axes = [axes]

    if img is not None and torch.is_tensor(img):
        if img.ndim == 3 and img.shape[0] >= 3:
            vis = _normalize_to_hwc(img[:3])
            axes[0].imshow(vis)
        elif img.ndim == 3 and img.shape[0] == 2:
            vis = _normalize_to_hwc(img[:1]).squeeze(-1)
            axes[0].imshow(vis, cmap="gray")
        else:
            vis = _normalize_to_hwc(img)
            if vis.ndim == 2:
                axes[0].imshow(vis, cmap="gray")
            else:
                axes[0].imshow(vis)
        axes[0].set_title(f"{dataset_name} image")
    else:
        axes[0].set_title(f"{dataset_name} image (n/a)")

    axes[0].axis("off")

    if mask is not None and torch.is_tensor(mask):
        axes[1].imshow(mask.squeeze().cpu(), cmap="Blues")
        axes[1].set_title("Mask")
        axes[1].axis("off")

    plt.show()