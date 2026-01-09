from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
import rioxarray  # noqa: F401 (attiva accessor .rio)
from rasterio.transform import from_bounds

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

# TODO: remove when these datasets are supported in dev
_NOT_DEV_DATASETS = {
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
    
    # TODO: remove when these datasets are supported in dev
    if dataset_name in _NOT_DEV_DATASETS:
        raise ValueError(f"Warning: Dataset '{dataset_name}' cannot be used yet.")

    if root is None:
        # Allineato al resto del progetto (pipeline/cli)
        root = Path("eo") / "data" / "torchgeo" / dataset_name
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    ds_cls = _DATASETS[dataset_name]
    return ds_cls(
        root=str(root),
        download=download,
        checksum=checksum if download else False,
    )


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, xr.DataArray):
        return np.asarray(x.values)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_xarray_dataarray(
    arr: Any,
    name: str,
    bbox: Any = None,
    crs: Any = None,
) -> xr.DataArray:
    data = _as_numpy(arr)

    # Normalizza shape in modo “geospatial friendly”
    # 2D -> (y, x)
    # 3D -> assume (band, y, x) se band piccolo, altrimenti (y, x, band)
    dims: Tuple[str, ...]
    if data.ndim == 2:
        dims = ("y", "x")
    elif data.ndim == 3:
        if data.shape[0] <= 32:  # euristica: band-first
            dims = ("band", "y", "x")
        else:  # likely HWC
            data = np.transpose(data, (2, 0, 1))
            dims = ("band", "y", "x")
    elif data.ndim == 4:
        # es: ChaBuD pre/post -> (time, band, y, x) oppure (time, y, x, band)
        if data.shape[1] <= 32:
            dims = ("time", "band", "y", "x")
        else:
            data = np.transpose(data, (0, 3, 1, 2))
            dims = ("time", "band", "y", "x")
    else:
        # fallback: lascia dims generiche
        dims = tuple(f"dim_{i}" for i in range(data.ndim))

    da = xr.DataArray(data, dims=dims, name=name)

    # Se ho bbox + crs e ho y/x, costruisco coordinate e metadati rio
    if bbox is not None and ("y" in da.dims and "x" in da.dims):
        # TorchGeo BoundingBox tipicamente ha minx/maxx/miny/maxy
        minx, maxx = float(bbox.minx), float(bbox.maxx)
        miny, maxy = float(bbox.miny), float(bbox.maxy)

        height = int(da.sizes["y"])
        width = int(da.sizes["x"])

        x = np.linspace(minx, maxx, width)
        y = np.linspace(maxy, miny, height)  # y decrescente (north->south)
        da = da.assign_coords(x=("x", x), y=("y", y))

        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        if crs is not None:
            da = da.rio.write_crs(crs, inplace=False)
        da = da.rio.write_transform(transform, inplace=False)

    return da


def _sample_to_xdataset(sample: Dict[str, Any]) -> xr.Dataset:
    # TorchGeo spesso include bbox/crs
    bbox = sample.get("bbox", None)
    crs = sample.get("crs", None)

    data_vars = {}
    for key in ["image", "mask"]:
        if key in sample and sample[key] is not None:
            data_vars[key] = _to_xarray_dataarray(sample[key], name=key, bbox=bbox, crs=crs)

    # Metti il resto come attrs (serializzando in modo semplice)
    attrs = {}
    for k, v in sample.items():
        if k in data_vars:
            continue
        # prova a rendere serializzabile
        if torch.is_tensor(v):
            attrs[k] = str(tuple(v.shape))
        else:
            attrs[k] = str(v)

    ds = xr.Dataset(data_vars=data_vars, attrs=attrs)
    return ds


def get_sample(
    dataset,
    sample_index: int = 0,
    chip_size: int = 256,
    seed: int = 0,
    return_xarray: bool = True,
) -> Union[Dict[str, Any], xr.Dataset]:
    # 1) Map-style
    try:
        sample = dataset[sample_index]
        return sample
    except Exception:
        pass

    # 2) GeoDatasets: chip sampling
    g = torch.Generator().manual_seed(seed + int(sample_index))
    sampler = RandomGeoSampler(dataset, size=chip_size, length=1, generator=g)
    loader = DataLoader(dataset, sampler=sampler, batch_size=1, collate_fn=stack_samples)
    batch = next(iter(loader))

    sample_out: Dict[str, Any] = {}
    for k, v in batch.items():
        sample_out[k] = v[0] if torch.is_tensor(v) and v.ndim >= 1 else v

    return _sample_to_xdataset(sample_out) if return_xarray else sample_out


def _normalize_hwc(img_hwc: np.ndarray) -> np.ndarray:
    """
    Convert (C,H,W) -> (H,W,C) and normalize to [0, 1] for plotting.

    Args:
        img: Image tensor.

    Returns:
        Float tensor in HWC format and [0,1] range.
    """
    img = img_hwc.astype(np.float32)
    mn, mx = float(np.nanmin(img)), float(np.nanmax(img))
    denom = (mx - mn) if (mx - mn) > 1e-8 else 1.0
    return (img - mn) / denom


def visualize_sample(
    dataset,
    dataset_name: str,
    sample_index: int = 0,
    chip_size: int = 256,
    seed: int = 0,
    figsize: Tuple[int, int] = (15, 5),
    show_plot: bool = False,
    save: bool = False,
    use_xarray: bool = True,
) -> None:
    sample = get_sample(
        dataset,
        sample_index=sample_index,
        chip_size=chip_size,
        seed=seed,
        return_xarray=use_xarray,
    )

    # Estrai image/mask in modo uniforme
    img = sample.get("image")
    mask = sample.get("mask")

    # --- helpers ---
    def _axes_list(axs):
        # garantisce lista di matplotlib.axes.Axes
        return list(np.atleast_1d(axs).ravel())
    
    def _to_numpy(x: Any) -> np.ndarray:
        if x is None:
            return None
        if isinstance(x, xr.DataArray):
            return np.asarray(x.values)
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _as_chw(arr: np.ndarray) -> np.ndarray:
        # vogliamo CHW per semplicità
        if arr.ndim == 2:
            return arr[None, ...]  # 1,H,W
        if arr.ndim == 3:
            # euristica: se ultimo asse è piccolo (<=32) potrebbe essere HWC
            if arr.shape[-1] <= 32 and arr.shape[0] > 32:
                return np.transpose(arr, (2, 0, 1))
            return arr  # assume CHW
        if arr.ndim == 4:
            # ChaBuD ecc: lascia gestire dal ramo dedicato
            return arr
        return arr

    def _norm01(a: np.ndarray) -> np.ndarray:
        a = a.astype(np.float32)
        mn, mx = np.nanmin(a), np.nanmax(a)
        denom = (mx - mn) if (mx - mn) > 1e-8 else 1.0
        return (a - mn) / denom

    def _mmflood_false_color(chw: np.ndarray) -> np.ndarray:
        # chw: (2,H,W) -> rgb: (H,W,3)
        vv = chw[0]  # co-polarization
        vh = chw[1]  # cross-polarization
        eps = 1e-8
        ratio = vv / (vh + eps)

        # stessa logica del plot TorchGeo MMFlood: clip(vv/0.3), clip(vh/0.05), clip(ratio/25) [page:2]
        vv_n = np.clip(vv / 0.3, 0.0, 1.0)
        vh_n = np.clip(vh / 0.05, 0.0, 1.0)
        ratio_n = np.clip(ratio / 25.0, 0.0, 1.0)

        rgb = np.stack([vv_n, vh_n, ratio_n], axis=-1)
        return rgb
    
    def save_image(arr: np.ndarray, out_path: Union[str, Path]) -> None:
        arr8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(arr8, mode="RGB").save(out_path)

    def save_mask(arr: np.ndarray, out_path: Union[str, Path]) -> None:
        mask2d = np.squeeze(arr)  # -> (512, 512) [web:272]
        mask8 = (mask2d.astype(np.uint8) * 255)
        Image.fromarray(mask8, mode="L").save(out_path)

    if dataset_name == "ChaBuD":
        img_np = _to_numpy(img)

        ## RGB images pre/post-fire + mask
        pre = _norm01(np.transpose(img_np[0, 3:0:-1], (1, 2, 0))) # RGB
        post = _norm01(np.transpose(img_np[1, 3:0:-1], (1, 2, 0))) # RGB
        mask = np.squeeze(_to_numpy(mask)) if mask is not None else None # MASK

        if save:
            save_image(pre, os.path.join(Path(__file__).resolve().parent.parent, "models", "BIT_CD", "samples", "A", str(sample_index) + ".png"))
            save_image(post, os.path.join(Path(__file__).resolve().parent.parent, "models", "BIT_CD", "samples", "B", str(sample_index) + ".png"))
            if mask is not None:
                save_mask(mask, os.path.join(Path(__file__).resolve().parent.parent, "models", "BIT_CD", "samples", "label", str(sample_index) + ".png"))

        if show_plot:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            axes = _axes_list(axes)

            if img_np is not None and img_np.ndim >= 4:
                # atteso: (time, band, H, W) oppure simile
                axes[0].imshow(pre)
                axes[1].imshow(post)
                axes[0].set_title("Pre-fire (RGB)")
                axes[1].set_title("Post-fire (RGB)")
            else:
                axes[0].set_title("Pre-fire (n/a)")
                axes[1].set_title("Post-fire (n/a)")

            if mask is not None:
                axes[2].imshow(mask, cmap="Reds")
                axes[2].set_title("Mask")
            else:
                axes[2].set_title("Mask (n/a)")

            for ax in axes:
                ax.axis("off")
            plt.show()
        return

    # --- ramo generico (MMFlood incluso) ---
    img_np = _as_chw(_to_numpy(img)) if img is not None else None
    mask_np = np.squeeze(_to_numpy(mask)) if mask is not None else None

    if img_np is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_title(f"{dataset_name} image (n/a)")
        ax.axis("off")
        plt.show()
        return

    # quante bande?
    if img_np.ndim == 3:
        n_bands = img_np.shape[0]
    else:
        n_bands = 1

    # Decide layout pannelli
    if dataset_name == "MMFlood" and n_bands == 2:
        ncols = 2 if mask_np is not None else 1
        fig, axes = plt.subplots(1, ncols, figsize=figsize)
        axes = _axes_list(axes)

        rgb = _mmflood_false_color(img_np)
        axes[0].imshow(rgb)
        axes[0].set_title("MMFlood false color (S1 VV/VH/ratio)")
        axes[0].axis("off")

        if mask_np is not None:
            # TorchGeo: ignore_index=255 da ignorare (spesso conviene metterlo a 0 in preview) [page:2]
            mask_vis = mask_np.copy()
            mask_vis[mask_vis == 255] = 0
            axes[1].imshow(mask_vis, cmap="gray")
            axes[1].set_title("Mask")
            axes[1].axis("off")

        plt.show()
        return

    # Fallback generalista
    if n_bands >= 3:
        rgb = np.transpose(img_np[:3], (1, 2, 0))
        fig, axes = plt.subplots(1, 2 if mask_np is not None else 1, figsize=figsize)
        axes = _axes_list(axes)
        axes[0].imshow(_norm01(rgb))
        axes[0].set_title(f"{dataset_name} RGB (bands 0-2)")
        axes[0].axis("off")
        if mask_np is not None:
            axes[1].imshow(mask_np, cmap="Blues")
            axes[1].set_title("Mask")
            axes[1].axis("off")
        plt.show()
        return

    if n_bands == 2:
        ncols = 3 if mask_np is not None else 2
        fig, axes = plt.subplots(1, ncols, figsize=figsize)
        axes = _axes_list(axes)

        axes[0].imshow(_norm01(img_np[0]), cmap="gray")
        axes[0].set_title(f"{dataset_name} band 0")
        axes[0].axis("off")

        axes[1].imshow(_norm01(img_np[1]), cmap="gray")
        axes[1].set_title(f"{dataset_name} band 1")
        axes[1].axis("off")

        if mask_np is not None:
            axes[2].imshow(mask_np, cmap="Blues")
            axes[2].set_title("Mask")
            axes[2].axis("off")

        plt.show()
        return

    # n_bands == 1
    fig, axes = plt.subplots(1, 2 if mask_np is not None else 1, figsize=figsize)
    axes = _axes_list(axes)
    axes[0].imshow(_norm01(np.squeeze(img_np)), cmap="gray")
    axes[0].set_title(f"{dataset_name} band 0")
    axes[0].axis("off")
    if mask_np is not None:
        axes[1].imshow(mask_np, cmap="Blues")
        axes[1].set_title("Mask")
        axes[1].axis("off")
    plt.show()
