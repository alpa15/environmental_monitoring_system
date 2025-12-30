"""
osgeo_data_loader.py - ChaBuD Dataset Loader (Burn Area Detection)
Pre-fire and Post-fire Satellite Image Analysis with Burn Mask
"""

import os, sys
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)

import torch
from torchgeo.datasets import ChaBuD, MMFlood, DigitalTyphoon, ADVANCE
from torchgeo.samplers import RandomGeoSampler
from torchgeo.datasets.utils import stack_samples
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def load_dataset(dataset_name: str = "ChaBuD", root: str = None, download: bool = True):
    """
    Load dataset from disk or download if not available
    
    Args:
        dataset_name: "ChaBuD", "MMFlood", "DigitalTyphoon", or "ADVANCE"
        root: Root directory for dataset. If None, uses default paths
        download: Whether to download if not available
    
    Returns:
        Loaded dataset object
    """
    if dataset_name == "ChaBuD":
        if root is None:
            root = "backend/data/ChaBuD"
        
        if not os.path.exists(root):
            print("📥 Downloading ChaBuD...")
            dataset = ChaBuD(root=root, download=download, checksum=True)
            print("✅ Download completed!")
        else:
            print("✅ ChaBuD dataset already present, opening...")
            dataset = ChaBuD(root=root, download=False, checksum=False)
    
    elif dataset_name == "MMFlood":
        if root is None:
            root = "backend/data/MMFlood"
        
        if not os.path.exists(root):
            print("📥 Downloading MMFlood...")
            dataset = MMFlood(root=root, download=download, checksum=True)
            print("✅ Download completed!")
        else:
            print("✅ MMFlood dataset already present, opening...")
            dataset = MMFlood(root=root, download=False, checksum=False)
    
    elif dataset_name == "DigitalTyphoon":
        if root is None:
            root = "backend/data/DigitalTyphoon"
        
        if not os.path.exists(root):
            print("📥 Downloading DigitalTyphoon...")
            dataset = DigitalTyphoon(root=root, download=download, checksum=True)
            print("✅ Download completed!")
        else:
            print("✅ DigitalTyphoon dataset already present, opening...")
            dataset = DigitalTyphoon(root=root, download=False, checksum=False)
    
    elif dataset_name == "ADVANCE":
        if root is None:
            root = "backend/data/ADVANCE"
        
        if not os.path.exists(root):
            print("📥 Downloading ADVANCE...")
            dataset = ADVANCE(root=root, download=download, checksum=True)
            print("✅ Download completed!")
        else:
            print("✅ ADVANCE dataset already present, opening...")
            dataset = ADVANCE(root=root, download=False, checksum=False)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'ChaBuD', 'MMFlood', 'DigitalTyphoon', or 'ADVANCE'")
    
    return dataset


def get_sample(dataset, sample_index: int = 100):
    """
    Load a specific sample from the dataset
    
    Handles both indexed datasets (ChaBuD) and geographic datasets (MMFlood, DigitalTyphoon, ADVANCE)
    """
    try:
        # Try direct indexing first (works for ChaBuD)
        sample = dataset[sample_index]
    except (TypeError, AttributeError, IndexError):
        # For geographic datasets, use RandomGeoSampler with DataLoader
        try:
            sampler = RandomGeoSampler(
                dataset,
                size=256,
                length=sample_index + 1
            )
            dataloader = DataLoader(
                dataset,
                sampler=sampler,
                collate_fn=stack_samples
            )
            
            # Get the last sample from the sampler (index sample_index)
            sample = None
            for batch in dataloader:
                sample = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        # Extract first item from batch dimension (remove batch dim)
                        sample[key] = value[0]
                    else:
                        sample[key] = value
            
            if sample is None:
                raise IndexError(f"Unable to retrieve sample at index {sample_index}")
        except Exception as e:
            # Fallback: standard DataLoader iteration
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            sample = None
            for idx, batch in enumerate(dataloader):
                if idx == sample_index:
                    sample = {}
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            sample[key] = value[0]
                        else:
                            sample[key] = value
                    break
            
            if sample is None:
                raise IndexError(f"Sample index {sample_index} out of range. Error: {str(e)}")

    return sample


def normalize_image(image_tensor):
    """Normalize image tensor or array to 0-1 range"""
    # Convert to numpy if it's a tensor
    if isinstance(image_tensor, torch.Tensor):
        image_array = image_tensor.numpy()
    else:
        image_array = image_tensor
    
    # Handle different shapes
    if image_array.ndim == 3 and image_array.shape[0] in [1, 2, 3, 4]:
        # Shape is (C, H, W)
        image_array = image_array.transpose(1, 2, 0)
    
    # Normalize to 0-1
    image_normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
    return image_normalized


def visualize_chabud_sample(sample, figsize=(15, 5)):
    """Visualize pre-fire, post-fire RGB and burn mask"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Pre-fire RGB
    pre_rgb = normalize_image(sample['image'][0, :3])
    axes[0].imshow(pre_rgb)
    axes[0].set_title("Pre-fire (RGB)")
    
    # Post-fire RGB  
    post_rgb = normalize_image(sample['image'][1, :3])
    axes[1].imshow(post_rgb)
    axes[1].set_title("Post-fire (RGB)")
    
    # Burn mask
    axes[2].imshow(sample['mask'].numpy().squeeze(), cmap='Reds')
    axes[2].set_title("Burned areas")
    
    for ax in axes:
        ax.axis('off')
    
    plt.show()


def visualize_mmflood_sample(sample, figsize=(15, 5)):
    """Visualize flood detection images and mask"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Handle image data - shape is (C, H, W)
    if 'image' in sample:
        image = sample['image']
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        # Image has C channels
        num_channels = image.shape[0]
        
        if num_channels >= 2:
            # Show first two channels as grayscale
            channel_0 = image[0]  # (H, W)
            channel_1 = image[1]  # (H, W)
            
            axes[0].imshow(channel_0, cmap='gray')
            axes[0].set_title("Pre-flood")
            
            axes[1].imshow(channel_1, cmap='gray')
            axes[1].set_title("Post-flood")
        elif num_channels == 1:
            # Single channel
            axes[0].imshow(image[0], cmap='gray')
            axes[0].set_title("Image")
            axes[1].axis('off')
        else:
            axes[0].axis('off')
            axes[1].axis('off')
    else:
        axes[0].axis('off')
        axes[1].axis('off')
    
    # Flood mask
    if 'mask' in sample:
        mask = sample['mask']
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        axes[2].imshow(mask.squeeze(), cmap='Blues')
        axes[2].set_title("Flooded areas")
    else:
        axes[2].axis('off')
    
    for ax in axes:
        ax.axis('off')
    
    plt.show()


def visualize_digitaltyphoon_sample(sample, figsize=(15, 5)):
    """Visualize typhoon satellite images"""
    fig, ax = plt.subplots(figsize=figsize)
    
    if 'image' in sample:
        image_data = normalize_image(sample['image'][:3] if sample['image'].shape[0] >= 3 else sample['image'])
        ax.imshow(image_data)
    
    ax.set_title("Typhoon Image")
    ax.axis('off')
    
    plt.show()


def visualize_advance_sample(sample, figsize=(15, 5)):
    """Visualize advanced disaster detection images"""
    fig, ax = plt.subplots(figsize=figsize)
    
    if 'image' in sample:
        image_data = normalize_image(sample['image'][:3] if sample['image'].shape[0] >= 3 else sample['image'])
        ax.imshow(image_data)
    
    ax.set_title("Disaster Detection Image")
    ax.axis('off')
    
    plt.show()


def visualize_sample(dataset, sample_index: int = 100, dataset_name: str = "ChaBuD", figsize=(15, 5)):
    """
    Generic visualization function that selects appropriate visualization based on dataset type
    
    Args:
        dataset: Loaded dataset object
        sample_index: Index of sample to visualize
        dataset_name: Name of the dataset ("ChaBuD", "MMFlood", "DigitalTyphoon", or "ADVANCE")
        figsize: Figure size for visualization
    """
    sample = get_sample(dataset, sample_index)
    
    if dataset_name == "ChaBuD":
        visualize_chabud_sample(sample, figsize)
    elif dataset_name == "MMFlood":
        visualize_mmflood_sample(sample, figsize)
    elif dataset_name == "DigitalTyphoon":
        visualize_digitaltyphoon_sample(sample, figsize)
    elif dataset_name == "ADVANCE":
        visualize_advance_sample(sample, figsize)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")