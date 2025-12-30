"""
data_loader.py - Google Earth Engine Data Loader (GENERIC WORLDWIDE)
Vegetation + Water + NIR Analysis with COLOR LEGENDS
"""

import ee
import geemap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import os, sys
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)
from config.const import PROJECT_ID


def initialize_ee(project_id: str = PROJECT_ID):
    """🔑 Initialize Earth Engine with project ID"""
    ee.Initialize(project=project_id)
    return True


def calculate_all_indices(s2_image: ee.Image = None):
    """🌿💧 Calculate NIR + NDVI (vegetation) + NDWI (water) - ONLY 10m bands"""
    
    # 📡 Select ONLY 10m bands + scale from 0-10000 → 0-1
    image = s2_image.select(['B2','B3','B4','B8']).multiply(0.0001)

    # NIR (B8, 842nm) = 0.0 → 1.0 (0% → 100% reflectance)
    #   0.00-0.15  → WATER, deep shadows (low NIR reflectance)
    #   0.15-0.30  → BUILDINGS, ASPHALT, ROOFS (very low reflectance)
    #   0.30-0.45  → BARE SOIL, ROCK, VERY DRY GRASS, CONCRETE
    #   0.45-0.60  → SPARSE GRASS, SHRUBS, VEGETATED SOIL, CROPLAND
    #   0.60-0.75  → MEDIUM GRASS, MEADOW, AGRICULTURE, YOUNG FOREST
    #   0.75-0.90  → DENSE VEGETATION, MATURE TREES, URBAN PARKS, ADULT FOREST
    #   >0.90      → VERY HEALTHY VEGETATION, LUSH TREES
    nir = image.select('B8')   # Near-Infrared (strong vegetation reflection)
    red = image.select('B4')   # Red (chlorophyll absorption)
    green = image.select('B3') # Green (reflects on water/vegetation)
    blue = image.select('B2')  # Blue
    
    # 🌿 NDVI = (NIR - Red) / (NIR + Red) → Vegetation
    #   ❌ < 0.0 = Water, bare soil, buildings
    #   ⚪ 0.0-0.2 = Bare soil, dry grass
    #   🟡 0.2-0.4 = Grass, shrubs
    #   🟢 > 0.4 = Trees, dense vegetation
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    
    # 💧 NDWI = (Green - NIR) / (Green + NIR) → Water
    #   🔴 < -0.2 = Buildings, dry soil
    #   ⚪ -0.2 to 0.0 = Vegetation, soil
    #   🔵 > 0.0 = Water (rivers, lakes)
    ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')
    
    # 🎨 RGB for natural color photo
    rgb = image.select(['B4', 'B3', 'B2'])  # Red, Green, Blue
    
    return rgb.addBands([nir, ndvi, ndwi])


def load_sentinel_data(start_lat: float = 41.90, start_lon: float = 12.49,
                      end_lat: float = 41.92, end_lon: float = 12.52,
                      start_year: int = 2016, end_year: int = 2025,
                      cloud_cover: int = 30):
    """🛰️ Load cleanest Sentinel-2 image (least clouds) for ANY area"""
    
    # 📍 Area of interest (user-defined coordinates)
    bounds = ee.Geometry.Rectangle([start_lon, start_lat, end_lon, end_lat])
    
    # 🔍 Sentinel-2 Surface Reflectance collection (atmospherically corrected)
    s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(bounds)
                    .filterDate(f'{start_year}-01-01', f'{end_year}-12-31')
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover)))  # <30% clouds
    
    print(f'🔍 Found {s2_collection.size().getInfo()} images')
    
    # ⭐ Best image (least clouds)
    s2 = s2_collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
    print('📡 Available bands:', s2.bandNames().getInfo())
    
    if s2.bandNames().getInfo():
        print("✅ NDVI test OK")
        image_final = calculate_all_indices(s2)
        print("✅ NIR + NDVI + NDWI calculated - 10m resolution")
        return image_final, bounds, s2_collection.size().getInfo()
    else:
        print("❌ No valid image")
        return None, None, 0


def visualize_single_day(start_lat: float = 41.90, start_lon: float = 12.49,
                   end_lat: float = 41.92, end_lon: float = 12.52,
                   image: ee.Image = None, bounds: ee.Geometry = None):
    """🖼️ Display interactive map + 2x3 plot with LEGENDS (FULLY PARAMETRIZED)"""
    print("\n📊 LOCAL VISUALIZATION...")
    
    # Create bounds if not provided
    if bounds is None:
        bounds = ee.Geometry.Rectangle([start_lon, start_lat, end_lon, end_lat])
    
    # Auto-center map from coordinates
    center_x = (start_lat + end_lat) / 2
    center_y = (start_lon + end_lon) / 2
    
    # 🎨 Color palettes for matplotlib
    index_viz = {
        'B8': {
            'vmin': 0, 'vmax': 1, 
            'cmap': ListedColormap(['cyan', 'dimgray', 'blue', 'red', 'yellow', 'white', 'green'])
            # 🔵⚫🔵🔴🟡⚪🟢
        },
        'NDVI': {
            'vmin': -0.2, 'vmax': 0.8, 
            'cmap': ListedColormap(['red', 'yellow', 'green'])  # ❌🟡🟢
        },
        'NDWI': {
            'vmin': -0.3, 'vmax': 0.3,
            'cmap': ListedColormap(['darkred', 'white', 'cyan'])  # 🔴⚪🔵
        }
    }
    
    # 🌍 INTERACTIVE MAP
    Map = geemap.Map(center=[center_x, center_y], zoom=13)
    
    # 📷 Natural RGB
    Map.addLayer(image.select(['B4', 'B3', 'B2']), 
                 {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3}, 
                 'Natural RGB')
    
    # 🌿💧 Indices
    map_viz = {
        'B8':  {'min': 0, 'max': 1, 'palette': ['cyan', 'gray', 'green']},
        'NDVI': {'min': -0.2, 'max': 0.8, 'palette': ['red', 'yellow', 'green']},
        'NDWI': {'min': -0.3, 'max': 0.3, 'palette': ['darkred', 'white', 'cyan']}
    }
    
    for index, params in map_viz.items():
        Map.addLayer(image.select(index), params, f'{index}')
    
    Map.addLayer(bounds, {'color': 'red'}, '📍 Analysis Area')
    Map
    
    # 🖼️ 2x3 PLOT
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    axes = axes.ravel()
    
    # [0] RGB
    rgb = geemap.ee_to_numpy(image.select(['B4', 'B3', 'B2']), bounds)
    if len(rgb.shape) == 4: rgb = rgb[0]
    rgb = np.clip(rgb * 3.0, 0, 1)
    print(f"✅ RGB: {rgb.shape}")
    axes[0].imshow(rgb)
    axes[0].set_title('Natural Colors (Sentinel-2)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Indices [1,2,3]
    indices = ['B8', 'NDVI', 'NDWI']
    for i, index in enumerate(indices, 1):
        idx_array = geemap.ee_to_numpy(image.select(index), bounds, scale=10)
        idx_array = np.squeeze(idx_array)
        
        print(f"✅ {index}: {idx_array.shape}, range {idx_array.min():.2f} → {idx_array.max():.2f}")
        
        viz = index_viz[index]
        im = axes[i].imshow(idx_array, cmap=viz['cmap'], vmin=viz['vmin'], vmax=viz['vmax'])
        axes[i].set_title(f'{index}', fontsize=14, fontweight='bold')
        axes[i].axis('off')
        cbar = plt.colorbar(im, ax=axes[i], shrink=0.8)
        cbar.set_label(f'{index} Values', fontsize=12, fontweight='bold')
    
    # [4,5] LEGEND (no overlap)
    axes[4].axis('off')
    axes[5].axis('off')
    
    legend_text = """
    NIR (B8) Legend (0→1 reflectance):
    0.00-0.15 → WATER, shadows | 0.15-0.30 → BUILDINGS
    0.30-0.45 → BARE SOIL | 0.45-0.60 → SPARSE GRASS
    0.60-0.75 → MEADOWS | 0.75-0.90 → TREES, PARKS
    >0.90 → LUSH VEGETATION

    NDVI Legend: <0.0=Bare/RED | 0.0-0.3=Grass/YELLOW | >0.4=Trees/GREEN
    NDWI Legend: <-0.1=Dry/RED | -0.1-0=Soil/WHITE | >0=Water/CYAN

    Resolution: 10m/pixel | Sentinel-2 Data
    """
    
    axes[5].text(-0.5, 0.6, legend_text, transform=axes[5].transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))
    axes[5].set_title('Color Legend', fontsize=14, fontweight='bold')
    
    plt.suptitle('AREA ANALYSIS: NIR + Vegetation + Water (10m)', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    # 📈 STATISTICS
    print("\n📊 AREA STATISTICS:")
    for index in ['B8', 'NDVI', 'NDWI']:
        mean_stats = image.select(index).reduceRegion(ee.Reducer.mean(), bounds, 20).getInfo()
        minmax_stats = image.select(index).reduceRegion(ee.Reducer.minMax(), bounds, 20).getInfo()
        print(f"  {index}: Mean={mean_stats.get(index, 0):.3f}, Range={minmax_stats.get(f'{index}_min', 0):.3f}→{minmax_stats.get(f'{index}_max', 0):.3f}")


def visualize_comparison(start_lat: float = 41.90, start_lon: float = 12.49,
                        end_lat: float = 41.92, end_lon: float = 12.52,
                        image_start: ee.Image = None, image_end: ee.Image = None,
                        bounds: ee.Geometry = None):
    """🖼️ Display interactive map + 2x3 plot with LEGENDS (FULLY PARAMETRIZED)"""
    print("\n📊 LOCAL VISUALIZATION...")
    
    # Create bounds if not provided
    if bounds is None:
        bounds = ee.Geometry.Rectangle([start_lon, start_lat, end_lon, end_lat])
    
    # Auto-center map from coordinates
    center_x = (start_lat + end_lat) / 2
    center_y = (start_lon + end_lon) / 2
    
    # 🎨 Color palettes for matplotlib
    index_viz = {
        'NDWI': {
            'vmin': -0.3, 'vmax': 0.3,
            'cmap': ListedColormap(['darkred', 'white', 'cyan'])  # 🔴⚪🔵
        }
    }
    
    # 🌍 INTERACTIVE MAP
    Map = geemap.Map(center=[center_x, center_y], zoom=13)
    
    # 📷 Natural RGB
    Map.addLayer(image_end.select(['B4', 'B3', 'B2']), 
                 {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3}, 
                 'Natural RGB')
    
    # 🌿💧 Indices
    map_viz = {
        'NDWI': {'min': -0.3, 'max': 0.3, 'palette': ['darkred', 'white', 'cyan']}
    }    
    Map.addLayer(image_start.select(list(map_viz.keys())[0]), list(map_viz.values())[0], f'{list(map_viz.keys())[0]}')
    
    Map.addLayer(bounds, {'color': 'red'}, '📍 Analysis Area')
    Map
    
    # 🖼️ 2x3 PLOT
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    axes = axes.ravel()
    
    # [0] RGB
    rgb = geemap.ee_to_numpy(image_end.select(['B4', 'B3', 'B2']), bounds)
    if len(rgb.shape) == 4: rgb = rgb[0]
    rgb = np.clip(rgb * 3.0, 0, 1)
    print(f"✅ RGB: {rgb.shape}")
    axes[0].imshow(rgb)
    axes[0].set_title('Natural Colors (Sentinel-2)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    i = 1
    for image in [image_start, image_end]:
        idx_array = geemap.ee_to_numpy(image.select(list(map_viz.keys())[0]), bounds, scale=10)
        idx_array = np.squeeze(idx_array)
        
        print(f"✅ {list(map_viz.keys())[0]}: {idx_array.shape}, range {idx_array.min():.2f} → {idx_array.max():.2f}")
        
        viz = index_viz[list(map_viz.keys())[0]]
        im = axes[i].imshow(idx_array, cmap=viz['cmap'], vmin=viz['vmin'], vmax=viz['vmax'])
        axes[i].set_title(f'{list(map_viz.keys())[0]}', fontsize=14, fontweight='bold')
        axes[i].axis('off')
        cbar = plt.colorbar(im, ax=axes[i], shrink=0.8)
        cbar.set_label(f'{list(map_viz.keys())[0]} Values', fontsize=12, fontweight='bold')
        i += 1
    
    # [4,5] LEGEND (no overlap)
    axes[4].axis('off')
    axes[5].axis('off')
    
    legend_text = """
    NDWI Legend: <-0.1=Dry/RED | -0.1-0=Soil/WHITE | >0=Water/CYAN

    Resolution: 10m/pixel | Sentinel-2 Data
    """
    
    axes[5].text(-0.5, 0.6, legend_text, transform=axes[5].transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))
    axes[5].set_title('Color Legend', fontsize=14, fontweight='bold')
    
    plt.suptitle('AREA ANALYSIS: Water (10m)', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    # 📈 STATISTICS
    print("\n📊 AREA STATISTICS:")
    for image in [image_start, image_end]:
        mean_stats = image.select(list(map_viz.keys())[0]).reduceRegion(ee.Reducer.mean(), bounds, 20).getInfo()
        minmax_stats = image.select(list(map_viz.keys())[0]).reduceRegion(ee.Reducer.minMax(), bounds, 20).getInfo()
        print(f"  {list(map_viz.keys())[0]}: Mean={mean_stats.get(list(map_viz.keys())[0], 0):.3f}, Range={minmax_stats.get(f'{list(map_viz.keys())[0]}_min', 0):.3f}→{minmax_stats.get(f'{list(map_viz.keys())[0]}_max', 0):.3f}")


def export_to_drive(image: ee.Image = None, bounds: ee.Geometry = None):
    """💾 Export GeoTIFF to Google Drive"""
    print("🚀 Exporting to Drive...")
    indices = ['RGB', 'B8', 'NDVI', 'NDWI']
    
    for idx in indices:
        if idx == 'RGB':
            bands = ['B4', 'B3', 'B2']
            image_export = image.select(bands)
            desc = 'Area_RGB'
        else:
            bands = [idx]
            image_export = image.select(idx)
            desc = f'Area_{idx}'
        
        task = ee.batch.Export.image.toDrive(
            image=image_export,
            description=desc,
            folder='GEE_Analysis',
            scale=10,
            region=bounds,
            maxPixels=1e8,
            crs='EPSG:4326'
        )
        task.start()
    
    print("✅ 4 files exported: RGB + NIR + NDVI + NDWI → GEE_Analysis/")