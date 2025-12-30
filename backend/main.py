from data_load.osgeo_data_loader import load_dataset, visualize_sample
from data_load.ee_data_loader import initialize_ee, load_sentinel_data, visualize_single_day, visualize_comparison, export_to_drive

from config.const import OSGEO_DB_NAME

def main_osgeo():
    """Main execution: load dataset, get sample, and visualize"""
    dataset = load_dataset(dataset_name=OSGEO_DB_NAME)
    visualize_sample(dataset, sample_index=100, dataset_name=OSGEO_DB_NAME)

def main_ee():
    export = False
    initialize_ee()

    start_lat = 41.85 # Rome start_lat = 41.85
    start_lon = 12.45 # Rome start_lon = 12.45
    end_lat = 41.95 # Rome end_lat = 41.95
    end_lon = 12.55 # Rome end_lon = 12.55
    
    best_image, bounds, _ = load_sentinel_data(start_lat=start_lat, start_lon=start_lon,
                                                end_lat=end_lat, end_lon=end_lon)
    image_init, _, _ = load_sentinel_data(start_lat=start_lat, start_lon=start_lon,
                                                end_lat=end_lat, end_lon=end_lon,
                                                start_year=2017, end_year=2017)
    image_end, _, _ = load_sentinel_data(start_lat=start_lat, start_lon=start_lon,
                                                end_lat=end_lat, end_lon=end_lon,
                                                start_year=2025, end_year=2025)
    if best_image:
        visualize_single_day(start_lat=start_lat, start_lon=start_lon,
                        end_lat=end_lat, end_lon=end_lon,
                        image=best_image, bounds=bounds)
        if export:
            export_to_drive(best_image, bounds)
    
    if image_init and image_end:
        visualize_comparison(start_lat=start_lat, start_lon=start_lon,
                        end_lat=end_lat, end_lon=end_lon,
                        image_start=image_init, image_end=image_end, bounds=bounds)

if __name__ == "__main__":
    main_osgeo()
    main_ee()