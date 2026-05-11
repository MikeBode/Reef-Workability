from data_processing.extraction.whacs_weather_extractor import WhacsWeatherExtractor
import pandas as pd
import pathlib

def extract_historical_whacs_for_centroids(centroids_path: pathlib.Path, whacs_base_path: pathlib.Path):
    whacs_extractor = WhacsWeatherExtractor(whacs_base_path)

    centroids = []

    with open(centroids_path, 'r') as f:
        centroids_txt = [line.strip() for line in f.readlines()]

        for txt_centroid in centroids_txt:
            if len(txt_centroid) == 0:
                continue

            lat_str, lon_str = txt_centroid.split(",")
            centroids.append((float(lat_str), float(lon_str)))
    
    historical_centroid_weather_data = pd.DataFrame(columns=["latitude", "longitude", "datetime", "wave_height", "u_wind", "v_wind", "wind_magnitude"])
    