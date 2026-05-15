from data_processing.extraction.whacs_weather_extractor import WhacsWeatherExtractor
import pandas as pd
import pathlib
from datetime import datetime, timedelta
import numpy as np
import multiprocessing

def extract_historical_whacs_for_centroids_in_month(centroids: list[tuple[float, float]], whacs_base_path: pathlib.Path, target_month_start: datetime, results_path: pathlib.Path):
    if results_path.exists():
        print(f"Loading cached historical WHACS data for month starting {target_month_start.strftime('%Y-%m-%d')} from {results_path}")
        return pd.read_csv(results_path)

    print(f"Extracting WHACS data for month starting {target_month_start.strftime('%Y-%m-%d')}")

    cur_date = target_month_start
    extractor = WhacsWeatherExtractor(whacs_base_path)
        
    rows = []

    while cur_date.month == target_month_start.month:
        wave_heights = extractor.extract_batch_daytime_hours_mean_by_parameter("hs", cur_date, np.array(centroids))
        u_winds = extractor.extract_batch_daytime_hours_mean_by_parameter("uwnd", cur_date, np.array(centroids))
        v_winds = extractor.extract_batch_daytime_hours_mean_by_parameter("vwnd", cur_date, np.array(centroids))

        for i in range(len(centroids)):
            x, y = centroids[i]

            rows.append({
                "centroid_index": i,
                "day_of_year": int(target_month_start.strftime('%j')),
                "month": target_month_start.month,
                "y": y,
                "x": x,
                "datetime": cur_date,
                "wave_height": wave_heights[i],
                "u_wind": u_winds[i],
                "v_wind": v_winds[i],
                "wind_magnitude": np.hypot(u_winds[i], v_winds[i])
            })

        cur_date += timedelta(days=1)
    
    print(f"Finished extracting WHACS data for month starting {target_month_start.strftime('%Y-%m-%d')}, saving to {results_path}")
    output_df = pd.DataFrame(rows)
    output_df.to_csv(results_path, index=False)
        
    return output_df

def extract_historical_whacs_for_centroids(centroids_path: pathlib.Path, whacs_base_path: pathlib.Path, partial_output_paths: pathlib.Path, start_date: datetime = datetime(2005,1,1), end_date: datetime = datetime(2023, 12, 31)):
    partial_output_paths.mkdir(parents=True, exist_ok=True)

    centroids = []

    with open(centroids_path, 'r') as f:
        centroids_txt = [line.strip() for line in f.readlines()]

        for txt_centroid in centroids_txt:
            if len(txt_centroid) == 0:
                continue

            lat_str, lon_str = txt_centroid.split(",")
            centroids.append((float(lat_str), float(lon_str)))
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        month_starts = []
        cur_date = start_date
        while cur_date <= end_date:
            month_starts.append(cur_date)
            if cur_date.month == 12:
                cur_date = datetime(cur_date.year + 1, 1, 1)
            else:
                cur_date = datetime(cur_date.year, cur_date.month + 1, 1)

        results = [pool.apply_async(extract_historical_whacs_for_centroids_in_month, args=(centroids, whacs_base_path, month_start, partial_output_paths / f"{month_start.strftime('%Y-%m')}.csv")) for month_start in month_starts]
        output_dfs = [result.get() for result in results]
    
    return pd.concat(output_dfs, ignore_index=True)