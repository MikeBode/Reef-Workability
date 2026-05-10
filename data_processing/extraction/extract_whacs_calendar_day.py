# Extracts WHACS hindcast wave and wind data for all reef centroids across multiple years
# for specific calendar days, writing per-day CSV outputs to a designated folder.

import pandas as pd
import numpy as np
import xarray as xr
import os
from datetime import datetime, timedelta
from scipy.spatial import KDTree
from calendar import monthrange
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import multiprocessing as mp


def process_specific_date(reef_df, month, day, start_year, end_year, output_folder):
    extractor = WeatherExtractor()

    try:
        print(f"\n=== Processing {month}-{day} across years {start_year}-{end_year} ===")

        coords_array = reef_df[['lon', 'lat']].values

        all_data = []

        for year in range(start_year, end_year + 1):
            try:
                current_date = datetime(year, month, day)
            except ValueError:
                print(f"  Skipping {year}-{month:02d}-{day:02d} (doesn't exist)")
                continue

            wave_file = extractor.get_nc_file_path("SignificantWaveHeight", current_date)
            u_wind_file = extractor.get_nc_file_path("UWind", current_date)
            v_wind_file = extractor.get_nc_file_path("VWind", current_date)

            print(f"  Files for {current_date.strftime('%Y-%m-%d')}:")
            print(f"    Wave: {wave_file}")
            print(f"    U-Wind: {u_wind_file}")
            print(f"    V-Wind: {v_wind_file}")

            wave_values = extractor.extract_batch_6_hours_mean(wave_file, current_date, coords_array, 'hs')
            u_wind_values = extractor.extract_batch_6_hours_mean(u_wind_file, current_date, coords_array, 'uwnd')
            v_wind_values = extractor.extract_batch_6_hours_mean(v_wind_file, current_date, coords_array, 'vwnd')

            year_data = pd.DataFrame({
                'date': current_date.strftime('%Y-%m-%d'),
                'year': year,
                'month': month,
                'day': day,
                'lat': reef_df['lat'].values,
                'lon': reef_df['lon'].values,
                'wave_height': wave_values,
                'u_wind': u_wind_values,
                'v_wind': v_wind_values
            })

            all_data.append(year_data)

            valid_wave = np.sum(~np.isnan(wave_values))
            valid_uwind = np.sum(~np.isnan(u_wind_values))
            valid_vwind = np.sum(~np.isnan(v_wind_values))
            total = len(reef_df)

            print(f"  {current_date.strftime('%Y-%m-%d')}: {total} records, "
                  f"valid: wave={valid_wave}, u_wind={valid_uwind}, v_wind={valid_vwind}")

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values(['year', 'lat', 'lon']).reset_index(drop=True)

            filename = f"{month}-{day}.csv"
            filepath = os.path.join(output_folder, filename)
            combined_df.to_csv(filepath, index=False)

            print(f"  Saved {len(combined_df)} total records ({len(all_data)} years x {len(reef_df)} reefs) to {filename}")

            return f"Successfully processed {month}-{day}: {len(combined_df)} records"
        else:
            return f"No data found for {month}-{day}"

    except Exception as e:
        return f"Error processing {month}-{day}: {e}"
    finally:
        extractor.cleanup_cache()


def main():
    try:
        reef_df = pd.read_csv('ReefCentroids.csv')
        print(f"Loaded {len(reef_df)} reef centroids")
        print(f"Columns: {list(reef_df.columns)}")
    except FileNotFoundError:
        print("Error: ReefCentroids.csv not found!")
        return
    except Exception as e:
        print(f"Error loading ReefCentroids.csv: {e}")
        return

    output_folder = "reef_Extracted_Multiyear_missing"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")

    start_year = 2014
    end_year = 2023

    target_dates = [
        (2, 16),
        (2, 17),
        (2, 18),
        (2, 19),
        (3, 11),
        (3, 12),
        (3, 13),
        (3, 14),
        (3, 15),
    ]

    print(f"Processing {len(target_dates)} specific dates for years {start_year}-{end_year}")
    print(f"Target dates: {target_dates}")
    print(f"Expected total records per file: {len(reef_df)} reefs x {end_year - start_year + 1} years "
          f"= {len(reef_df) * (end_year - start_year + 1)}")

    results = []
    for month, day in target_dates:
        try:
            result = process_specific_date(reef_df, month, day, start_year, end_year, output_folder)
            results.append(result)
            print(f"Result: {result}")

        except Exception as e:
            error_msg = f"Error processing {month}-{day}: {e}"
            results.append(error_msg)
            print(error_msg)

    print(f"\n=== Processing Complete ===")
    print("Results summary:")
    for i, result in enumerate(results):
        month, day = target_dates[i]
        print(f"  {month}-{day}: {result}")

    if os.path.exists(output_folder):
        files = sorted([f for f in os.listdir(output_folder) if f.endswith('.csv')])
        print(f"\nFiles created: {len(files)}")
        for file in files:
            print(f"  {file}")

        if files:
            sample_file = os.path.join(output_folder, files[0])
            try:
                sample_df = pd.read_csv(sample_file)
                print(f"\nSample from {files[0]}:")
                print(f"Shape: {sample_df.shape}")
                print(f"Years present: {sorted(sample_df['year'].unique())}")
                print(f"Date range: {sample_df['date'].min()} to {sample_df['date'].max()}")
                print(f"First few rows:")
                print(sample_df.head())
            except Exception as e:
                print(f"Error reading sample file: {e}")


if __name__ == "__main__":
    main()
