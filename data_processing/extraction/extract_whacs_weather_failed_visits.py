# Extracts wave height, u-wind, and v-wind from WHACS hindcast NetCDF files for failed
# COTS survey visits between 2020 and 2024, using KDTree lookup to the nearest grid point.

import pandas as pd
import numpy as np
import xarray as xr
import os
from datetime import datetime
from scipy.spatial import KDTree


def get_nc_file_path(parameter_folder, date):
    year_month = date.strftime('%Y%m')

    if parameter_folder == "SignificantWaveHeight":
        prefix = "hs"
    elif parameter_folder == "UWind":
        prefix = "uwnd"
    elif parameter_folder == "VWind":
        prefix = "vwnd"
    else:
        raise ValueError(f"Unknown parameter folder: {parameter_folder}")

    for file in os.listdir(parameter_folder):
        if file.startswith(prefix) and year_month in file:
            return os.path.join(parameter_folder, file)

    return None


def extract_first_6_hours_mean(nc_file_path, date, x, y):
    if nc_file_path is None:
        return np.nan

    try:
        ds = xr.open_dataset(nc_file_path)

        date_str = date.strftime('%Y-%m-%d')
        start_time = f"{date_str}T06:00:00"
        end_time = f"{date_str}T12:00:00"

        ds_subset = ds.sel(time=slice(start_time, end_time))

        if len(ds_subset.time) == 0:
            print(f"No data found for date {date_str} in file {nc_file_path}")
            return np.nan

        lon = ds_subset.longitude.values
        lat = ds_subset.latitude.values

        lon_grid, lat_grid = np.meshgrid(lon, lat)
        grid_points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))

        tree = KDTree(grid_points)
        distance, index = tree.query([x, y])
        closest_i, closest_j = np.unravel_index(index, lon_grid.shape)

        if 'hs_' in nc_file_path:
            param = 'hs'
        elif 'uwnd_' in nc_file_path:
            param = 'uwnd'
        elif 'vwnd_' in nc_file_path:
            param = 'vwnd'
        else:
            raise ValueError(f"Could not determine parameter from file name: {nc_file_path}")

        mean_value = ds_subset[param].isel(latitude=closest_i, longitude=closest_j).mean().item()

        ds.close()

        return mean_value

    except Exception as e:
        print(f"Error processing file {nc_file_path}: {e}")
        return np.nan


def main():
    cots_df = pd.read_excel('COTS INLOC Weather impacts-WithCoor.xlsx')
    print(cots_df.info())

    cots_df = cots_df.dropna(subset=['x', 'y'])

    if not pd.api.types.is_datetime64_any_dtype(cots_df['Date']):
        cots_df['Date'] = pd.to_datetime(cots_df['Date'])

    filtered_df = cots_df[(cots_df['Date'] >= '2020-01-01') & (cots_df['Date'] < '2024-01-01')]

    if len(filtered_df) == 0:
        print("No data found between 2020 and 2024.")
        return

    filtered_df['wave_height'] = np.nan
    filtered_df['u_wind'] = np.nan
    filtered_df['v_wind'] = np.nan

    for idx, row in filtered_df.iterrows():
        date = row['Date']
        x_coord = row['x']
        y_coord = row['y']

        print(f"Processing reef {row['Reef']} on {date.strftime('%Y-%m-%d')}")

        wave_file = get_nc_file_path("SignificantWaveHeight", date)
        u_wind_file = get_nc_file_path("UWind", date)
        v_wind_file = get_nc_file_path("VWind", date)

        wave_height = extract_first_6_hours_mean(wave_file, date, x_coord, y_coord)
        u_wind = extract_first_6_hours_mean(u_wind_file, date, x_coord, y_coord)
        v_wind = extract_first_6_hours_mean(v_wind_file, date, x_coord, y_coord)

        filtered_df.at[idx, 'wave_height'] = wave_height
        filtered_df.at[idx, 'u_wind'] = u_wind
        filtered_df.at[idx, 'v_wind'] = v_wind

    filtered_df.to_csv('cots_weather_analysis.csv', index=False)


if __name__ == "__main__":
    main()
