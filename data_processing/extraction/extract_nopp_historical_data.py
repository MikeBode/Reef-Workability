# Extracts wave height, wind speed, and wavelength from NOPP-phase2 NetCDF files
# for reef survey visits between 1990 and 2008 using spatial interpolation.

import netCDF4
from cftime import num2date
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import os
import glob
from datetime import datetime
from tqdm import tqdm
import traceback

CSV_PATH = "Data/surveyData[63].csv"


def extract_oceanographic_data(df, netcdf_folder):
    print("Starting oceanographic data extraction process...")

    reef_df = df.copy()

    reef_df['date'] = pd.to_datetime(reef_df['date'])

    date_mask = (reef_df['date'] >= '1990-01-01') & (reef_df['date'] <= '2008-12-31')
    filtered_df = reef_df[date_mask].copy()

    if len(filtered_df) == 0:
        print("No reef visits found between 1990 and 2008")
        return pd.DataFrame()

    print(f"Processing {len(filtered_df)} reef visits between 1990 and 2008")

    filtered_df['year_month'] = filtered_df['date'].dt.strftime('%Y%m')
    filtered_df['date_key'] = filtered_df['date'].dt.strftime('%Y-%m-%d')

    nc_files = glob.glob(os.path.join(netcdf_folder, '*.nc'))

    if len(nc_files) == 0:
        print(f"No NetCDF files found in {netcdf_folder}")
        return pd.DataFrame()

    print(f"Found {len(nc_files)} NetCDF files")

    file_mapping = {}
    for file_path in nc_files:
        filename = os.path.basename(file_path)
        if 'multi_reanal.partition.oz_10m.' in filename:
            year_month = filename.split('multi_reanal.partition.oz_10m.')[-1].split('.')[0]
            file_mapping[year_month] = file_path

    print(f"Created mapping for {len(file_mapping)} NetCDF files")

    lat_min, lat_max = -25, -10
    lon_min, lon_max = 142, 154

    needed_year_months = filtered_df['year_month'].unique()
    print(f"Need to process {len(needed_year_months)} unique year-months")

    results = []

    sample_file = None
    for year_month in needed_year_months:
        if year_month in file_mapping:
            sample_file = file_mapping[year_month]
            break

    if sample_file is None:
        print("Could not find any matching NetCDF files for the required dates")
        return pd.DataFrame()

    try:
        with netCDF4.Dataset(sample_file, 'r') as nco:
            lat = nco['latitude'][:]
            lon = nco['longitude'][:]

            lat_mask = (lat >= lat_min) & (lat <= lat_max)
            lon_mask = (lon >= lon_min) & (lon <= lon_max)

            print("Available variables in NetCDF files:")
            for var_name in nco.variables:
                print(f"  - {var_name}")
    except Exception as e:
        print(f"Error examining sample file: {str(e)}")
        lat_mask = slice(None)
        lon_mask = slice(None)

    for year_month in tqdm(needed_year_months, desc="Processing year-months"):
        if year_month not in file_mapping:
            print(f"No NetCDF file found for {year_month}")
            continue

        file_path = file_mapping[year_month]
        month_visits = filtered_df[filtered_df['year_month'] == year_month]
        date_groups = month_visits.groupby('date_key')

        try:
            with netCDF4.Dataset(file_path, 'r') as nco:
                nc_dates = nco['date']
                time_units = nc_dates.units
                calendar = getattr(nc_dates, 'calendar', 'standard')

                try:
                    all_dates = num2date(nc_dates[:], units=time_units, calendar=calendar)
                    date_to_idx = {d.strftime('%Y-%m-%d'): i for i, d in enumerate(all_dates)}
                except Exception as e:
                    print(f"Error converting dates in file {file_path}: {str(e)}")
                    continue

                try:
                    lat = nco['latitude'][:]
                    lon = nco['longitude'][:]

                    masked_lat = lat[lat_mask]
                    masked_lon = lon[lon_mask]

                    lon_grid, lat_grid = np.meshgrid(masked_lon, masked_lat)
                    grid_points = np.column_stack((lon_grid.flatten(), lat_grid.flatten()))
                except Exception as e:
                    print(f"Error creating grid for file {file_path}: {str(e)}")
                    continue

                for date_key, day_visits in date_groups:
                    if date_key not in date_to_idx:
                        print(f"Date {date_key} not found in file {file_path}")
                        continue

                    day_idx = date_to_idx[date_key]

                    try:
                        wavelength = nco['wavelength'][day_idx, 1, lat_mask, :][:, lon_mask]
                        sig_wave_height = nco['significant_wave_height'][day_idx, 1, lat_mask, :][:, lon_mask]
                        wind_speed = nco['wind_speed'][day_idx, lat_mask, :][:, lon_mask]

                        wavelength_flat = wavelength.flatten()
                        swh_flat = sig_wave_height.flatten()
                        wind_flat = wind_speed.flatten()

                        visit_points = np.column_stack((day_visits['x'].values, day_visits['y'].values))

                        wavelength_values = griddata(grid_points, wavelength_flat, visit_points, method='linear')
                        swh_values = griddata(grid_points, swh_flat, visit_points, method='linear')
                        wind_values = griddata(grid_points, wind_flat, visit_points, method='linear')

                        for i, row in enumerate(day_visits.itertuples()):
                            results.append({
                                'reefName': row.reefName,
                                'date': row.date,
                                'longitude': row.x,
                                'latitude': row.y,
                                'source': row.source,
                                'wavelength': wavelength_values[i],
                                'significant_wave_height': swh_values[i],
                                'wind_speed': wind_values[i]
                            })
                    except Exception as e:
                        print(f"Error processing date {date_key} in file {file_path}: {str(e)}")
                        continue
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            print(traceback.format_exc())
            continue

    if not results:
        print("No data could be extracted.")
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    print(f"Successfully extracted data for {len(result_df)} reef visits")

    return result_df


if __name__ == "__main__":
    netcdf_folder = 'nopp-phase2'

    df = pd.read_csv(CSV_PATH)

    result_df = extract_oceanographic_data(df, netcdf_folder)

    if not result_df.empty:
        output_file = 'reef_visits_ocean_data.csv'
        result_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No results to save.")
