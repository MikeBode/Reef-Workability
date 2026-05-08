# Extracts ERA5 wind and wave variables from GRIB files for reef survey locations,
# matching each record to the nearest annual GRIB file and closest spatial grid point.

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from scipy.spatial import KDTree
from tqdm import tqdm

DATA_DIR = "Data"
CSV_PATH = "Data/surveyData[63].csv"
OUTPUT_CSV = "surveyData[63]_with_metrics_post_2015.csv"

variables = {
    'u10': '10m_u_component_of_wind',
    'v10': '10m_v_component_of_wind',
    'mwp': 'mean_wave_period',
    'swh': 'significant_height_of_combined_wind_waves_and_swell',
}


def find_closest_grib_date(target_date, grib_files):
    if isinstance(target_date, str):
        try:
            target_date = datetime.strptime(target_date, '%Y-%m-%d')
        except ValueError:
            print(f"Invalid date format: {target_date}")
            return None

    target_year = target_date.year

    year_files = []
    for file in grib_files:
        basename = os.path.basename(file)
        basename = basename.split('.')[0]

        try:
            if basename.isdigit() and len(basename) == 4:
                year = int(basename)
                year_files.append((file, year, year))

            elif '-' in basename:
                parts = basename.split('-')
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    start_year = int(parts[0])
                    end_year = int(parts[1])
                    year_files.append((file, start_year, end_year))
        except Exception:
            continue

    if not year_files:
        print(f"No suitable GRIB file found for date: {target_date}")
        return None

    matching_files = []

    for file, start_year, end_year in year_files:
        if target_year >= start_year and target_year <= end_year:
            matching_files.append((file, abs(target_year - start_year) + abs(target_year - end_year)))

    if not matching_files:
        for file, start_year, end_year in year_files:
            dist_to_start = abs(target_year - start_year)
            dist_to_end = abs(target_year - end_year)
            min_dist = min(dist_to_start, dist_to_end)
            matching_files.append((file, min_dist))

    matching_files.sort(key=lambda x: x[1])

    return matching_files[0][0] if matching_files else None


def get_data_at_location(ds, lat, lon, variable_shortnames):
    try:
        lat_dim_candidates = [dim for dim in ds.dims if 'lat' in dim.lower()]
        lon_dim_candidates = [dim for dim in ds.dims if 'lon' in dim.lower()]

        if not lat_dim_candidates or not lon_dim_candidates:
            lat_dim_candidates = [dim for dim in ds.dims if 'y' == dim.lower()]
            lon_dim_candidates = [dim for dim in ds.dims if 'x' == dim.lower()]

        if not lat_dim_candidates or not lon_dim_candidates:
            for var_name in ds.variables:
                var = ds[var_name]
                if hasattr(var, 'standard_name'):
                    if var.standard_name == 'latitude':
                        lat_dim_candidates = [var_name]
                    elif var.standard_name == 'longitude':
                        lon_dim_candidates = [var_name]

        if not lat_dim_candidates or not lon_dim_candidates:
            print("Could not identify latitude/longitude dimensions in the dataset")
            print(f"Available dimensions: {list(ds.dims)}")
            return {shortname: np.nan for shortname in variable_shortnames if shortname in variables}

        lat_name = lat_dim_candidates[0]
        lon_name = lon_dim_candidates[0]

        print(f"Using coordinates: {lat_name}, {lon_name}")

        lats = ds[lat_name].values
        lons = ds[lon_name].values

        if lats.ndim == 1 and lons.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(lons, lats)
        else:
            lat_grid, lon_grid = lats, lons

        points = np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T
        tree = KDTree(points)

        _, index = tree.query([lat, lon])
        i, j = np.unravel_index(index, lat_grid.shape)

        if lats.ndim == 1:
            nearest_lat = lats[i]
            nearest_lon = lons[j]
        else:
            nearest_lat = lats[i, j]
            nearest_lon = lons[i, j]

        results = {}
        for shortname, fullname in variables.items():
            if shortname in variable_shortnames:
                try:
                    var_found = False
                    for var_name in [fullname, shortname]:
                        if var_name in ds:
                            var = ds[var_name]
                            if lat_name in var.dims and lon_name in var.dims:
                                value = var.isel({lat_name: i, lon_name: j}).values
                                var_found = True
                                break

                    if not var_found:
                        for var_name in ds.variables:
                            var = ds[var_name]
                            if hasattr(var, 'standard_name') and (
                                var.standard_name == fullname or var.standard_name == shortname
                            ):
                                if lat_name in var.dims and lon_name in var.dims:
                                    value = var.isel({lat_name: i, lon_name: j}).values
                                    var_found = True
                                    break

                    if not var_found:
                        print(f"Variable {shortname}/{fullname} not found in dataset")
                        value = np.nan

                    if hasattr(value, 'size') and value.size == 1:
                        value = float(value)

                    results[shortname] = value
                except Exception as e:
                    print(f"Error extracting {shortname}: {e}")
                    results[shortname] = np.nan

        return results
    except Exception as e:
        print(f"Error processing location ({lat}, {lon}): {e}")
        return {shortname: np.nan for shortname in variable_shortnames if shortname in variables}


def process_locations_data():
    print(f"Reading CSV file: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    df['date'] = pd.to_datetime(df['date'])

    df = df[df['date'].dt.year > 2015]
    print(f"Filtered data to include only years after 2015. Remaining rows: {len(df)}")

    if len(df) == 0:
        print("No data remains after filtering for years > 2015")
        return df

    grib_files = glob.glob(os.path.join(DATA_DIR, "*.grib")) + \
                 glob.glob(os.path.join(DATA_DIR, "*.grib2"))

    if not grib_files:
        raise FileNotFoundError(f"No GRIB files found in {DATA_DIR}")

    print(f"Found {len(grib_files)} GRIB files")
    print(f"GRIB files: {grib_files}")

    df['year'] = df['date'].dt.year
    year_groups = df.groupby('year')

    for shortname in variables.keys():
        df[shortname] = np.nan

    dataset_cache = {}

    for year, group in year_groups:
        print(f"Processing year: {year}")

        year_file = None
        for file in grib_files:
            basename = os.path.basename(file).split('.')[0]

            if basename.isdigit() and int(basename) == year:
                year_file = file
                break

            elif '-' in basename:
                parts = basename.split('-')
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    start_year = int(parts[0])
                    end_year = int(parts[1])
                    if year >= start_year and year <= end_year:
                        year_file = file
                        break

        if not year_file:
            closest_year_diff = float('inf')
            for file in grib_files:
                basename = os.path.basename(file).split('.')[0]

                if basename.isdigit():
                    file_year = int(basename)
                    year_diff = abs(file_year - year)
                    if year_diff < closest_year_diff:
                        closest_year_diff = year_diff
                        year_file = file

                elif '-' in basename:
                    parts = basename.split('-')
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        start_year = int(parts[0])
                        end_year = int(parts[1])
                        dist_to_start = abs(year - start_year)
                        dist_to_end = abs(year - end_year)
                        year_diff = min(dist_to_start, dist_to_end)
                        if year_diff < closest_year_diff:
                            closest_year_diff = year_diff
                            year_file = file

        if not year_file:
            print(f"No suitable GRIB file found for year: {year}")
            continue

        print(f"Using file {year_file} for year {year}")

        if year_file not in dataset_cache:
            try:
                dataset_cache[year_file] = xr.open_dataset(year_file, engine='cfgrib')
                print(f"Successfully opened {year_file}")

                ds = dataset_cache[year_file]
                print(f"Dataset dimensions: {ds.dims}")
                print(f"Dataset variables: {list(ds.variables)}")

            except Exception as e:
                print(f"Error opening GRIB file {year_file}: {e}")
                continue

        ds = dataset_cache[year_file]

        for idx, row in tqdm(group.iterrows(), total=len(group), desc=f"Processing {year}"):
            try:
                if pd.isna(row['x']) or pd.isna(row['y']):
                    continue

                lon, lat = row['x'], row['y']
                data = get_data_at_location(ds, lat, lon, variables.keys())

                for shortname, value in data.items():
                    df.loc[idx, shortname] = value

            except Exception as e:
                print(f"Error processing row {idx}: {e}")

    for ds in dataset_cache.values():
        ds.close()

    original_len = len(df)
    df = df.dropna(subset=list(variables.keys()))
    print(f"Dropped {original_len - len(df)} rows with missing metric values")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved processed data to {OUTPUT_CSV}")

    return df


def create_yearly_maps(df, metric):
    if metric not in variables:
        raise ValueError(f"Invalid metric: {metric}. Available metrics: {list(variables.keys())}")

    years = sorted(df['year'].unique())

    for year in years:
        if year <= 2015:
            continue

        year_data = df[df['year'] == year]

        if len(year_data) == 0:
            print(f"No data for year {year}")
            continue

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, alpha=0.5)
        ax.add_feature(cfeature.OCEAN)

        scatter = ax.scatter(
            year_data['x'],
            year_data['y'],
            c=year_data[metric],
            cmap='viridis',
            transform=ccrs.PlateCarree(),
            s=50,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )

        cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
        cbar.set_label(f"{variables[metric]}")

        plt.title(f"{variables[metric]} - {year}")

        output_dir = "maps"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{metric}_{year}_map.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Created map for {metric} in {year}")


def inspect_grib_file(file_path):
    try:
        print(f"\nInspecting GRIB file: {file_path}")
        ds = xr.open_dataset(file_path, engine='cfgrib')

        print("Dimensions:")
        for dim_name, dim in ds.dims.items():
            print(f"  {dim_name}: {dim}")

        print("\nCoordinates:")
        for coord_name, coord in ds.coords.items():
            print(f"  {coord_name}: shape={coord.shape}, dtype={coord.dtype}")
            if coord.size < 10:
                print(f"    Values: {coord.values}")
            else:
                print(f"    Range: {coord.values.min()} to {coord.values.max()}")

        print("\nData Variables:")
        for var_name, var in ds.data_vars.items():
            print(f"  {var_name}: shape={var.shape}, dtype={var.dtype}")
            if hasattr(var, 'standard_name'):
                print(f"    Standard name: {var.standard_name}")
            if hasattr(var, 'long_name'):
                print(f"    Long name: {var.long_name}")
            if hasattr(var, 'units'):
                print(f"    Units: {var.units}")

        ds.close()
    except Exception as e:
        print(f"Error inspecting file {file_path}: {e}")


def main():
    grib_files = glob.glob(os.path.join(DATA_DIR, "*.grib")) + \
                 glob.glob(os.path.join(DATA_DIR, "*.grib2"))

    if not grib_files:
        print(f"No GRIB files found in {DATA_DIR}")
        return

    print(f"Found {len(grib_files)} GRIB files: {grib_files}")

    if len(grib_files) > 0:
        inspect_grib_file(grib_files[0])

    df = process_locations_data()

    if df is None or len(df) == 0:
        print("No data available to create maps")
        return

    for metric in variables.keys():
        create_yearly_maps(df, metric)
        print(f"Completed maps for {metric}")


if __name__ == "__main__":
    main()
