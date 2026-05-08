# Merges reef visit records with weather station data using a spatial grid search
# and KDTree nearest-neighbour lookup within a configurable radius and date window.

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from geopy.distance import geodesic
from scipy.spatial import KDTree


def merge_weather_data_grid(reef_file, weather_file, output_file, search_radius_km=50, date_window_days=1):
    reef_df = pd.read_excel(reef_file)
    weather_df = pd.read_csv(weather_file)

    print(f"Loaded reef dataset with {len(reef_df)} rows")
    print(f"Loaded weather dataset with {len(weather_df)} rows")
    print("Reef dataset columns:", reef_df.columns.tolist())

    date_column_candidates = ['Date', 'date', 'DATE', 'DateTime', 'Datetime']
    reef_date_column = None

    for col in date_column_candidates:
        if col in reef_df.columns:
            reef_date_column = col
            break

    if reef_date_column is None:
        for col in reef_df.columns:
            sample_values = reef_df[col].dropna().head(5).astype(str).tolist()
            if sample_values and any('/' in str(val) or '-' in str(val) for val in sample_values):
                reef_date_column = col
                print(f"Guessing date column as '{col}' based on content")
                print(f"Sample values: {sample_values}")
                break

    if reef_date_column is None:
        raise ValueError("Could not identify a date column in the reef dataset.")

    print(f"Using '{reef_date_column}' as the date column in reef dataset")

    missing_x_count = reef_df['x'].isna().sum()
    missing_y_count = reef_df['y'].isna().sum()
    print(f"Reef data has {missing_x_count} missing x values and {missing_y_count} missing y values")

    reef_df_with_coords = reef_df.dropna(subset=['x', 'y'])
    missing_coords = len(reef_df) - len(reef_df_with_coords)
    print(f"Dropped {missing_coords} reef entries without coordinates")

    date_formats = [
        '%d/%m/%Y %I:%M:%S %p',
        '%d/%m/%Y',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d'
    ]

    sample_dates = reef_df_with_coords[reef_date_column].dropna().head(5).astype(str).tolist()
    print(f"Sample date values: {sample_dates}")

    converted = False
    for date_format in date_formats:
        try:
            reef_df_with_coords['parsed_date'] = pd.to_datetime(
                reef_df_with_coords[reef_date_column],
                format=date_format,
                errors='raise'
            )
            print(f"Successfully parsed dates with format: {date_format}")
            converted = True
            break
        except (ValueError, TypeError) as e:
            print(f"Format {date_format} failed: {e}")
            continue

    if not converted:
        reef_df_with_coords['parsed_date'] = pd.to_datetime(
            reef_df_with_coords[reef_date_column],
            errors='coerce',
            dayfirst=True
        )
        print("Used flexible date parsing")

    nan_dates = reef_df_with_coords['parsed_date'].isna().sum()
    if nan_dates > 0:
        print(f"Warning: {nan_dates} dates could not be parsed and will be excluded")
        reef_df_with_coords = reef_df_with_coords.dropna(subset=['parsed_date'])

    try:
        weather_df['parsed_date'] = pd.to_datetime(weather_df['DATE'], errors='coerce')
    except Exception as e:
        print(f"Error converting weather dates: {e}")
        weather_df['parsed_date'] = pd.to_datetime(weather_df['DATE'], errors='coerce', dayfirst=True)

    weather_df = weather_df.dropna(subset=['LATITUDE', 'LONGITUDE', 'parsed_date'])

    weather_columns = ['SWELL_HGT', 'SWELL_PERIOD', 'WAVE_HGT', 'WAVE_PERIOD', 'WIND_SPEED']

    for col in weather_columns:
        missing_count = weather_df[col].isna().sum()
        missing_pct = (missing_count / len(weather_df)) * 100
        print(f"{col}: {missing_count} missing values ({missing_pct:.1f}%)")

    for col in weather_columns:
        reef_df_with_coords[col] = np.nan

    reef_df_with_coords['weather_stations_count'] = 0
    reef_df_with_coords['weather_avg_distance_km'] = np.nan
    reef_df_with_coords['weather_date_offset'] = np.nan

    def find_grid_weather_data(reef_row, date_offset=0):
        target_date = reef_row['parsed_date'] + timedelta(days=date_offset)
        date_weather = weather_df[weather_df['parsed_date'].dt.date == target_date.date()]

        if len(date_weather) == 0:
            return None

        weather_coords = date_weather[['LATITUDE', 'LONGITUDE']].values

        if len(weather_coords) > 0:
            tree = KDTree(weather_coords)
            reef_coords = np.array([reef_row['y'], reef_row['x']]).reshape(1, -1)
            radius_in_degrees = search_radius_km / 111.0
            indices = tree.query_ball_point(reef_coords[0], radius_in_degrees)

            if not indices:
                return None

            nearby_stations = date_weather.iloc[indices]
            distances = []
            valid_stations = []

            for idx, station in nearby_stations.iterrows():
                station_coords = (station['LATITUDE'], station['LONGITUDE'])
                reef_location = (reef_row['y'], reef_row['x'])
                try:
                    distance = geodesic(reef_location, station_coords).kilometers
                    if distance <= search_radius_km:
                        distances.append(distance)
                        valid_stations.append(station)
                except Exception as e:
                    print(f"Error calculating distance: {e}")
                    continue

            if not valid_stations:
                return None

            valid_stations_df = pd.DataFrame(valid_stations)
            result = {}
            for col in weather_columns:
                valid_values = valid_stations_df[col].dropna()
                result[col] = valid_values.mean() if len(valid_values) > 0 else np.nan

            result['stations_count'] = len(valid_stations)
            result['avg_distance_km'] = sum(distances) / len(distances) if distances else np.nan
            result['date_offset'] = date_offset

            return result

        return None

    total_rows = len(reef_df_with_coords)
    print(f"Processing {total_rows} reef entries to find matching weather data...")

    batch_size = max(1, min(1000, total_rows // 20))
    match_count = 0

    for i in range(0, total_rows, batch_size):
        end_idx = min(i + batch_size, total_rows)
        print(f"Processing entries {i + 1} to {end_idx} of {total_rows}...")

        batch_matches = 0

        for j in range(i, end_idx):
            if j >= total_rows:
                break

            reef_row = reef_df_with_coords.iloc[j]
            weather_data = find_grid_weather_data(reef_row, date_offset=0)

            if weather_data is None or all(pd.isna(weather_data.get(col, np.nan)) for col in weather_columns):
                for offset in range(-date_window_days, date_window_days + 1):
                    if offset == 0:
                        continue
                    weather_data = find_grid_weather_data(reef_row, date_offset=offset)
                    if weather_data is not None and any(
                        not pd.isna(weather_data.get(col, np.nan)) for col in weather_columns
                    ):
                        break

            if weather_data:
                for col in weather_columns:
                    if col in weather_data and not pd.isna(weather_data[col]):
                        reef_df_with_coords.at[reef_row.name, col] = weather_data[col]

                reef_df_with_coords.at[reef_row.name, 'weather_stations_count'] = weather_data['stations_count']
                reef_df_with_coords.at[reef_row.name, 'weather_avg_distance_km'] = weather_data['avg_distance_km']
                reef_df_with_coords.at[reef_row.name, 'weather_date_offset'] = weather_data['date_offset']

                batch_matches += 1
                match_count += 1

        print(f"  Found matches for {batch_matches} entries in this batch")

    match_rate = (match_count / total_rows) * 100
    print(f"\nFound matching weather data for {match_count} out of {total_rows} reef entries ({match_rate:.1f}%)")

    for col in weather_columns:
        filled_count = reef_df_with_coords[col].notna().sum()
        filled_pct = (filled_count / total_rows) * 100
        print(f"{col}: {filled_count} values filled ({filled_pct:.1f}%)")

    if match_count > 0:
        matched_rows = reef_df_with_coords['weather_stations_count'] > 0
        stations_stats = reef_df_with_coords.loc[matched_rows, 'weather_stations_count'].describe()
        distance_stats = reef_df_with_coords.loc[matched_rows, 'weather_avg_distance_km'].describe()

        print("\nWeather stations per reef statistics:")
        print(f"  Min: {stations_stats['min']:.0f}")
        print(f"  Mean: {stations_stats['mean']:.1f}")
        print(f"  Max: {stations_stats['max']:.0f}")

        print("\nDistance statistics (km) to weather stations:")
        print(f"  Min: {distance_stats['min']:.2f}")
        print(f"  Mean: {distance_stats['mean']:.2f}")
        print(f"  Median: {distance_stats['50%']:.2f}")
        print(f"  Max: {distance_stats['max']:.2f}")

    reef_df_with_coords = reef_df_with_coords.drop(columns=['parsed_date'])
    reef_df_with_coords.to_csv(output_file, index=False)
    print(f"\nMerged dataset saved to {output_file}")

    return reef_df_with_coords


merged_df = merge_weather_data_grid(
    'COTS INLOC Weather impacts-WithCoor.xlsx',
    '3984558.csv',
    'COTS_Weather_Complete.csv',
    search_radius_km=50,
    date_window_days=1
)
