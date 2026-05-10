# Extracts wave height, u-wind, and v-wind from WHACS hindcast NetCDF files for failed
# COTS survey visits between 2020 and 2024, using KDTree lookup to the nearest grid point.

import pandas as pd
import numpy as np
import pathlib

from whacs_weather_extractor import WhacsWeatherExtractor

def construct_csv_with_weather_data(cots_with_coords: pd.DataFrame, whacs_base_path: pathlib.Path) -> pd.DataFrame:
    # First let's filter our df down.
    cots_df = cots_with_coords.dropna(subset=['x', 'y'])
    dropped_num = len(cots_with_coords) - len(cots_df)
    if dropped_num > 0:
        print(f"Dropping {dropped_num} rows due to missing coordinates.")
    
    # The one absolute incompatibility between our surveyData and our COTS data is the name of the date column.
    # Otherwise, both can use this function just fine.
    if "Date" in cots_df.columns:
        cots_with_coords.rename(columns={"Date": "date"}, inplace=True)
        cots_with_coords['date'] = pd.to_datetime(cots_with_coords['date'], dayfirst=True, errors='raise')
    elif "date" in cots_df.columns:
        cots_with_coords['date'] = pd.to_datetime(cots_with_coords['date'], format="ISO8601", errors='raise')
    else:
        raise Exception("No date column found in COTS data.")
    
    filtered_df = cots_df[(cots_df['date'] >= '2020-01-01') & (cots_df['date'] < '2024-01-01')]
    dropped_num = len(cots_df) - len(filtered_df)
    if dropped_num > 0:
        print(f"Dropping {dropped_num} rows falling outside of 2020-2023.")
    
    cots_df = filtered_df

    if len(cots_df) == 0:
        raise Exception("No COTS visits found after filtering.")
    
    # Initializing our WhacsWeatherExtractor.
    whacs_extractor = WhacsWeatherExtractor(whacs_base_path)
    
    # Adding and filling our new weather columns.
    cots_df['wave_height'] = np.nan
    cots_df['u_wind'] = np.nan
    cots_df['v_wind'] = np.nan

    for idx, row in cots_df.iterrows():
        date = row['date']
        x_coord = row['x']
        y_coord = row['y']

        wave_height = whacs_extractor.extract_batch_6_hours_mean_by_parameter("hs", date, np.array([[x_coord, y_coord]]))[0]
        u_wind = whacs_extractor.extract_batch_6_hours_mean_by_parameter("uwnd", date, np.array([[x_coord, y_coord]]))[0]
        v_wind = whacs_extractor.extract_batch_6_hours_mean_by_parameter("vwnd", date, np.array([[x_coord, y_coord]]))[0]

        cots_df.at[idx, 'wave_height'] = wave_height
        cots_df.at[idx, 'u_wind'] = u_wind
        cots_df.at[idx, 'v_wind'] = v_wind

    return cots_df