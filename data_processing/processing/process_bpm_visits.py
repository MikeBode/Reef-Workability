import pandas as pd
import pathlib
from data_processing.extraction.weather_extractor import NoaaWW3Extractor
import numpy as np
import math
import re

def wind_to_u_v(wind_str):
    if not isinstance(wind_str, str) or "?" in wind_str:
        return math.nan, math.nan
    
    # The stated wind is a string description.
    # Something like "17-23 knots SE". The direction substring can be anywhere from 1 to 3 characters long with standard meanings.
    # We take the smaller number, and the direction, then convert that wind into a (u,v) representation.
    direction_map = {
        "N":   0,
        "NNE": 22.5,
        "NE":  45,
        "ENE": 67.5,
        "E":   90,
        "ESE": 112.5,
        "SE":  135,
        "SSE": 157.5,
        "S":   180,
        "SSW": 202.5,
        "SW":  225,
        "WSW": 247.5,
        "W":   270,
        "WNW": 292.5,
        "NW":  315,
        "NNW": 337.5,
    }

    # Extract the first (smaller) speed value
    speed_match = re.search(r'(\d+)', wind_str)
    if not speed_match:
        raise ValueError(f"No speed found in wind string: '{wind_str}'")
    speed = float(speed_match.group(1)) * 0.514444

    # Extract the direction (1–3 uppercase letters)
    dir_match = re.search(r'[A-Z]{1,3}', wind_str)
    if not dir_match:
        raise ValueError(f"No direction found in wind string: '{wind_str}'")
    direction = dir_match.group(0)

    if direction not in direction_map:
        raise ValueError(f"Unknown wind direction: '{direction}'")

    # Convert met convention: wind FROM bearing -> u/v components
    # bearing is degrees clockwise from North
    bearing_rad = math.radians(direction_map[direction])
    u = -speed * math.sin(bearing_rad)   # westward -> negative u
    v = -speed * math.cos(bearing_rad)   # southward -> negative v

    return (u, v)

def process_bpm_visits(bpm_failure_path: pathlib.Path, noaa_ww3_base_path: pathlib.Path):
    bpm_failures = pd.read_csv(bpm_failure_path)

    noaa_weather_extractor = NoaaWW3Extractor(noaa_ww3_base_path)

    bpm_failures["date"] = pd.to_datetime(bpm_failures['Date'], dayfirst=True)
    bpm_failures.drop(columns=["Date"], inplace=True)
    bpm_failures['wave_height'] = np.nan
    bpm_failures['u_wind'] = np.nan
    bpm_failures['v_wind'] = np.nan

    for idx, row in bpm_failures.iterrows():
        date = row['date']
        x_coord = row['x']
        y_coord = row['y']
        wave_height = noaa_weather_extractor.extract_batch_daytime_hours_mean_by_parameter("Thgt", date, np.array([[x_coord, y_coord]]))[0]
        u_wind, v_wind = wind_to_u_v(row["stated_wind"])
        
        bpm_failures.at[idx, 'wave_height'] = wave_height
        bpm_failures.at[idx, 'u_wind'] = u_wind
        bpm_failures.at[idx, 'v_wind'] = v_wind
    
    bpm_failures = bpm_failures[bpm_failures["u_wind"].notna() & bpm_failures["wave_height"].notna()]
    return bpm_failures