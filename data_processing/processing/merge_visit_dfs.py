import pandas as pd
import numpy as np

def merge_visit_dfs(visit_dfs: list[pd.DataFrame], names: list[str], was_successful: list[bool]) -> pd.DataFrame:
    if len(visit_dfs) != len(names):
        raise ValueError("Length of visit_dfs and names must be the same.")
    
    if len(visit_dfs) != len(was_successful):
        raise ValueError("Length of visit_dfs and was_successful must be the same.")
    
    for df, name, success in zip(visit_dfs, names, was_successful):
        df['source_name'] = name
        df['was_successful'] = success
    
    merged_df = pd.concat(visit_dfs, ignore_index=True)
    
    merged_df['day_of_year'] = merged_df['date'].dt.strftime('%j').map(lambda l: int(l))
    merged_df['month'] = merged_df['date'].dt.month
    merged_df['year'] = merged_df['date'].dt.year
    merged_df['wind_magnitude'] = np.hypot(merged_df['u_wind'], merged_df['v_wind'])

    merged_df_nas_dropped = merged_df.dropna(subset=['wave_height', 'u_wind', 'v_wind'])

    if len(merged_df_nas_dropped) < len(merged_df):
        print(f"Warning: Dropping {len(merged_df) - len(merged_df_nas_dropped)} rows due to missing weather data.")

    return merged_df_nas_dropped