# Calculates realistic wind component ranges for given wave heights from WHACS hindcast data applied to reef centroids.
import pandas as pd

class WindWaveConstraintAnalysis:
    def __init__(self, reef_centroid_weather_df: pd.DataFrame):
        self.reef_centroid_weather_df = reef_centroid_weather_df

    def compute_constraints(self):
        # We get the 1st and 99th percentile of u, v, wind magnitude, and wave height for each month.
        # This is to draw a realistic box of weather conditions for later graphing.
        self.quantiles_by_month = self.reef_centroid_weather_df[["month", "wave_height", "u_wind", "v_wind", "wind_magnitude"]].groupby('month').quantile([0.01, 0.99]).unstack(level=1)
    
    def get_quantiles_for_month(self, month, attribute):
        if not hasattr(self, 'quantiles_by_month'):
            raise ValueError("Must call compute_constraints() before getting quantiles.")
        
        return self.quantiles_by_month.loc[month][attribute]
