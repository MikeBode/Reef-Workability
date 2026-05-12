from visualization.wind_wave_constraint_analysis import WindWaveConstraintAnalysis
from models.model_training import ModelAndCalibrationCurve
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib

def generate_plot_for_month_waveheight(wave_height, month, constraint_analysis: WindWaveConstraintAnalysis, model_and_calibration_curve: ModelAndCalibrationCurve, maximum_u_wind, maximum_v_wind, save_directory):
    plt.title(f"Predicted Workability Probability for Month {month} at Wave Height {wave_height:.2f}m")
    plt.xlabel("U Wind Component (m/s?)")
    plt.ylabel("V Wind Component (m/s?)")

    # First, let's predict the probability of workability across a grid of u and v wind values, at the given wave height and month.
    grid_size = 100
    data = pd.DataFrame({
        "wave_height": [wave_height] * grid_size**2,
        "month": [month] * grid_size**2,
        "u_wind": np.linspace(0, maximum_u_wind, grid_size).repeat(grid_size),
        "v_wind": np.tile(np.linspace(0, maximum_v_wind, grid_size), grid_size)
    })
    data['wind_magnitude'] = np.hypot(data['u_wind'], data['v_wind'])
    predicted_probabilities = model_and_calibration_curve.predict_proba(data)
    plt.contourf(data['u_wind'], data['v_wind'], predicted_probabilities, levels=50, cmap='RdBu')

    magnitude_range = constraint_analysis.get_quantiles_for_month(month, "wind_magnitude")
    u_wind_range = constraint_analysis.get_quantiles_for_month(month, "u_wind")
    v_wind_range = constraint_analysis.get_quantiles_for_month(month, "v_wind")

    magnitude_min_xs = np.linspace(0, u_wind_range[1], grid_size)
    magnitude_min_ys = np.sqrt(np.maximum(0, magnitude_range[0]**2 - magnitude_min_xs**2))
    magnitude_max_xs = np.linspace(0, u_wind_range[1], grid_size)
    magnitude_max_ys = np.sqrt(np.maximum(0, magnitude_range[1]**2 - magnitude_max_xs**2))

    # Let's plot a box of realistic wind conditions based on our wind magnitude analysis.
    plt.plot(magnitude_min_xs, magnitude_min_ys, color='black', linestyle='-', zorder=10)
    plt.plot(magnitude_max_xs, magnitude_max_ys, color='black', linestyle='-', zorder=10)
    plt.plot([u_wind_range[0], u_wind_range[0]], [v_wind_range[0], v_wind_range[1]], color='black', linestyle='-', zorder=10)
    plt.plot([u_wind_range[1], u_wind_range[1]], [v_wind_range[0], v_wind_range[1]], color='black', linestyle='-', zorder=10)
    

    plt.legend()

    # Saving our plot.
    save_directory.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_directory / f"workability_heatmap_month_{month}_waveheight_{wave_height:.2f}.png")
    pass

def plot_workability_heatmaps_with_constraints(best_model_path, centroid_weather_data, save_directory=pathlib.Path("PlotOutputs/heatmaps")):
    with open(best_model_path, 'rb') as f:
        best_model = pickle.load(f)

    constraint_analysis = WindWaveConstraintAnalysis(centroid_weather_data)
    constraint_analysis.compute_constraints()

    for month in range(1, 13):
        wave_height_quantiles = constraint_analysis.get_quantiles_for_month(month, "wave_height")
        for wave_height in np.arange(int(wave_height_quantiles[0]*10)/10, int(wave_height_quantiles[1]*10)/10 + 0.1, 0.1):
            generate_plot_for_month_waveheight(wave_height, month, constraint_analysis, best_model, maximum_u_wind=20, maximum_v_wind=20, save_directory=save_directory)