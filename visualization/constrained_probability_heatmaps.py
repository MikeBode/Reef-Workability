from visualization.wind_wave_constraint_analysis import WindWaveConstraintAnalysis
from models.model_training import ModelAndCalibrationCurve
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib

def generate_plot_for_month_waveheight(wave_height, month, constraint_analysis: WindWaveConstraintAnalysis, model_and_calibration_curve: ModelAndCalibrationCurve, maximum_u_wind, maximum_v_wind, save_directory):
    plt.title(f"Predicted Workability Probability for Month {month} at Wave Height {wave_height:.2f}m")
    plt.xlabel("U Wind Component (m/s)")
    plt.ylabel("V Wind Component (m/s)")

    # First, let's predict the probability of workability across a grid of u and v wind values, at the given wave height and month.
    grid_size = 100
    data = pd.DataFrame({
        "wave_height": [wave_height] * grid_size**2,
        "u_wind": np.linspace(-maximum_u_wind, maximum_u_wind, grid_size).repeat(grid_size),
        "v_wind": np.tile(np.linspace(-maximum_v_wind, maximum_v_wind, grid_size), grid_size),
    })
    data['wind_magnitude'] = np.hypot(data['u_wind'], data['v_wind'])
    data['month'] = month
    
    predicted_probabilities = model_and_calibration_curve.predict_proba(data)[:,1]
    contour = plt.contourf(np.linspace(-maximum_u_wind, maximum_u_wind, grid_size), np.linspace(-maximum_v_wind, maximum_v_wind, grid_size), predicted_probabilities.reshape(grid_size, grid_size), levels=20, cmap='RdBu', vmin=0, vmax=1)

    cbar = plt.colorbar(contour, ticks=[0, 0.25, 0.5, 0.75, 1])
    cbar.set_label('Dive Success Probability')
    
    magnitude_range = constraint_analysis.get_quantiles_for_month(month, "wind_magnitude")
    u_wind_range = constraint_analysis.get_quantiles_for_month(month, "u_wind")
    v_wind_range = constraint_analysis.get_quantiles_for_month(month, "v_wind")


    # Let's plot a box of realistic wind conditions based on our wind magnitude analysis.
    plt.plot([u_wind_range[0.01], u_wind_range[0.01]], [v_wind_range[0.01], v_wind_range[0.99]], color='black', linestyle='-', zorder=10)
    plt.plot([u_wind_range[0.99], u_wind_range[0.99]], [v_wind_range[0.01], v_wind_range[0.99]], color='black', linestyle='-', zorder=10)
    plt.plot([u_wind_range[0.01], u_wind_range[0.99]], [v_wind_range[0.01], v_wind_range[0.01]], color='black', linestyle='-', zorder=10)
    plt.plot([u_wind_range[0.01], u_wind_range[0.99]], [v_wind_range[0.99], v_wind_range[0.99]], color='black', linestyle='-', zorder=10)
    
    # Saving our plot.
    save_directory.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_directory / f"workability_heatmap_month_{month}_waveheight_{wave_height:.2f}.png")
    plt.close()

def plot_workability_heatmaps_with_constraints(best_model_path, centroid_weather_data, save_directory=pathlib.Path("PlotOutputs/heatmaps")):
    with open(best_model_path, 'rb') as f:
        best_model = pickle.load(f)

    constraint_analysis = WindWaveConstraintAnalysis(centroid_weather_data)
    constraint_analysis.compute_constraints()

    for month in range(1, 13):
        wave_height_quantiles = constraint_analysis.get_quantiles_for_month(month, "wave_height")
        for wave_height in np.arange(int(wave_height_quantiles[0.01]*10)/10, int(wave_height_quantiles[0.99]*10)/10 + 0.1, 0.1):
            generate_plot_for_month_waveheight(wave_height, month, constraint_analysis, best_model, maximum_u_wind=20, maximum_v_wind=20, save_directory=save_directory)