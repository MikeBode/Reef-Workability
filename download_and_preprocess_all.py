from data_processing.collection.download_australian_whacs import download_whacs_files
from data_processing.processing.merge_reef_coordinates import merge_reef_datasets
from data_processing.extraction.extract_whacs_for_visits import construct_csvs_with_whacs_weather_data
from data_processing.processing.merge_visit_dfs import merge_visit_dfs
import pandas as pd
from models.model_training import train_and_evaluate_probability_models
from data_processing.extraction.extract_historical_whacs_for_centroids import extract_historical_whacs_for_centroids
from data_processing.processing.predict_reef_date_success_probabilities import predict_success_prob_for_reef_visits
from visualization.constrained_probability_heatmaps import plot_workability_heatmaps_with_constraints
from data_processing.collection.download_noaa_ww3 import download_noaa_ww3
from data_processing.processing.process_bpm_visits import process_bpm_visits
from data_processing.collection.download_era5_data import download_all_era5_data

import pathlib

def download_and_process_all_data(download_folder: pathlib.Path = pathlib.Path(__file__).parent / "Data"):
    # First we download WHACS data.
    print("Starting WHACS download, this may take a long while...")
    whacs_milestone = download_folder / "meta" / "whacs_download_complete.txt"
    if not whacs_milestone.exists():
        year_months = [(year, month) for year in range(2005, 2024) for month in range(1, 13)]
        download_whacs_files(year_months, download_folder / "whacs", max_workers=5)
        whacs_milestone.parent.mkdir(parents=True, exist_ok=True)
        whacs_milestone.touch()
    
    # Then the noaa data.
    #print("Starting NOAA WW3 download")
    #noaa_milestone = download_folder / "meta"/ "noaa_download_complete.txt"
    #if not noaa_milestone.exists():
    #    download_noaa_ww3(download_folder / "noaa_ww3")
    
    # Then ERA5 (will remove other sources soon if this works well).
    download_all_era5_data(download_folder / "era5")
    
    print("Downloads finished")

    # Now, we merge the coordinates from our surveyData and our COTS data.
    cots_with_coords = merge_reef_datasets(download_folder / 'surveyData[63].csv',
        download_folder / 'COTS INLOC Weather impacts.xlsx'
    )
    survey_data = pd.read_csv(download_folder / 'surveyData[63].csv')

    # We add wind/wave data to all of our datasets.
    bpm_visits_with_weather = process_bpm_visits(download_folder / "BPM_weather_failures.csv", download_folder / "noaa_ww3")

    dfs_with_whacs_weather = construct_csvs_with_whacs_weather_data([survey_data, cots_with_coords], download_folder / "whacs")

    merged_df = merge_visit_dfs(dfs_with_whacs_weather + [bpm_visits_with_weather], ["surveyData", "COTS", "BPM"], [True, False, False])
    merged_df.to_csv(download_folder / "combined_visits_with_weather.csv", index=False)

    # Training our models.
    best_model_path = download_folder / "best_model.pickle"
    model_stats = download_folder / "model_stats.pickle"
    if not best_model_path.exists():
        print("Training best unconstrained model")
        train_and_evaluate_probability_models(merged_df, best_model_path, model_stats)
    else:
        print("Best model already exists, skipping training.")
    
    # Now, let's do the setup for batch workability prediction.
    centroid_whacs_path = download_folder / "centroid_historical_whacs.csv"

    if centroid_whacs_path.exists():
        print(f"Loading cached historical WHACS data from {centroid_whacs_path}")
        historical_centroid_weather_data = pd.read_csv(centroid_whacs_path)
    else:
        print("Extracting historical WHACS data for centroids, this may take a long while...")
        historical_centroid_weather_data = extract_historical_whacs_for_centroids(
            pathlib.Path(__file__).parent / "Data" / "ReefCentroids.csv",
            download_folder / "whacs",
            download_folder / "partial_historical_whacs"
        )
        historical_centroid_weather_data.to_csv(centroid_whacs_path, index=False)
    
    # Output graphs
    plot_workability_heatmaps_with_constraints(best_model_path, historical_centroid_weather_data, save_directory=pathlib.Path("PlotOutputs/heatmaps"))

    # Predict workability for each centroid-day combination, using the best model.
    predicted_workability_path = download_folder / "predicted_workability_for_centroids.csv"
    if predicted_workability_path.exists():
        print(f"Loading cached predicted workability for centroids from {predicted_workability_path}")
        predicted_workability = pd.read_csv(predicted_workability_path)
    else:
        print("Predicting workability for each centroid-day combination, this also may take a long while...")
        predicted_workability = predict_success_prob_for_reef_visits(best_model_path, historical_centroid_weather_data)
        predicted_workability.to_csv(download_folder / "predicted_workability_for_centroids.csv", index=False)

if __name__ == "__main__":
    download_and_process_all_data()