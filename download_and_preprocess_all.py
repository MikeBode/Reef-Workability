from data_processing.processing.merge_reef_coordinates import merge_reef_datasets
from data_processing.extraction.extract_era5_for_visits import construct_csvs_with_era5_weather_data
from data_processing.processing.merge_visit_dfs import merge_visit_dfs
import pandas as pd
from models.model_training import train_and_evaluate_probability_models
from data_processing.extraction.extract_historical_era5_for_centroids import extract_historical_era5_for_centroids
from data_processing.processing.predict_reef_date_success_probabilities import predict_success_prob_for_reef_visits
from visualization.constrained_probability_heatmaps import plot_workability_heatmaps_with_constraints
from data_processing.collection.download_era5_data import download_all_era5_data
from data_processing.extraction.split_into_daily_csvs import split_into_daily_subsets
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

import pathlib


def download_and_process_all_data(download_folder: pathlib.Path = pathlib.Path(__file__).parent / "Data"):        
    download_all_era5_data(download_folder / "era5")
    
    print("Downloads finished")

    # Now, we merge the coordinates from our surveyData and our COTS data.
    cots_with_coords = merge_reef_datasets(download_folder / 'surveyData[63].csv',
        download_folder / 'COTS INLOC Weather impacts.xlsx'
    )
    survey_data = pd.read_csv(download_folder / 'surveyData[63].csv')
    bpm_data = pd.read_csv(download_folder / "BPM_weather_failures.csv")

    # We add wind/wave data to all of our datasets.
    dfs_with_whacs_weather = construct_csvs_with_era5_weather_data([survey_data, cots_with_coords, bpm_data], download_folder / "era5")

    merged_df = merge_visit_dfs(dfs_with_whacs_weather, ["AIMS", "COTS", "BPM"], [True, False, False])
    merged_df.to_csv(download_folder / "combined_visits_with_weather.csv", index=False)

    # Training our models.
    best_model_path = download_folder / "best_model.pickle"
    model_stats = download_folder / "model_stats.pickle"
    if not best_model_path.exists():
        print("Training best unconstrained model")
        train_and_evaluate_probability_models(merged_df, best_model_path, model_stats)
    else:
        print("Best model already exists, skipping training.")

    # Regenerate visualization notebook
    print("Regenerating notebook")
    with open("report_visualizations.ipynb") as ff:
        nb_in = nbformat.read(ff, nbformat.NO_CONVERT)
        
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    ep.preprocess(nb_in)
    
    with open('report_visualizations.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb_in, f)
    print("Notebook fully regenerated")
    
    # Now, let's do the setup for batch workability prediction.
    centroid_era5_path = download_folder / "centroid_historical_era5.csv"

    if centroid_era5_path.exists():
        print(f"Loading cached historical ERA5 data from {centroid_era5_path}")
        historical_centroid_weather_data = pd.read_csv(centroid_era5_path)
    else:
        print("Extracting historical ERA5 data for centroids, this may take a long while...")
        historical_centroid_weather_data = extract_historical_era5_for_centroids(
            pathlib.Path(__file__).parent / "Data" / "ReefCentroids.csv",
            download_folder / "era5",
            download_folder / "partial_historical_era5"
        )
        historical_centroid_weather_data.to_csv(centroid_era5_path, index=False)
    
    # Predict workability for each centroid-day combination, using the best model.
    predicted_workability_path = download_folder / "predicted_workability_for_centroids.csv"
    if predicted_workability_path.exists():
        print(f"Loading cached predicted workability for centroids from {predicted_workability_path}")
        predicted_workability = pd.read_csv(predicted_workability_path)
    else:
        print("Predicting workability for each centroid-day combination, this also may take a long while...")
        predicted_workability = predict_success_prob_for_reef_visits(best_model_path, historical_centroid_weather_data)
        predicted_workability.to_csv(download_folder / "predicted_workability_for_centroids.csv", index=False)
    predicted_workability["datetime"] = pd.to_datetime(predicted_workability["datetime"], format="%Y-%m-%d %H:%M:%S", errors='coerce')

    # Output daily predicted workabilities.
    #split_into_daily_subsets(2013, predicted_workability, download_folder / "yearly_subsetted_data")

    # Output graphs
    plot_workability_heatmaps_with_constraints(predicted_workability, save_directory=pathlib.Path("PlotOutputs/heatmaps"))

if __name__ == "__main__":
    download_and_process_all_data()