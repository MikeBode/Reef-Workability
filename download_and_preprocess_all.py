from data_processing.collection.download_nopp_hindcast import download_files as download_nopp_hindcast_files
from data_processing.collection.download_australian_whacs import download_whacs_files
from data_processing.processing.merge_reef_coordinates import merge_reef_datasets
from data_processing.extraction.extract_whacs_for_visits import construct_csvs_with_weather_data
from data_processing.processing.merge_visit_dfs import merge_visit_dfs
import pandas as pd
from models.model_training import train_and_evaluate_probability_models

import pathlib

# TODO: Look into failures. Make it throw on failure, accumulate all throws at end.

def download_and_process_all_data(download_folder: pathlib.Path = pathlib.Path(__file__).parent / "Data"):
    # This is a little hacky, but keeps us from repeating a download.
    nopp_milestone = download_folder / "meta" / "nopp_hindcast_download_complete.txt"
    if not nopp_milestone.exists():
        # First we download the NOPP hindcast data.
        download_nopp_hindcast_files(1990, 2008, download_folder / "nopp-phase2", max_workers=5)
        nopp_milestone.parent.mkdir(parents=True, exist_ok=True)
        nopp_milestone.touch()

    # Then we download WHACS data.
    print("Starting WHACS download, this may take a long while...")
    whacs_milestone = download_folder / "meta" / "whacs_download_complete.txt"
    if not whacs_milestone.exists():
        year_months = [(year, month) for year in range(2005, 2024) for month in range(1, 13)]
        download_whacs_files(year_months, download_folder / "whacs", max_workers=5)
        whacs_milestone.parent.mkdir(parents=True, exist_ok=True)
        whacs_milestone.touch()

    # Now, we merge the coordinates from our surveyData and our COTS data.
    cots_with_coords = merge_reef_datasets(download_folder / 'surveyData[63].csv',
        download_folder / 'COTS INLOC Weather impacts.xlsx'
    )
    survey_data = pd.read_csv(download_folder / 'surveyData[63].csv')

    # We add wind/wave data to both of our datasets.
    dfs_with_weather = construct_csvs_with_weather_data([survey_data, cots_with_coords], download_folder / "whacs")

    merged_df = merge_visit_dfs(dfs_with_weather, ["surveyData", "COTS"], [True, False])
    merged_df.to_csv(download_folder / "combined_visits_with_weather.csv", index=False)

    # Training our models.
    train_and_evaluate_probability_models(merged_df, pathlib.Path(__file__).parent / "Data" / "best_model.pickle")

if __name__ == "__main__":
    download_and_process_all_data()