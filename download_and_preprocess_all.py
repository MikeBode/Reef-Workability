from data_processing.collection.download_nopp_hindcast import download_files as download_nopp_hindcast_files
from data_processing.collection.download_australian_whacs import download_whacs_files
from data_processing.processing.merge_reef_coordinates import merge_reef_datasets
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
    """     print("Starting WHACS download, this may take a long while...")
    whacs_milestone = download_folder / "meta" / "whacs_download_complete.txt"
    if not whacs_milestone.exists():
        year_months = [(year, month) for year in range(2020, 2024) for month in range(1, 13)]
        download_whacs_files(year_months, download_folder / "whacs", max_workers=1)    # For whatever reason, in my experience the WHACs data downloads at very low bitrates when done simultaneously.
        whacs_milestone.parent.mkdir(parents=True, exist_ok=True)
        whacs_milestone.touch() """

    # Now, we merge the coordinates from our surveyData and our COTS data.
    merge_reef_datasets(download_folder / 'surveyData[63].csv',
    download_folder / 'COTS INLOC Weather impacts.xlsx',
    download_folder / 'COTS INLOC Weather impacts-WithCoor.xlsx'
    )

if __name__ == "__main__":
    download_and_process_all_data()