from collection.download_nopp_hindcast import download_files as download_nopp_hindcast_files
import pathlib

def download_and_process_all_data(download_folder: pathlib.Path = pathlib.Path(__file__).parent.parent / "Data"):
    # First we download the NOPP hindcast data.
    download_nopp_hindcast_files(1990, 2008, download_folder / "nopp-phase2", max_workers=5)

    # Then we download WHACS data.

if __name__ == "__main__":
    download_and_process_all_data()