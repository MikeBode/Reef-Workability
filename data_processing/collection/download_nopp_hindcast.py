# Downloads NOPP-phase2 hindcast NetCDF files from the NOAA polar server for a specified
# year range using concurrent HTTP requests with per-file progress tracking.

import requests
from tqdm import tqdm
import concurrent.futures
import pathlib
from data_processing.collection.download_tools import DownloadProgress, download_file

def download_files(start_year: int, end_year: int, folder: pathlib.Path, max_workers: int = 5):
    folder.mkdir(parents=True, exist_ok=True)

    base_url = (
        "https://polar.ncep.noaa.gov/waves/hindcasts/nopp-phase2/{date}/partitions/"
        "multi_reanal.partition.oz_10m.{date}.nc"
    )

    download_tasks = []
    for year in range(start_year, end_year + 1):
        start_month = 1
        
        for month in range(start_month, 13):
            date = f"{year}{month:02d}"
            url = base_url.format(date=date)
            download_tasks.append(url)

    progress = DownloadProgress()
    progress.total_files = len(download_tasks)

    print(f"Starting download of {len(download_tasks)} files with {max_workers} concurrent workers")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_file, url, folder, progress)
            for url in download_tasks
        ]

        concurrent.futures.wait(futures)

    successful = sum(1 for future in futures if future.result())
    failed = len(download_tasks) - successful

    print(f"\nDownload summary:")
    print(f"- Total files: {len(download_tasks)}")
    print(f"- Successfully downloaded: {successful}")
    print(f"- Failed: {failed}")

    if failed > 0:
        exceptions = []
        for future in futures:
            ex = future.exception()  # This will raise any exceptions that occurred during download
            if ex:
                exceptions.append(ex)
        raise ExceptionGroup("One or more downloads failed", exceptions)