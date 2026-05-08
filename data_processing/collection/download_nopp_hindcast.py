# Downloads NOPP-phase2 hindcast NetCDF files from the NOAA polar server for a specified
# year range using concurrent HTTP requests with per-file progress tracking.

import requests
from datetime import datetime, timedelta
from tqdm import tqdm
import concurrent.futures
import threading
import pathlib

BLOCK_SIZE = 8192 * 64


class DownloadProgress:
    def __init__(self):
        self._lock = threading.Lock()
        self.total_files = 0
        self.completed_files = 0
        self.start_time = datetime.now()

    def update(self):
        with self._lock:
            self.completed_files += 1
            elapsed_time = datetime.now() - self.start_time
            if self.completed_files > 0:
                avg_time_per_file = elapsed_time / self.completed_files
                estimated_total_time = avg_time_per_file * self.total_files
                remaining_time = estimated_total_time - elapsed_time
                print(f"Completed: {self.completed_files}/{self.total_files} files | Remaining time: {remaining_time}")


def download_file(url: str, folder: pathlib.Path, progress_tracker: DownloadProgress) -> bool:
    try:
        session = requests.Session()
        response = session.get(url, stream=True, timeout=30)

        if response.status_code == 200:
            filename = url.split('/')[-1]
            filepath: pathlib.Path = folder / filename

            if filepath.exists() and filepath.stat().st_size == int(response.headers.get('content-length', 0)):
                print(f"File {filename} already exists and is complete. Skipping download.")
                progress_tracker.update()
                return True

            total_size = int(response.headers.get('content-length', 0))

            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as progress_bar:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=BLOCK_SIZE):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))

            if total_size > 0 and filepath.stat().st_size != total_size:
                print(f"Size mismatch for {filename}. Download may be incomplete.")
                return False

            print(f"Successfully downloaded {filename}")
            progress_tracker.update()
            return True
        else:
            print(f"Failed to download {url}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False


def download_files(start_year: int, end_year: int, folder: pathlib.Path, max_workers: int = 5):
    folder.mkdir(parents=True, exist_ok=True)

    base_url = (
        "https://polar.ncep.noaa.gov/waves/hindcasts/nopp-phase2/{date}/partitions/"
        "multi_reanal.partition.oz_10m.{date}.nc"
    )

    download_tasks = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
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
        print("\nSome downloads failed. You may want to retry them.")