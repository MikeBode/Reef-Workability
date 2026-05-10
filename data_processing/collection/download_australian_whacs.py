# Uwind: https://data-cbr.csiro.au/thredds/catalog/catch_all/ACS_WP3_WHACS/ACS_hindcast_DRS/gridded/release/WP3/WHACS/BoM-CSIRO/hindcast/ERA5/ERA5/WHACS/WWIII-v6.07/aust-R16/1hr/uwnd/catalog.html
# Vwind: https://data-cbr.csiro.au/thredds/catalog/catch_all/ACS_WP3_WHACS/ACS_hindcast_DRS/gridded/release/WP3/WHACS/BoM-CSIRO/hindcast/ERA5/ERA5/WHACS/WWIII-v6.07/aust-R16/1hr/vwnd/catalog.html
# Wave Height: https://data-cbr.csiro.au/thredds/catalog/catch_all/ACS_WP3_WHACS/ACS_hindcast_DRS/gridded/release/WP3/WHACS/BoM-CSIRO/hindcast/ERA5/ERA5/WHACS/WWIII-v6.07/aust-R16/1hr/hs/catalog.html

# We're only interested in WHACS data from the beginning of 2020, to the end of 2023 (which is as far as WHACs currently provides, as of writing).
from data_processing.collection.download_tools import DownloadProgress, download_file
from tqdm import tqdm
import concurrent.futures
import pathlib

def download_whacs_files(year_months: list[tuple[int, int]], base_folder: pathlib.Path, max_workers: int = 5):
    base_url = "https://data-cbr.csiro.au/thredds/fileServer/catch_all/ACS_WP3_WHACS/ACS_hindcast_DRS/gridded/release/WP3/WHACS/BoM-CSIRO/hindcast/ERA5/ERA5/WHACS/WWIII-v6.07/aust-R16/1hr"
    variables = ['uwnd', 'vwnd', 'hs']
    url_files = []

    for year, month in year_months:
        for var in variables:
            url = f"{base_url}/{var}/{var}_WHACS_hindcast_WHACS_ERA5_1hr_{year}{month:02d}010000-{year}{month:02d}312300.nc"
            url_files.append((url, base_folder / var))

    progress_tracker = DownloadProgress()
    progress_tracker.total_files = len(url_files)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_file, url, folder, progress_tracker) for url, folder in url_files]
        concurrent.futures.wait(futures)

        exceptions = []
        for future in futures:
            ex = future.exception()  # This will raise any exceptions that occurred during download
            if ex:
                exceptions.append(ex)
        
        if len(exceptions) > 0:
            raise ExceptionGroup("One or more downloads failed", exceptions)