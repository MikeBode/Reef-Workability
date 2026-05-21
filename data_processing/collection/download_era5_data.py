import cdsapi
import pathlib
import multiprocessing

def download_single_year(year: int, variable, temp_download_path: pathlib.Path, download_path: pathlib.Path):
    if download_path.exists():
        print(f"{download_path} already exists. Aborting download")
        return
    print(f"Downloading {download_path}")

    dataset = "reanalysis-era5-single-levels"

    request = {
        "product_type": ["reanalysis"],
        "variable": [variable],
        "year": [str(year)],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": [
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00"
        ],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [-10.5, 113, -44, 154]
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request, temp_download_path)
    temp_download_path.rename(download_path)

def download_all_era5_data(download_base_path: pathlib.Path):
    download_base_path.mkdir(parents=True, exist_ok=True)

    variables = ["10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_wave_direction",
        "mean_wave_period",
        "significant_height_of_combined_wind_waves_and_swell",
        "total_precipitation"]
    years = [i for i in range(2005, 2026)]
    download_calls = []

    for year in years:
        for variable in variables:
            download_calls.append((year, variable, download_base_path / f"{year}_{variable}.nc.temp", download_base_path / f"{year}_{variable}.nc"))
    
    with multiprocessing.Pool(processes=10) as pool:
        pool.starmap(download_single_year, download_calls)
