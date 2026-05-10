import pathlib
from scipy.spatial import KDTree
import numpy as np
import xarray as xr

class WhacsWeatherExtractor:
    def __init__(self, data_base_path):
        self.data_base_path = pathlib.Path(data_base_path)

        self.dataset_cache = {}
        self.kdtree_cache = {}
        self.grid_cache = {}

    def get_nc_file_path(self, parameter, date):
        year_month = date.strftime('%Y%m')

        parameter_path: pathlib.Path = self.data_base_path / parameter
        if not parameter_path.exists():
            return None

        for file in parameter_path.iterdir():
            if file.is_file() and file.name.startswith(parameter) and year_month in file.name:
                return str(file)

        return None

    def load_and_cache_dataset(self, nc_file_path):
        path = pathlib.Path(nc_file_path)
        if nc_file_path is None or not path.exists():
            return None

        cache_key = nc_file_path

        if cache_key not in self.dataset_cache:
            try:
                print(f"Loading dataset: {path.name}")
                ds = xr.open_dataset(nc_file_path)
                self.dataset_cache[cache_key] = ds

                lon = ds.longitude.values
                lat = ds.latitude.values
                lon_grid, lat_grid = np.meshgrid(lon, lat)
                grid_points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))

                self.grid_cache[cache_key] = {
                    'lon_grid': lon_grid,
                    'lat_grid': lat_grid,
                    'grid_points': grid_points
                }

                self.kdtree_cache[cache_key] = KDTree(grid_points)

            except Exception as e:
                print(f"Error loading dataset {nc_file_path}: {e}")
                return None

        return self.dataset_cache[cache_key]
    
    def extract_batch_6_hours_mean_by_parameter(self, parameter, date, coords_array):
        nc_file_path = self.get_nc_file_path(parameter, date)
        return self.extract_batch_6_hours_mean(nc_file_path, date, coords_array, parameter)

    def extract_batch_6_hours_mean(self, nc_file_path, date, coords_array, param_name):
        ds = self.load_and_cache_dataset(nc_file_path)
        if ds is None:
            return np.full(len(coords_array), np.nan)

        try:
            date_str = date.strftime('%Y-%m-%d')
            start_time = f"{date_str}T06:00:00"
            end_time = f"{date_str}T12:00:00"

            ds_subset = ds.sel(time=slice(start_time, end_time))

            if len(ds_subset.time) == 0:
                return np.full(len(coords_array), np.nan)

            cache_key = nc_file_path
            tree = self.kdtree_cache[cache_key]
            grid_info = self.grid_cache[cache_key]

            _, indices = tree.query(coords_array)
            closest_indices = [np.unravel_index(idx, grid_info['lon_grid'].shape) for idx in indices]

            results = []
            for closest_i, closest_j in closest_indices:
                try:
                    mean_value = ds_subset[param_name].isel(
                        latitude=closest_i, longitude=closest_j
                    ).mean().item()
                    results.append(mean_value)
                except:
                    results.append(np.nan)

            return np.array(results)

        except Exception as e:
            print(f"Error processing batch for {date_str}: {e}")
            return np.full(len(coords_array), np.nan)

    def cleanup_cache(self):
        for ds in self.dataset_cache.values():
            try:
                ds.close()
            except:
                pass
        self.dataset_cache.clear()
        self.kdtree_cache.clear()
        self.grid_cache.clear()